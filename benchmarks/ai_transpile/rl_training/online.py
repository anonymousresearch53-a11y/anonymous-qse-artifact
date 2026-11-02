"""Online rollout helpers for conservative policy evaluation and collection."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qiskit import qasm2
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS

from ..chain_executor import ChainStep, execute_chain
from ..rl_trajectory.database import (
    CircuitRecord,
    OptimizerRecord,
    TrajectoryDatabase,
    TrajectoryStepRecord,
)
from ..rl_trajectory.grid_search import OPTIMIZER_CONFIGS
from ..rl_trajectory.reward import RewardConfig, compute_all_rewards, compute_improvement_percentage
from ..rl_trajectory.state import RLState
from ..transpilers import CircuitMetrics, analyze_circuit
from .normalization import NormalizationStats


def make_chain_step(optimizer_name: str) -> ChainStep:
    """Build a chain step for a named optimizer."""
    if optimizer_name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    opt_config = OPTIMIZER_CONFIGS[optimizer_name]
    return ChainStep(
        runner_type=str(opt_config["runner_type"]),
        options=dict(opt_config["options"]),
        name=optimizer_name,
    )


def normalize_state_for_policy(
    metrics: CircuitMetrics,
    circuit_record: CircuitRecord,
    *,
    steps_taken: int,
    time_budget_remaining: float,
    norm_stats: NormalizationStats,
    device: torch.device,
) -> torch.Tensor:
    """Convert raw circuit metrics into a normalized state tensor."""
    state = RLState.from_metrics(
        metrics=metrics,
        num_qubits=circuit_record.num_qubits,
        category=circuit_record.category,
        steps_taken=steps_taken,
        time_budget_remaining=time_budget_remaining,
    ).to_vector()
    normalized = norm_stats.normalize(state[None, :])[0]
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)


def trainer_score_distribution(
    trainer,  # noqa: ANN001
    state: torch.Tensor,
    *,
    mc_dropout_passes: int = 1,
) -> tuple[np.ndarray, float]:
    """Estimate action scores and uncertainty for a single state."""
    if state.dim() == 1:
        state = state.unsqueeze(0)

    num_passes = max(int(mc_dropout_passes), 1)
    samples: list[np.ndarray] = []

    if num_passes > 1:
        trainer._set_train_mode()
    else:
        trainer._set_eval_mode()

    try:
        with torch.no_grad():
            for _ in range(num_passes):
                scores = _forward_action_scores(trainer, state)
                probs = torch.softmax(scores, dim=-1)
                samples.append(probs.squeeze(0).detach().cpu().numpy())
    finally:
        trainer._set_train_mode()

    probs_arr = np.stack(samples, axis=0)
    mean_probs = probs_arr.mean(axis=0)
    uncertainty = float(probs_arr.std(axis=0).mean()) if len(samples) > 1 else 0.0
    return mean_probs, uncertainty


def select_action_with_uncertainty(
    trainer,  # noqa: ANN001
    state: torch.Tensor,
    *,
    exploration_rate: float = 0.0,
    uncertainty_threshold: float = 0.0,
    mc_dropout_passes: int = 1,
    rng: random.Random | None = None,
) -> tuple[int, dict[str, Any]]:
    """Select an action with optional uncertainty-triggered exploration."""
    rng = rng or random.Random()
    probs, uncertainty = trainer_score_distribution(
        trainer,
        state,
        mc_dropout_passes=mc_dropout_passes,
    )
    greedy_action = int(np.argmax(probs))

    explored = False
    action_idx = greedy_action
    should_explore = (
        exploration_rate > 0
        and uncertainty >= uncertainty_threshold
        and rng.random() < exploration_rate
    )
    if should_explore:
        action_idx = int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])
        explored = action_idx != greedy_action

    return action_idx, {
        "action_probabilities": probs.tolist(),
        "uncertainty": uncertainty,
        "greedy_action": greedy_action,
        "explored": explored,
    }


def rollout_policy(
    trainer,  # noqa: ANN001
    circuit_record: CircuitRecord,
    *,
    action_names: list[str],
    norm_stats: NormalizationStats,
    max_steps: int = 1,
    time_budget: float = 300.0,
    degradation_threshold: float | None = None,
    exploration_rate: float = 0.0,
    uncertainty_threshold: float = 0.0,
    mc_dropout_passes: int = 1,
    output_root: Path | None = None,
    save_intermediates: bool = False,
    reward_config: RewardConfig | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Roll out a policy on a single circuit."""
    rng = rng or random.Random()
    reward_cfg = reward_config or RewardConfig()
    circuit_kind = "artifact" if circuit_record.name.startswith("artifact_") else "original"

    result: dict[str, Any] = {
        "circuit_name": circuit_record.name,
        "circuit_id": circuit_record.id,
        "circuit_kind": circuit_kind,
        "success": False,
        "per_step": [],
    }

    if circuit_record.qasm_path is None:
        result["error"] = "missing_qasm_path"
        return result

    qasm_path = Path(circuit_record.qasm_path)
    if not qasm_path.exists():
        result["error"] = f"missing_qasm_file:{qasm_path}"
        return result

    try:
        current_circuit = qasm2.loads(qasm_path.read_text(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception as exc:  # pragma: no cover - parser failures depend on external inputs
        result["error"] = f"qasm_load_error:{exc}"
        return result

    initial_metrics = analyze_circuit(current_circuit)
    current_metrics = initial_metrics
    total_duration = 0.0
    chosen_actions: list[str] = []
    start_time = time.perf_counter()
    terminated_reason = "max_steps"
    reward_summaries = {
        "efficiency": 0.0,
        "efficiency_normalized": 0.0,
        "improvement_only": 0.0,
    }

    for step_index in range(max_steps):
        time_remaining = max(time_budget - total_duration, 0.0)
        if time_remaining <= 0:
            terminated_reason = "time_budget_exhausted"
            break

        state_tensor = normalize_state_for_policy(
            current_metrics,
            circuit_record,
            steps_taken=step_index,
            time_budget_remaining=time_remaining,
            norm_stats=norm_stats,
            device=trainer.device,
        )
        action_idx, selection_info = select_action_with_uncertainty(
            trainer,
            state_tensor,
            exploration_rate=exploration_rate,
            uncertainty_threshold=uncertainty_threshold,
            mc_dropout_passes=mc_dropout_passes,
            rng=rng,
        )
        optimizer_name = action_names[action_idx]
        chosen_actions.append(optimizer_name)

        chain_name = f"{circuit_record.name}_online_step{step_index}_{optimizer_name}"
        step_output_dir = None
        if output_root is not None:
            step_output_dir = output_root / circuit_record.name / chain_name

        try:
            chain_result = execute_chain(
                current_circuit,
                steps=[make_chain_step(optimizer_name)],
                chain_name=chain_name,
                output_dir=step_output_dir,
                save_intermediates=save_intermediates,
            )
        except Exception as exc:  # pragma: no cover - depends on external toolchain
            result["error"] = f"execution_error:{exc}"
            terminated_reason = "execution_error"
            break

        step_result = chain_result.step_results[0]
        next_metrics = step_result.output_metrics
        next_total_duration = total_duration + step_result.duration_seconds
        relative_degradation = 0.0
        if current_metrics.two_qubit_gates > 0 and next_metrics.two_qubit_gates > current_metrics.two_qubit_gates:
            relative_degradation = (
                next_metrics.two_qubit_gates - current_metrics.two_qubit_gates
            ) / current_metrics.two_qubit_gates
        is_final_step = (
            step_index == max_steps - 1
            or next_total_duration >= time_budget
            or (
                degradation_threshold is not None
                and relative_degradation > degradation_threshold
            )
        )
        rewards = compute_all_rewards(
            prev_metrics=current_metrics,
            new_metrics=next_metrics,
            time_cost=step_result.duration_seconds,
            initial_metrics=initial_metrics,
            is_final_step=is_final_step,
            config=reward_cfg,
            category=circuit_record.category,
            time_budget=time_budget,
            optimizer_name=optimizer_name,
        )
        reward_summaries["efficiency"] += rewards.efficiency
        reward_summaries["efficiency_normalized"] += rewards.efficiency_normalized
        reward_summaries["improvement_only"] += rewards.improvement_only

        result["per_step"].append({
            "step_index": step_index,
            "optimizer": optimizer_name,
            "duration_s": step_result.duration_seconds,
            "state_time_budget_remaining": time_remaining,
            "next_state_time_budget_remaining": max(time_budget - next_total_duration, 0.0),
            "initial_2q": current_metrics.two_qubit_gates,
            "final_2q": next_metrics.two_qubit_gates,
            "improvement": (
                (current_metrics.two_qubit_gates - next_metrics.two_qubit_gates)
                / max(current_metrics.two_qubit_gates, 1)
            ),
            "relative_degradation": relative_degradation,
            "reward_improvement_only": rewards.improvement_only,
            "reward_efficiency": rewards.efficiency,
            "reward_multi_objective": rewards.multi_objective,
            "reward_sparse_final": rewards.sparse_final,
            "reward_efficiency_normalized": rewards.efficiency_normalized,
            "action_probabilities": selection_info["action_probabilities"],
            "uncertainty": selection_info["uncertainty"],
            "explored": selection_info["explored"],
            "artifact_path": str(step_result.artifact_path) if step_result.artifact_path else None,
            "state_metrics": _metrics_to_dict(current_metrics),
            "next_state_metrics": _metrics_to_dict(next_metrics),
        })

        total_duration = next_total_duration
        current_circuit = chain_result.final_circuit
        current_metrics = next_metrics

        if degradation_threshold is not None and relative_degradation > degradation_threshold:
            terminated_reason = "degradation_threshold"
            break
        if total_duration >= time_budget:
            terminated_reason = "time_budget_exhausted"
            break

    final_improvement_pct = compute_improvement_percentage(
        initial_metrics.two_qubit_gates,
        current_metrics.two_qubit_gates,
    )
    result.update({
        "success": "error" not in result,
        "chain_name": "online_" + "__".join(chosen_actions) if chosen_actions else "online_empty",
        "optimizers": chosen_actions,
        "num_steps": len(chosen_actions),
        "terminated_reason": terminated_reason,
        "initial_2q": initial_metrics.two_qubit_gates,
        "final_2q": current_metrics.two_qubit_gates,
        "improvement": final_improvement_pct / 100.0,
        "improvement_pct": final_improvement_pct,
        "elapsed_s": time.perf_counter() - start_time,
        "total_duration_s": total_duration,
        "total_reward_efficiency": reward_summaries["efficiency"],
        "total_reward_efficiency_normalized": reward_summaries["efficiency_normalized"],
        "total_reward_improvement_only": reward_summaries["improvement_only"],
    })
    return result


def summarize_rollouts(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rollout results into summary metrics."""
    executed = [item for item in results if item.get("success")]
    improvements = [float(item["improvement"]) for item in executed]
    improved = sum(1 for value in improvements if value > 0)

    summary: dict[str, Any] = {
        "num_circuits": len(results),
        "num_executed": len(executed),
        "num_improved": improved,
        "success_rate": len(executed) / max(len(results), 1),
        "mean_2q_improvement": float(np.mean(improvements)) if improvements else 0.0,
        "max_2q_improvement": float(np.max(improvements)) if improvements else 0.0,
        "mean_num_steps": float(np.mean([item["num_steps"] for item in executed])) if executed else 0.0,
        "per_circuit": results,
    }

    by_kind: dict[str, dict[str, Any]] = {}
    for circuit_kind in ("original", "artifact"):
        kind_results = [item for item in executed if item.get("circuit_kind") == circuit_kind]
        kind_improvements = [float(item["improvement"]) for item in kind_results]
        by_kind[circuit_kind] = {
            "num_executed": len(kind_results),
            "mean_2q_improvement": float(np.mean(kind_improvements)) if kind_improvements else 0.0,
            "success_rate": len(kind_results) / max(
                len([r for r in results if r.get("circuit_kind") == circuit_kind]), 1
            ),
        }
    summary["by_circuit_kind"] = by_kind
    return summary


def sync_circuit_record(target_db: TrajectoryDatabase, circuit_record: CircuitRecord) -> CircuitRecord:
    """Ensure a circuit record exists in the target DB."""
    existing = target_db.get_circuit_by_name(circuit_record.name)
    if existing is not None:
        return existing

    circuit_id = target_db.insert_circuit(
        CircuitRecord(
            id=None,
            name=circuit_record.name,
            category=circuit_record.category,
            source=circuit_record.source,
            qasm_path=circuit_record.qasm_path,
            num_qubits=circuit_record.num_qubits,
            initial_depth=circuit_record.initial_depth,
            initial_two_qubit_gates=circuit_record.initial_two_qubit_gates,
            initial_two_qubit_depth=circuit_record.initial_two_qubit_depth,
            initial_total_gates=circuit_record.initial_total_gates,
            gate_density=circuit_record.gate_density,
            two_qubit_ratio=circuit_record.two_qubit_ratio,
        )
    )
    synced = target_db.get_circuit_by_id(circuit_id)
    if synced is None:
        raise RuntimeError(f"Failed to sync circuit {circuit_record.name}")
    return synced


def sync_optimizers(target_db: TrajectoryDatabase, action_names: list[str]) -> dict[str, int]:
    """Ensure rollout optimizers exist in the target DB."""
    mapping: dict[str, int] = {}
    for action_name in action_names:
        if action_name not in OPTIMIZER_CONFIGS:
            raise ValueError(f"Unknown optimizer in action space: {action_name}")
        opt_cfg = OPTIMIZER_CONFIGS[action_name]
        optimizer_id = target_db.get_or_create_optimizer(
            OptimizerRecord(
                id=None,
                name=action_name,
                runner_type=str(opt_cfg["runner_type"]),
                options=dict(opt_cfg["options"]),
                description=str(opt_cfg.get("description", "")),
            )
        )
        mapping[action_name] = optimizer_id
    return mapping


def record_rollout(
    target_db: TrajectoryDatabase,
    circuit_record: CircuitRecord,
    rollout: dict[str, Any],
    *,
    action_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> int:
    """Persist a rollout into a trajectory database."""
    if not rollout.get("success"):
        raise ValueError("Only successful rollouts can be recorded")
    if not rollout.get("per_step"):
        raise ValueError("Cannot record an empty rollout")

    synced_circuit = sync_circuit_record(target_db, circuit_record)
    optimizer_ids = sync_optimizers(target_db, action_names)
    circuit_id = synced_circuit.id
    if circuit_id is None:
        raise RuntimeError("Synced circuit is missing an ID")

    chain_name = str(rollout["chain_name"])
    if target_db.trajectory_exists(circuit_id, chain_name):
        return -1

    trajectory_id = target_db.insert_trajectory(
        circuit_id=circuit_id,
        chain_name=chain_name,
        num_steps=int(rollout["num_steps"]),
        initial_depth=int(rollout["per_step"][0]["state_metrics"]["depth"]),
        initial_two_qubit_gates=int(rollout["initial_2q"]),
        initial_two_qubit_depth=int(rollout["per_step"][0]["state_metrics"]["two_qubit_depth"]),
        initial_total_gates=int(rollout["per_step"][0]["state_metrics"]["total_gates"]),
        final_depth=int(rollout["per_step"][-1]["next_state_metrics"]["depth"]),
        final_two_qubit_gates=int(rollout["final_2q"]),
        final_two_qubit_depth=int(rollout["per_step"][-1]["next_state_metrics"]["two_qubit_depth"]),
        final_total_gates=int(rollout["per_step"][-1]["next_state_metrics"]["total_gates"]),
        total_duration_seconds=float(rollout["total_duration_s"]),
        total_reward=float(rollout["total_reward_efficiency"]),
        improvement_percentage=float(rollout["improvement_pct"]),
        metadata={
            "circuit_kind": rollout["circuit_kind"],
            "terminated_reason": rollout["terminated_reason"],
            "optimizers": rollout["optimizers"],
            **(metadata or {}),
        },
    )

    for step in rollout["per_step"]:
        optimizer_name = str(step["optimizer"])
        optimizer_id = optimizer_ids[optimizer_name]
        state_metrics = step["state_metrics"]
        next_state_metrics = step["next_state_metrics"]

        state = RLState.from_metrics(
            metrics=_metrics_from_dict(state_metrics),
            num_qubits=synced_circuit.num_qubits,
            category=synced_circuit.category,
            steps_taken=int(step["step_index"]),
            time_budget_remaining=float(step["state_time_budget_remaining"]),
        )
        next_state = RLState.from_metrics(
            metrics=_metrics_from_dict(next_state_metrics),
            num_qubits=synced_circuit.num_qubits,
            category=synced_circuit.category,
            steps_taken=int(step["step_index"]) + 1,
            time_budget_remaining=float(step["next_state_time_budget_remaining"]),
        )

        target_db.insert_trajectory_step(
            TrajectoryStepRecord(
                trajectory_id=trajectory_id,
                step_index=int(step["step_index"]),
                optimizer_id=optimizer_id,
                state_depth=int(state_metrics["depth"]),
                state_two_qubit_gates=int(state_metrics["two_qubit_gates"]),
                state_two_qubit_depth=int(state_metrics["two_qubit_depth"]),
                state_total_gates=int(state_metrics["total_gates"]),
                state_num_qubits=synced_circuit.num_qubits,
                state_gate_density=state.gate_density,
                state_two_qubit_ratio=state.two_qubit_ratio,
                state_steps_taken=int(step["step_index"]),
                state_time_budget_remaining=float(state.time_budget_remaining),
                state_category=state.category_encoding,
                next_state_depth=int(next_state_metrics["depth"]),
                next_state_two_qubit_gates=int(next_state_metrics["two_qubit_gates"]),
                next_state_two_qubit_depth=int(next_state_metrics["two_qubit_depth"]),
                next_state_total_gates=int(next_state_metrics["total_gates"]),
                next_state_gate_density=next_state.gate_density,
                next_state_two_qubit_ratio=next_state.two_qubit_ratio,
                next_state_steps_taken=int(step["step_index"]) + 1,
                next_state_time_budget_remaining=float(next_state.time_budget_remaining),
                reward_improvement_only=float(step["reward_improvement_only"]),
                reward_efficiency=float(step["reward_efficiency"]),
                reward_multi_objective=float(step["reward_multi_objective"]),
                reward_sparse_final=float(step["reward_sparse_final"]),
                reward_efficiency_normalized=float(step["reward_efficiency_normalized"]),
                done=int(step["step_index"]) == int(rollout["num_steps"]) - 1,
                duration_seconds=float(step["duration_s"]),
            )
        )

    return trajectory_id


def _forward_action_scores(trainer, state: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    if hasattr(trainer, "policy"):
        return trainer.policy(state)
    if hasattr(trainer, "q_network"):
        return trainer.q_network(state)
    if hasattr(trainer, "network"):
        target_return = float(getattr(trainer.config, "target_return", 0.3))
        rtg_seq = torch.tensor([[[target_return]]], dtype=torch.float32, device=trainer.device)
        action_seq = torch.zeros(1, 1, dtype=torch.long, device=trainer.device)
        timestep_seq = torch.zeros(1, 1, dtype=torch.long, device=trainer.device)
        return trainer.network(rtg_seq, state.unsqueeze(1), action_seq, timestep_seq)[:, 0, :]
    raise ValueError("Unsupported trainer type for action scoring")


def _metrics_to_dict(metrics: CircuitMetrics) -> dict[str, int]:
    return {
        "depth": int(metrics.depth),
        "two_qubit_gates": int(metrics.two_qubit_gates),
        "two_qubit_depth": int(metrics.two_qubit_depth),
        "total_gates": int(metrics.total_gates),
    }


def _metrics_from_dict(data: dict[str, Any]) -> CircuitMetrics:
    return CircuitMetrics(
        depth=int(data["depth"]),
        two_qubit_gates=int(data["two_qubit_gates"]),
        two_qubit_depth=int(data["two_qubit_depth"]),
        total_gates=int(data["total_gates"]),
    )
