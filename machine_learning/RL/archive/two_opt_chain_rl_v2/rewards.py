"""Reward functions for the v2 two-step chain RL setup.

This module keeps reward logic centralized so we can easily compare
different reward definitions without changing the environment logic.

The system now uses one consistent pattern:
- local step reward during the episode
- terminal correction based on final result versus the original circuit
"""

CURRENT_REWARD = "current_reward"
TWO_QUBIT_ONLY = "two_qubit_only"
DEPTH_ONLY = "depth_only"
MIXED_WEIGHTED = "mixed_weighted"
MIXED_EQUAL = "mixed_equal"
LEXICOGRAPHIC = "lexicographic"
NO_RUNTIME = "no_runtime"

SUPPORTED_REWARD_TYPES = [
    CURRENT_REWARD,
    TWO_QUBIT_ONLY,
    DEPTH_ONLY,
    MIXED_WEIGHTED,
    MIXED_EQUAL,
    LEXICOGRAPHIC,
    NO_RUNTIME,
]


def _value(row, key):
    """Read a field from either a pandas row-like object or a dict."""
    if isinstance(row, dict):
        return row[key]
    return getattr(row, key)


def _two_qubit_score(row):
    """Return current-step two-qubit reduction as a 0-1 style score."""
    return float(_value(row, "current_step_two_qubit_gate_reduction_pct")) / 100.0


def _depth_score(row):
    """Return current-step depth reduction as a 0-1 style score."""
    return float(_value(row, "current_step_depth_reduction_pct")) / 100.0


def _runtime_score(row):
    """Return the normalized current-step runtime penalty."""
    return float(_value(row, "runtime_score"))


def _total_two_qubit_score(row):
    """Return total two-qubit reduction versus original as a 0-1 style score."""
    return float(_value(row, "total_two_qubit_gate_reduction_pct")) / 100.0


def _total_depth_score(row):
    """Return total depth reduction versus original as a 0-1 style score."""
    return float(_value(row, "depth_reduction_pct")) / 100.0


def compute_step_reward(
    row,
    reward_type=CURRENT_REWARD,
    runtime_weight=0.1,
    two_qubit_weight=0.5,
    depth_weight=0.5,
):
    """Compute one step reward for the chosen reward type."""
    two_qubit_score = _two_qubit_score(row)
    depth_score = _depth_score(row)
    runtime_score = _runtime_score(row)

    if reward_type in {CURRENT_REWARD, TWO_QUBIT_ONLY}:
        return two_qubit_score - float(runtime_weight) * runtime_score

    if reward_type == DEPTH_ONLY:
        return depth_score - float(runtime_weight) * runtime_score

    if reward_type == MIXED_WEIGHTED:
        return (
            float(two_qubit_weight) * two_qubit_score
            + float(depth_weight) * depth_score
            - float(runtime_weight) * runtime_score
        )

    if reward_type == MIXED_EQUAL:
        return (
            0.5 * two_qubit_score
            + 0.5 * depth_score
            - float(runtime_weight) * runtime_score
        )

    if reward_type == LEXICOGRAPHIC:
        return (
            two_qubit_score
            + float(depth_weight) * depth_score
            - float(runtime_weight) * runtime_score
        )

    if reward_type == NO_RUNTIME:
        return two_qubit_score

    raise ValueError(
        f"Unsupported reward_type: {reward_type}. "
        f"Supported values: {SUPPORTED_REWARD_TYPES}"
    )


def compute_final_correction(
    row,
    reward_type=CURRENT_REWARD,
    final_reward_weight=0.5,
    two_qubit_weight=0.5,
    depth_weight=0.5,
):
    """Compute the terminal correction based on final result vs original."""
    total_two_qubit_score = _total_two_qubit_score(row)
    total_depth_score = _total_depth_score(row)

    if reward_type in {CURRENT_REWARD, TWO_QUBIT_ONLY, NO_RUNTIME}:
        return float(final_reward_weight) * total_two_qubit_score

    if reward_type == DEPTH_ONLY:
        return float(final_reward_weight) * total_depth_score

    if reward_type == MIXED_WEIGHTED:
        return float(final_reward_weight) * (
            float(two_qubit_weight) * total_two_qubit_score
            + float(depth_weight) * total_depth_score
        )

    if reward_type == MIXED_EQUAL:
        return float(final_reward_weight) * (
            0.5 * total_two_qubit_score + 0.5 * total_depth_score
        )

    if reward_type == LEXICOGRAPHIC:
        return float(final_reward_weight) * (
            total_two_qubit_score + float(depth_weight) * total_depth_score
        )

    return 0.0


def single_step_reward(
    row,
    reward_type=CURRENT_REWARD,
    runtime_weight=0.1,
    two_qubit_weight=0.5,
    depth_weight=0.5,
):
    """Reward for one optimization step from the current row alone."""
    return compute_step_reward(
        row,
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
    )


def incremental_step_reward(
    current_row,
    previous_row,
    reward_type=CURRENT_REWARD,
    runtime_weight=0.1,
    two_qubit_weight=0.5,
    depth_weight=0.5,
):
    """Reward for the second optimization step only."""
    del previous_row
    return compute_step_reward(
        current_row,
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
    )


def apply_reward(
    frame,
    reward_type=CURRENT_REWARD,
    runtime_weight=0.1,
    two_qubit_weight=0.5,
    depth_weight=0.5,
):
    """Add step-local reward bookkeeping columns to a dataframe.

    Training itself uses the environment's step rewards plus terminal
    final correction. This dataframe-level helper stores the local
    step reward only so analysis code still has a per-row reward view.
    """
    df = frame.copy()
    df["two_qubit_score"] = df["current_step_two_qubit_gate_reduction_pct"] / 100.0
    df["depth_score"] = df["current_step_depth_reduction_pct"] / 100.0
    df["local_step_reward"] = df.apply(
        lambda row: compute_step_reward(
            row,
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        ),
        axis=1,
    )
    df["utility_reward"] = df["local_step_reward"]
    return df
