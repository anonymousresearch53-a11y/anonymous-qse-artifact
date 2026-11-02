"""Notebook helpers for llm_v3 experiment evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from machine_learning.LLM.llm_v3.data import (
        ORIGINAL_CHAIN_NAME,
        candidate_rows_for_circuit,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from machine_learning.LLM.llm_v3.evaluate import score_circuit_candidates
    from machine_learning.LLM.llm_v3.objective import ObjectiveConfig
except ImportError:
    from .data import (
        ORIGINAL_CHAIN_NAME,
        candidate_rows_for_circuit,
        load_dataset_bundle,
        split_frame_for_name,
    )
    from .evaluate import score_circuit_candidates
    from .objective import ObjectiveConfig


PAPER_BG = "#FAFBFD"
PAPER_PANEL_BG = "#F2F5FA"
PAPER_BAR = "#5B7DB1"
PAPER_ALT_BAR = "#86A8D7"
PAPER_RUNTIME = "#6FAFA0"
PAPER_GRID = "#D7DEEA"
PAPER_TEXT = "#243042"
PAPER_EDGE = "#A4B3C7"
EMPTY_GROUP_SUMMARY_COLUMNS = [
    "group",
    "circuits",
    "mean_objective_score",
    "mean_regret",
    "original_selection_rate",
    "shortlist_oracle_recall",
    "mean_confidence",
]


def resolve_latest_run(artifact_root: Path, split_name: str) -> Path:
    """Return the most recent saved run for a given split."""
    candidates: list[tuple[str, Path]] = []
    for run_dir in sorted(artifact_root.iterdir()) if artifact_root.exists() else []:
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "predictions.csv"
        if not (manifest_path.exists() and summary_path.exists() and predictions_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        if str(manifest.get("split")) != split_name:
            continue
        candidates.append((str(manifest.get("created_at_utc", run_dir.name)), run_dir))
    if not candidates:
        raise FileNotFoundError(f"No saved runs found for split={split_name}.")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Load one saved run manifest."""
    return json.loads((run_dir / "manifest.json").read_text())


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load flat prediction rows for one saved run."""
    return pd.read_csv(run_dir / "predictions.csv")


def load_summary(run_dir: Path) -> dict[str, Any]:
    """Load aggregate summary for one saved run."""
    return json.loads((run_dir / "summary.json").read_text())


def objective_config_from_manifest(manifest: dict[str, Any]) -> ObjectiveConfig:
    """Rebuild the deterministic objective config used for a saved run."""
    objective = manifest["objective"]
    return ObjectiveConfig(
        reward_type=objective["reward_type"],
        runtime_weight=float(objective["runtime_weight"]),
        two_qubit_weight=float(objective["two_qubit_weight"]),
        depth_weight=float(objective["depth_weight"]),
        final_reward_weight=float(objective["final_reward_weight"]),
        objective_version=str(objective["objective_version"]),
    )


def score_split_frame(frame: pd.DataFrame, objective_config: ObjectiveConfig) -> pd.DataFrame:
    """Score every circuit in a split frame for all candidate actions."""
    scored_frames: list[pd.DataFrame] = []
    for circuit_id in sorted(frame["original_circuit"].astype(str).unique()):
        circuit_rows = candidate_rows_for_circuit(frame, circuit_id)
        scored_rows = score_circuit_candidates(circuit_rows, objective_config).copy()
        scored_rows["original_circuit"] = circuit_id
        scored_frames.append(scored_rows)
    return pd.concat(scored_frames, ignore_index=True)


def align_selected_rows(
    eval_scored: pd.DataFrame,
    predictions: pd.DataFrame,
    method_name: str,
) -> pd.DataFrame:
    """Join saved predictions back onto the scored evaluation rows."""
    selected_rows = eval_scored.merge(
        predictions,
        left_on=["original_circuit", "chain_id"],
        right_on=["circuit_id", "selected_chain_id"],
        how="inner",
        suffixes=("", "_pred"),
    )
    if len(selected_rows) != len(predictions):
        missing = sorted(
            set(predictions["circuit_id"].astype(str)) - set(selected_rows["original_circuit"].astype(str))
        )
        raise ValueError(
            f"Expected {len(predictions)} selected rows after merge, found {len(selected_rows)}. "
            f"Missing circuits: {missing[:10]}"
        )
    selected_rows = selected_rows.copy()
    selected_rows["method"] = method_name
    selected_rows["selected_chain"] = selected_rows["chain_name"]
    selected_rows["confidence"] = pd.to_numeric(
        selected_rows.get("confidence"),
        errors="coerce",
    )
    selected_rows["used_fallback"] = (
        selected_rows.get("used_fallback", False).astype(str).str.lower().eq("true")
    )
    return selected_rows.sort_values("original_circuit").reset_index(drop=True)


def fixed_chain_rows(frame: pd.DataFrame, chain_name: str, method_name: str) -> pd.DataFrame:
    """Return the rows corresponding to one fixed baseline chain."""
    rows = frame[frame["chain_name"] == chain_name].copy()
    rows["method"] = method_name
    rows["selected_chain"] = rows["chain_name"]
    rows["regret"] = np.nan
    rows["used_fallback"] = False
    rows["confidence"] = np.nan
    return rows


def best_possible_rows(frame: pd.DataFrame, column: str, method_name: str) -> pd.DataFrame:
    """Return the per-circuit oracle rows for one target metric."""
    rows = frame.loc[frame.groupby("original_circuit")[column].idxmax()].copy()
    rows["method"] = method_name
    rows["selected_chain"] = rows["chain_name"]
    rows["regret"] = 0.0 if column == "objective_score" else np.nan
    rows["used_fallback"] = False
    rows["confidence"] = np.nan
    return rows


def summarize_selected(rows: pd.DataFrame, method_name: str) -> dict[str, Any]:
    """Summarize one selected-row frame into a single table row."""
    return {
        "method": method_name,
        "circuits": len(rows),
        "avg_objective_score": float(pd.to_numeric(rows["objective_score"], errors="coerce").mean()),
        "avg_regret": float(pd.to_numeric(rows["regret"], errors="coerce").mean())
        if rows["regret"].notna().any()
        else np.nan,
        "avg_two_qubit_gate_reduction_pct": float(
            pd.to_numeric(rows["total_two_qubit_gate_reduction_pct"], errors="coerce").mean()
        ),
        "avg_depth_reduction_pct": float(
            pd.to_numeric(rows["total_depth_reduction_pct"], errors="coerce").mean()
        ),
        "avg_runtime_seconds": float(
            pd.to_numeric(rows["total_runtime_seconds"], errors="coerce").mean()
        ),
        "avg_runtime_score": float(
            pd.to_numeric(rows["total_runtime_score"], errors="coerce").mean()
        ),
        "fallback_rate": float(
            pd.to_numeric(rows.get("used_fallback", False), errors="coerce").fillna(0).mean()
        ),
        "mean_confidence": float(pd.to_numeric(rows.get("confidence"), errors="coerce").mean())
        if "confidence" in rows
        else np.nan,
        "original_selection_rate": float(
            pd.to_numeric(rows.get("selected_is_original"), errors="coerce").mean()
        )
        if "selected_is_original" in rows
        else np.nan,
        "shortlist_oracle_recall": float(
            pd.to_numeric(rows.get("oracle_in_shortlist"), errors="coerce").mean()
        )
        if "oracle_in_shortlist" in rows
        else np.nan,
        "mean_oracle_shortlist_rank": float(
            pd.to_numeric(rows.get("oracle_shortlist_rank"), errors="coerce").mean()
        )
        if "oracle_shortlist_rank" in rows
        else np.nan,
    }


def build_model_evaluator_tables(
    run_dir: Path,
) -> dict[str, Any]:
    """Load one run and build the tables needed for the model evaluator notebook."""
    manifest = load_manifest(run_dir)
    predictions = load_predictions(run_dir)
    bundle = load_dataset_bundle()
    objective_config = objective_config_from_manifest(manifest)

    train_df = split_frame_for_name(bundle, "train")
    eval_df = split_frame_for_name(bundle, str(manifest["split"]))
    predicted_ids = set(predictions["circuit_id"].astype(str))
    eval_df = eval_df[eval_df["original_circuit"].astype(str).isin(predicted_ids)].copy()

    train_scored = score_split_frame(train_df, objective_config)
    eval_scored = score_split_frame(eval_df, objective_config)

    train_chain_means = (
        train_scored.groupby(["chain_id", "chain_name"], as_index=False)["objective_score"]
        .mean()
        .sort_values(["objective_score", "chain_id"], ascending=[False, True])
    )
    global_best_average_chain_name = str(train_chain_means.iloc[0]["chain_name"])
    method_name = f"LLM v3 ({manifest.get('model_name', 'unknown model')})"
    selected_rows = align_selected_rows(eval_scored, predictions, method_name)
    selection_checks = pd.DataFrame(
        [
            {
                "selected_circuits": int(len(selected_rows)),
                "expected_circuits": int(len(predictions)),
                "mean_objective_score": float(
                    pd.to_numeric(selected_rows["objective_score"], errors="coerce").mean()
                ),
                "mean_regret": float(
                    pd.to_numeric(selected_rows["regret"], errors="coerce").mean()
                ),
                "fallback_rate": float(
                    pd.to_numeric(selected_rows.get("used_fallback"), errors="coerce").mean()
                ),
                "mean_confidence": float(
                    pd.to_numeric(selected_rows.get("confidence"), errors="coerce").mean()
                ),
                "shortlist_oracle_recall": float(
                    pd.to_numeric(selected_rows.get("oracle_in_shortlist"), errors="coerce").mean()
                ),
                "mean_oracle_shortlist_rank": float(
                    pd.to_numeric(selected_rows.get("oracle_shortlist_rank"), errors="coerce").mean()
                ),
            }
        ]
    )

    single_chains = sorted(
        name
        for name in train_scored[train_scored["length"] == 1]["chain_name"].astype(str).unique()
        if name != ORIGINAL_CHAIN_NAME
    )
    double_chains = sorted(
        name
        for name in train_scored[train_scored["length"] == 2]["chain_name"].astype(str).unique()
    )
    best_fixed_reward_chain = str(
        train_scored.groupby("chain_name")["objective_score"].mean().sort_values(ascending=False).index[0]
    )
    best_single_depth = str(
        train_scored[train_scored["chain_name"].isin(single_chains)]
        .groupby("chain_name")["total_depth_reduction_pct"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    best_single_twoq = str(
        train_scored[train_scored["chain_name"].isin(single_chains)]
        .groupby("chain_name")["total_two_qubit_gate_reduction_pct"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    best_chain_depth = str(
        train_scored[train_scored["chain_name"].isin(double_chains)]
        .groupby("chain_name")["total_depth_reduction_pct"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    best_chain_twoq = str(
        train_scored[train_scored["chain_name"].isin(double_chains)]
        .groupby("chain_name")["total_two_qubit_gate_reduction_pct"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    baseline_frames = {
        f"Best fixed reward ({best_fixed_reward_chain})": fixed_chain_rows(
            eval_scored, best_fixed_reward_chain, f"Best fixed reward ({best_fixed_reward_chain})"
        ),
        f"Best depth chain ({best_chain_depth})": fixed_chain_rows(
            eval_scored, best_chain_depth, f"Best depth chain ({best_chain_depth})"
        ),
        f"Best 2Q chain ({best_chain_twoq})": fixed_chain_rows(
            eval_scored, best_chain_twoq, f"Best 2Q chain ({best_chain_twoq})"
        ),
        f"Best depth single ({best_single_depth})": fixed_chain_rows(
            eval_scored, best_single_depth, f"Best depth single ({best_single_depth})"
        ),
        f"Best 2Q single ({best_single_twoq})": fixed_chain_rows(
            eval_scored, best_single_twoq, f"Best 2Q single ({best_single_twoq})"
        ),
        "Best possible objective": best_possible_rows(
            eval_scored, "objective_score", "Best possible objective"
        ),
        "Best possible depth": best_possible_rows(
            eval_scored, "total_depth_reduction_pct", "Best possible depth"
        ),
        "Best possible 2Q": best_possible_rows(
            eval_scored,
            "total_two_qubit_gate_reduction_pct",
            "Best possible 2Q",
        ),
    }

    comparison = pd.DataFrame(
        [summarize_selected(selected_rows, method_name)]
        + [summarize_selected(frame, label) for label, frame in baseline_frames.items()]
    ).sort_values("avg_objective_score", ascending=False).reset_index(drop=True)

    return {
        "manifest": manifest,
        "predictions": predictions,
        "train_scored": train_scored,
        "eval_scored": eval_scored,
        "selected_rows": selected_rows,
        "selection_checks": selection_checks,
        "baseline_frames": baseline_frames,
        "comparison": comparison,
        "global_best_average_chain_name": global_best_average_chain_name,
        "worst_regret_rows": selected_rows.sort_values("regret", ascending=False).head(20),
        "family_summary": pd.DataFrame(columns=EMPTY_GROUP_SUMMARY_COLUMNS),
    }


def prettify_label(name: str, wrap_width: int | None = 24) -> str:
    """Make chain and method labels cleaner for paper-style plots."""
    text = str(name)
    if text == ORIGINAL_CHAIN_NAME:
        return "No Optimization"
    text = text.replace("__", " -> ").replace("_", " ")
    replacements = {
        "qiskit standard": "Qiskit",
        "qiskit ai": "Qiskit AI",
        "wisq rules": "WISQ Rules",
        "wisq bqskit": "WISQ BQSKit",
        "tket": "TKET",
        "best fixed reward": "Best Fixed Reward",
        "best depth chain": "Best Depth Chain",
        "best 2q chain": "Best 2Q Chain",
        "best depth single": "Best Depth Single",
        "best 2q single": "Best 2Q Single",
        "best possible objective": "Best Possible Objective",
        "best possible depth": "Best Possible Depth",
        "best possible 2q": "Best Possible 2Q",
        "llm v3": "LLM v3",
    }
    lowered = text.lower()
    for old, new in replacements.items():
        lowered = lowered.replace(old, new)
    pretty = lowered
    if wrap_width is None:
        return pretty
    return textwrap.fill(pretty, width=wrap_width)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save a figure as both PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def style_axis(ax: plt.Axes) -> None:
    """Apply the shared paper-style axis treatment."""
    ax.set_facecolor(PAPER_PANEL_BG)
    ax.grid(axis="x", color=PAPER_GRID, linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PAPER_EDGE)
    ax.spines["bottom"].set_color(PAPER_EDGE)
    ax.tick_params(colors=PAPER_TEXT)
    ax.xaxis.label.set_color(PAPER_TEXT)
    ax.yaxis.label.set_color(PAPER_TEXT)
    ax.title.set_color(PAPER_TEXT)


def horizontal_bar(
    data: pd.DataFrame,
    x: str,
    xlabel: str,
    title: str,
    color: str = PAPER_BAR,
    fmt: str = "{:.3f}",
    wrap_width: int | None = 26,
) -> plt.Figure:
    """Build a clean standalone horizontal bar chart."""
    plot_data = data.sort_values(x).copy()
    plot_data["method_pretty"] = plot_data["method"].apply(
        lambda name: prettify_label(name, wrap_width=wrap_width)
    )

    fig_height = max(3.2, 0.58 * len(plot_data))
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    fig.patch.set_facecolor(PAPER_BG)
    bars = ax.barh(
        plot_data["method_pretty"],
        plot_data[x],
        color=color,
        edgecolor=PAPER_EDGE,
        linewidth=0.8,
    )
    style_axis(ax)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)

    max_value = float(plot_data[x].max())
    min_value = float(plot_data[x].min())
    value_range = max_value - min_value
    if value_range == 0:
        value_range = abs(max_value) if max_value != 0 else 1.0
    padding = 0.03 * value_range

    for bar, value in zip(bars, plot_data[x], strict=True):
        ax.text(
            float(bar.get_width()) + padding,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            fontsize=8.5,
            color=PAPER_TEXT,
        )

    ax.set_xlim(min(0, min_value) - 0.05 * value_range, max_value + 0.24 * value_range)
    fig.tight_layout()
    return fig


def draw_metric_panel(
    comparison: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """Build the shared 2x2 evaluator panel."""
    specs = [
        ("avg_objective_score", "Average objective score", "{:.3f}", PAPER_BAR),
        ("avg_two_qubit_gate_reduction_pct", "Average 2Q reduction (%)", "{:.1f}", PAPER_ALT_BAR),
        ("avg_depth_reduction_pct", "Average depth reduction (%)", "{:.1f}", "#7A93C8"),
        ("avg_runtime_seconds", "Average runtime (seconds)", "{:.2f}", PAPER_RUNTIME),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4))
    fig.patch.set_facecolor(PAPER_BG)
    axes = axes.flatten()
    for ax, (column, xlabel, fmt, color) in zip(axes, specs, strict=True):
        plot_data = comparison.sort_values(column).copy()
        plot_data["method_pretty"] = plot_data["method"].apply(
            lambda name: prettify_label(name, wrap_width=22)
        )
        bars = ax.barh(
            plot_data["method_pretty"],
            plot_data[column],
            color=color,
            edgecolor=PAPER_EDGE,
            linewidth=0.8,
        )
        style_axis(ax)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
        ax.tick_params(axis="y", labelsize=8.5)
        value_range = float(plot_data[column].max() - plot_data[column].min())
        if value_range == 0:
            value_range = abs(float(plot_data[column].max())) or 1.0
        padding = 0.025 * value_range
        for bar, value in zip(bars, plot_data[column], strict=True):
            ax.text(
                float(bar.get_width()) + padding,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
                fontsize=8,
                color=PAPER_TEXT,
            )
    fig.suptitle(title, fontsize=15, y=0.99, x=0.5, ha="center", color=PAPER_TEXT)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.92, bottom=0.08, wspace=0.52, hspace=0.36)
    return fig


def draw_chain_usage_heatmap(plot_frames: dict[str, pd.DataFrame], split_name: str) -> plt.Figure:
    """Plot selected-chain counts across the compared methods."""
    selected_chain_counts = (
        pd.concat(
            [frame[["method", "selected_chain"]] for frame in plot_frames.values()],
            ignore_index=True,
        )
        .groupby(["method", "selected_chain"])
        .size()
        .reset_index(name="circuits")
    )
    grid = selected_chain_counts.pivot_table(
        index="method",
        columns="selected_chain",
        values="circuits",
        fill_value=0,
        aggfunc="sum",
    )
    grid = grid.loc[
        grid.sum(axis=1).sort_values(ascending=False).index,
        grid.sum(axis=0).sort_values(ascending=False).index,
    ]
    grid.index = [prettify_label(name, wrap_width=22) for name in grid.index]
    grid.columns = [prettify_label(name, wrap_width=16) for name in grid.columns]

    fig, ax = plt.subplots(figsize=(max(12, 0.65 * grid.shape[1]), 5.8))
    fig.patch.set_facecolor(PAPER_BG)
    ax.set_facecolor(PAPER_PANEL_BG)
    sns.heatmap(
        grid,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(PAPER_BAR, as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Evaluated {split_name} circuits"},
        ax=ax,
    )
    ax.set_xlabel("Selected optimizer chain", fontsize=10)
    ax.set_ylabel("Method", fontsize=10)
    ax.set_title("Selected Optimizer Chain Counts by Method", fontsize=12, pad=10, color=PAPER_TEXT)
    plt.xticks(rotation=40, ha="right")
    return fig


def draw_confidence_regret_scatter(selected_rows: pd.DataFrame) -> plt.Figure:
    """Plot confidence versus regret for one run."""
    plot_data = selected_rows.copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    fig.patch.set_facecolor(PAPER_BG)
    ax.set_facecolor(PAPER_PANEL_BG)
    ax.scatter(
        pd.to_numeric(plot_data["confidence"], errors="coerce"),
        pd.to_numeric(plot_data["regret"], errors="coerce"),
        color=PAPER_BAR,
        edgecolors=PAPER_EDGE,
        alpha=0.82,
    )
    style_axis(ax)
    ax.set_xlabel("Model confidence", fontsize=10)
    ax.set_ylabel("Regret", fontsize=10)
    ax.set_title("Confidence vs Regret", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def draw_family_original_rate(family_summary: pd.DataFrame) -> plt.Figure:
    """Plot original-selection rate by an arbitrary grouping column."""
    required_columns = {"original_selection_rate"}
    if family_summary.empty or not required_columns.issubset(family_summary.columns):
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        fig.patch.set_facecolor(PAPER_BG)
        ax.set_facecolor(PAPER_PANEL_BG)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No grouping summary available for this run.",
            ha="center",
            va="center",
            fontsize=11,
            color=PAPER_TEXT,
        )
        return fig
    plot_data = family_summary.sort_values("original_selection_rate").copy()

    label_column = next(
        (
            column
            for column in plot_data.columns
            if column not in {"circuits", "mean_objective_score", "mean_regret", "original_selection_rate", "shortlist_oracle_recall", "mean_confidence"}
        ),
        None,
    )
    if label_column is None:
        label_column = "group"
        plot_data[label_column] = [f"group_{index + 1}" for index in range(len(plot_data))]

    fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.42 * len(plot_data))))
    fig.patch.set_facecolor(PAPER_BG)
    bars = ax.barh(
        plot_data[label_column].astype(str),
        plot_data["original_selection_rate"],
        color=PAPER_ALT_BAR,
        edgecolor=PAPER_EDGE,
        linewidth=0.8,
    )
    style_axis(ax)
    ax.set_xlabel("Original-selection rate", fontsize=10)
    ax.set_title("Original Selection by Group", fontsize=12, pad=10)
    for bar, value in zip(bars, plot_data["original_selection_rate"], strict=True):
        ax.text(
            float(bar.get_width()) + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.2f}",
            va="center",
            fontsize=8,
            color=PAPER_TEXT,
        )
    fig.tight_layout()
    return fig


def collect_run_table(artifact_root: Path, split_name: str | None = None) -> pd.DataFrame:
    """Collect manifest and summary rows across saved runs."""
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(artifact_root.iterdir()) if artifact_root.exists() else []:
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        if not (manifest_path.exists() and summary_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        if split_name is not None and str(manifest.get("split")) != str(split_name):
            continue
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": manifest.get("run_name"),
                "created_at_utc": manifest.get("created_at_utc"),
                "split": manifest.get("split"),
                "provider": manifest.get("provider"),
                "model_name": manifest.get("model_name"),
                "prompt_mode": manifest.get("prompt_mode"),
                "feature_subset": manifest.get("feature_subset"),
                "retrieval_k": manifest.get("retrieval_k"),
                "retrieval_top_outcomes": manifest.get("retrieval_top_outcomes"),
                "shortlist_size": manifest.get("shortlist_size"),
                "min_confidence": manifest.get("min_confidence"),
                "reward_type": manifest["objective"]["reward_type"],
                "runtime_weight": manifest["objective"]["runtime_weight"],
                "two_qubit_weight": manifest["objective"]["two_qubit_weight"],
                "depth_weight": manifest["objective"]["depth_weight"],
                "final_reward_weight": manifest["objective"]["final_reward_weight"],
                **summary,
            }
        )
    return pd.DataFrame(rows).sort_values("created_at_utc").reset_index(drop=True)


def aggregate_run_table(run_table: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Average repeated runs into one summary row per comparison group."""
    if run_table.empty:
        return run_table.copy()

    metrics = [
        "mean_objective_score",
        "mean_regret",
        "p95_regret",
        "max_regret",
        "mean_two_qubit_gate_reduction_pct",
        "mean_depth_reduction_pct",
        "mean_runtime_seconds",
        "mean_runtime_score",
        "fallback_rate",
        "mean_confidence",
        "original_selection_rate",
        "shortlist_oracle_recall",
    ]
    available_metrics = [metric for metric in metrics if metric in run_table.columns]
    return (
        run_table.groupby(group_col, as_index=False)[available_metrics]
        .mean(numeric_only=True)
        .sort_values("mean_objective_score", ascending=False)
        .reset_index(drop=True)
    )
