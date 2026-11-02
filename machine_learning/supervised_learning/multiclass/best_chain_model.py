"""Random-forest baseline for selecting an optimizer chain per circuit.

The classifier predicts one complete choice from the RL v4 action set:
the original circuit, four single optimizers, or sixteen two-optimizer
chains. Targets are defined by the weighted final utility of each candidate;
there is intentionally no RL-style terminal correction.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (  # noqa: E402
    BASELINE_FEATURE_COLUMNS,
    DYNAMIC_PROGRESS_FEATURE_COLUMNS,
    RANDOM_STATE,
    prepare_datasets,
)


FEATURE_BATCH = "plus_all_features"
ORIGINAL_CLASS = "__original__"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "machine_learning"
    / "supervised_learning"
    / "multiclass"
    / "artifacts"
    / "best_chain_random_forest"
)

DEFAULT_MODEL_CONFIG = {
    "n_estimators": 250,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}

SEARCH_VALUES = {
    "n_estimators": [100, 250],
    "max_depth": [None, 12, 24],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5],
}


def parse_bool(value):
    """Parse common command-line boolean spellings."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def chain_order(optimizer_names):
    """Return the deterministic class order used by the baseline."""
    return [
        ORIGINAL_CLASS,
        *optimizer_names,
        *(f"{first}__{second}" for first in optimizer_names for second in optimizer_names),
    ]


def build_feature_frame(frame, bundle):
    """Build one initial-state feature row per circuit.

    Static features come from their initial_* columns. Dynamic progress
    features are zero at the first decision, matching the RL observation.
    """
    feature_columns = list(bundle["observation_feature_columns"])
    first_rows = (
        frame.sort_values("original_circuit")
        .groupby("original_circuit", as_index=False)
        .first()
    )
    features = pd.DataFrame(index=first_rows["original_circuit"])

    for column in feature_columns:
        if column in DYNAMIC_PROGRESS_FEATURE_COLUMNS:
            features[column] = 0.0
            continue

        source_column = f"initial_{column}"
        if source_column not in first_rows.columns:
            raise ValueError(f"Missing initial feature column: {source_column}")
        features[column] = pd.to_numeric(
            first_rows.set_index("original_circuit")[source_column],
            errors="coerce",
        )

    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing numeric feature values in: {missing_columns}")

    return features.astype(float)


def build_candidate_frame(frame, optimizer_names, runtime_denominator):
    """Return all 21 selectable candidates and their weighted components."""
    candidates = frame[
        [
            "original_circuit",
            "chain_name",
            "total_two_qubit_gate_reduction_pct",
            "total_depth_reduction_pct",
            "total_runtime_seconds",
            "total_runtime_score",
        ]
    ].copy()

    circuits = sorted(frame["original_circuit"].unique())
    original_rows = pd.DataFrame(
        {
            "original_circuit": circuits,
            "chain_name": ORIGINAL_CLASS,
            "total_two_qubit_gate_reduction_pct": 0.0,
            "total_depth_reduction_pct": 0.0,
            "total_runtime_seconds": 0.0,
            "total_runtime_score": 0.0,
        }
    )
    candidates = pd.concat([original_rows, candidates], ignore_index=True)

    expected_chains = set(chain_order(optimizer_names)[1:])
    counts = candidates.groupby("original_circuit")["chain_name"].apply(set)
    invalid = counts[counts.map(lambda values: values != expected_chains | {ORIGINAL_CLASS})]
    if not invalid.empty:
        examples = ", ".join(
            f"{circuit}={sorted(values)}" for circuit, values in invalid.head(3).items()
        )
        raise ValueError(f"Candidate set is incomplete. Examples: {examples}")

    candidates["two_qubit_score"] = (
        candidates["total_two_qubit_gate_reduction_pct"] / 100.0
    )
    candidates["depth_score"] = candidates["total_depth_reduction_pct"] / 100.0
    candidates["runtime_score"] = pd.to_numeric(
        candidates["total_runtime_score"], errors="coerce"
    ).fillna(
        np.log1p(candidates["total_runtime_seconds"].clip(lower=0))
        / runtime_denominator
    )
    return candidates


def add_utility_and_targets(
    candidates,
    two_qubit_weight,
    depth_weight,
    runtime_weight,
    class_order,
):
    """Compute weighted utility and the oracle class for every circuit."""
    result = candidates.copy()
    result["utility"] = (
        float(two_qubit_weight) * result["two_qubit_score"]
        + float(depth_weight) * result["depth_score"]
        - float(runtime_weight) * result["runtime_score"]
    )
    result["class_order"] = pd.Categorical(
        result["chain_name"], categories=class_order, ordered=True
    ).codes
    result = result.sort_values(["original_circuit", "utility", "class_order"])
    targets = (
        result.groupby("original_circuit", sort=True, as_index=False)
        .tail(1)
        .set_index("original_circuit")["chain_name"]
        .rename("target_chain")
    )
    return result, targets


def fit_classifier(features, targets, config, random_state):
    """Fit one classifier configuration."""
    model = RandomForestClassifier(
        **config,
        random_state=int(random_state),
        n_jobs=-1,
    )
    model.fit(features.loc[targets.index], targets)
    return model


def evaluate_predictions(
    model,
    features,
    candidates,
    targets,
    split_name,
    class_order,
):
    """Map predicted classes back to measured candidate outcomes."""
    predicted = pd.Series(
        model.predict(features), index=features.index, name="predicted_chain"
    )
    target_values = targets.reindex(features.index)
    lookup = candidates.set_index(["original_circuit", "chain_name"])
    selected = lookup.loc[
        list(zip(features.index, predicted)),
        [
            "total_two_qubit_gate_reduction_pct",
            "total_depth_reduction_pct",
            "total_runtime_seconds",
            "runtime_score",
            "utility",
        ],
    ].reset_index()
    selected["target_chain"] = target_values.to_numpy()
    selected["predicted_chain"] = predicted.to_numpy()
    selected["correct"] = selected["predicted_chain"] == selected["target_chain"]
    selected["split"] = split_name
    selected["class_order"] = selected["predicted_chain"].map(
        {name: index for index, name in enumerate(class_order)}
    )
    return selected


def summary_row(predictions, split_name):
    """Summarize metrics in the same units used by RL v4 reporting."""
    return {
        "split": split_name,
        "utility": float(predictions["utility"].mean()),
        "avg_two_qubit_gate_reduction_pct": float(
            predictions["total_two_qubit_gate_reduction_pct"].mean()
        ),
        "avg_depth_reduction_pct": float(predictions["total_depth_reduction_pct"].mean()),
        "avg_runtime_seconds": float(predictions["total_runtime_seconds"].mean()),
        "avg_runtime_score": float(predictions["runtime_score"].mean()),
        "classification_accuracy": float(predictions["correct"].mean()),
        "circuits": int(len(predictions)),
    }


def search_configs(train_features, train_targets, validation_features, validation_candidates, validation_targets, class_order, random_state):
    """Search the explicit grid using validation utility."""
    rows = []
    keys = list(SEARCH_VALUES)
    values = [SEARCH_VALUES[key] for key in keys]
    total = np.prod([len(items) for items in values])
    for index, combination in enumerate(product(*values), start=1):
        config = dict(zip(keys, combination))
        model = fit_classifier(train_features, train_targets, config, random_state)
        predictions = evaluate_predictions(
            model,
            validation_features,
            validation_candidates,
            validation_targets,
            "validation",
            class_order,
        )
        row = {**config, **summary_row(predictions, "validation")}
        row["search_index"] = index
        row["search_total"] = int(total)
        rows.append(row)
        print(
            f"Grid search {index}/{int(total)}: "
            f"validation utility={row['utility']:.6f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["utility", "classification_accuracy"], ascending=False
    )
    return results.reset_index(drop=True)


def format_weight_for_name(value):
    """Return a filesystem-friendly weight string."""
    return format(float(value), "g")


def resolve_output_dir(output_dir, seed, two_qubit_weight, depth_weight, runtime_weight):
    """Append the reward weights and seed to the default artifact directory name."""
    output_dir = Path(output_dir)
    if output_dir == DEFAULT_OUTPUT_DIR:
        suffix = (
            f"_2q{format_weight_for_name(two_qubit_weight)}"
            f"_depth{format_weight_for_name(depth_weight)}"
            f"_rt{format_weight_for_name(runtime_weight)}"
            f"_seed{int(seed)}"
        )
        return output_dir.with_name(f"{output_dir.name}{suffix}")
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a multiclass random-forest optimizer-chain baseline."
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-search", type=parse_bool, default=False)
    parser.add_argument(
        "--seed",
        "--random-state",
        dest="seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for the classifier and hyperparameter search.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=RANDOM_STATE,
        help="Circuit split seed. Keep this at 42 to match RL v4.",
    )
    parser.add_argument("--two-qubit-weight", type=float, default=0.5)
    parser.add_argument("--depth-weight", type=float, default=0.5)
    parser.add_argument("--runtime-weight", type=float, default=0.1)
    return parser


def validate_weights(two_qubit_weight, depth_weight, runtime_weight):
    weights = [two_qubit_weight, depth_weight, runtime_weight]
    if any(weight < 0 for weight in weights):
        raise ValueError("Reward weights must be non-negative.")
    if two_qubit_weight == 0 and depth_weight == 0 and runtime_weight == 0:
        raise ValueError("At least one reward weight must be positive.")


def run_experiment(args):
    validate_weights(args.two_qubit_weight, args.depth_weight, args.runtime_weight)
    bundle = prepare_datasets(
        csv_path=args.csv_path,
        random_state=args.split_seed,
        save=False,
        feature_batch=FEATURE_BATCH,
    )
    class_order = chain_order(bundle["optimizer_names"])

    split_data = {}
    for split_name in ("train", "validation", "test"):
        frame = bundle[f"{split_name}_df"]
        features = build_feature_frame(frame, bundle)
        candidates = build_candidate_frame(
            frame,
            bundle["optimizer_names"],
            bundle["runtime_denominator"],
        )
        candidates, targets = add_utility_and_targets(
            candidates,
            args.two_qubit_weight,
            args.depth_weight,
            args.runtime_weight,
            class_order,
        )
        split_data[split_name] = {
            "features": features,
            "candidates": candidates,
            "targets": targets,
        }

    circuit_sets = {
        name: set(data["features"].index) for name, data in split_data.items()
    }
    if circuit_sets["train"] & circuit_sets["validation"]:
        raise ValueError("Train and validation circuits overlap.")
    if circuit_sets["train"] & circuit_sets["test"]:
        raise ValueError("Train and test circuits overlap.")
    if circuit_sets["validation"] & circuit_sets["test"]:
        raise ValueError("Validation and test circuits overlap.")

    if args.grid_search:
        search_results = search_configs(
            split_data["train"]["features"],
            split_data["train"]["targets"],
            split_data["validation"]["features"],
            split_data["validation"]["candidates"],
            split_data["validation"]["targets"],
            class_order,
            args.seed,
        )
        model_config = {
            key: search_results.iloc[0][key] for key in SEARCH_VALUES
        }
        model_config["max_depth"] = (
            None if pd.isna(model_config["max_depth"]) else int(model_config["max_depth"])
        )
        model_config["n_estimators"] = int(model_config["n_estimators"])
        model_config["min_samples_leaf"] = int(model_config["min_samples_leaf"])
    else:
        search_results = pd.DataFrame()
        model_config = DEFAULT_MODEL_CONFIG.copy()

    model = fit_classifier(
        split_data["train"]["features"],
        split_data["train"]["targets"],
        model_config,
        args.seed,
    )
    validation_predictions = evaluate_predictions(
        model,
        split_data["validation"]["features"],
        split_data["validation"]["candidates"],
        split_data["validation"]["targets"],
        "validation",
        class_order,
    )
    test_predictions = evaluate_predictions(
        model,
        split_data["test"]["features"],
        split_data["test"]["candidates"],
        split_data["test"]["targets"],
        "test",
        class_order,
    )

    output_dir = resolve_output_dir(
        args.output_dir,
        args.seed,
        args.two_qubit_weight,
        args.depth_weight,
        args.runtime_weight,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    search_results.to_csv(output_dir / "hyperparameter_search.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    summaries = pd.DataFrame(
        [
            summary_row(validation_predictions, "validation"),
            summary_row(test_predictions, "test"),
        ]
    )
    summaries.to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(
        {
            "chain_name": class_order,
            "class_index": range(len(class_order)),
        }
    ).to_csv(output_dir / "class_mapping.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_batch": FEATURE_BATCH,
                "seed": args.seed,
                "random_state": args.seed,
                "split_seed": args.split_seed,
                "runtime_denominator": bundle["runtime_denominator"],
                "two_qubit_weight": args.two_qubit_weight,
                "depth_weight": args.depth_weight,
                "runtime_weight": args.runtime_weight,
                "final_reward_correction": False,
                "grid_search": args.grid_search,
                "model_config": model_config,
                "feature_columns": list(split_data["train"]["features"].columns),
            },
            handle,
            indent=2,
        )

    print(summaries.to_string(index=False))
    print(f"Artifacts saved to: {output_dir}")
    return summaries


def main():
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
