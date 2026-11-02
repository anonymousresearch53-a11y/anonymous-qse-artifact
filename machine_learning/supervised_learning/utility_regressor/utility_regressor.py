"""Supervised utility regression baseline for optimizer-chain selection.

The model predicts the v4 local-reward utility of each supported optimizer
chain from pre-optimization circuit features and chain identity.  Runtime,
reduction, final-metric, and dynamic-progress columns are used only to build
the target, never as model inputs.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (
        feature_batch_names,
        prepare_datasets,
    )
    from machine_learning.RL.two_opt_chain_rl_v4.rewards import (
        MIXED_WEIGHTED,
        SUPPORTED_REWARD_TYPES,
        compute_step_reward,
    )
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from machine_learning.RL.two_opt_chain_rl_v4.data_processing import (
        feature_batch_names,
        prepare_datasets,
    )
    from machine_learning.RL.two_opt_chain_rl_v4.rewards import (
        MIXED_WEIGHTED,
        SUPPORTED_REWARD_TYPES,
        compute_step_reward,
    )


RANDOM_STATE = 42
DEFAULT_FEATURE_BATCH = "plus_all_features"
DEFAULT_REWARD_TYPE = MIXED_WEIGHTED
DEFAULT_RUNTIME_WEIGHT = 0.1
DEFAULT_TWO_QUBIT_WEIGHT = 0.5
DEFAULT_DEPTH_WEIGHT = 0.5

CHAIN_CATEGORICAL_FEATURES = ["chain_name", "opt_1", "opt_2"]
CHAIN_NUMERIC_FEATURES = ["chain_length"]

RIDGE_SEARCH_GRID = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}
GRADIENT_BOOSTING_SEARCH_GRID = {
    "model__n_estimators": [100, 250, 500],
    "model__learning_rate": [0.03, 0.1],
    "model__max_depth": [2, 3],
    "model__min_samples_leaf": [1, 3],
    "model__subsample": [0.8, 1.0],
}


def parse_chain_name(chain_name: str) -> tuple[str, str, int]:
    """Return first optimizer, second optimizer, and chain length."""
    parts = [part for part in str(chain_name).split("__") if part]
    if len(parts) not in {1, 2}:
        raise ValueError(f"Expected a one- or two-optimizer chain: {chain_name}")
    return parts[0], parts[1] if len(parts) == 2 else "__none__", len(parts)


def add_chain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add inference-available metadata derived only from chain identity."""
    df = frame.copy()
    parsed = df["chain_name"].map(parse_chain_name)
    df[["opt_1", "opt_2", "chain_length"]] = pd.DataFrame(
        parsed.tolist(), index=df.index
    )
    return df


def utility_target(
    frame: pd.DataFrame,
    reward_type: str = DEFAULT_REWARD_TYPE,
    runtime_weight: float = DEFAULT_RUNTIME_WEIGHT,
    two_qubit_weight: float = DEFAULT_TWO_QUBIT_WEIGHT,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
) -> pd.Series:
    """Compute direct utility as the sum of local v4 rewards, without correction."""
    rows_by_circuit = {
        circuit: rows.set_index("chain_name")
        for circuit, rows in frame.groupby("original_circuit")
    }

    targets = []
    for row in frame.itertuples(index=False):
        circuit_rows = rows_by_circuit[row.original_circuit]
        first_optimizer, _, chain_length = parse_chain_name(row.chain_name)
        reward = compute_step_reward(
            circuit_rows.loc[first_optimizer],
            reward_type=reward_type,
            runtime_weight=runtime_weight,
            two_qubit_weight=two_qubit_weight,
            depth_weight=depth_weight,
        )
        if chain_length == 2:
            reward += compute_step_reward(
                circuit_rows.loc[row.chain_name],
                reward_type=reward_type,
                runtime_weight=runtime_weight,
                two_qubit_weight=two_qubit_weight,
                depth_weight=depth_weight,
            )
        targets.append(float(reward))
    return pd.Series(targets, index=frame.index, name="utility")


def feature_columns(bundle: dict) -> tuple[list[str], list[str]]:
    """Return inference-safe numeric and categorical feature columns."""
    numeric = [f"initial_{column}" for column in bundle["baseline_feature_columns"]]
    numeric.extend(f"initial_{column}" for column in bundle["static_feature_columns"])
    numeric.extend(CHAIN_NUMERIC_FEATURES)
    return numeric, list(CHAIN_CATEGORICAL_FEATURES)


def build_model_pipeline(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build one preprocessing/model pipeline."""
    model_params = dict(model_params or {})
    if model_name == "ridge":
        model = Ridge(**model_params)
        numeric_transformer = StandardScaler()
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=random_state, **model_params)
        numeric_transformer = "passthrough"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def model_search_grid(model_name: str) -> dict[str, list]:
    """Return the explicit v4-style grid for a model."""
    if model_name == "ridge":
        return RIDGE_SEARCH_GRID
    if model_name == "gradient_boosting":
        return GRADIENT_BOOSTING_SEARCH_GRID
    raise ValueError(f"Unsupported model: {model_name}")


def expand_grid(search_grid: dict[str, list]) -> Iterable[dict]:
    """Yield one concrete hyperparameter configuration at a time."""
    keys = list(search_grid)
    for values in itertools.product(*(search_grid[key] for key in keys)):
        yield dict(zip(keys, values))


def model_features(
    frame: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Select the model feature matrix after chain metadata is present."""
    return frame[numeric + categorical]


def select_best_chains(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_names: tuple[list[str], list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all rows and select one highest-scoring chain per circuit."""
    numeric, categorical = feature_names
    scored = frame.copy()
    scored["predicted_utility"] = model.predict(model_features(scored, numeric, categorical))
    selected_indices = scored.groupby("original_circuit")["predicted_utility"].idxmax()
    return scored, scored.loc[selected_indices].copy()


def selection_summary(selected: pd.DataFrame, all_rows: pd.DataFrame) -> dict:
    """Summarize selected-chain utility and outcome metrics."""
    oracle = all_rows.groupby("original_circuit")["utility"].max()
    selected_indexed = selected.set_index("original_circuit")
    return {
        "circuits": int(len(selected)),
        "mean_utility": float(selected["utility"].mean()),
        "oracle_utility": float(oracle.mean()),
        "mean_regret": float((oracle - selected_indexed["utility"]).mean()),
        "mean_two_qubit_reduction_pct": float(
            selected["total_two_qubit_gate_reduction_pct"].mean()
        ),
        "mean_depth_reduction_pct": float(
            selected["total_depth_reduction_pct"].mean()
        ),
        "mean_runtime_seconds": float(selected["total_runtime_seconds"].mean()),
        "selected_chain_counts": selected["chain_name"].value_counts().to_dict(),
    }


def regression_metrics(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_names: tuple[list[str], list[str]],
) -> dict:
    """Compute row-level regression metrics."""
    numeric, categorical = feature_names
    predictions = model.predict(model_features(frame, numeric, categorical))
    return {
        "mae": float(mean_absolute_error(frame["utility"], predictions)),
        "rmse": float(np.sqrt(mean_squared_error(frame["utility"], predictions))),
        "r2": float(r2_score(frame["utility"], predictions)),
    }


def search_model(
    model_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_names: tuple[list[str], list[str]],
    random_state: int = RANDOM_STATE,
) -> tuple[dict, pd.DataFrame]:
    """Choose hyperparameters by validation selected-chain utility."""
    numeric, categorical = feature_names
    results = []
    for overrides in expand_grid(model_search_grid(model_name)):
        params = {
            key.removeprefix("model__"): value for key, value in overrides.items()
        }
        candidate = build_model_pipeline(
            model_name, numeric, categorical, params, random_state=random_state
        )
        candidate.fit(
            model_features(train_df, numeric, categorical), train_df["utility"]
        )
        _, selected = select_best_chains(candidate, validation_df, feature_names)
        summary = selection_summary(selected, validation_df)
        results.append({**params, **summary})
    results_frame = pd.DataFrame(results).sort_values(
        "mean_utility", ascending=False
    ).reset_index(drop=True)
    best_params = {
        parameter.removeprefix("model__"): results_frame.iloc[0][
            parameter.removeprefix("model__")
        ]
        for parameter in model_search_grid(model_name)
    }
    return best_params, results_frame


def prepare_frame(
    frame: pd.DataFrame,
    reward_type: str,
    runtime_weight: float,
    two_qubit_weight: float,
    depth_weight: float,
) -> pd.DataFrame:
    """Add chain metadata and the supervised target."""
    prepared = add_chain_features(frame)
    prepared["utility"] = utility_target(
        prepared,
        reward_type=reward_type,
        runtime_weight=runtime_weight,
        two_qubit_weight=two_qubit_weight,
        depth_weight=depth_weight,
    )
    return prepared


def run_experiment(
    csv_path: str | Path | None = None,
    feature_batch: str = DEFAULT_FEATURE_BATCH,
    reward_type: str = DEFAULT_REWARD_TYPE,
    runtime_weight: float = DEFAULT_RUNTIME_WEIGHT,
    two_qubit_weight: float = DEFAULT_TWO_QUBIT_WEIGHT,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    random_state: int = RANDOM_STATE,
    run_search: bool = True,
) -> dict:
    """Prepare data, search models, fit finalists, and evaluate selection."""
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(f"Unsupported reward type: {reward_type}")
    if any(
        weight < 0
        for weight in (runtime_weight, two_qubit_weight, depth_weight)
    ):
        raise ValueError("Reward weights must be non-negative")
    if reward_type == MIXED_WEIGHTED and two_qubit_weight == depth_weight == 0:
        raise ValueError("mixed_weighted needs a positive metric weight")
    bundle = prepare_datasets(
        csv_path=csv_path,
        random_state=random_state,
        save=False,
        feature_batch=feature_batch,
    )
    frames = {
        name: prepare_frame(
            bundle[f"{name}_df"],
            reward_type,
            runtime_weight,
            two_qubit_weight,
            depth_weight,
        )
        for name in ("train", "validation", "test")
    }
    numeric, categorical = feature_columns(bundle)
    feature_names = (numeric, categorical)

    finalists = {}
    for model_name in ("ridge", "gradient_boosting"):
        if run_search:
            best_params, search_results = search_model(
                model_name,
                frames["train"],
                frames["validation"],
                feature_names,
                random_state=random_state,
            )
        else:
            best_params, search_results = {}, pd.DataFrame()
        model = build_model_pipeline(
            model_name,
            numeric,
            categorical,
            best_params,
            random_state=random_state,
        )
        model.fit(
            model_features(frames["train"], numeric, categorical),
            frames["train"]["utility"],
        )
        _, validation_selected = select_best_chains(
            model, frames["validation"], feature_names
        )
        _, test_selected = select_best_chains(model, frames["test"], feature_names)
        finalists[model_name] = {
            "model": model,
            "best_params": best_params,
            "search_results": search_results,
            "train_metrics": regression_metrics(model, frames["train"], feature_names),
            "validation_metrics": regression_metrics(
                model, frames["validation"], feature_names
            ),
            "test_metrics": regression_metrics(model, frames["test"], feature_names),
            "validation_selection": selection_summary(validation_selected, frames["validation"]),
            "test_selection": selection_summary(test_selected, frames["test"]),
            "validation_selected": validation_selected,
            "test_selected": test_selected,
        }

    return {
        "bundle": bundle,
        "frames": frames,
        "feature_names": feature_names,
        "models": finalists,
        "config": {
            "feature_batch": feature_batch,
            "reward_type": reward_type,
            "runtime_weight": runtime_weight,
            "two_qubit_weight": two_qubit_weight,
            "depth_weight": depth_weight,
            "random_state": random_state,
            "run_search": run_search,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument(
        "--feature-batch", choices=feature_batch_names(), default=DEFAULT_FEATURE_BATCH
    )
    parser.add_argument(
        "--reward-type", choices=SUPPORTED_REWARD_TYPES, default=DEFAULT_REWARD_TYPE
    )
    parser.add_argument("--runtime-weight", type=float, default=DEFAULT_RUNTIME_WEIGHT)
    parser.add_argument("--two-qubit-weight", type=float, default=DEFAULT_TWO_QUBIT_WEIGHT)
    parser.add_argument("--depth-weight", type=float, default=DEFAULT_DEPTH_WEIGHT)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--search", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_experiment(
        csv_path=args.csv_path,
        feature_batch=args.feature_batch,
        reward_type=args.reward_type,
        runtime_weight=args.runtime_weight,
        two_qubit_weight=args.two_qubit_weight,
        depth_weight=args.depth_weight,
        random_state=args.random_state,
        run_search=args.search,
    )
    print(json.dumps({
        model_name: {
            "best_params": values["best_params"],
            "test_metrics": values["test_metrics"],
            "test_selection": values["test_selection"],
        }
        for model_name, values in results["models"].items()
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
