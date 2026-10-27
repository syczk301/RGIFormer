#!/usr/bin/env python3
"""Train auditable traditional baselines under the manuscript data protocol.

The script deliberately keeps the test set outside model and hyperparameter
selection.  Hyperparameters are selected by validation accuracy (then
validation macro-F1 as a deterministic tie breaker), matching the checkpoint
selection criterion used by the neural baselines.
"""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPOSITORY_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DATA_ROOT = REPO_ROOT.parent / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "01_model_comparison/traditional"
DEFAULT_SEEDS = (42, 3407, 2024, 2025, 2026)
MODEL_ORDER = (
    "logistic_regression",
    "rbf_svm",
    "knn",
    "random_forest",
    "extra_trees",
)
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF-SVM",
    "knn": "kNN",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    feature_names: tuple[str, ...]
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray | None
    test_labels: np.ndarray | None
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class PreparedSplit:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    scaler: StandardScaler


def parse_csv_values(text: str, *, cast=str) -> list[Any]:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one comma-separated value is required.")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def load_dataset(name: str) -> DatasetBundle:
    if name == "Paper":
        data_path = DATA_ROOT / "Paper" / "prediction_dataset_full_11000.csv"
        frame = pd.read_csv(data_path)
        feature_names = tuple(
            column for column in frame.columns if column not in {"fault_id", "fault_name"}
        )
        features = frame.loc[:, feature_names].to_numpy(dtype=np.float64)
        labels = frame["fault_id"].to_numpy(dtype=np.int64)
        return DatasetBundle(
            name=name,
            feature_names=feature_names,
            train_features=features,
            train_labels=labels,
            test_features=None,
            test_labels=None,
            source_paths=(data_path,),
        )

    if name == "BSM2":
        train_path = DATA_ROOT / "BSM2" / "bsm2_train_data.csv"
        test_path = DATA_ROOT / "BSM2" / "bsm2_test_data.csv"
        train_frame = pd.read_csv(train_path)
        test_frame = pd.read_csv(test_path)
        feature_names = tuple(
            column for column in train_frame.columns if column not in {"time_day", "label"}
        )
        if tuple(
            column for column in test_frame.columns if column not in {"time_day", "label"}
        ) != feature_names:
            raise ValueError("BSM2 train/test feature columns differ.")
        return DatasetBundle(
            name=name,
            feature_names=feature_names,
            train_features=train_frame.loc[:, feature_names].to_numpy(dtype=np.float64),
            train_labels=train_frame["label"].to_numpy(dtype=np.int64),
            test_features=test_frame.loc[:, feature_names].to_numpy(dtype=np.float64),
            test_labels=test_frame["label"].to_numpy(dtype=np.int64),
            source_paths=(train_path, test_path),
        )

    raise ValueError(f"Unknown dataset: {name}")


def prepare_split(bundle: DatasetBundle, seed: int) -> PreparedSplit:
    all_indices = np.arange(len(bundle.train_labels), dtype=np.int64)
    if bundle.name == "Paper":
        train_indices, temp_indices = train_test_split(
            all_indices,
            test_size=0.30,
            random_state=seed,
            stratify=bundle.train_labels,
        )
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.50,
            random_state=seed,
            stratify=bundle.train_labels[temp_indices],
        )
        raw_train = bundle.train_features[train_indices]
        raw_val = bundle.train_features[val_indices]
        raw_test = bundle.train_features[test_indices]
        y_train = bundle.train_labels[train_indices]
        y_val = bundle.train_labels[val_indices]
        y_test = bundle.train_labels[test_indices]
    else:
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=0.20,
            random_state=seed,
            stratify=bundle.train_labels,
        )
        if bundle.test_features is None or bundle.test_labels is None:
            raise ValueError("BSM2 fixed test archive is unavailable.")
        test_indices = np.arange(len(bundle.test_labels), dtype=np.int64)
        raw_train = bundle.train_features[train_indices]
        raw_val = bundle.train_features[val_indices]
        raw_test = bundle.test_features
        y_train = bundle.train_labels[train_indices]
        y_val = bundle.train_labels[val_indices]
        y_test = bundle.test_labels

    scaler = StandardScaler()
    x_train = scaler.fit_transform(raw_train)
    x_val = scaler.transform(raw_val)
    x_test = scaler.transform(raw_test)
    return PreparedSplit(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        scaler=scaler,
    )


def parameter_grid(model_name: str, quick: bool = False) -> list[dict[str, Any]]:
    grids: dict[str, list[dict[str, Any]]] = {
        "logistic_regression": [{"C": value} for value in (0.1, 1.0, 10.0)],
        "rbf_svm": [
            {"C": 1.0, "gamma": "scale"},
            {"C": 10.0, "gamma": "scale"},
            {"C": 100.0, "gamma": "scale"},
            {"C": 10.0, "gamma": 0.01},
        ],
        "knn": [
            {"n_neighbors": value, "weights": "distance"}
            for value in (3, 5, 11, 21)
        ],
        "random_forest": [
            {"max_depth": None, "min_samples_leaf": 1},
            {"max_depth": None, "min_samples_leaf": 2},
            {"max_depth": 24, "min_samples_leaf": 1},
            {"max_depth": 24, "min_samples_leaf": 2},
        ],
        "extra_trees": [
            {"max_depth": None, "min_samples_leaf": 1},
            {"max_depth": None, "min_samples_leaf": 2},
            {"max_depth": 24, "min_samples_leaf": 1},
            {"max_depth": 24, "min_samples_leaf": 2},
        ],
    }
    grid = grids[model_name]
    return grid[:1] if quick else grid


def build_estimator(
    model_name: str,
    params: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=float(params["C"]),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            tol=1e-4,
            random_state=seed,
        )
    if model_name == "rbf_svm":
        return SVC(
            C=float(params["C"]),
            gamma=params["gamma"],
            kernel="rbf",
            class_weight="balanced",
            cache_size=2048,
            probability=False,
            random_state=seed,
        )
    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(params["n_neighbors"]),
            weights=str(params["weights"]),
            metric="minkowski",
            p=2,
            n_jobs=n_jobs,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown model: {model_name}")


def score_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_split_artifacts(
    dataset_dir: Path,
    bundle: DatasetBundle,
    split: PreparedSplit,
    seed: int,
) -> None:
    split_path = dataset_dir / "splits" / f"seed_{seed}.npz"
    scaler_path = dataset_dir / "scalers" / f"seed_{seed}.npz"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_path,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
        train_labels=split.y_train,
        val_labels=split.y_val,
        test_labels=split.y_test,
        test_index_scope=(
            "same_archive" if bundle.name == "Paper" else "fixed_test_archive"
        ),
    )
    np.savez_compressed(
        scaler_path,
        mean=split.scaler.mean_,
        scale=split.scaler.scale_,
        var=split.scaler.var_,
        n_features=len(bundle.feature_names),
        fitted_samples=len(split.y_train),
    )


def train_one(
    bundle: DatasetBundle,
    split: PreparedSplit,
    model_name: str,
    seed: int,
    output_dir: Path,
    *,
    n_jobs: int,
    quick: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = parameter_grid(model_name, quick=quick)
    search_rows: list[dict[str, Any]] = []
    best_estimator = None
    best_key = None
    best_params: dict[str, Any] | None = None
    best_val_metrics: dict[str, float] | None = None
    total_fit_seconds = 0.0

    for index, params in enumerate(candidates):
        estimator = build_estimator(model_name, params, seed=seed, n_jobs=n_jobs)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(split.x_train, split.y_train)
        fit_seconds = time.perf_counter() - started
        total_fit_seconds += fit_seconds
        val_predictions = estimator.predict(split.x_val)
        val_metrics = score_predictions(split.y_val, val_predictions)
        key = (val_metrics["accuracy"], val_metrics["macro_f1"], -index)
        search_rows.append(
            {
                "candidate": index,
                "params_json": json.dumps(params, sort_keys=True),
                **{f"val_{metric}": value for metric, value in val_metrics.items()},
                "fit_seconds": fit_seconds,
                "warnings": " | ".join(str(item.message) for item in caught),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_estimator = estimator
            best_params = dict(params)
            best_val_metrics = val_metrics

    if best_estimator is None or best_params is None or best_val_metrics is None:
        raise RuntimeError(f"No fitted candidate for {model_name}, seed {seed}.")

    search_frame = pd.DataFrame(search_rows)
    search_frame["selected"] = search_frame["params_json"].eq(
        json.dumps(best_params, sort_keys=True)
    )
    search_frame.to_csv(output_dir / "validation_search.csv", index=False)

    test_started = time.perf_counter()
    predictions = best_estimator.predict(split.x_test).astype(np.int64)
    test_seconds = time.perf_counter() - test_started
    test_metrics = score_predictions(split.y_test, predictions)
    report = classification_report(
        split.y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(split.y_test, predictions)

    joblib.dump(best_estimator, output_dir / "model.joblib", compress=3)
    np.savez_compressed(
        output_dir / "predictions.npz",
        labels=split.y_test.astype(np.int64),
        preds=predictions,
    )
    np.save(output_dir / "confusion_matrix.npy", matrix)
    write_json(output_dir / "classification_report.json", report)

    model_size = int((output_dir / "model.joblib").stat().st_size)
    metrics = {
        "dataset": bundle.name,
        "model": model_name,
        "model_display_name": MODEL_DISPLAY_NAMES[model_name],
        "seed": seed,
        "selection_metric": "validation_accuracy_then_macro_f1",
        "selected_params": best_params,
        "validation": best_val_metrics,
        "test": test_metrics,
        "train_samples": len(split.y_train),
        "val_samples": len(split.y_val),
        "test_samples": len(split.y_test),
        "n_features": split.x_train.shape[1],
        "n_classes": int(len(np.unique(split.y_test))),
        "candidate_count": len(candidates),
        "fit_seconds_total": total_fit_seconds,
        "test_prediction_seconds": test_seconds,
        "model_size_bytes": model_size,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def existing_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_model(model_dir: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    run_rows = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        run_rows.append(
            {
                "seed": row["seed"],
                **row["test"],
                **{f"val_{key}": value for key, value in row["validation"].items()},
                "selected_params_json": json.dumps(row["selected_params"], sort_keys=True),
                "fit_seconds_total": row["fit_seconds_total"],
                "test_prediction_seconds": row["test_prediction_seconds"],
                "model_size_bytes": row["model_size_bytes"],
            }
        )
    runs_frame = pd.DataFrame(run_rows)
    runs_frame.to_csv(model_dir / "runs.csv", index=False)

    n = len(runs_frame)
    t_value = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    summary_rows = []
    for metric in METRICS:
        values = runs_frame[metric].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        half_width = t_value * std / math.sqrt(n) if n > 1 else 0.0
        summary_rows.append(
            {
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "ci95_half_width": half_width,
                "mean_percent": mean * 100.0,
                "std_percent": std * 100.0,
                "ci95_half_width_percent": half_width * 100.0,
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(model_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return summary_frame


def write_protocol(
    dataset_dir: Path,
    bundle: DatasetBundle,
    models: list[str],
    seeds: list[int],
    *,
    quick: bool,
) -> None:
    if bundle.name == "Paper":
        split_description = (
            "stratified 70/15/15 train/validation/test split, recreated for each seed"
        )
    else:
        split_description = (
            "repository-fixed BSM2 test archive; stratified 20% validation subset "
            "from the training archive for each seed"
        )
    lines = [
        f"# Traditional baselines: {bundle.name}",
        "",
        f"- Source files: {', '.join(f'`{path}`' for path in bundle.source_paths)}",
        f"- Features: {len(bundle.feature_names)} numeric features.",
        f"- Split: {split_description}.",
        "- Scaling: `StandardScaler` fitted on the training subset only.",
        "- Hyperparameter selection: highest validation accuracy, then validation macro-F1; the test set is never used for selection.",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}.",
        f"- Models: {', '.join(MODEL_DISPLAY_NAMES[name] for name in models)}.",
        "- Class handling: balanced class weights for logistic regression, SVM, and tree ensembles; distance weighting for kNN. Both datasets are class-balanced, so balanced weights are effectively neutral.",
        "- BSM2 evidence boundary: 53 single-snapshot variables only; no lag, difference, rolling, future, or label-derived features.",
        f"- Quick mode: {quick}.",
        "",
        "## Validation grids",
        "",
    ]
    for model_name in models:
        lines.extend(
            [
                f"### {MODEL_DISPLAY_NAMES[model_name]}",
                "",
                "```json",
                json.dumps(parameter_grid(model_name, quick=quick), indent=2),
                "```",
                "",
            ]
        )
    (dataset_dir / "protocol.md").write_text("\n".join(lines), encoding="utf-8")


def write_dataset_comparison(
    dataset_dir: Path,
    models: list[str],
    summaries: dict[str, pd.DataFrame],
) -> None:
    rows = []
    for model_name in models:
        frame = summaries[model_name].set_index("metric")
        row: dict[str, Any] = {
            "model": model_name,
            "model_display_name": MODEL_DISPLAY_NAMES[model_name],
            "n": int(frame.loc["accuracy", "n"]),
        }
        for metric in METRICS:
            row[f"{metric}_mean_percent"] = float(frame.loc[metric, "mean_percent"])
            row[f"{metric}_std_percent"] = float(frame.loc[metric, "std_percent"])
            row[f"{metric}_ci95_half_width_percent"] = float(
                frame.loc[metric, "ci95_half_width_percent"]
            )
        rows.append(row)
    pd.DataFrame(rows).sort_values(
        "accuracy_mean_percent", ascending=False
    ).to_csv(dataset_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="Paper,BSM2")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true", help="Use one candidate per model for smoke tests")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_csv_values(args.datasets)
    models = parse_csv_values(args.models)
    seeds = parse_csv_values(args.seeds, cast=int)
    unknown_datasets = sorted(set(datasets) - {"Paper", "BSM2"})
    unknown_models = sorted(set(models) - set(MODEL_ORDER))
    if unknown_datasets or unknown_models:
        raise ValueError(
            f"Unknown datasets={unknown_datasets}, unknown models={unknown_models}"
        )

    print(
        json.dumps(
            {
                "datasets": datasets,
                "models": models,
                "seeds": seeds,
                "output_dir": str(args.output_dir),
                "n_jobs": args.n_jobs,
                "quick": args.quick,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "n_jobs": args.n_jobs,
        "quick": args.quick,
        "data_files": {},
    }

    for dataset_name in datasets:
        bundle = load_dataset(dataset_name)
        dataset_dir = args.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest["data_files"][dataset_name] = [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in bundle.source_paths
        ]
        write_protocol(
            dataset_dir,
            bundle,
            models,
            seeds,
            quick=args.quick,
        )
        rows_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}

        for seed in seeds:
            split = prepare_split(bundle, seed)
            save_split_artifacts(dataset_dir, bundle, split, seed)
            print(
                f"DATASET={dataset_name} SEED={seed} "
                f"train={len(split.y_train)} val={len(split.y_val)} "
                f"test={len(split.y_test)} features={split.x_train.shape[1]}",
                flush=True,
            )
            for model_name in models:
                seed_dir = dataset_dir / model_name / f"seed_{seed}"
                metrics_path = seed_dir / "metrics.json"
                metrics = None if args.overwrite else existing_metrics(metrics_path)
                if metrics is None:
                    started = time.perf_counter()
                    metrics = train_one(
                        bundle,
                        split,
                        model_name,
                        seed,
                        seed_dir,
                        n_jobs=args.n_jobs,
                        quick=args.quick,
                    )
                    elapsed = time.perf_counter() - started
                    print(
                        f"DONE {dataset_name}/{MODEL_DISPLAY_NAMES[model_name]} "
                        f"seed={seed} acc={metrics['test']['accuracy'] * 100:.4f}% "
                        f"f1={metrics['test']['macro_f1'] * 100:.4f}% "
                        f"wall={elapsed:.1f}s params={metrics['selected_params']}",
                        flush=True,
                    )
                else:
                    print(
                        f"SKIP existing {dataset_name}/{MODEL_DISPLAY_NAMES[model_name]} seed={seed}",
                        flush=True,
                    )
                rows_by_model[model_name].append(metrics)

        summaries = {}
        for model_name in models:
            summaries[model_name] = summarize_model(
                dataset_dir / model_name,
                rows_by_model[model_name],
            )
        write_dataset_comparison(dataset_dir, models, summaries)

    write_json(args.output_dir / "run_manifest.json", manifest)
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
