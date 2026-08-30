"""
Benchmarks several ML classifiers on the 3-class PMU power-system event
dataset (Natural / Attack / NoEvents), evaluating both classification
quality and edge-deployment practicality.

Two latency budgets are used as pass/fail thresholds, both derived from
real grid communication/protection standards rather than picked arbitrarily:

  - Streaming budget (16.7 ms): PMU streams report at up to 60 frames/sec
    per IEEE C37.118.2. A model has to finish classifying one frame before
    the next one arrives, or it falls behind the live stream.
  - Trip-critical budget (3 ms): IEC 61850-5's TT6 message class, the
    timing requirement for messages allowed to gate a protection trip
    decision rather than just raise an operator alert.

Run after scripts/download_data.py. Writes everything to results/.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pmu_dataset.csv"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.2
LATENCY_SAMPLES = 200
LATENCY_WARMUP = 20

STREAMING_BUDGET_MS = 1000 / 60  # IEEE C37.118.2, 60 fps reporting rate
TRIP_BUDGET_MS = 3.0             # IEC 61850-5, TT6 message class

TABPFN_ENABLE = os.environ.get("TABPFN_ENABLE", "0") == "1"
TABPFN_SUBSAMPLE = 2000
TABPFN_SEEDS = SEEDS[:2]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/download_data.py first."
        )
    df = pd.read_csv(DATA_PATH)
    y_raw = df["marker"]
    X = df.drop(columns=["marker"])

    # A handful of impedance columns (R*-PA:Z) divide by near-zero current
    # and produce +inf. Treat those as missing; the imputer below fills
    # them in per training fold so there's no leakage from test rows.
    X = X.replace([np.inf, -np.inf], np.nan)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    return X, y, encoder


def build_model_specs():
    """Returns {name: (estimator_factory(seed), uses_sample_weight)}."""
    return {
        "Logistic Regression": (
            lambda seed: LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            ),
            False,
        ),
        "Decision Tree": (
            lambda seed: DecisionTreeClassifier(
                class_weight="balanced", random_state=seed
            ),
            False,
        ),
        "Random Forest": (
            lambda seed: RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost": (
            lambda seed: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
            ),
            True,
        ),
        "MLP (Full)": (
            lambda seed: MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=300,
                random_state=seed,
            ),
            False,
        ),
        "MLP (Compact)": (
            lambda seed: MLPClassifier(
                hidden_layer_sizes=(16,),
                max_iter=300,
                random_state=seed,
            ),
            False,
        ),
    }


def make_pipeline(estimator):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", estimator),
        ]
    )


def measure_latency(pipeline, X_test):
    # Single-row predictions, one at a time -- this mirrors how an edge
    # device consumes one PMU frame at a time, not a batch.
    rng = np.random.RandomState(0)
    n = min(LATENCY_SAMPLES + LATENCY_WARMUP, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    rows = X_test.iloc[idx]

    times_ms = []
    for i in range(n):
        row = rows.iloc[[i]]
        start = time.perf_counter()
        pipeline.predict(row)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if i >= LATENCY_WARMUP:
            times_ms.append(elapsed_ms)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def measure_size_kb(pipeline):
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump(pipeline, tmp_path)
        size_kb = os.path.getsize(tmp_path) / 1024
    finally:
        os.remove(tmp_path)
    return size_kb


def run_core_models(X, y, class_names):
    specs = build_model_specs()
    per_run_rows = []
    confusion_totals = {name: np.zeros((len(class_names), len(class_names)), dtype=int) for name in specs}
    last_split_fits = {}  # keep split-1 fitted pipelines for feature importance

    for split_i, seed in enumerate(SEEDS):
        print(f"\n=== Split {split_i + 1}/{len(SEEDS)} (seed={seed}) ===")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, (factory, uses_sample_weight) in specs.items():
            pipeline = make_pipeline(factory(seed))

            if uses_sample_weight:
                weights = compute_sample_weight("balanced", y_train)
                pipeline.fit(X_train, y_train, clf__sample_weight=weights)
            else:
                pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
            latency_mean, latency_std = measure_latency(pipeline, X_test)
            size_kb = measure_size_kb(pipeline)

            confusion_totals[name] += confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

            per_run_rows.append(
                {
                    "model": name,
                    "split": split_i,
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "latency_ms_mean": latency_mean,
                    "latency_ms_std": latency_std,
                    "size_kb": size_kb,
                }
            )

            print(
                f"  {name:<20} acc={acc:.3f}  f1_macro={f1_macro:.3f}  "
                f"latency={latency_mean:.3f}ms  size={size_kb:.1f}KB"
            )

            if split_i == 0 and name in ("Random Forest", "XGBoost"):
                last_split_fits[name] = (pipeline, X_test, y_test)

    return pd.DataFrame(per_run_rows), confusion_totals, last_split_fits


def run_tabpfn(X, y, class_names):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        print("\nTABPFN_ENABLE=1 was set but the 'tabpfn' package is not installed.")
        print("Install it with: pip install tabpfn")
        return pd.DataFrame()

    print("\n=== TabPFN (extreme case, subsampled) ===")
    print(f"TabPFN is a tabular foundation model, not designed for {len(X)}-row datasets.")
    print(f"Subsampling to {TABPFN_SUBSAMPLE} rows per split to keep runtime reasonable.")

    rows = []
    rng = np.random.RandomState(0)
    for split_i, seed in enumerate(TABPFN_SEEDS):
        idx = rng.choice(len(X), size=min(TABPFN_SUBSAMPLE, len(X)), replace=False)
        X_sub, y_sub = X.iloc[idx], y[idx]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(splitter.split(X_sub, y_sub))
        X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
        y_train, y_test = y_sub[train_idx], y_sub[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        clf = TabPFNClassifier()
        start_fit = time.perf_counter()
        clf.fit(X_train_imp, y_train)
        fit_time_s = time.perf_counter() - start_fit

        y_pred = clf.predict(X_test_imp)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        start_pred = time.perf_counter()
        clf.predict(X_test_imp[:1])
        latency_ms = (time.perf_counter() - start_pred) * 1000

        print(
            f"  split {split_i + 1}: acc={acc:.3f}  f1_macro={f1_macro:.3f}  "
            f"fit_time={fit_time_s:.1f}s  single_predict={latency_ms:.1f}ms"
        )

        rows.append(
            {
                "model": "TabPFN",
                "split": split_i,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "latency_ms_mean": latency_ms,
                "latency_ms_std": 0.0,
                "size_kb": float("nan"),  # not a locally-persisted model
            }
        )

    return pd.DataFrame(rows)


def aggregate_results(per_run_df):
    agg = per_run_df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        f1_weighted_mean=("f1_weighted", "mean"),
        precision_macro_mean=("precision_macro", "mean"),
        recall_macro_mean=("recall_macro", "mean"),
        latency_ms_mean=("latency_ms_mean", "mean"),
        latency_ms_std=("latency_ms_mean", "std"),
        size_kb_mean=("size_kb", "mean"),
        n_splits=("split", "count"),
    ).reset_index()

    agg["meets_streaming_budget"] = agg["latency_ms_mean"] <= STREAMING_BUDGET_MS
    agg["meets_trip_budget"] = agg["latency_ms_mean"] <= TRIP_BUDGET_MS
    agg = agg.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    return agg


def save_confusion_matrices(confusion_totals, class_names):
    rows = []
    for model, matrix in confusion_totals.items():
        for i, true_label in enumerate(class_names):
            for j, pred_label in enumerate(class_names):
                rows.append(
                    {
                        "model": model,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(matrix[i, j]),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False)


def save_feature_importance(last_split_fits, feature_names):
    columns = {}
    for model_name, (pipeline, X_test, y_test) in last_split_fits.items():
        # Permutation importance operates on the fitted pipeline directly,
        # so imputation/scaling are already accounted for.
        result = permutation_importance(
            pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring="f1_macro", n_jobs=-1
        )
        key = "importance_" + model_name.lower().replace(" ", "_")
        columns[key] = result.importances_mean

    df = pd.DataFrame({"feature": feature_names, **columns})
    score_cols = [c for c in df.columns if c != "feature"]
    df["average_importance"] = df[score_cols].mean(axis=1)
    df = df.sort_values("average_importance", ascending=False).reset_index(drop=True)
    df.head(20).to_csv(RESULTS_DIR / "feature_importance.csv", index=False)


def make_plots(agg_df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bar chart: macro-F1 per model, with std error bars
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(agg_df["model"], agg_df["f1_macro_mean"], yerr=agg_df["f1_macro_std"], capsize=4)
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification quality by model (mean +/- std across splits)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison_bars.png", dpi=150)
    plt.close(fig)

    # Scatter: latency (log x) vs macro-F1, with budget reference lines
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(agg_df["latency_ms_mean"], agg_df["f1_macro_mean"], s=80)
    for _, row in agg_df.iterrows():
        ax.annotate(row["model"], (row["latency_ms_mean"], row["f1_macro_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(TRIP_BUDGET_MS, color="red", linestyle="--", linewidth=1,
               label=f"Trip-critical budget ({TRIP_BUDGET_MS:.1f} ms, IEC 61850-5 TT6)")
    ax.axvline(STREAMING_BUDGET_MS, color="orange", linestyle="--", linewidth=1,
               label=f"Streaming budget ({STREAMING_BUDGET_MS:.1f} ms, IEEE C37.118.2 @ 60fps)")
    ax.set_xscale("log")
    ax.set_xlabel("Mean single-sample inference latency (ms, log scale)")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Accuracy vs. edge-deployment latency")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_vs_latency.png", dpi=150)
    plt.close(fig)


def print_summary(agg_df):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = f"{'Model':<20}{'Acc':>8}{'F1-macro':>10}{'Latency(ms)':>13}{'Size(KB)':>10}{'Stream OK':>11}{'Trip OK':>9}"
    print(header)
    print("-" * len(header))
    for _, row in agg_df.iterrows():
        print(
            f"{row['model']:<20}"
            f"{row['accuracy_mean']:>8.3f}"
            f"{row['f1_macro_mean']:>10.3f}"
            f"{row['latency_ms_mean']:>13.3f}"
            f"{row['size_kb_mean']:>10.1f}"
            f"{'yes' if row['meets_streaming_budget'] else 'no':>11}"
            f"{'yes' if row['meets_trip_budget'] else 'no':>9}"
        )
    print("=" * 100)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    X, y, encoder = load_data()
    class_names = list(encoder.classes_)
    print(f"Loaded {len(X)} rows, {X.shape[1]} features, classes={class_names}")

    per_run_df, confusion_totals, last_split_fits = run_core_models(X, y, class_names)

    if TABPFN_ENABLE:
        tabpfn_df = run_tabpfn(X, y, class_names)
        if not tabpfn_df.empty:
            per_run_df = pd.concat([per_run_df, tabpfn_df], ignore_index=True)

    per_run_df.to_csv(RESULTS_DIR / "per_split_results.csv", index=False)

    agg_df = aggregate_results(per_run_df)
    agg_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    save_confusion_matrices(confusion_totals, class_names)
    save_feature_importance(last_split_fits, list(X.columns))
    make_plots(agg_df)

    with open(RESULTS_DIR / "run_config.json", "w") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "test_size": TEST_SIZE,
                "streaming_budget_ms": STREAMING_BUDGET_MS,
                "trip_budget_ms": TRIP_BUDGET_MS,
                "tabpfn_enabled": TABPFN_ENABLE,
                "n_rows": int(len(X)),
                "n_features": int(X.shape[1]),
                "class_names": class_names,
            },
            f,
            indent=2,
        )

    print_summary(agg_df)
    print(f"\nAll results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
