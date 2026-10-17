"""
Shared helpers for the PMU attack-classification benchmark suite.

Every experiment script imports from here so that data loading, splits,
model definitions, metric capture and timing are identical across scripts.

Environment variables
  PMU_TAG     run label; results go to results/runs/<tag>/   (default: "local")
  PMU_SMOKE   "1" = tiny data/models for a quick end-to-end check of the code
"""

import json
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# sklearn 1.9's RandomForestClassifier.predict() emits a UserWarning about
# `sklearn.utils.parallel.delayed`/`Parallel` pairing from inside its joblib
# threading workers, once per tree per predict() call. Normal warnings
# filters (including PYTHONWARNINGS=ignore and filterwarnings()) do not
# suppress it -- it bypasses filter matching -- so during the df-native
# latency mode (n_jobs=-1, 1000 samples x 5 splits) it was writing >100MB of
# log per model. Patching warnings.warn() directly is the only thing that
# stops it; everything else (ConvergenceWarning, FutureWarning, etc.) still
# passes through unchanged.
_orig_warn = warnings.warn
def _filtered_warn(message, category=UserWarning, *args, **kwargs):
    if "sklearn.utils.parallel.delayed" in str(message):
        return None
    return _orig_warn(message, category, *args, **kwargs)
warnings.warn = _filtered_warn

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pmu_dataset.csv"

TAG = os.environ.get("PMU_TAG", "local")
SMOKE = os.environ.get("PMU_SMOKE", "0") == "1"
RESULTS_DIR = ROOT / "results" / "runs" / (TAG + ("-smoke" if SMOKE else ""))
RAW_DIR = RESULTS_DIR / "raw"

SEEDS = [42, 43] if SMOKE else [42, 43, 44, 45, 46]
TEST_SIZE = 0.2

# Latency measurement protocol (see latency.py)
LATENCY_SAMPLES = 60 if SMOKE else 1000
LATENCY_WARMUP = 10 if SMOKE else 100

# Reference budgets (ms). Streaming is the unconditional one.
STREAMING_BUDGET_MS = 1000 / 60  # IEEE C37.118.2 @ 60 frames/s
TRIP_BUDGET_MS = 3.0             # IEC 61850-5 TT6 (illustrative only)
ALERTING_BUDGET_MS = 100.0       # IEC 61850-5 TT3 (SOC / control-centre tier)

SVM_SUBSAMPLE = 2000
CLASS_NAMES = ["Attack", "Natural", "NoEvents"]  # LabelEncoder order

MAIN_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "MLP (Full)",
    "MLP (Compact)",
]
SUPPLEMENTARY_MODELS = ["MLP (Full, oversampled)", "MLP (Compact, oversampled)"]


# --------------------------------------------------------------------- data
def blocked_split(groups, seed, chunk=250, purge=25, test_size=TEST_SIZE):
    """Contiguous-block split that prevents neighbouring-frame leakage.

    Every source file is cut into consecutive chunks of `chunk` frames; a
    random `test_size` share of the chunks becomes the test set. Training rows
    within `purge` frames of any test chunk are dropped, so no training frame
    sits next to a test frame. (A plain chronological 80/20 split is degenerate
    on this data: the last 20% of each file is ~92% Natural events.)"""
    groups = np.asarray(groups)
    pos = pd.Series(groups).groupby(groups).cumcount().to_numpy()
    chunk_id = pos // chunk
    rng = np.random.RandomState(seed)
    test_mask = np.zeros(len(groups), dtype=bool)
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        chunks = np.unique(chunk_id[rows])
        n_test = max(1, int(round(test_size * len(chunks))))
        picked = rng.choice(chunks, size=n_test, replace=False)
        test_mask[rows[np.isin(chunk_id[rows], picked)]] = True
    dilated = np.zeros(len(groups), dtype=bool)
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]  # file rows are contiguous and in time order
        dilated[rows] = np.convolve(test_mask[rows].astype(float), np.ones(2 * purge + 1), mode="same") > 0
    return np.where(~dilated)[0], np.where(test_mask)[0]


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def feature_group(col):
    """Physical channel family of a feature column (used for grouped importance)."""
    if re.match(r"^R\d-PM\d+:V$", col):
        return "Voltage magnitude (V)"
    if re.match(r"^R\d-PM\d+:I$", col):
        return "Current magnitude (I)"
    if re.match(r"^R\d-PA\d+:VH$", col):
        return "Voltage phase angle (VH)"
    if re.match(r"^R\d-PA\d+:IH$", col):
        return "Current phase angle (IH)"
    if re.match(r"^R\d:F$", col) or re.match(r"^R\d:DF$", col):
        return "Frequency (F, dF/dt)"
    if re.match(r"^R\d-PA:Z$", col):
        return "Impedance magnitude (Z)"
    if re.match(r"^R\d-PA:ZH$", col):
        return "Impedance angle (ZH)"
    if re.match(r"^R\d:S$", col):
        return "Status flag (S)"
    return "Log flags (control panel / relay / Snort)"


def relay_of(col):
    m = re.match(r"^R(\d)[-:]", col)
    if m:
        return int(m.group(1))
    m = re.match(r"^relay(\d)_log$", col)
    if m:
        return int(m.group(1))
    return None


def is_log_flag(col):
    return col.startswith(("control_panel_log", "relay", "snort_log"))


def load_data(smoke_rows=6000):
    """Returns X (DataFrame, 128 features), y (int array), groups (source_file),
    row_order (position of each row inside its own source file, 0..1)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/download_data.py first.")
    df = pd.read_csv(DATA_PATH)
    if "source_file" not in df.columns:
        raise RuntimeError(
            "data/pmu_dataset.csv has no 'source_file' column. Re-run scripts/download_data.py "
            "(delete data/pmu_dataset.csv first)."
        )
    if SMOKE:
        # keep temporal structure: take every k-th row
        step = max(1, len(df) // smoke_rows)
        df = df.iloc[::step].reset_index(drop=True)

    groups = df["source_file"].to_numpy()
    pos = df.groupby("source_file").cumcount().to_numpy()
    size = df.groupby("source_file")["marker"].transform("size").to_numpy()
    rel_pos = pos / size

    X = df.drop(columns=["marker", "source_file"])
    # Impedance columns (R*-PA:Z) divide by ~0 current -> +inf. Treat as missing;
    # the pipeline imputer fills them using training-fold medians only.
    X = X.replace([np.inf, -np.inf], np.nan)

    y = df["marker"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise RuntimeError("Unexpected label values in 'marker'")
    y = y.astype(int)
    return X, y, groups, rel_pos


PROTOCOLS = ("random", "blocked", "group5")


def protocol_suffix(protocol):
    """File-name suffix so non-default protocols never overwrite random-split outputs."""
    return "" if protocol == "random" else f"_{protocol}"


def iter_splits(protocol, X, y, groups):
    """Yield (split_i, seed, train_idx, test_idx) for an evaluation protocol.

    random   five stratified random 80/20 row splits (the submitted paper's protocol)
    blocked  five contiguous-chunk splits with a purge gap (see blocked_split)
    group5   five folds that hold out 3 of the 15 source files each (leave-scenarios-out)
    """
    if protocol == "random":
        for i, seed in enumerate(SEEDS):
            tr, te = stratified_split(X, y, seed)
            yield i, seed, tr, te
    elif protocol == "blocked":
        for i, seed in enumerate(SEEDS):
            tr, te = blocked_split(groups, seed)
            yield i, seed, tr, te
    elif protocol == "group5":
        for i, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
            yield i, SEEDS[0], tr, te
    else:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")


def stratified_split(X, y, seed):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    return next(splitter.split(X, y))


# ------------------------------------------------------------------- models
class OversampledClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-compatible wrapper: random-oversamples minority classes to the
    majority count inside fit(), so cross-validation never leaks. Used for the
    MLPs because sklearn's MLPClassifier has no class/sample weighting."""

    def __init__(self, base, seed=0):
        self.base = base
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        target = counts.max()
        idx = []
        for c in classes:
            members = np.where(y == c)[0]
            extra = rng.choice(members, size=target - len(members), replace=True) if len(members) < target else []
            idx.append(np.concatenate([members, extra]).astype(int))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        self.base.fit(np.asarray(X)[idx], y[idx])
        self.classes_ = self.base.classes_
        return self

    def predict(self, X):
        return self.base.predict(X)

    def predict_proba(self, X):
        return self.base.predict_proba(X)


def build_model_specs(smoke=SMOKE):
    """{name: (factory(seed) -> estimator, weight_mode)}; weight_mode is
    'class_weight' (native), 'sample_weight' (XGBoost fit kwarg) or 'none'."""
    n_est = 20 if smoke else 200
    mlp_iter = 30 if smoke else 300
    return {
        "Logistic Regression": (
            lambda s: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=s),
            "class_weight",
        ),
        "Decision Tree": (
            lambda s: DecisionTreeClassifier(class_weight="balanced", random_state=s),
            "class_weight",
        ),
        "Random Forest": (
            lambda s: RandomForestClassifier(n_estimators=n_est, class_weight="balanced", random_state=s, n_jobs=-1),
            "class_weight",
        ),
        "XGBoost": (
            lambda s: XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.1,
                                    eval_metric="mlogloss", random_state=s, n_jobs=-1),
            "sample_weight",
        ),
        "MLP (Full)": (
            lambda s: MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=mlp_iter, random_state=s),
            "none",
        ),
        "MLP (Compact)": (
            lambda s: MLPClassifier(hidden_layer_sizes=(16,), max_iter=mlp_iter, random_state=s),
            "none",
        ),
        "MLP (Full, oversampled)": (
            lambda s: OversampledClassifier(
                MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=mlp_iter, random_state=s), seed=s),
            "none",
        ),
        "MLP (Compact, oversampled)": (
            lambda s: OversampledClassifier(
                MLPClassifier(hidden_layer_sizes=(16,), max_iter=mlp_iter, random_state=s), seed=s),
            "none",
        ),
    }


def make_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", estimator),
    ])


def fit_pipeline(name, spec, X_train, y_train, seed):
    factory, weight_mode = spec
    pipe = make_pipeline(factory(seed))
    t0 = time.perf_counter()
    if weight_mode == "sample_weight":
        pipe.fit(X_train, y_train, clf__sample_weight=compute_sample_weight("balanced", y_train))
    else:
        pipe.fit(X_train, y_train)
    return pipe, time.perf_counter() - t0


def final_estimator(pipe):
    est = pipe.named_steps["clf"]
    return getattr(est, "base", est)  # unwrap OversampledClassifier


def set_single_thread(pipe):
    """Force the estimator to one thread (n_jobs / nthread) for latency timing."""
    est = final_estimator(pipe)
    params = est.get_params() if hasattr(est, "get_params") else {}
    if "n_jobs" in params:
        est.set_params(n_jobs=1)
    return pipe


def model_complexity(pipe):
    """Descriptive size info: tree/node/parameter counts."""
    est = final_estimator(pipe)
    info = {}
    if isinstance(est, DecisionTreeClassifier):
        info.update(n_nodes=int(est.tree_.node_count), depth=int(est.get_depth()), n_leaves=int(est.get_n_leaves()))
    elif isinstance(est, RandomForestClassifier):
        info.update(n_trees=len(est.estimators_),
                    n_nodes=int(sum(t.tree_.node_count for t in est.estimators_)),
                    depth=int(max(t.get_depth() for t in est.estimators_)))
    elif isinstance(est, XGBClassifier):
        booster = est.get_booster()
        info.update(n_trees=int(len(booster.get_dump())), depth=int(est.get_params().get("max_depth") or 0))
    elif isinstance(est, MLPClassifier):
        info.update(n_params=int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_)))
    elif isinstance(est, LogisticRegression):
        info.update(n_params=int(est.coef_.size + est.intercept_.size))
    elif hasattr(est, "n_support_"):
        info.update(n_support_vectors=int(est.n_support_.sum()))
    return info


def size_kb(pipe):
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        path = tmp.name
    try:
        joblib.dump(pipe, path)
        return os.path.getsize(path) / 1024
    finally:
        os.remove(path)


# ------------------------------------------------------------------ metrics
def full_metrics(y_true, y_pred, proba=None):
    """Everything we might want later, flat dict (one row)."""
    labels = list(range(len(CLASS_NAMES)))
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "precision_macro": precision_recall_fscore_support(y_true, y_pred, average="macro", labels=labels, zero_division=0)[0],
        "recall_macro": precision_recall_fscore_support(y_true, y_pred, average="macro", labels=labels, zero_division=0)[1],
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, c in enumerate(CLASS_NAMES):
        out[f"precision_{c}"], out[f"recall_{c}"], out[f"f1_{c}"], out[f"support_{c}"] = p[i], r[i], f[i], int(s[i])

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    # one-vs-rest TP/FP/TN/FN and rates for each class
    for i, c in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out.update({
            f"tp_{c}": int(tp), f"fp_{c}": int(fp), f"tn_{c}": int(tn), f"fn_{c}": int(fn),
            f"tpr_{c}": tp / (tp + fn) if tp + fn else np.nan,
            f"tnr_{c}": tn / (tn + fp) if tn + fp else np.nan,
            f"fpr_{c}": fp / (fp + tn) if fp + tn else np.nan,
            f"fnr_{c}": fn / (fn + tp) if fn + tp else np.nan,
        })
    # Security-oriented binary view: Attack vs everything else
    out["attack_missed_as_natural"] = cm[0, 1] / cm[0].sum() if cm[0].sum() else np.nan
    out["attack_missed_as_noevents"] = cm[0, 2] / cm[0].sum() if cm[0].sum() else np.nan
    out["natural_flagged_as_attack"] = cm[1, 0] / cm[1].sum() if cm[1].sum() else np.nan

    if proba is not None:
        # AUC computed per class one-vs-rest (version-proof; avoids multi_class kwargs)
        try:
            aucs = []
            for i in labels:
                yi = (y_true == i).astype(int)
                if yi.min() == yi.max():
                    aucs.append(np.nan)
                else:
                    aucs.append(roc_auc_score(yi, proba[:, i]))
            for i, c in enumerate(CLASS_NAMES):
                out[f"roc_auc_{c}_vs_rest"] = aucs[i]
            out["roc_auc_ovr_macro"] = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else np.nan
            out["log_loss"] = log_loss(y_true, proba, labels=labels)
        except Exception:  # noqa: BLE001
            pass
    return out


def confusion_long(model, split, y_true, y_pred, extra=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    rows = []
    for i, t in enumerate(CLASS_NAMES):
        for j, p in enumerate(CLASS_NAMES):
            row = {"model": model, "split": split, "true_label": t, "predicted_label": p, "count": int(cm[i, j])}
            if extra:
                row.update(extra)
            rows.append(row)
    return rows


# ------------------------------------------------------------------ latency
def _stats(times_ms):
    t = np.asarray(times_ms)
    return {
        "mean": float(t.mean()), "std": float(t.std()),
        "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
        "p99": float(np.percentile(t, 99)), "max": float(t.max()), "n": int(len(t)),
    }


def _time_calls(fn, rows, warmup):
    times = []
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        fn(row)
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(dt)
    return times


def measure_latency(pipe, X_test, n=None, warmup=None, seed=0):
    """Latency of single-frame inference, several ways, in ms.

    pipeline_df_native : DataFrame row -> full pipeline, estimator as trained
                         (reproduces the original submission's protocol incl.
                         n_jobs=-1 thread dispatch for RF/XGBoost)
    pipeline_np_1thread: numpy row -> full pipeline, estimator forced to 1 thread
    model_only_1thread : pre-imputed/scaled numpy row -> bare estimator, 1 thread
    preprocess_only    : imputer + scaler on one numpy row
    Reporting mean/std/p50/p95/p99/max for each.
    """
    n = n or LATENCY_SAMPLES
    warmup = LATENCY_WARMUP if warmup is None else warmup
    total = n + warmup
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_test), size=min(total, len(X_test)), replace=total > len(X_test))

    out = {}
    df_rows = [X_test.iloc[[i]] for i in idx]
    np_rows = [X_test.iloc[[i]].to_numpy(dtype=float) for i in idx]

    out["pipeline_df_native"] = _stats(_time_calls(pipe.predict, df_rows, warmup))

    import copy
    pipe1 = set_single_thread(copy.deepcopy(pipe))
    # numpy rows carry no column names; drop the fitted names so sklearn does
    # not emit a (slow) warning on every call.
    for _, step in pipe1.steps:
        if hasattr(step, "feature_names_in_"):
            del step.feature_names_in_
    out["pipeline_np_1thread"] = _stats(_time_calls(pipe1.predict, np_rows, warmup))

    prep = Pipeline(pipe1.steps[:-1])
    est = pipe1.steps[-1][1]
    out["preprocess_only"] = _stats(_time_calls(prep.transform, np_rows, warmup))
    scaled = [prep.transform(r) for r in np_rows]
    out["model_only_1thread"] = _stats(_time_calls(est.predict, scaled, warmup))
    return out


def flatten_latency(lat, prefix="lat"):
    row = {}
    for mode, st in lat.items():
        for k, v in st.items():
            if k != "n":
                row[f"{prefix}_{mode}_{k}"] = v
    return row


def meets_budgets(latency_ms):
    return {
        "meets_streaming": latency_ms <= STREAMING_BUDGET_MS,
        "meets_trip_budget": latency_ms <= TRIP_BUDGET_MS,
        "meets_alerting": latency_ms <= ALERTING_BUDGET_MS,
    }


# ---------------------------------------------------------------- bookkeeping
def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def env_summary():
    import sklearn, xgboost, scipy
    info = {
        "tag": TAG, "smoke": SMOKE,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_mhz_current"] = round(freq.current)
            info["cpu_freq_mhz_max"] = round(freq.max)
    except Exception:
        pass
    return info


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
