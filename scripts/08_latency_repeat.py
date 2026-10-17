"""
Latency-only repeatability study (step 08). Run this on the machine whose latency
the paper reports (the edge device), with the machine idle.

Why: in the first full run the same 200-tree Random Forest measured 9.8 ms, 14.5 ms
and 16.2 ms (1 thread, numpy row) in different steps, and step 01 ran while a
sklearn warning flood was being written to the log. This script re-measures the
latency of the models that matter, several times, each repeat in a FRESH Python
process (cold caches, fresh allocator), in a different random model order, with a
sentinel model measured first and last in every repeat. It records CPU frequency
and CPU load around every measurement, so drift is visible instead of guessed.

Models are fitted once (split 0, seed 42, same data/protocol as step 01), saved to
results/runs/<tag>/.cache_models/, and loaded by each worker process.

Measured per model, in four modes (see common.measure_latency):
  pipeline_df_native   DataFrame row -> pipeline, estimator threads as trained
                       (the protocol of the submitted paper)
  pipeline_np_1thread  numpy row -> pipeline, one thread
  preprocess_only      imputer + scaler
  model_only_1thread   bare estimator, one thread
each with mean/std/p50/p95/p99/max over LATENCY_SAMPLES calls after warm-up.

Outputs (results/runs/<tag>/):
  latency_repeat_raw.csv       one row per repeat x model x mode
  latency_repeat_summary.csv   per model x mode: mean of repeat means, std and CV across
                               repeats, min/max of repeat means, mean p99
  latency_repeat_models.csv    size and complexity of each fitted model

Usage:
  python scripts/run_all.py --tag edge-dell-rep --only 00 08
  python scripts/08_latency_repeat.py --repeats 3         # direct, uses PMU_TAG
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import common as C

SENTINEL = "Random Forest (200)"
MODEL_DIR = C.RESULTS_DIR / ".cache_models"
TUNED_XGB = {  # best configuration found by step 06 on the edge device (random protocol)
    "n_estimators": 400, "max_depth": 6, "learning_rate": 0.3, "subsample": 0.7, "colsample_bytree": 0.8,
}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def model_builders():
    """name -> (builder(seed) -> estimator, weight_mode)"""
    specs = C.build_model_specs()
    out = {
        "Logistic Regression": (specs["Logistic Regression"][0], "class_weight"),
        "Decision Tree": (specs["Decision Tree"][0], "class_weight"),
        "MLP (Full)": (specs["MLP (Full)"][0], "none"),
        "MLP (Compact)": (specs["MLP (Compact)"][0], "none"),
        "XGBoost (200, depth 6)": (specs["XGBoost"][0], "sample_weight"),
        "XGBoost (tuned)": (lambda s: XGBClassifier(eval_metric="mlogloss", random_state=s, n_jobs=-1, **TUNED_XGB),
                            "sample_weight"),
    }
    for n in (200, 100, 50, 25, 10):
        out[f"Random Forest ({n})"] = (
            lambda s, n=n: RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=s, n_jobs=-1),
            "class_weight")
    out["SVM (RBF, n=2000)"] = (lambda s: SVC(kernel="rbf", class_weight="balanced", random_state=s), "svm")
    return out


def load_split0():
    X, y, groups, _ = C.load_data()
    seed = C.SEEDS[0]
    tr, te = C.stratified_split(X, y, seed)
    return X, y, tr, te, seed


def cpu_snapshot():
    snap = {"cpu_freq_mhz": np.nan, "cpu_load_pct": np.nan}
    try:
        import psutil
        f = psutil.cpu_freq()
        snap["cpu_freq_mhz"] = round(f.current) if f else np.nan
        snap["cpu_load_pct"] = psutil.cpu_percent(interval=1.0)
    except Exception:  # noqa: BLE001
        pass
    return snap


def fit_all():
    C.ensure_dirs()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y, tr, te, seed = load_split0()
    rows = []
    for name, (builder, mode) in model_builders().items():
        C.log(f"fitting {name}")
        t0 = time.perf_counter()
        if mode == "svm":
            rng = np.random.RandomState(0)
            idx = rng.choice(len(X), size=C.SVM_SUBSAMPLE, replace=False)
            Xs, ys = X.iloc[idx], y[idx]
            tr_s, _ = C.stratified_split(Xs, ys, seed)
            pipe = C.make_pipeline(builder(seed))
            pipe.fit(Xs.iloc[tr_s], ys[tr_s])
        elif mode == "sample_weight":
            pipe = C.make_pipeline(builder(seed))
            pipe.fit(X.iloc[tr], y[tr], clf__sample_weight=compute_sample_weight("balanced", y[tr]))
        else:
            pipe = C.make_pipeline(builder(seed))
            pipe.fit(X.iloc[tr], y[tr])
        fit_s = time.perf_counter() - t0
        joblib.dump(pipe, MODEL_DIR / f"{slug(name)}.joblib")
        proba_ok = hasattr(pipe, "predict_proba")
        yp = pipe.predict(X.iloc[te])
        rows.append({"model": name, "fit_s": fit_s, "size_kb": C.size_kb(pipe),
                     "accuracy_split0": float((yp == y[te]).mean()),
                     "f1_macro_split0": C.full_metrics(y[te], yp)["f1_macro"], **C.model_complexity(pipe)})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "latency_repeat_models.csv", index=False)
    return list(model_builders().keys())


def worker(repeat, names):
    X, y, tr, te, seed = load_split0()
    X_te = X.iloc[te]
    rng = np.random.RandomState(1000 + repeat)
    order = [n for n in names if n != SENTINEL]
    rng.shuffle(order)
    order = [SENTINEL] + order + [SENTINEL]
    rows = []
    for pos, name in enumerate(order):
        pipe = joblib.load(MODEL_DIR / f"{slug(name)}.joblib")
        before = cpu_snapshot()
        lat = C.measure_latency(pipe, X_te)
        after = cpu_snapshot()
        for mode, st in lat.items():
            rows.append({"repeat": repeat, "position": pos, "is_sentinel_repeat": name == SENTINEL and pos > 0 and pos == len(order) - 1,
                         "model": name, "mode": mode, **{k: v for k, v in st.items()},
                         "cpu_freq_before": before["cpu_freq_mhz"], "cpu_load_before": before["cpu_load_pct"],
                         "cpu_freq_after": after["cpu_freq_mhz"], "cpu_load_after": after["cpu_load_pct"],
                         "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S")})
        C.log(f"[repeat {repeat}] {pos + 1}/{len(order)} {name:<26} "
              f"np1t={lat['pipeline_np_1thread']['mean']:.2f}ms model1t={lat['model_only_1thread']['mean']:.3f}ms "
              f"df={lat['pipeline_df_native']['mean']:.2f}ms  cpu {before['cpu_freq_mhz']}MHz/{before['cpu_load_pct']}%")
        del pipe
    pd.DataFrame(rows).to_csv(C.RAW_DIR / f"latency_repeat_{repeat}.csv", index=False)


def summarize(repeats):
    frames = [pd.read_csv(C.RAW_DIR / f"latency_repeat_{r}.csv") for r in range(repeats)]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(C.RESULTS_DIR / "latency_repeat_raw.csv", index=False)
    main = raw[~raw.is_sentinel_repeat]            # sentinel's closing measurement handled separately
    g = main.groupby(["model", "mode", "repeat"], sort=False)["mean"].mean().reset_index()
    s = g.groupby(["model", "mode"], sort=False)["mean"].agg(
        repeat_mean="mean", repeat_std="std", repeat_min="min", repeat_max="max").reset_index()
    s["cv_pct"] = 100 * s.repeat_std / s.repeat_mean
    p99 = main.groupby(["model", "mode"], sort=False)["p99"].mean().rename("p99_mean").reset_index()
    p50 = main.groupby(["model", "mode"], sort=False)["p50"].mean().rename("p50_mean").reset_index()
    s = s.merge(p50, on=["model", "mode"]).merge(p99, on=["model", "mode"])
    s["meets_streaming_mean"] = s.repeat_mean <= C.STREAMING_BUDGET_MS
    s["meets_streaming_p99"] = s.p99_mean <= C.STREAMING_BUDGET_MS
    s["meets_trip_mean"] = s.repeat_mean <= C.TRIP_BUDGET_MS
    s.to_csv(C.RESULTS_DIR / "latency_repeat_summary.csv", index=False)

    sent = raw[(raw.model == SENTINEL) & (raw["mode"] == "pipeline_np_1thread")][["repeat", "position", "mean", "cpu_freq_before", "cpu_load_before"]]
    sent.to_csv(C.RESULTS_DIR / "latency_repeat_sentinel_drift.csv", index=False)
    C.log("Summary (mean of repeat means, ms):")
    view = s.pivot(index="model", columns="mode", values="repeat_mean").round(3)
    print(view.to_string())
    C.log("Across-repeat CV% (pipeline_np_1thread):")
    print(s[s["mode"] == "pipeline_np_1thread"].set_index("model")["cv_pct"].round(1).to_string())
    C.log("Sentinel drift (RF-200, pipeline_np_1thread) per repeat/position:")
    print(sent.round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cooldown-s", type=float, default=20.0, help="pause between repeats")
    ap.add_argument("--worker", type=int, default=None, help="internal: run one repeat in this process")
    ap.add_argument("--keep-models", action="store_true", help="keep the cached fitted models")
    args = ap.parse_args()

    C.ensure_dirs()
    names = list(model_builders().keys())
    if args.worker is not None:
        worker(args.worker, names)
        return

    C.log(f"Latency repeatability: {args.repeats} repeats, {C.LATENCY_SAMPLES} samples (+{C.LATENCY_WARMUP} warm-up) per mode")
    C.log("Make sure the machine is idle (no browsers, no other benchmarks).")
    fit_all()
    for r in range(args.repeats):
        if r:
            C.log(f"cooling down {args.cooldown_s:.0f}s")
            time.sleep(args.cooldown_s)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(r)]
        C.log(f"=== repeat {r + 1}/{args.repeats} in a fresh process ===")
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=os.environ.copy())
        if proc.returncode != 0:
            raise SystemExit(f"worker {r} failed with exit code {proc.returncode}")
    summarize(args.repeats)
    if not args.keep_models:
        for f in MODEL_DIR.glob("*.joblib"):
            f.unlink()
        try:
            MODEL_DIR.rmdir()
        except OSError:
            pass
    C.save_json({"env": C.env_summary(), "repeats": args.repeats, "latency_samples": C.LATENCY_SAMPLES,
                 "latency_warmup": C.LATENCY_WARMUP, "tuned_xgb_params": TUNED_XGB,
                 "sentinel": SENTINEL}, C.RESULTS_DIR / "latency_repeat_config.json")
    C.log("Done.")


if __name__ == "__main__":
    main()
