"""
Main benchmark (E1 + fuller metric capture).

For every model x 5 stratified splits (same seeds/protocol as the submitted
paper) it records:
  * all classification metrics (accuracy, balanced acc., macro/weighted P/R/F1,
    per-class P/R/F1, one-vs-rest TP/FP/TN/FN/TPR/TNR/FPR/FNR, MCC, kappa,
    ROC-AUC, log-loss)
  * the confusion matrix of every split (long format) and raw predictions
  * training time, serialized size, model complexity (nodes/trees/params)
  * single-frame latency in four modes (see common.measure_latency), with
    mean/std/p50/p95/p99/max -- including the original protocol so the new
    numbers can be compared with the submitted ones.

Outputs (results/runs/<tag>/):
  main_per_split.csv, main_summary.csv, main_confusion_long.csv, raw/pred_*.npz
"""

import re
import time

import numpy as np
import pandas as pd
from sklearn.svm import SVC

import common as C


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def summarize(per_split):
    num = per_split.select_dtypes("number").drop(columns=["split"], errors="ignore")
    g = per_split.groupby("model", sort=False)
    mean = g[num.columns].mean().add_suffix("_mean")
    std = g[num.columns].std().add_suffix("_std")
    out = pd.concat([mean, std], axis=1).reset_index()
    out["n_splits"] = g.size().to_numpy()
    return out


def evaluate(name, pipe, fit_s, X_test, y_test, split_i, seed, n_train, rows, conf_rows):
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    row = {"model": name, "split": split_i, "seed": seed, "n_train": n_train, "n_test": len(y_test)}
    row.update(C.full_metrics(y_test, y_pred, proba))
    row["train_time_s"] = fit_s
    row["size_kb"] = C.size_kb(pipe)
    row.update(C.model_complexity(pipe))
    lat = C.measure_latency(pipe, X_test)
    row.update(C.flatten_latency(lat))
    rows.append(row)
    conf_rows.extend(C.confusion_long(name, split_i, y_test, y_pred))
    np.savez_compressed(
        C.RAW_DIR / f"pred_{slug(name)}_split{split_i}.npz",
        y_true=y_test, y_pred=y_pred, proba=np.asarray(proba, dtype=np.float32) if proba is not None else np.empty(0),
    )
    C.log(f"  {name:<28} acc={row['accuracy']:.3f} f1m={row['f1_macro']:.3f} "
          f"lat(df,native)={lat['pipeline_df_native']['mean']:.2f}ms "
          f"lat(model,1t)={lat['model_only_1thread']['mean']:.3f}ms size={row['size_kb']:.0f}KB")


def main():
    C.ensure_dirs()
    X, y, groups, _ = C.load_data()
    C.log(f"Loaded {len(X)} rows x {X.shape[1]} features")
    specs = C.build_model_specs()
    names = C.MAIN_MODELS + C.SUPPLEMENTARY_MODELS

    rows, conf_rows = [], []
    for split_i, seed in enumerate(C.SEEDS):
        C.log(f"=== split {split_i + 1}/{len(C.SEEDS)} seed={seed} ===")
        tr, te = C.stratified_split(X, y, seed)
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
        for name in names:
            pipe, fit_s = C.fit_pipeline(name, specs[name], X_tr, y_tr, seed)
            evaluate(name, pipe, fit_s, X_te, y_te, split_i, seed, len(tr), rows, conf_rows)

    # Kernel SVM: same subsampling protocol as the submitted paper (random
    # 2,000-row subsample per split, stratified 80/20 inside it).
    C.log("=== SVM (RBF) on 2,000-row subsample, paper protocol ===")
    rng = np.random.RandomState(0)
    for split_i, seed in enumerate(C.SEEDS):
        idx = rng.choice(len(X), size=min(C.SVM_SUBSAMPLE, len(X)), replace=False)
        Xs, ys = X.iloc[idx], y[idx]
        tr, te = C.stratified_split(Xs, ys, seed)
        pipe = C.make_pipeline(SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=seed))
        t0 = time.perf_counter()
        pipe.fit(Xs.iloc[tr], ys[tr])
        fit_s = time.perf_counter() - t0
        evaluate("SVM (RBF, n=2000)", pipe, fit_s, Xs.iloc[te], ys[te], split_i, seed, len(tr), rows, conf_rows)

    per_split = pd.DataFrame(rows)
    per_split.to_csv(C.RESULTS_DIR / "main_per_split.csv", index=False)
    summarize(per_split).to_csv(C.RESULTS_DIR / "main_summary.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(C.RESULTS_DIR / "main_confusion_long.csv", index=False)
    C.save_json({"env": C.env_summary(), "seeds": C.SEEDS, "test_size": C.TEST_SIZE,
                 "latency_samples": C.LATENCY_SAMPLES, "latency_warmup": C.LATENCY_WARMUP},
                C.RESULTS_DIR / "main_config.json")
    C.log(f"Done. Wrote main_*.csv to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
