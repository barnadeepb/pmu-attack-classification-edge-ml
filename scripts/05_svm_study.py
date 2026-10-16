"""
SVM study (E5): why the SVM was benchmarked on 2,000 rows, and what that hides.

Kernel-SVM training scales between O(n^2) and O(n^3) with training size, and
single-sample inference cost grows with the number of support vectors, so a
2,000-row SVM is both less accurate AND artificially fast/small compared with
an SVM fitted on the full data. This script quantifies both effects.

Part A  like-for-like: every model trained on the same 2,000-row subsamples
        (so the SVM sits on equal footing with the others)
Part B  learning curve: SVM / DT / RF / XGBoost at growing training sizes;
        records fit time, support-vector count, accuracy, latency, size.
        The SVM sweep stops once a single fit exceeds --svm-time-cap-s.
Part C  scalable kernel approximation: Nystroem features + linear SVM (SGD)
        trained on the FULL training split.

Outputs: svm_like_for_like.csv, svm_learning_curve.csv, svm_nystroem.csv
Flags:   --svm-time-cap-s 1200   --max-n 20000 (use 78377 for the full dataset; SVM may need hours)
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import common as C

SIZES = [400, 800] if C.SMOKE else [2000, 5000, 10000, 20000, 40000, 78377]


def subsample(X, y, n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X.iloc[idx], y[idx]


def eval_pipe(pipe, name, X_te, y_te, fit_s, extra):
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_te)
    batch_s = time.perf_counter() - t0
    proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
    row = dict(extra, model=name, n_test=len(y_te), train_time_s=fit_s, batch_predict_s=batch_s)
    row.update(C.full_metrics(y_te, y_pred, proba))
    row["size_kb"] = C.size_kb(pipe)
    row.update(C.model_complexity(pipe))
    lat = C.measure_latency(pipe, X_te, n=60 if len(X_te) < 5000 else 200, warmup=10)
    row.update(C.flatten_latency(lat))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svm-time-cap-s", type=float, default=1200.0)
    ap.add_argument("--max-n", type=int, default=20000,
                    help="largest total sample size in the learning curve (78377 = full data; can take hours for the SVM)")
    args = ap.parse_args()

    C.ensure_dirs()
    X, y, groups, _ = C.load_data()
    specs = C.build_model_specs()

    # ---------------------------------------------------------- Part A
    C.log("Part A: like-for-like at 2,000 rows")
    rowsA = []
    n_small = SIZES[0]
    for split_i, seed in enumerate(C.SEEDS):
        Xs, ys = subsample(X, y, n_small, seed)
        tr, te = C.stratified_split(Xs, ys, seed)
        for name in C.MAIN_MODELS:
            pipe, fit_s = C.fit_pipeline(name, specs[name], Xs.iloc[tr], ys[tr], seed)
            rowsA.append(eval_pipe(pipe, name, Xs.iloc[te], ys[te], fit_s, {"n_total": n_small, "split": split_i}))
        pipe = C.make_pipeline(SVC(kernel="rbf", class_weight="balanced", random_state=seed))
        t0 = time.perf_counter(); pipe.fit(Xs.iloc[tr], ys[tr]); fit_s = time.perf_counter() - t0
        rowsA.append(eval_pipe(pipe, "SVM (RBF)", Xs.iloc[te], ys[te], fit_s, {"n_total": n_small, "split": split_i}))
        C.log(f"  split {split_i} done")
    pd.DataFrame(rowsA).to_csv(C.RESULTS_DIR / "svm_like_for_like.csv", index=False)

    # ---------------------------------------------------------- Part B
    C.log("Part B: learning curve")
    rowsB = []
    svm_capped = False
    for n in SIZES:
        if n > args.max_n:
            continue
        seeds = C.SEEDS[:3] if n <= 10000 else C.SEEDS[:1]
        for split_i, seed in enumerate(seeds):
            Xs, ys = (X, y) if n >= len(X) else subsample(X, y, n, seed)
            tr, te = C.stratified_split(Xs, ys, seed)
            for name in ("Decision Tree", "Random Forest", "XGBoost"):
                pipe, fit_s = C.fit_pipeline(name, specs[name], Xs.iloc[tr], ys[tr], seed)
                rowsB.append(eval_pipe(pipe, name, Xs.iloc[te], ys[te], fit_s, {"n_total": n, "split": split_i}))
            if svm_capped:
                continue
            pipe = C.make_pipeline(SVC(kernel="rbf", class_weight="balanced", random_state=seed))
            t0 = time.perf_counter(); pipe.fit(Xs.iloc[tr], ys[tr]); fit_s = time.perf_counter() - t0
            row = eval_pipe(pipe, "SVM (RBF)", Xs.iloc[te], ys[te], fit_s, {"n_total": n, "split": split_i})
            rowsB.append(row)
            C.log(f"  n={n} SVM fit={fit_s:.0f}s SVs={row.get('n_support_vectors')} "
                  f"acc={row['accuracy']:.3f} lat(model,1t)={row['lat_model_only_1thread_mean']:.2f}ms")
            if fit_s > args.svm_time_cap_s:
                svm_capped = True
                C.log(f"  SVM fit exceeded {args.svm_time_cap_s:.0f}s; larger SVM sizes skipped")
        pd.DataFrame(rowsB).to_csv(C.RESULTS_DIR / "svm_learning_curve.csv", index=False)

    # ---------------------------------------------------------- Part C
    C.log("Part C: Nystroem + linear SVM on full training split")
    rowsC = []
    comps = [50, 100] if C.SMOKE else [100, 300, 1000]
    for split_i, seed in enumerate(C.SEEDS[:3]):
        tr, te = C.stratified_split(X, y, seed)
        for k in comps:
            pipe = Pipeline([
                ("impute", C.SimpleImputer(strategy="median")),
                ("scale", C.StandardScaler()),
                ("clf", Pipeline([
                    ("nys", Nystroem(kernel="rbf", gamma=1.0 / X.shape[1], n_components=k, random_state=seed)),
                    ("sgd", SGDClassifier(loss="modified_huber", alpha=1e-5, class_weight="balanced",
                                          max_iter=30, tol=None, random_state=seed)),
                ])),
            ])
            t0 = time.perf_counter(); pipe.fit(X.iloc[tr], y[tr]); fit_s = time.perf_counter() - t0
            rowsC.append(eval_pipe(pipe, f"Nystroem-{k} + linear SVM", X.iloc[te], y[te], fit_s,
                                   {"n_total": len(X), "split": split_i, "n_components": k}))
            C.log(f"  split {split_i} k={k} acc={rowsC[-1]['accuracy']:.3f} f1m={rowsC[-1]['f1_macro']:.3f}")
    pd.DataFrame(rowsC).to_csv(C.RESULTS_DIR / "svm_nystroem.csv", index=False)
    C.log("Done.")


if __name__ == "__main__":
    main()
