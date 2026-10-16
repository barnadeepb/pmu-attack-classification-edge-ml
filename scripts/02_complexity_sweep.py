"""
Complexity sweep (E2): does a *smaller* ensemble fit the edge budgets?

Sweeps tree count / depth for Random Forest, XGBoost and depth for the
Decision Tree, on the same 5 splits as the main benchmark. For each setting it
records the full metric set, serialized size and latency (all four modes), so
the accuracy-latency-footprint Pareto front can be drawn.

Output: sweep_per_split.csv, sweep_summary.csv
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import common as C

if C.SMOKE:
    RF_TREES, RF_DEPTHS = [5, 10], [None, 12]
    XGB_TREES, XGB_DEPTHS = [10, 20], [3, 6]
    DT_DEPTHS = [6, None]
else:
    RF_TREES = [5, 10, 25, 50, 100, 200]
    RF_DEPTHS = [None, 20, 12]
    XGB_TREES = [10, 25, 50, 100, 200]
    XGB_DEPTHS = [3, 4, 6, 8]
    DT_DEPTHS = [4, 6, 8, 10, 12, 16, 20, None]


def settings():
    for d in DT_DEPTHS:
        yield ("Decision Tree", f"depth={d}", {"max_depth": d},
               lambda s, d=d: DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=s), False)
    for n in RF_TREES:
        for d in RF_DEPTHS:
            yield ("Random Forest", f"trees={n},depth={d}", {"n_trees": n, "max_depth": d},
                   lambda s, n=n, d=d: RandomForestClassifier(n_estimators=n, max_depth=d, class_weight="balanced",
                                                              random_state=s, n_jobs=-1), False)
    for n in XGB_TREES:
        for d in XGB_DEPTHS:
            yield ("XGBoost", f"trees={n},depth={d}", {"n_trees": n, "max_depth": d},
                   lambda s, n=n, d=d: XGBClassifier(n_estimators=n, max_depth=d, learning_rate=0.1,
                                                     eval_metric="mlogloss", random_state=s, n_jobs=-1), True)


def main():
    C.ensure_dirs()
    X, y, groups, _ = C.load_data()
    rows = []
    for split_i, seed in enumerate(C.SEEDS):
        C.log(f"=== split {split_i + 1}/{len(C.SEEDS)} ===")
        tr, te = C.stratified_split(X, y, seed)
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
        for family, label, params, factory, uses_sw in settings():
            pipe = C.make_pipeline(factory(seed))
            t0 = time.perf_counter()
            if uses_sw:
                pipe.fit(X_tr, y_tr, clf__sample_weight=compute_sample_weight("balanced", y_tr))
            else:
                pipe.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            y_pred = pipe.predict(X_te)
            row = {"family": family, "setting": label, "split": split_i, "seed": seed}
            row.update({f"param_{k}": v for k, v in params.items()})
            row.update(C.full_metrics(y_te, y_pred, pipe.predict_proba(X_te)))
            row["train_time_s"] = fit_s
            row["size_kb"] = C.size_kb(pipe)
            row.update(C.model_complexity(pipe))
            lat = C.measure_latency(pipe, X_te)
            row.update(C.flatten_latency(lat))
            rows.append(row)
            C.log(f"  {family:<14}{label:<22} f1m={row['f1_macro']:.3f} "
                  f"lat(np,1t)={lat['pipeline_np_1thread']['mean']:.2f}ms size={row['size_kb']:.0f}KB")

    per = pd.DataFrame(rows)
    per.to_csv(C.RESULTS_DIR / "sweep_per_split.csv", index=False)
    num = per.select_dtypes("number").drop(columns=["split", "seed"], errors="ignore")
    g = per.groupby(["family", "setting"], sort=False)
    summary = pd.concat([g[num.columns].mean().add_suffix("_mean"), g[num.columns].std().add_suffix("_std")], axis=1).reset_index()
    summary.to_csv(C.RESULTS_DIR / "sweep_summary.csv", index=False)
    C.log("Done.")


if __name__ == "__main__":
    main()
