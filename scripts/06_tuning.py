"""
Hyper-parameter tuning (E6): are the comparisons fair?

The submitted paper used library-default-ish settings for every model. This
script runs a small randomized search per model family (3-fold CV, macro-F1,
on a 20,000-row stratified subsample of split-0's training data), then refits
the best configuration on each of the five full training splits and records
the same metrics/latency as the main benchmark so the numbers are directly
comparable.

MLP is tuned with random oversampling of minority classes (scikit-learn's MLP
has no class/sample weights), applied inside each CV training fold only.

Outputs: tuning_best_params.json, tuning_cv_results.csv,
         tuned_per_split.csv, tuned_summary.csv, tuned_confusion_long.csv
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import common as C

N_ITER = 2 if C.SMOKE else 12
SEARCH_ROWS = 3000 if C.SMOKE else 20000


def search_space():
    mlp_iter = 30 if C.SMOKE else 200
    return {
        "Logistic Regression (tuned)": (
            lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": loguniform(1e-3, 1e2)}, False),
        "Decision Tree (tuned)": (
            lambda: DecisionTreeClassifier(class_weight="balanced"),
            {"clf__max_depth": [6, 8, 10, 12, 16, 20, 30, None],
             "clf__min_samples_leaf": [1, 2, 5, 10, 20],
             "clf__criterion": ["gini", "entropy"]}, False),
        "Random Forest (tuned)": (
            lambda: RandomForestClassifier(class_weight="balanced", n_jobs=-1),
            {"clf__n_estimators": [10, 25, 50, 100, 200],
             "clf__max_depth": [10, 15, 20, None],
             "clf__min_samples_leaf": [1, 2, 5],
             "clf__max_features": ["sqrt", 0.3]}, False),
        "XGBoost (tuned)": (
            lambda: XGBClassifier(eval_metric="mlogloss", n_jobs=-1),
            {"clf__n_estimators": [50, 100, 200, 400],
             "clf__max_depth": [3, 4, 6, 8, 10],
             "clf__learning_rate": [0.05, 0.1, 0.2, 0.3],
             "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.5, 0.8, 1.0]}, True),
        "MLP (tuned, oversampled)": (
            lambda: C.OversampledClassifier(MLPClassifier(max_iter=mlp_iter, early_stopping=True, random_state=0), seed=0),
            {"clf__base__hidden_layer_sizes": [(16,), (32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
             "clf__base__alpha": loguniform(1e-6, 1e-2),
             "clf__base__learning_rate_init": loguniform(1e-4, 1e-2)}, False),
    }


def set_params_on_pipe(pipe, params):
    pipe.set_params(**params)
    return pipe


def main():
    C.ensure_dirs()
    X, y, groups, _ = C.load_data()
    seed0 = C.SEEDS[0]
    tr0, _ = C.stratified_split(X, y, seed0)

    # stratified search subsample drawn from split-0 training rows only
    rng = np.random.RandomState(seed0)
    sub = []
    for c in np.unique(y[tr0]):
        members = tr0[y[tr0] == c]
        take = max(10, int(round(SEARCH_ROWS * len(members) / len(tr0))))
        sub.append(rng.choice(members, size=min(take, len(members)), replace=False))
    sub = np.concatenate(sub)
    X_s, y_s = X.iloc[sub], y[sub]
    C.log(f"Search subsample: {len(sub)} rows")

    best_params, cv_rows, rows, conf_rows = {}, [], [], []
    for name, (factory, space, uses_sw) in search_space().items():
        C.log(f"--- tuning {name} ({N_ITER} candidates x 3 folds)")
        pipe = C.make_pipeline(factory())
        search = RandomizedSearchCV(pipe, space, n_iter=N_ITER, scoring="f1_macro",
                                    cv=StratifiedKFold(3, shuffle=True, random_state=seed0),
                                    random_state=seed0, n_jobs=1, refit=False, error_score="raise")
        fit_kwargs = {"clf__sample_weight": compute_sample_weight("balanced", y_s)} if uses_sw else {}
        t0 = time.perf_counter()
        search.fit(X_s, y_s, **fit_kwargs)
        C.log(f"    search took {time.perf_counter() - t0:.0f}s, best CV macro-F1={search.best_score_:.3f}")
        best_params[name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()}
        res = pd.DataFrame(search.cv_results_)
        res.insert(0, "model", name)
        cv_rows.append(res)

        for split_i, seed in enumerate(C.SEEDS):
            tr, te = C.stratified_split(X, y, seed)
            pipe = C.make_pipeline(factory())
            pipe.set_params(**search.best_params_)
            est = C.final_estimator(pipe)
            if "random_state" in est.get_params():
                est.set_params(random_state=seed)
            t0 = time.perf_counter()
            if uses_sw:
                pipe.fit(X.iloc[tr], y[tr], clf__sample_weight=compute_sample_weight("balanced", y[tr]))
            else:
                pipe.fit(X.iloc[tr], y[tr])
            fit_s = time.perf_counter() - t0
            y_pred = pipe.predict(X.iloc[te])
            proba = pipe.predict_proba(X.iloc[te])
            row = {"model": name, "split": split_i, "seed": seed, "n_train": len(tr), "n_test": len(te)}
            row.update(C.full_metrics(y[te], y_pred, proba))
            row["train_time_s"] = fit_s
            row["size_kb"] = C.size_kb(pipe)
            row.update(C.model_complexity(pipe))
            row.update(C.flatten_latency(C.measure_latency(pipe, X.iloc[te])))
            rows.append(row)
            conf_rows.extend(C.confusion_long(name, split_i, y[te], y_pred))
            C.log(f"    split {split_i}: acc={row['accuracy']:.3f} f1m={row['f1_macro']:.3f} "
                  f"lat(np,1t)={row['lat_pipeline_np_1thread_mean']:.2f}ms size={row['size_kb']:.0f}KB")

        pd.concat(cv_rows).to_csv(C.RESULTS_DIR / "tuning_cv_results.csv", index=False)
        C.save_json(best_params, C.RESULTS_DIR / "tuning_best_params.json")
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "tuned_per_split.csv", index=False)

    per = pd.DataFrame(rows)
    num = per.select_dtypes("number").drop(columns=["split"], errors="ignore")
    g = per.groupby("model", sort=False)
    summary = pd.concat([g[num.columns].mean().add_suffix("_mean"), g[num.columns].std().add_suffix("_std")], axis=1).reset_index()
    summary.to_csv(C.RESULTS_DIR / "tuned_summary.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(C.RESULTS_DIR / "tuned_confusion_long.csv", index=False)
    C.log("Done.")


if __name__ == "__main__":
    main()
