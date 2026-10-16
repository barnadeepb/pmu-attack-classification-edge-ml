"""
Scenario evaluation (E4): how do the models behave beyond the random 80/20 split?

Part 1 - protocol x feature-set grid
  protocols     random      stratified random rows (the submitted paper's protocol)
                blocked     contiguous blocks: each scenario file is cut into
                            250-frame chunks, 20% of chunks form the test set,
                            training rows within 25 frames of a test chunk are
                            purged (no neighbouring-frame leakage)
                group5      leave-scenarios-out: 5 folds, 3 of the 15 source
                            files held out per fold (unseen runs)
  feature sets  all128      128 features as in the paper
                pmu116      PMU measurement channels only (no Snort/relay/panel logs)
                logs12      the 12 log flags only (leakage diagnostic)

Part 2 - degradation tests (random split, all features, train on clean data):
  relay_loss_k   all PMU channels of relay k replaced by missing -> train-median
                 imputation (communication loss / failed PMU)
  logs_lost      all 12 log flags missing
  noise_s        zero-mean Gaussian noise, std = s x (per-feature training std),
                 added to the PMU channels only (measurement noise; NOT an
                 adversarial attack)

Outputs: scenario_results.csv, scenario_summary.csv, scenario_confusion_long.csv
Flags:   --quick   skip the two MLPs (they dominate runtime)
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as C

NOISE_LEVELS = [0.05, 0.1, 0.25, 0.5]
RANDOM_SEEDS = C.SEEDS[:3]
BLOCKED_SEEDS = C.SEEDS[:3]


def featuresets(columns):
    pmu = [c for c in columns if not C.is_log_flag(c)]
    logs = [c for c in columns if C.is_log_flag(c)]
    return {"all128": list(columns), "pmu116": pmu, "logs12": logs}


def run_one(model_name, specs, X_tr, y_tr, X_te, y_te, seed):
    pipe, fit_s = C.fit_pipeline(model_name, specs[model_name], X_tr, y_tr, seed)
    y_pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)
    return pipe, fit_s, y_pred, proba


def record(rows, conf_rows, meta, model_name, y_te, y_pred, proba, fit_s):
    row = dict(meta, model=model_name, n_test=len(y_te), train_time_s=fit_s)
    row.update(C.full_metrics(y_te, y_pred, proba))
    rows.append(row)
    conf_rows.extend(C.confusion_long(model_name, meta.get("fold", 0), y_te, y_pred,
                                      extra={k: meta[k] for k in ("scenario", "protocol", "featureset", "seed")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip MLP models")
    args = ap.parse_args()

    C.ensure_dirs()
    X, y, groups, _ = C.load_data()
    specs = C.build_model_specs()
    models = [m for m in C.MAIN_MODELS if not (args.quick and m.startswith("MLP"))]
    fsets = featuresets(X.columns)
    rows, conf_rows = [], []

    # ------------------------------------------------- part 1: protocol grid
    for fs_name, cols in fsets.items():
        Xf = X[cols]
        # random
        for seed in RANDOM_SEEDS:
            if fs_name == "all128":
                continue  # covered in part 2 (clean baseline) and by the main benchmark
            tr, te = C.stratified_split(Xf, y, seed)
            for m in models:
                _, fit_s, yp, pr = run_one(m, specs, Xf.iloc[tr], y[tr], Xf.iloc[te], y[te], seed)
                meta = {"scenario": f"random/{fs_name}", "protocol": "random", "featureset": fs_name,
                        "seed": seed, "fold": 0, "n_train": len(tr)}
                record(rows, conf_rows, meta, m, y[te], yp, pr, fit_s)
            C.log(f"random/{fs_name} seed={seed} done")
        # blocked (contiguous chunks, purged)
        for seed in BLOCKED_SEEDS:
            tr, te = C.blocked_split(groups, seed)
            for m in models:
                _, fit_s, yp, pr = run_one(m, specs, Xf.iloc[tr], y[tr], Xf.iloc[te], y[te], seed)
                meta = {"scenario": f"blocked/{fs_name}", "protocol": "blocked", "featureset": fs_name,
                        "seed": seed, "fold": 0, "n_train": len(tr)}
                record(rows, conf_rows, meta, m, y[te], yp, pr, fit_s)
            C.log(f"blocked/{fs_name} seed={seed} done")
        # leave-scenarios-out (5 folds x 3 files)
        gkf = GroupKFold(n_splits=5)
        for fold, (tr, te) in enumerate(gkf.split(Xf, y, groups)):
            for m in models:
                _, fit_s, yp, pr = run_one(m, specs, Xf.iloc[tr], y[tr], Xf.iloc[te], y[te], C.SEEDS[0])
                meta = {"scenario": f"group5/{fs_name}", "protocol": "group5", "featureset": fs_name,
                        "seed": C.SEEDS[0], "fold": fold, "n_train": len(tr),
                        "heldout_files": ",".join(str(int(g)) for g in np.unique(groups[te]))}
                record(rows, conf_rows, meta, m, y[te], yp, pr, fit_s)
            C.log(f"group5/{fs_name} fold={fold} done")
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "scenario_results.csv", index=False)  # checkpoint

    # --------------------------------------------- part 2: degradation tests
    pmu_cols = fsets["pmu116"]
    log_cols = fsets["logs12"]
    for seed in RANDOM_SEEDS:
        tr, te = C.stratified_split(X, y, seed)
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        std = X_tr[pmu_cols].std(skipna=True).to_numpy()
        for m in models:
            pipe, fit_s, yp, pr = run_one(m, specs, X_tr, y[tr], X_te, y[te], seed)
            base = {"protocol": "random", "featureset": "all128", "seed": seed, "fold": 0, "n_train": len(tr)}
            record(rows, conf_rows, dict(base, scenario="clean"), m, y[te], yp, pr, fit_s)

            for k in (1, 2, 3, 4):
                Xp = X_te.copy()
                cols = [c for c in pmu_cols if C.relay_of(c) == k]
                Xp[cols] = np.nan
                record(rows, conf_rows, dict(base, scenario=f"relay_loss_{k}"), m, y[te],
                       pipe.predict(Xp), pipe.predict_proba(Xp), 0.0)

            Xp = X_te.copy()
            Xp[log_cols] = np.nan
            record(rows, conf_rows, dict(base, scenario="logs_lost"), m, y[te],
                   pipe.predict(Xp), pipe.predict_proba(Xp), 0.0)

            rng = np.random.RandomState(seed)
            for s in NOISE_LEVELS:
                Xp = X_te.copy()
                noise = rng.standard_normal(size=(len(Xp), len(pmu_cols))) * (s * np.nan_to_num(std))
                Xp[pmu_cols] = Xp[pmu_cols].to_numpy() + noise
                record(rows, conf_rows, dict(base, scenario=f"noise_{s}"), m, y[te],
                       pipe.predict(Xp), pipe.predict_proba(Xp), 0.0)
        C.log(f"degradation seed={seed} done")

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "scenario_results.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(C.RESULTS_DIR / "scenario_confusion_long.csv", index=False)
    num = res.select_dtypes("number").drop(columns=["seed", "fold"], errors="ignore")
    g = res.groupby(["scenario", "model"], sort=False)
    summary = pd.concat([g[num.columns].mean().add_suffix("_mean"), g[num.columns].std().add_suffix("_std")], axis=1).reset_index()
    summary["n_runs"] = g.size().to_numpy()
    summary.to_csv(C.RESULTS_DIR / "scenario_summary.csv", index=False)
    C.log("Done.")


if __name__ == "__main__":
    main()
