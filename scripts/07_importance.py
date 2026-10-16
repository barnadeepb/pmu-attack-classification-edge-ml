"""
Feature importance (E7): which measurements carry the signal, and is it stable?

* Permutation importance (macro-F1 drop) for Decision Tree, Random Forest and
  XGBoost on EVERY split (not just split 1), for all 128 features
* Grouped permutation importance: all columns of a physical channel family
  (voltage magnitude, current angle, impedance, frequency, log flags, ...)
  are shuffled together, which is the physically meaningful quantity
* Model-native importances (Gini / gain) for comparison
* Rank stability across splits (Spearman correlation)

Outputs: importance_permutation_per_split.csv (feature x model x split),
         importance_permutation_mean.csv, importance_grouped.csv,
         importance_native.csv, importance_rank_stability.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

import common as C

N_REPEATS = 2 if C.SMOKE else 3
EVAL_ROWS = 1500 if C.SMOKE else 6000
MODELS = ["Decision Tree", "Random Forest", "XGBoost"]


def permute_importance(pipe, X_te, y_te, cols_groups, rng):
    """cols_groups: {label: [column names]} -> {label: mean macro-F1 drop}."""
    base = f1_score(y_te, pipe.predict(X_te), average="macro")
    out = {}
    for label, cols in cols_groups.items():
        drops = []
        for _ in range(N_REPEATS):
            Xp = X_te.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]   # shuffle the group jointly (rows stay aligned within group)
            drops.append(base - f1_score(y_te, pipe.predict(Xp), average="macro"))
        out[label] = float(np.mean(drops))
    return out, base


def main():
    C.ensure_dirs()
    X, y, groups, _ = C.load_data()
    specs = C.build_model_specs()
    feat_groups = {c: [c] for c in X.columns}
    fam = {}
    for c in X.columns:
        fam.setdefault(C.feature_group(c), []).append(c)

    perm_rows, group_rows, native_rows = [], [], []
    for split_i, seed in enumerate(C.SEEDS):
        C.log(f"=== split {split_i + 1}/{len(C.SEEDS)} ===")
        tr, te = C.stratified_split(X, y, seed)
        rng_eval = np.random.RandomState(seed)
        ev = rng_eval.choice(len(te), size=min(EVAL_ROWS, len(te)), replace=False)
        X_te, y_te = X.iloc[te].iloc[ev], y[te][ev]
        for name in MODELS:
            pipe, _ = C.fit_pipeline(name, specs[name], X.iloc[tr], y[tr], seed)
            rng = np.random.RandomState(seed)
            imp, base = permute_importance(pipe, X_te, y_te, feat_groups, rng)
            for f, v in imp.items():
                perm_rows.append({"model": name, "split": split_i, "feature": f, "group": C.feature_group(f),
                                  "relay": C.relay_of(f), "importance": v, "baseline_f1_macro": base})
            gimp, _ = permute_importance(pipe, X_te, y_te, fam, rng)
            for g, v in gimp.items():
                group_rows.append({"model": name, "split": split_i, "group": g, "n_features": len(fam[g]), "importance": v})
            est = C.final_estimator(pipe)
            if hasattr(est, "feature_importances_"):
                for f, v in zip(X.columns, est.feature_importances_):
                    native_rows.append({"model": name, "split": split_i, "feature": f, "native_importance": float(v)})
            C.log(f"  {name}: done (baseline macro-F1 on eval rows = {base:.3f})")

    perm = pd.DataFrame(perm_rows)
    perm.to_csv(C.RESULTS_DIR / "importance_permutation_per_split.csv", index=False)
    mean = (perm.groupby(["model", "feature", "group", "relay"], dropna=False)["importance"]
            .agg(["mean", "std"]).reset_index().rename(columns={"mean": "importance_mean", "std": "importance_std"}))
    mean = mean.sort_values(["model", "importance_mean"], ascending=[True, False])
    mean.to_csv(C.RESULTS_DIR / "importance_permutation_mean.csv", index=False)
    pd.DataFrame(group_rows).to_csv(C.RESULTS_DIR / "importance_grouped.csv", index=False)
    pd.DataFrame(native_rows).to_csv(C.RESULTS_DIR / "importance_native.csv", index=False)

    stab = []
    for name in MODELS:
        piv = perm[perm.model == name].pivot(index="feature", columns="split", values="importance")
        cols = list(piv.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                rho = spearmanr(piv[cols[i]], piv[cols[j]]).correlation
                stab.append({"model": name, "split_a": cols[i], "split_b": cols[j], "spearman": rho})
    pd.DataFrame(stab).to_csv(C.RESULTS_DIR / "importance_rank_stability.csv", index=False)
    C.log("Done.")


if __name__ == "__main__":
    main()
