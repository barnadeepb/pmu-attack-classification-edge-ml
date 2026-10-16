"""
Tier-architecture study (E3).

The paper proposes a small-footprint model at the substation edge plus a larger
model at a bigger, more reliable node. Two tiers is the minimal instance; this
script asks what happens with ONE, TWO and THREE tiers, and how the
edge->upstream escalation policy changes accuracy, false positives/negatives,
the share of frames that must leave the edge, and expected latency.

Tier models (all trained on the same stratified split, class-balanced):
  A  shallow Decision Tree (depth 6)           IED-class, a few KB
  B  full Decision Tree (the paper's edge model)
  C  Random Forest, 200 trees (the paper's upstream model)
  M  mid-size Random Forest (25 trees, depth 12)  intermediate tier

Configurations:
  1-tier:  A | B | M | C
  2-tier:  A->C | B->C | M->C | B->M
  3-tier:  A->B->C | A->M->C | B->M->C

Escalation policies for a non-final tier:
  conf<t     escalate frames whose top-class probability is below t
             (t in CONF_THRESHOLDS); needs a tree with soft leaves, so the
             confidence-driven tiers use min_samples_leaf>1 variants "A", "B*"
  attack     escalate every frame the tier labels Attack ("secondary
             confirmation of alerts", the use the paper describes)
  nonquiet   escalate every frame the tier labels Attack or Natural
             (only confidently quiet frames stay at the edge)

Expected end-to-end latency per frame =
    t_tier1 + p_escalate_1 * (RTT + t_tier2 + p_escalate_2 * (RTT + t_tier3))
with model-only 1-thread latencies measured on THIS machine (so the upstream
tier is pessimistic: a real control-centre node is faster) and RTT_MS_GRID as
an explicit assumed link round trip, swept rather than claimed.

Output: tier_results.csv (one row per split x config x policy x threshold),
        tier_summary.csv, tier_model_latency.csv
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import common as C

CONF_THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 0.95]
RTT_MS_GRID = [0.0, 5.0, 20.0, 50.0]
N_CENTRAL = 20 if C.SMOKE else 200

TIER_FACTORIES = {
    # name: factory(seed); "soft" variants have probability resolution for confidence gating
    "A": lambda s: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=s),
    "B": lambda s: DecisionTreeClassifier(class_weight="balanced", random_state=s),
    "B*": lambda s: DecisionTreeClassifier(min_samples_leaf=10, class_weight="balanced", random_state=s),
    "M": lambda s: RandomForestClassifier(n_estimators=10 if C.SMOKE else 25, max_depth=12,
                                          class_weight="balanced", random_state=s, n_jobs=1),
    "C": lambda s: RandomForestClassifier(n_estimators=N_CENTRAL, class_weight="balanced",
                                          random_state=s, n_jobs=1),
}

CONFIGS = [
    ["A"], ["B"], ["M"], ["C"],
    ["A", "C"], ["B", "C"], ["M", "C"], ["B", "M"],
    ["A", "B", "C"], ["A", "M", "C"], ["B", "M", "C"],
]
# tiers whose confidence we can gate on use their soft variant
SOFT = {"B": "B*"}


def escalation_mask(policy, thr, y_pred, proba):
    if policy == "attack":
        return y_pred == 0
    if policy == "nonquiet":
        return y_pred != 2
    if policy == "conf":
        return proba.max(axis=1) < thr
    raise ValueError(policy)


def run_cascade(tier_names, policy, thr, P, lat, y_true):
    """P[tier] = (y_pred, proba) on the full test set; lat[tier] = ms.
    Returns final predictions and per-tier traffic shares."""
    n = len(y_true)
    handled = np.ones(n, dtype=bool)   # frames still in play at the current tier
    final = np.zeros(n, dtype=int)
    shares = []
    for k, t in enumerate(tier_names):
        y_pred, proba = P[t]
        shares.append(handled.mean())
        if k == len(tier_names) - 1:
            final[handled] = y_pred[handled]
        else:
            esc = escalation_mask(policy, thr, y_pred, proba) & handled
            keep = handled & ~esc
            final[keep] = y_pred[keep]
            handled = esc
    return final, shares


def expected_latency(tier_names, shares, lat, rtt):
    total = lat[tier_names[0]]
    for k in range(1, len(tier_names)):
        total += shares[k] * (rtt + lat[tier_names[k]])
    return total


def main():
    C.ensure_dirs()
    X, y, groups, _ = C.load_data()
    rows = []
    lat_rows = []
    for split_i, seed in enumerate(C.SEEDS):
        C.log(f"=== split {split_i + 1}/{len(C.SEEDS)} ===")
        tr, te = C.stratified_split(X, y, seed)
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]

        fitted, P, lat, size = {}, {}, {}, {}
        for name, factory in TIER_FACTORIES.items():
            pipe = C.make_pipeline(factory(seed))
            pipe.fit(X_tr, y_tr)
            fitted[name] = pipe
            P[name] = (pipe.predict(X_te), pipe.predict_proba(X_te))
            lat_all = C.measure_latency(pipe, X_te)
            st, full = lat_all["model_only_1thread"], lat_all["pipeline_np_1thread"]
            lat[name] = full["mean"]          # preprocessing + model, 1 thread, numpy row
            size[name] = C.size_kb(pipe)
            lat_rows.append({"split": split_i, "tier_model": name, "model_only_ms": st["mean"],
                             "pipeline_np_1thread_ms": full["mean"], "p99_ms": full["p99"], "size_kb": size[name]})
            C.log(f"  tier model {name:<3} f1m={C.full_metrics(y_te, P[name][0])['f1_macro']:.3f} "
                  f"lat={full['mean']:.2f}ms size={size[name]:.0f}KB")

        for cfg in CONFIGS:
            soft_cfg = [SOFT.get(t, t) if i < len(cfg) - 1 else t for i, t in enumerate(cfg)]
            if len(cfg) == 1:
                policies = [("none", 0.0)]
            else:
                policies = [("attack", 0.0), ("nonquiet", 0.0)] + [("conf", t) for t in CONF_THRESHOLDS]
            for policy, thr in policies:
                tiers = soft_cfg if policy == "conf" else cfg
                final, shares = run_cascade(tiers, policy, thr, P, lat, y_te)
                m = C.full_metrics(y_te, final)
                row = {"split": split_i, "config": "->".join(cfg), "n_tiers": len(cfg),
                       "policy": policy, "conf_threshold": thr,
                       "share_tier1": shares[0], "share_tier2": shares[1] if len(shares) > 1 else 0.0,
                       "share_tier3": shares[2] if len(shares) > 2 else 0.0,
                       "edge_size_kb": size[tiers[0]], "central_size_kb": size[tiers[-1]],
                       "edge_latency_ms": lat[tiers[0]]}
                for rtt in RTT_MS_GRID:
                    row[f"exp_latency_ms_rtt{int(rtt)}"] = expected_latency(tiers, shares, lat, rtt)
                row.update({k: m[k] for k in (
                    "accuracy", "balanced_accuracy", "f1_macro", "mcc",
                    "recall_Attack", "recall_Natural", "recall_NoEvents",
                    "precision_Attack", "fpr_Attack", "fnr_Attack",
                    "attack_missed_as_natural", "natural_flagged_as_attack")})
                rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "tier_results.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(C.RESULTS_DIR / "tier_model_latency.csv", index=False)
    keys = ["config", "n_tiers", "policy", "conf_threshold"]
    num = res.select_dtypes("number").drop(columns=["split"] + [k for k in keys if k in res], errors="ignore")
    g = res.groupby(keys, sort=False)
    summary = pd.concat([g[num.columns].mean().add_suffix("_mean"), g[num.columns].std().add_suffix("_std")], axis=1).reset_index()
    summary.to_csv(C.RESULTS_DIR / "tier_summary.csv", index=False)
    C.log("Done.")


if __name__ == "__main__":
    main()
