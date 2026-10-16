# Edge run notes — tag `edge-dell`

Run for ICSPCRE 2026 paper ID 1527 camera-ready revision. Full protocol, all
8 steps, no skips.

## 1. Device

| | |
|---|---|
| Manufacturer / Model | **Dell Inc. / OptiPlex 7050** |
| CPU | Intel(R) Core(TM) i5-7500T @ 2.70GHz, 4 cores / 4 logical processors |
| RAM | **7.9 GiB** |
| OS | Microsoft Windows 10 Pro, 10.0.19045 |
| BIOS | Dell Inc., 1.27.0 |
| Disk | INTEL SSDSC2BW240A4 (SATA SSD), 240 GB |
| Python | 3.12.10 |
| numpy / pandas / scikit-learn / xgboost / scipy / joblib | 2.5.2 / 3.0.5 / 1.9.0 / 3.4.1 / 1.18.1 / 1.5.3 |

**Matches "Dell OptiPlex 7050 / i5-7500T" — confirmed**, with one correction:
the original `results/system_specs.md` reports **78.9 GiB** RAM, which was
implausible for this model. This run's own `system_info.json`/`system_specs.md`
reports **7.9 GiB**, consistent with an OptiPlex 7050 and with the 8 GiB
memory module detected. The original figure looks like a stray-digit
reporting bug, not a different machine.

## 2. Conditions

- Power plan: switched to **High performance** before the run
  (`powercfg /setactive SCHEME_MIN`), confirmed active throughout.
- AC power: desktop, no battery — always AC.
- Background software: Chrome (~15-30 processes), Edge + WebView2, OneDrive,
  and Google Drive File Stream were running. Attempted to close them before
  the full run; Chrome/Edge/OneDrive respawned immediately on every attempt
  (access-restricted / Windows-managed helper processes — `Stop-Process`
  returned no error but the PIDs reappeared). Left running. Total system CPU
  load with them present was only ~17-25%, so unlikely to have materially
  skewed the large-model latency numbers, though some sub-millisecond jitter
  on the fastest models (LR, DT) is possible.
- Antivirus: Windows Defender real-time protection was **on** throughout
  (not disabled, per the rule to leave security software alone without the
  owner's say-so).
- Thermal/throttling: checked twice mid-run during step 02's heavy load —
  CPU clock held at its max 2712 MHz both times (no throttling), thermal
  zone temperatures ~28-30°C. No signs of thermal degradation despite ~7
  hours of continuous multi-core load.

## 3. Steps run / wall time

All 8 steps (00-07) ran to completion. Nothing skipped, nothing forced.

| Step | Wall time |
|---|---|
| 00 system info | 8s |
| 01 main benchmark | 5435s (90.6 min) |
| 02 complexity sweep | 5483s (91.4 min) |
| 03 tier study | 1017s (17.0 min) |
| 04 scenarios | 8224s (137.1 min) |
| 05 svm study | 421s (7.0 min) |
| 06 tuning | 2880s (48.0 min) |
| 07 importance | 683s (11.4 min) |
| **Total (sum of steps)** | **24151s ≈ 6h 43m** |

Wall-clock from first launch to completion was closer to ~7h 10m, the
difference being environment setup, the smoke check, and the restart at the
01→02 boundary described below.

## 4. Code changes made

**One change, `scripts/common.py`, lines added right after the existing
`warnings.filterwarnings(...)` call at the top of the file.**

Reason: scikit-learn 1.9's `RandomForestClassifier.predict()` emits a
`UserWarning` about `sklearn.utils.parallel.delayed`/`Parallel` pairing from
inside its joblib threading workers — once per tree, per `predict()` call.
During step 01's `pipeline_df_native` latency mode (`n_jobs=-1`, 1000 samples
× 5 splits, ~280 trees per Random Forest), this wrote **823 MB / 3.12 million
warning lines** for that one step. Verified in isolation that neither the
`PYTHONWARNINGS=ignore` environment variable (already set by `run_all.py`)
nor an explicit `warnings.filterwarnings("ignore", message=...)` call
suppresses it — it bypasses normal filter matching. Overriding
`warnings.warn()` itself was the only thing that worked; confirmed it does
**not** affect any measured value and leaves other warnings (e.g. the MLP
`ConvergenceWarning`, the SVM `FutureWarning` about `probability=`) visible
and unaffected.

Applied at the 01→02 step boundary with zero lost progress (run_all.py's
`.done/` markers meant 00 and 01 were skipped on restart). No other code,
protocol parameter, seed, split size, model definition, or metric was
touched anywhere in the suite.

## 5. Comparison against section 3 (original submission numbers)

Seeds 42-46, test size 0.2, `pipeline_df_native` latency mode — same
protocol, same device.

| Model | Acc % (orig → new) | Macro-F1 % (orig → new) | Latency ms, df/native (orig → new) | Size (orig → new) | Verdict |
|---|---|---|---|---|---|
| Logistic Regression | 49.5 → 49.5 | 47.7 → 47.7 | 2.27 → 1.95 (−14%) | 10.3 KB → 26.0 KB | reproduces acc/F1 exactly; **size differs (+152%)** — flagged below |
| Decision Tree | 85.4 → 85.4 | 82.8 → 82.8 | 2.18 → 1.94 (−11%) | 1.14 MB → 1.14 MB | reproduces |
| Random Forest | 91.8 → 91.8 | 91.4 → 91.4 | 48.21 → 38.24 (−21%) | 175 MB → 175 MB | reproduces; still fails both latency budgets — unchanged conclusion |
| XGBoost | 76.3 → 76.3 | 74.1 → 74.1 | 3.48 → 2.89 (−17%) | 2.11 MB → 2.11 MB | reproduces acc/F1/size; **latency drop flips the 3ms budget verdict** (was fail, now pass) — see below |
| MLP (Full) | 81.0 → 81.0 | 77.3 → 77.3 | 2.45 → 2.08 (−15%) | 862 KB → 862 KB | reproduces |
| MLP (Compact) | 74.2 → 74.2 | 59.0 → 59.0 | 2.39 → 2.03 (−15%) | 84 KB → 84 KB | reproduces |
| SVM (RBF, 2000 rows) | 43.3 → 43.3 | 37.0 → 37.0 | 2.60 → 2.28 (−12%) | 1.6 MB → ~1.5 MB | reproduces |

Accuracy and macro-F1 reproduce to within a few hundredths of a point for
every model — essentially exact, well inside the ±1 point tolerance.
Latency came in **consistently 11-21% lower** than the original submission
across every single model, which is within the ±30% "don't worry about it"
band individually, but the fact that *all seven* models moved in the same
direction by a similar amount is a systematic effect, not noise — most
likely explained by the newer library versions on this run (scikit-learn
1.9.0 vs. 1.7.2, numpy 2.5.2 vs. 2.2, xgboost 3.4.1 vs. 3.2 in the original)
rather than anything about the device itself. None of this was "fixed" —
it's reported as measured.

## 6. Three most surprising / concerning observations

1. **Random-split accuracy numbers substantially overstate true
   generalization — exactly the leakage the context file warned about.**
   Switching from the original random row-level split to the blocked
   (purged contiguous chunks) or leave-5-scenarios-out protocol collapses
   every model's accuracy by 24-32 points: Random Forest 91.8% → 67.6%
   (blocked) / 68.7% (leave-scenarios-out); Decision Tree 85.4% → 65.3%;
   XGBoost 76.3% → 61.5-62.0%. Macro-F1 drops similarly (e.g. RF 91.4% →
   59.3%/59.7%). This means the paper's headline accuracy figures, measured
   on randomly-split near-duplicate frames, do not represent how these
   models would perform on genuinely new attack scenarios — this is the
   single most paper-relevant finding from the whole run and should be
   front-and-center in the revision, not a footnote.

2. **The 12-feature logs-only subset produces worse-than-chance accuracy
   for every tree-based model, suspiciously identical across model types.**
   Decision Tree, Random Forest, and XGBoost all land at 11.3-11.5% accuracy
   on logs-only features (random-split protocol) — *worse* than the ~71%
   you'd get by always predicting the majority class, and suspiciously
   near-identical across three unrelated model families. Logistic
   Regression also collapses (10.4-10.6%), but MLP does not (71.4%,
   sane). This smells like an interaction between `class_weight="balanced"`
   and a low-information/near-constant feature subset rather than a real
   result, but it was not altered or investigated further — only run and
   reported as instructed. Worth a careful look before citing any logs-only
   number in the paper.

3. **XGBoost's measured latency flips a budget-pass/fail conclusion from
   the submitted paper**, and does so as part of a broader, systematic
   pattern. The original submission reported XGBoost at 3.48 ms, narrowly
   *failing* the 3 ms IEC 61850-5 TT6 trip-critical budget — a specific
   claim in the paper. Measured here, under the same protocol on the same
   physical device, it comes in at 2.89 ms — comfortably *passing* the same
   budget. This is not an isolated anomaly: every model's latency dropped
   11-21% in the same direction (see section 5), pointing to the newer
   library stack rather than hardware variance. Random Forest's conclusion
   is unaffected (38.24 ms still fails both budgets by a wide margin), but
   the XGBoost budget claim as currently written in the paper is now
   contradicted by this measurement and needs to be revisited.

## 7. Everything else

No step failed. No disk-space, stuck-process, or crash issues over the
~7-hour run (watched continuously). All output files listed in the context
file's section 6 are present under this folder. Nothing under `results/`
outside this `runs/edge-dell/` folder was modified, deleted, or overwritten.
Nothing was committed or pushed.
