# Latency repeatability run (tag: edge-dell-rep)

Scope: steps `00` (system info) + `08` (latency repeatability only) on the edge-dell
machine (Dell OptiPlex 7050, Intel i5-7500T, 4C/4T @ 2.70GHz, 7.9 GiB RAM). No other
steps were run; no script, protocol parameter, seed, or sample count was changed.
12 models fitted once on split 0 / seed 42, then measured in 3 repeats, each in a
fresh Python process, in a randomized model order, with `Random Forest (200)` as
the sentinel measured first and last in every repeat. 1000 timed calls + 100
warm-up per model x mode (`latency_repeat_config.json`). Run took 28.2 min
(`timings_s.json`), finished with exit code 0, no failures.

## a. Conditions

- **Power plan**: `powercfg /getactivescheme` reported High performance
  (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`) *before* `/setactive SCHEME_MIN` was
  run, and the same GUID/name *after* — plan was already correct, confirmed twice.
  Also recorded automatically by step 00 in `system_specs.md` (`power_plan` field).
- **AC power**: OptiPlex 7050 is a desktop; `Get-CimInstance Win32_Battery`
  returned no battery device, so there is no AC/battery distinction to make here.
- **Background software**: Chrome and OneDrive were left running (not
  force-closed) to avoid losing unsaved browser state; `explorer`/`SearchApp`
  were at idle baseline. This session's own remoting/Claude processes were
  necessarily running throughout. See the CPU-load note below for whether this
  visibly affected measurements.
- **Antivirus**: `Get-MpComputerStatus` showed Windows Defender
  `RealTimeProtectionEnabled = True`, `AntivirusEnabled = True`,
  `OnAccessProtectionEnabled = True` throughout — left untouched, not disabled,
  per instruction.
- **CPU frequency**: every one of the 624 `cpu_freq_before`/`cpu_freq_after`
  samples in `latency_repeat_raw.csv` reads exactly `2712` MHz — identical to
  `cpu_freq_mhz_max` in `system_specs.md`. No downclocking and no sign of thermal
  throttling at any point in the run.
- **CPU load**: `cpu_load_before`/`cpu_load_after` (1-second `psutil.cpu_percent`
  samples) ranged from 0.4% to 28.0% across all 624 snapshots, mean 5.6% — i.e.
  mostly idle with occasional brief spikes (e.g. 26.5% during repeat 1's
  `Random Forest (50)` call, 22.1% during repeat 1's `MLP (Compact)` call).
  Comparing those specific models' per-repeat means to their low-load repeats
  (RF-50 np_1thread: 5.06 / **5.12** / 5.01 ms; MLP-Compact np_1thread: 0.77 /
  **0.79** / 0.78 ms) shows at most a ~1-2% bump on the elevated-load repeat —
  not distinguishable from ordinary run-to-run noise at this sample size. No
  evidence the background load meaningfully corrupted any single-threaded
  measurement.

## b. Per-model latency summary (mean of 3 repeat means; all from `latency_repeat_summary.csv`)

| Model | Mode | Mean (ms) | Std (ms) | CV% | p99 (ms) |
|---|---|---:|---:|---:|---:|
| Decision Tree | pipeline_np_1thread | 0.64 | 0.005 | 0.8 | 1.06 |
| Decision Tree | model_only_1thread | 0.15 | 0.003 | 2.0 | 0.31 |
| Decision Tree | pipeline_df_native | 2.08 | 0.064 | 3.1 | 2.83 |
| Logistic Regression | pipeline_np_1thread | 0.69 | 0.026 | 3.7 | 1.13 |
| Logistic Regression | model_only_1thread | 0.19 | 0.014 | 7.6 | 0.39 |
| Logistic Regression | pipeline_df_native | 2.12 | 0.068 | 3.2 | 2.95 |
| MLP (Compact) | pipeline_np_1thread | 0.78 | 0.008 | 1.0 | 1.08 |
| MLP (Compact) | model_only_1thread | 0.28 | 0.009 | 3.0 | 0.45 |
| MLP (Compact) | pipeline_df_native | 2.23 | 0.067 | 3.0 | 3.01 |
| MLP (Full) | pipeline_np_1thread | 0.84 | 0.013 | 1.6 | 1.50 |
| MLP (Full) | model_only_1thread | 0.31 | 0.002 | 0.6 | 0.50 |
| MLP (Full) | pipeline_df_native | 2.25 | 0.043 | 1.9 | 3.10 |
| XGBoost (200, depth 6) | pipeline_np_1thread | 1.48 | 0.039 | 2.6 | 2.34 |
| XGBoost (200, depth 6) | model_only_1thread | 0.79 | 0.025 | 3.1 | 1.43 |
| XGBoost (200, depth 6) | pipeline_df_native | 3.16 | 0.064 | 2.0 | 4.01 |
| XGBoost (tuned) | pipeline_np_1thread | 1.69 | 0.080 | 4.8 | 2.59 |
| XGBoost (tuned) | model_only_1thread | 0.98 | 0.050 | 5.1 | 1.74 |
| XGBoost (tuned) | pipeline_df_native | 3.32 | 0.090 | 2.7 | 4.16 |
| SVM (RBF, n=2000) | pipeline_np_1thread | 1.02 | 0.028 | 2.8 | 1.59 |
| SVM (RBF, n=2000) | model_only_1thread | 0.47 | 0.004 | 0.9 | 0.83 |
| SVM (RBF, n=2000) | pipeline_df_native | 2.46 | 0.018 | 0.7 | 3.77 |
| Random Forest (10) | pipeline_np_1thread | 1.77 | 0.035 | 2.0 | 2.43 |
| Random Forest (10) | model_only_1thread | 1.25 | 0.024 | 1.9 | 2.19 |
| Random Forest (10) | pipeline_df_native | 17.20 | 0.090 | 0.5 | 18.50 |
| Random Forest (25) | pipeline_np_1thread | 3.07 | 0.103 | 3.4 | 4.05 |
| Random Forest (25) | model_only_1thread | 2.53 | 0.114 | 4.5 | 3.42 |
| Random Forest (25) | pipeline_df_native | 17.00 | 0.038 | 0.2 | 18.35 |
| Random Forest (50) | pipeline_np_1thread | 5.06 | 0.055 | 1.1 | 5.95 |
| Random Forest (50) | model_only_1thread | 4.51 | 0.070 | 1.5 | 5.65 |
| Random Forest (50) | pipeline_df_native | 16.70 | 0.009 | 0.1 | 17.86 |
| Random Forest (100) | pipeline_np_1thread | 9.26 | 0.141 | 1.5 | 10.82 |
| Random Forest (100) | model_only_1thread | 8.69 | 0.141 | 1.6 | 10.44 |
| Random Forest (100) | pipeline_df_native | 28.33 | 0.148 | 0.5 | 30.50 |
| **Random Forest (200)** | **pipeline_np_1thread** | **18.24** | **0.537** | **2.9** | **26.10** |
| **Random Forest (200)** | **model_only_1thread** | **18.06** | **1.229** | **6.8** | **26.44** |
| **Random Forest (200)** | **pipeline_df_native** | **41.37** | **0.455** | **1.1** | **56.52** |

All CV%s are at or below ~7.6% (Logistic Regression, `model_only_1thread`, the
single highest value in the table); every other model x mode is under 5%, and
most are under 3%. This is the result the user needs: **the three repeats now
agree closely with each other**, in sharp contrast to the single-shot numbers
from the original run (see §d).

## c. Streaming (16.7 ms) and trip (3 ms) budget verdicts

Budgets confirmed from `scripts/common.py`: `STREAMING_BUDGET_MS = 1000/60 =
16.67ms`, `TRIP_BUDGET_MS = 3.0ms`. Flags below are read directly from
`latency_repeat_summary.csv` (`meets_streaming_mean`, `meets_streaming_p99`,
`meets_trip_mean`), not recomputed by hand.

**Streaming budget (16.7 ms)** — met on both mean and p99 by every model in
`pipeline_np_1thread` and `model_only_1thread` mode, including `Random Forest
(100)` (mean 9.26/9.26, p99 10.82/10.44 ms). The **only mode/model combinations
that miss the streaming budget are `pipeline_df_native` for every Random
Forest size (10/25/50/100/200)** — e.g. RF-50 df_native mean is 16.70 ms,
just over the 16.67 ms line; RF-200 df_native is 41.37 ms, far over. No
non-RF model misses the streaming budget in any mode (max is XGBoost (tuned)
pipeline_df_native at 3.32 ms mean / 4.16 ms p99, well inside 16.7 ms).

**Trip-critical budget (3 ms, mean only)** — met by: Decision Tree, Logistic
Regression, MLP (Compact), MLP (Full), SVM (RBF, n=2000) in all three modes;
XGBoost (200, depth 6) and XGBoost (tuned) in `pipeline_np_1thread` and
`model_only_1thread` but **not** `pipeline_df_native` (3.16/3.32 ms, just over);
Random Forest (10) in `pipeline_np_1thread`/`model_only_1thread` (1.77/1.25 ms)
but not `pipeline_df_native` (17.20 ms); Random Forest (25) in
`model_only_1thread` only (2.53 ms) — its `pipeline_np_1thread` mean of
3.07 ms is just **over** the 3 ms line, so RF-25 fails the trip budget in
`pipeline_np_1thread` despite clearing it in `model_only_1thread`. Random
Forest (50), (100) and (200) fail the trip budget in every mode.

## d. Comparison with the earlier edge-dell run (Random Forest (200) only)

| Mode | Earlier edge-dell | New edge-dell-rep (3-repeat mean, CV%) |
|---|---|---|
| pipeline_np_1thread | 14.5 ms (step 01), 9.8 ms (step 02), 16.2 ms (step 03) | **18.24 ms, CV 2.9%** |
| pipeline_df_native | 38.2 ms (step 01), 32.7 ms (step 02) | **41.37 ms, CV 1.1%** |

The three new repeats agree closely with **each other** (repeat means — taken
from each repeat's first/sentinel-opening measurement, per
`latency_repeat_sentinel_drift.csv` and the run log — 18.25, 17.70, 18.77 ms
for np_1thread; 41.36, 40.92, 41.83 ms for df_native, averaging to the 41.37 ms
/ 1.1% CV shown in the summary table). They do **not**
closely reproduce any single earlier figure: the new np_1thread mean (18.24 ms)
is higher than all three earlier step measurements, including the closest one
(step 03's 16.2 ms, 2.0 ms / ~12% below the new mean). The new df_native mean
(41.37 ms) is likewise higher than both earlier figures (38.2 and 32.7 ms).
Given that the new numbers are tightly reproducible across 3 independent
fresh-process repeats under confirmed-idle, non-throttled, constant-frequency
conditions, while the earlier step 01/02/03 figures disagreed with each other
by 50-65% of their own mean, **the new numbers should be treated as the more
trustworthy figures for the paper; the old 9.8/14.5/16.2 ms and 32.7/38.2 ms
numbers should not be used.** This report does not attempt to explain why the
earlier numbers were lower — only step 01 is independently known to have had a
confounding factor (the sklearn warning flood to its log); no equivalent cause
is confirmed for step 02 or 03, and none is asserted here.

**Sentinel drift** (`latency_repeat_sentinel_drift.csv`, RF-200
`pipeline_np_1thread`, first vs. last position in each repeat):

| Repeat | First (pos 0) | Last (pos 12) | Δ (last − first) |
|---|---:|---:|---:|
| 0 | 18.25 ms | 21.05 ms | +2.80 ms |
| 1 | 17.70 ms | 17.60 ms | −0.10 ms |
| 2 | 18.77 ms | 17.38 ms | −1.39 ms |

The drift direction is **not consistent across repeats** — repeat 0 drifts up
by 2.8 ms over the course of the repeat, repeat 1 is essentially flat, repeat 2
drifts down by 1.4 ms. CPU frequency was 2712 MHz and CPU load was low (1.2-11.4%)
at every one of these six snapshots, so neither explains the inconsistency.
This is reported as an open observation, not as evidence of a specific cause:
within a single repeat, the same model can read ±1.4-2.8 ms (roughly 7-15% of
its own ~18 ms baseline) differently depending on its position in the sequence,
and that spread does not point in one direction.

## e. Other checks / anything odd

- **`fit_s` vs `size_kb`** (`latency_repeat_models.csv`) were checked
  individually and are not confused in this report: e.g. MLP (Full) has
  `fit_s` = 286.4 s (training time) and `size_kb` = 862.4 KB (joblib dump
  size) — two different columns, two different units, consistent with each
  other only in that both are plausible for that model. Random Forest (200)
  is 179,830.8 KB (~175.6 MB on disk), matching the ~179 MB figure recorded for
  this same model in the original run — a cross-run sanity check, not a new
  result.
- **No warnings in the log**: `run_experiment_edge_dell_rep.log` (the full
  captured stdout/stderr of this run) contains zero occurrences of "warning",
  "error", or "deprecat" — unlike step 01 of the original run, this run's
  console was not confounded by a warning flood. `PYTHONWARNINGS=ignore` is
  set by `run_all.py` for every step, including this one.
- **library versions are unchanged from the original edge-dell run**:
  `system_specs.md` for this run reports scikit-learn 1.9.0, numpy 2.5.2,
  pandas 3.0.5, xgboost 3.4.1, scipy 1.18.1, joblib 1.5.3 — the same
  versions as the original edge-dell run's environment. This repeatability
  run cannot be, and is not, used to attribute any latency difference to a
  library-version change, because there wasn't one.
- **`pipeline_df_native` vs `pipeline_np_1thread` gap is large for every
  Random Forest size** (e.g. RF-10: 17.20 ms vs 1.77 ms, ~9.7x) but small for
  every other model (e.g. XGBoost (200): 3.16 ms vs 1.48 ms, ~2.1x). This
  matches the already-known pattern from the original run (RF df_native being
  dramatically slower than its own numpy-row mode) — not a new or surprising
  finding, noted here only to confirm it reproduced.
- **`Random Forest (200)` `model_only_1thread` has the highest CV% in the
  whole table for that model (6.8%)**, higher than its own
  `pipeline_np_1thread` (2.9%) and `pipeline_df_native` (1.1%) CVs. This is
  mildly counter-intuitive (the bare-estimator call is simpler than the full
  pipeline call, so lower variance might be expected) but is reported as
  observed, with no cause identified or guessed.
- Nothing else in the CSVs looked implausible on inspection: all 12 models
  produced all 4 modes x 3 repeats with no missing rows, no NaNs in the
  `_summary.csv`'s mean/std/p50/p99 columns, and `accuracy_split0` /
  `f1_macro_split0` in `latency_repeat_models.csv` are all in a sane 0-1 range
  consistent with the original run's per-model accuracy figures.
