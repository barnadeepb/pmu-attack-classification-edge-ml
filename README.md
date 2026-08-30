# PMU Attack Classification for Edge ML

Benchmarking machine learning classifiers for real-time cyber-attack and
fault discrimination in power system substations, with a specific focus on
whether each model is actually practical to run on substation edge hardware
rather than a centralized control-center server.

This is a companion study to
[grids-stability-study-for-edge-ml](https://github.com/barnadeepb/grids-stability-study-for-edge-ml):
that project asked which model best predicts grid stability under an edge
latency budget; this one asks the same deployability question for a
different, security-critical task -- telling a real power system fault
apart from a cyber-attack made to look like one.

## Motivation

Wide-area monitoring and protection systems rely on PMUs (phasor
measurement units) streaming voltage/current phase angle, magnitude, and
frequency data at up to 60 frames/sec. Protective relays act on that same
stream. If an attacker can inject a false trip command or spoof the
telemetry so it resembles a natural fault, the protection logic has no way
to tell the difference -- this is not hypothetical, it's the attack pattern
used against the Ukrainian grid in 2015/2016.

Today, that kind of discrimination almost always happens after the fact, at
a centralized SOC, minutes after the event. Modern substation edge
platforms and IEDs (the P360-class devices increasingly deployed in the
field, for instance) carry real general-purpose compute headroom now --
which raises the question this project tries to answer with data instead of
assumption: which ML models are both accurate *and* small/fast enough to
run directly at the substation, flagging a suspicious event before it
propagates, without needing a round trip to a control center?

## Dataset

[Power System Attack Datasets](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets),
Mississippi State University / Oak Ridge National Laboratory (Pokhrel,
Adhikari, Morris et al.) -- the standard public benchmark for ML-based
power system cyberattack detection research. Real synchrophasor logs from
a physical power system testbed with staged faults, switching events, and
attacks, not a synthetic simulation.

- 15 source files, 78,377 rows combined
- 128 features: PMU voltage/current phase angle & magnitude, frequency,
  and impedance from 4 relays, plus Snort/relay/control-panel log flags
- 3-class label (`marker`): `Attack`, `Natural`, `NoEvents`
  (roughly 74% / 21% / 5% of the data -- real class imbalance, not
  rebalanced in the source data)

Note: 4 impedance columns (`R*-PA:Z`) contain `inf` values where current is
near zero (division blows up). These are treated as missing and
median-imputed within each training fold -- see `scripts/run_experiment.py`.

## What this measures

For each model: accuracy, macro/weighted F1, precision, recall, a
confusion matrix, model size on disk, and single-sample inference latency
(one PMU frame at a time, matching how an edge device actually consumes the
stream -- not a batch).

Two latency budgets, both from real grid standards, not picked arbitrarily:

| Budget | Threshold | Source | Meaning |
|---|---|---|---|
| Streaming | 16.7 ms | IEEE C37.118.2, 60 fps reporting rate | Model must finish classifying one frame before the next arrives, or it falls behind the live stream |
| Trip-critical | 3 ms | IEC 61850-5, TT6 message class | Required if the model's output is allowed to gate a protection trip decision, not just raise an alert |

A model that misses the streaming budget can still be useful for delayed
alerting/forensics; a model that misses the trip-critical budget just can't
sit inline with protection logic.

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Running it

```bash
python scripts/download_data.py
python scripts/run_experiment.py
```

`download_data.py` fetches the dataset (~19.7 MB archive) from the UAH
host, extracts it, and combines the 15 source files into
`data/pmu_dataset.csv`. If the download fails (it's an old university
server, it does go down occasionally), the script tells you the direct URL
-- grab it manually, drop the `.7z` into `data/raw/`, and re-run; it picks
up from there.

`run_experiment.py` runs the full benchmark: 6 models x 5 stratified
train/test splits (seeds 42-46), aggregates results, and writes everything
to `results/`. On a modest quad-core machine with 8GB RAM, expect roughly
20-45 minutes total -- the two MLP variants are the slowest to train, tree
ensembles and logistic regression are fast.

### Optional: TabPFN extreme case

The companion grid-stability study included TabPFN (a tabular foundation
model) as a documented extreme case, and this project can do the same. It's
off by default because it needs an internet connection and the dataset
(78k rows) is far outside what TabPFN is meant for, so it gets aggressively
subsampled (2,000 rows/split, 2 splits only) just to get a data point on
the chart.

```bash
pip install tabpfn
set TABPFN_ENABLE=1          # Windows (cmd)
$env:TABPFN_ENABLE=1         # Windows (PowerShell)
export TABPFN_ENABLE=1       # macOS/Linux

python scripts/run_experiment.py
```

## Output

Everything lands in `results/`:

- `model_comparison.csv` -- one row per model: mean/std accuracy, macro-F1,
  precision, recall, latency, model size, and whether it clears each
  latency budget
- `per_split_results.csv` -- the same metrics per individual split, before
  aggregation
- `confusion_matrices.csv` -- summed confusion matrix per model across all
  splits, long format (`model, true_label, predicted_label, count`)
- `feature_importance.csv` -- top 20 PMU features by permutation importance
  (Random Forest and XGBoost), i.e. which measurements actually carry the
  attack-vs-fault signal
- `run_config.json` -- exact seeds, budgets, and dataset shape used for the
  run, for reproducibility
- `plots/model_comparison_bars.png` -- macro-F1 per model with error bars
- `plots/accuracy_vs_latency.png` -- the key figure: F1 vs. latency
  (log scale) with both budget lines marked, i.e. the deployability
  trade-off in one chart

## Project layout

```
pmu-attack-classification-edge-ml/
  scripts/
    download_data.py     fetches and combines the dataset
    run_experiment.py    trains, evaluates, benchmarks, plots
  data/                   generated at runtime, not tracked in git
  results/                generated at runtime (csv + plots)
  requirements.txt
  LICENSE
```

## License

MIT, see `LICENSE`.
