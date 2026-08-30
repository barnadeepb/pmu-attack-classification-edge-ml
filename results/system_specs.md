# Benchmark environment

Hardware and software environment used to produce the results in this directory
(`model_comparison.csv`, `per_split_results.csv`, latency/size figures, plots).

## Hardware

- CPU: Intel(R) Core(TM) i5-7500T @ 2.70GHz
- Cores / threads: 4 physical / 4 logical (no hyperthreading)
- RAM: 78.9 GiB (84,678,287,360 bytes)
- OS: Microsoft Windows 10 Pro, 64-bit, version 10.0.19045

## Software

- Python 3.12.10
- pandas 3.0.5
- numpy 2.5.2
- scikit-learn 1.9.0
- xgboost 3.4.1
- py7zr 1.1.3
- joblib 1.5.3
- matplotlib 3.11.1
- requests 2.34.2

Full dependency set pinned in `requirements.txt` (minimum versions) and resolvable
exactly via `pip freeze` against the `.venv` used for this run.

## Notes

Latency numbers in `model_comparison.csv` are single-sample (one PMU frame at a
time) CPU inference times measured on this machine, not a GPU or an actual
substation IED. They are a benchmarking-hardware proxy, not a claim about
performance on any specific edge device.
