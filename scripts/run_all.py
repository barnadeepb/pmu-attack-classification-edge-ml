"""
Runs the whole experiment suite in order, resumable.

    python scripts/run_all.py --tag edge-optiplex7050
    python scripts/run_all.py --tag edge-optiplex7050 --only 01 04
    python scripts/run_all.py --tag edge-optiplex7050 --skip 05 06
    python scripts/run_all.py --tag edge-optiplex7050 --force 04      # redo one step
    python scripts/run_all.py --smoke                                 # 2-minute code check

A step that finished earlier leaves results/runs/<tag>/.done/<step> behind and
is skipped on re-run, so an interrupted run can simply be started again.
Each step's console output is saved to results/runs/<tag>/logs/<step>.log.

Steps
  00 system info            ~1 min     hardware/software inventory
  01 main benchmark         ~1-2 h     8 models x 5 splits + SVM, full metrics + 4-mode latency
  02 complexity sweep       ~1-2 h     RF/XGB/DT size sweeps
  03 tier study             ~30-60 min 1-, 2-, 3-tier cascades
  04 scenarios              ~2-4 h     blocked / leave-scenario-out / feature sets / relay loss / noise
  05 svm study              ~1-3 h     like-for-like, learning curve, Nystroem
  06 tuning                 ~1-3 h     randomized search + refit on 5 splits
  07 importance             ~30-60 min permutation importance on every split
(durations are rough guesses for a 4-core i5-7500T; the log shows real progress)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

STEPS = {
    "00": ("00_system_info.py", []),
    "01": ("01_main_benchmark.py", []),
    "02": ("02_complexity_sweep.py", []),
    "03": ("03_tier_study.py", []),
    "04": ("04_scenarios.py", []),
    "05": ("05_svm_study.py", []),
    "06": ("06_tuning.py", []),
    "07": ("07_importance.py", []),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="run label, e.g. edge-optiplex7050")
    ap.add_argument("--smoke", action="store_true", help="tiny data/models, just checks the code path")
    ap.add_argument("--only", nargs="*", default=None, help="step numbers to run, e.g. 01 04")
    ap.add_argument("--skip", nargs="*", default=[], help="step numbers to skip")
    ap.add_argument("--force", nargs="*", default=[], help="step numbers to re-run even if finished")
    ap.add_argument("--quick-scenarios", action="store_true", help="step 04 without the two MLPs")
    ap.add_argument("--svm-max-n", type=int, default=None, help="step 05: largest learning-curve size")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.tag:
        env["PMU_TAG"] = args.tag
    env.setdefault("PMU_TAG", "local")
    if args.smoke:
        env["PMU_SMOKE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"

    tag = env["PMU_TAG"] + ("-smoke" if args.smoke else "")
    out_dir = ROOT / "results" / "runs" / tag
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / ".done").mkdir(exist_ok=True)

    timings = {}
    for key, (script, extra) in STEPS.items():
        if args.only is not None and key not in args.only:
            continue
        if key in args.skip:
            continue
        marker = out_dir / ".done" / key
        if marker.exists() and key not in args.force:
            print(f"[{key}] already finished ({marker.read_text().strip()}), skipping. Use --force {key} to redo.")
            continue

        cmd = [sys.executable, str(HERE / script)] + list(extra)
        if key == "04" and args.quick_scenarios:
            cmd.append("--quick")
        if key == "05" and args.svm_max_n:
            cmd += ["--max-n", str(args.svm_max_n)]
        print(f"\n===== step {key}: {' '.join(cmd[1:])} =====", flush=True)
        t0 = time.time()
        with open(out_dir / "logs" / f"{key}.log", "w", encoding="utf8") as log:
            proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        elapsed = time.time() - t0
        timings[key] = round(elapsed)
        if proc.returncode != 0:
            print(f"\n[{key}] FAILED with exit code {proc.returncode} after {elapsed:.0f}s. See {out_dir / 'logs' / (key + '.log')}")
            print("Fix the problem and re-run the same command; finished steps are skipped.")
            sys.exit(proc.returncode)
        marker.write_text(f"finished in {elapsed:.0f}s at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[{key}] finished in {elapsed / 60:.1f} min", flush=True)

    prev = {}
    tfile = out_dir / "timings_s.json"
    if tfile.exists():
        prev = json.loads(tfile.read_text())
    prev.update(timings)
    tfile.write_text(json.dumps(prev, indent=2))
    print(f"\nAll requested steps finished. Results in {out_dir}")


if __name__ == "__main__":
    main()
