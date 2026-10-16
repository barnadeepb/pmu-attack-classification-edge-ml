"""
Builds a zip with everything needed to run the benchmark on another machine
(e.g. the edge device): scripts, requirements, docs, the original dataset
archive (so no internet is needed) and the submitted-paper reference results.

    python scripts/make_edge_bundle.py [output.zip]

Excluded: .git, __pycache__, virtual envs, extracted/combined data (rebuilt on
the device by download_data.py), and previous run folders under results/runs.
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "pmu-edge-bundle.zip"

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".claude"}


def wanted(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & SKIP_DIRS:
        return False
    if rel.parts[:2] == ("results", "runs"):
        return False
    if rel.parts[:2] == ("data", "raw") and rel.name != "triple.7z":
        return False
    if rel.parts[0] == "data" and rel.name == "pmu_dataset.csv":
        return False
    return path.is_file()


def main():
    files = sorted(p for p in ROOT.rglob("*") if wanted(p))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path(ROOT.name) / p.relative_to(ROOT))
    total = sum(p.stat().st_size for p in files)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB zipped, {total / 1e6:.1f} MB raw, {len(files)} files)")
    for p in files:
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
