"""Check that atlas_bench reproduces AlphaFlow's evaluation numerically.

Runs the original ``scripts/analyze_ensembles.py`` of AlphaFlow and
``atlas_bench.metrics.analyze`` on the same ensembles and compares every
per-target quantity, then both benchmark summaries.

    python scripts/verify_against_alphaflow.py --alphaflow_repo third_party/alphaflow \
        --atlas_dir data/atlas --pdbdir runs/esmflow_md_full/ensembles --pdb_id 6o2v_A 7ead_A

(the original script needs matplotlib, sklearn and mdtraj in the current env)
"""
import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas_bench.metrics import analyze, report  # noqa: E402


def max_abs_diff(a, b):
    if isinstance(a, dict):
        return max(max_abs_diff(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    d = np.abs(a.astype(complex) - b.astype(complex))
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphaflow_repo", required=True)
    p.add_argument("--atlas_dir", required=True)
    p.add_argument("--pdbdir", required=True)
    p.add_argument("--pdb_id", nargs="+", required=True)
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dirs = {}
        for tag in ("orig", "ours"):
            dirs[tag] = os.path.join(tmp, tag)
            os.makedirs(dirs[tag])
            for name in args.pdb_id:
                shutil.copy(os.path.join(args.pdbdir, f"{name}.pdb"), dirs[tag])
        subprocess.run(
            [sys.executable, os.path.join(args.alphaflow_repo, "scripts", "analyze_ensembles.py"),
             "--atlas_dir", args.atlas_dir, "--pdbdir", dirs["orig"], "--pdb_id", *args.pdb_id],
            check=True,
        )
        analyze.analyze_directory(args.atlas_dir, dirs["ours"], args.pdb_id)
        with open(os.path.join(dirs["orig"], "out.pkl"), "rb") as f:
            orig = pickle.load(f)
        with open(os.path.join(dirs["ours"], "out.pkl"), "rb") as f:
            ours = pickle.load(f)

    worst = 0.0
    for name in args.pdb_id:
        for key, val in orig[name].items():
            d = max_abs_diff(val, ours[name][key])
            worst = max(worst, d)
            flag = "" if d <= args.tol else "   <-- MISMATCH"
            print(f"{name:10s} {key:30s} max|diff| = {d:.3g}{flag}")
    if len(args.pdb_id) > 1:
        s_orig, s_ours = report.summarize(orig), report.summarize(ours)
        print("\nsummary (original vs atlas_bench):")
        for k, v in s_orig.items():
            print(f"  {k:28s} {v:10.4f} {s_ours[k]:10.4f}")
    print(f"\nworst per-target difference: {worst:.3g} (tolerance {args.tol})")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
