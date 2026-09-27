"""Bridge ConfDiff's OpenFold-representation writer and loader (stdlib only).

``pretrain_repr/openfold/make_openfold_repr.py`` (ConfDiff @ 9cfae1c) writes

    {root}/{name[:2]}/{name}.node_repr.recycle{N}.npy
    {root}/{name[:2]}/{name}.edge_repr.recycle{N}.npy
    {root}/seqres_to_index.worker{k}.csv     # rows "seqres,{name[:2]}/{name}", no header

while ``src/data/full_atom/feat_loader.py::OpenFoldReprLoader`` reads

    {root}/seqres_to_index.recycle{N}.csv    # header with a "seqres" column
    {root}/{prefix}/{prefix}_recycle{N}_single_repr.npy
    {root}/{prefix}/{prefix}_recycle{N}_pair_repr.npy

This script merges the worker indices into the loader's index (prefix = chain
name) and symlinks the arrays to the paths the loader expects.

Usage: python confdiff_repr_index.py ROOT NUM_RECYCLES
"""
import csv
import glob
import os
import sys


def main(root, n):
    index_path = os.path.join(root, f"seqres_to_index.recycle{n}.csv")
    entries = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            for row in csv.DictReader(f):
                entries[row["seqres"]] = row["prefix"]
    workers = sorted(glob.glob(os.path.join(root, "seqres_to_index.worker*.csv")))
    for wpath in workers:
        with open(wpath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("seqres,"):
                    continue
                seqres, rel = line.split(",", 1)
                name = os.path.basename(rel)
                for src_kind, dst_kind in (("node_repr", "single_repr"), ("edge_repr", "pair_repr")):
                    src = os.path.join(root, f"{rel}.{src_kind}.recycle{n}.npy")
                    dst_dir = os.path.join(root, name)
                    dst = os.path.join(dst_dir, f"{name}_recycle{n}_{dst_kind}.npy")
                    if not os.path.exists(src):
                        raise FileNotFoundError(src)
                    os.makedirs(dst_dir, exist_ok=True)
                    if not os.path.lexists(dst):
                        os.symlink(os.path.relpath(src, dst_dir), dst)
                entries[seqres] = name
    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seqres", "prefix"])
        for seqres, prefix in sorted(entries.items(), key=lambda kv: kv[1]):
            w.writerow([seqres, prefix])
    for wpath in workers:
        os.remove(wpath)
    print(f"indexed {len(entries)} sequences -> {index_path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
