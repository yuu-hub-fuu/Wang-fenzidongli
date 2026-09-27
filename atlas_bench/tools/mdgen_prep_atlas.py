"""Prepare MDGen's ATLAS inference inputs ({name}_R{i}.npy atom14 arrays).

Mirrors ``scripts/prep_sims.py --atlas`` of MDGen @ 81482a4 (whose ATLAS
branch references an undefined ``args.atlas_dir``): load the fitted
production replica, drop hydrogens, superpose on its first frame and convert
to the atom14 layout. ``sim_inference.py`` only reads frame 0 of
``{name}{suffix}.npy``, so by default only the first frame is stored.

Run with the MDGen environment and repository on PYTHONPATH:
    python mdgen_prep_atlas.py --split CSV --sim_dir ATLAS --outdir DIR [--all_frames]
"""
import argparse
import os

import mdtraj
import numpy as np
import pandas as pd
from mdgen import residue_constants as rc


def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.float16)
    for i, resi in enumerate(traj.top.residues):
        for at in resi.atoms:
            if at.name not in rc.restype_name_to_atom14_names[resi.name]:
                print(resi.name, at.name, "not found")
                continue
            j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
            arr[:, i, j] = traj.xyz[:, at.index] * 10.0
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--sim_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--all_frames", action="store_true")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    names = pd.read_csv(args.split, index_col="name").index
    for name in names:
        top = f"{args.sim_dir}/{name}/{name}.pdb"
        for i in args.replicas:
            out = f"{args.outdir}/{name}_R{i}.npy"
            if os.path.exists(out):
                continue
            xtc = f"{args.sim_dir}/{name}/{name}_prod_R{i}_fit.xtc"
            traj = mdtraj.load(xtc, top=top) if args.all_frames else mdtraj.load_frame(xtc, 0, top=top)
            traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != "H"], True)
            traj.superpose(traj)
            np.save(out, traj_to_atom14(traj)[:: args.stride])
            print("saved", out)


if __name__ == "__main__":
    main()
