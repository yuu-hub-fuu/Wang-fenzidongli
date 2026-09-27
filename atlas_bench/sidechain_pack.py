"""Side-chain packing of backbone-only ensembles with FASPR.

Str2Str's README packs its backbone samples with FASPR
(https://github.com/tommyhuangthu/FASPR, ``scripts/pack.py``); we apply the
same packer frame by frame so the SASA-based Table 1 rows (Exposed residue J,
Exposed MI matrix rs) can be computed for backbone-only models. Unlike
``scripts/pack.py`` the frame order is preserved and failures are reported.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

BACKBONE = ("N", "CA", "C", "O")


def find_faspr(faspr_bin: Optional[str] = None) -> str:
    for cand in (faspr_bin, os.environ.get("FASPR_BIN"), shutil.which("FASPR")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "FASPR binary not found: build it with scripts/envs/str2str.sh (third_party/FASPR/FASPR), "
        "or set options.faspr_bin / $FASPR_BIN"
    )


def _pack_one(faspr: str, in_path: str, out_path: str) -> None:
    # FASPR reads its rotamer library (dun2010bbdep.bin) from the binary's directory
    res = subprocess.run([faspr, "-i", in_path, "-o", out_path], cwd=os.path.dirname(faspr),
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"FASPR failed on {in_path}: {res.stderr.strip()[-500:]}")


def faspr_pack_ensemble(in_pdb: str, out_pdb: str, faspr_bin: Optional[str] = None, workers: int = 8) -> int:
    """Pack every frame of a multi-model PDB; returns the number of frames written."""
    import mdtraj
    import numpy as np

    faspr = os.path.abspath(find_faspr(faspr_bin))
    traj = mdtraj.load(in_pdb)
    traj = traj.atom_slice([a.index for a in traj.top.atoms if a.name in BACKBONE])
    with tempfile.TemporaryDirectory() as tmp:
        jobs = []
        for i in range(traj.n_frames):
            src, dst = os.path.join(tmp, f"in_{i}.pdb"), os.path.join(tmp, f"out_{i}.pdb")
            traj[i].save_pdb(src)
            jobs.append((src, dst))
        with ThreadPoolExecutor(max(1, workers)) as ex:
            list(ex.map(lambda j: _pack_one(faspr, *j), jobs))
        frames = [mdtraj.load(dst) for _, dst in jobs]
    top = frames[0].top
    keys0 = [(a.residue.index, a.name) for a in top.atoms]
    xyz = []
    for f in frames:
        keys = [(a.residue.index, a.name) for a in f.top.atoms]
        if keys != keys0:
            lookup = {k: i for i, k in enumerate(keys)}
            f = f.atom_slice([lookup[k] for k in keys0])
        xyz.append(f.xyz)
    packed = mdtraj.Trajectory(np.concatenate(xyz), top)
    os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
    packed.save_pdb(out_pdb)
    return packed.n_frames
