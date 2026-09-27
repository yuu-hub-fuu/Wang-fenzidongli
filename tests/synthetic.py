"""Synthetic ATLAS-like targets and mock baseline outputs for tests."""
from __future__ import annotations

import os

import mdtraj
import numpy as np

SIDECHAINS = {
    "GLY": [],
    "ALA": [("CB", "C")],
    "SER": [("CB", "C"), ("OG", "O")],
    "VAL": [("CB", "C"), ("CG1", "C"), ("CG2", "C")],
    "LEU": [("CB", "C"), ("CG", "C"), ("CD1", "C"), ("CD2", "C")],
    "LYS": [("CB", "C"), ("CG", "C"), ("CD", "C"), ("CE", "C"), ("NZ", "N")],
    "PHE": [("CB", "C"), ("CG", "C"), ("CD1", "C"), ("CD2", "C"), ("CE1", "C"), ("CE2", "C"), ("CZ", "C")],
    "GLU": [("CB", "C"), ("CG", "C"), ("CD", "C"), ("OE1", "O"), ("OE2", "O")],
}
ONE = {"G": "GLY", "A": "ALA", "S": "SER", "V": "VAL", "L": "LEU", "K": "LYS", "F": "PHE", "E": "GLU"}
DEFAULT_SEQ = "MKLVAEFLKSGLEAVKLFSEAGKLVLE".replace("M", "A")


def build_protein(seq: str = DEFAULT_SEQ, backbone_only: bool = False, first_resseq: int = 1, chain_id: str = "A",
                  with_hydrogens: bool = False, with_cb: bool = False):
    """Ideal-helix protein (1 frame). Coordinates in nm."""
    top = mdtraj.Topology()
    chain = top.add_chain(chain_id)
    xyz = []
    el = mdtraj.element
    for i, aa in enumerate(seq):
        resname = ONE[aa]
        res = top.add_residue(resname, chain, resSeq=first_resseq + i)
        th = np.deg2rad(100.0 * i)
        z = 1.5 * i

        def put(name, element, r, dth, dz):
            top.add_atom(name, getattr(el, element), res)
            xyz.append([r * np.cos(th + np.deg2rad(dth)), r * np.sin(th + np.deg2rad(dth)), z + dz])

        put("N", "nitrogen", 1.55, -28, -0.8)
        if with_hydrogens:
            put("H", "hydrogen", 1.2, -40, -1.2)
        put("CA", "carbon", 2.3, 0, 0.0)
        put("C", "carbon", 1.6, 28, 0.8)
        put("O", "oxygen", 1.9, 45, 2.0)
        if backbone_only and with_cb and SIDECHAINS[resname]:
            put("CB", "carbon", 3.5, 0, -0.3)
        if not backbone_only:
            for k, (name, e) in enumerate(SIDECHAINS[resname]):
                put(name, {"C": "carbon", "N": "nitrogen", "O": "oxygen"}[e], 3.5 + 1.3 * k, 8 * (k % 2), -0.3)
        if i == len(seq) - 1:
            put("OXT", "oxygen", 1.9, 15, 1.6)
    return mdtraj.Trajectory(np.array(xyz, dtype=np.float32)[None] / 10.0, top)


def _modes(n_atoms, res_index, k, rng):
    L = res_index.max() + 1
    phase = rng.rand(k, 3) * 2 * np.pi
    freq = rng.rand(k) * 2 + 0.3
    return np.stack([np.sin(2 * np.pi * freq[i] * res_index[:, None] / L + phase[i]) for i in range(k)])


def perturbed(base, n_frames, amp=0.06, noise=0.015, seed=0, modes=None):
    rng = np.random.RandomState(seed)
    res_index = np.array([a.residue.index for a in base.top.atoms])
    M = modes if modes is not None else _modes(base.n_atoms, res_index, 3, np.random.RandomState(123))
    c = rng.randn(n_frames, M.shape[0]) * amp
    xyz = base.xyz[0][None] + np.einsum("fk,knd->fnd", c, M) + rng.randn(n_frames, base.n_atoms, 3) * noise
    return mdtraj.Trajectory(xyz.astype(np.float32), base.top)


def make_atlas_target(atlas_dir: str, name: str = "1abc_A", seq: str = DEFAULT_SEQ, n_frames: int = 120):
    """Write {atlas_dir}/{name}/{name}.pdb (with H) and three replica xtc files."""
    d = os.path.join(atlas_dir, name)
    os.makedirs(d, exist_ok=True)
    ref = build_protein(seq, with_hydrogens=True)
    ref.save_pdb(os.path.join(d, f"{name}.pdb"))
    for r in (1, 2, 3):
        perturbed(ref, n_frames, seed=r).save_xtc(os.path.join(d, f"{name}_prod_R{r}_fit.xtc"))
    return ref


def split_csv(path: str, names_seqs):
    with open(path, "w") as f:
        f.write("name,seqres,release_date,msa_id,seqlen\n")
        for name, seq in names_seqs:
            f.write(f"{name},{seq},2022-01-01,{name},{len(seq)}\n")
    return path


def save_cif(traj, path):
    import tempfile

    import biotite.structure.io as bsio

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name
    traj.save_pdb(tmp_path)
    bsio.save_structure(path, bsio.load_structure(tmp_path))
    os.remove(tmp_path)
