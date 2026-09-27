"""Convert heterogeneous baseline outputs into the evaluator's ensemble format.

Every baseline ends up as ``{ensemble_dir}/{name}.pdb``: one multi-model PDB per
ATLAS target, heavy atoms only, single chain, with residue names/numbers and
chain id copied from the ATLAS starting structure so that AlphaFlow's
atom-matching (``RESNAMEresSeq-ATOMNAME``) finds every common atom.
"""
from __future__ import annotations

import glob
import os
import tempfile
from typing import Iterable, List, Optional, Sequence

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # protonation / force-field variants
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K", "MSE": "M",
}


def _md():
    import mdtraj

    return mdtraj


def sequence_of(top) -> str:
    return "".join(THREE_TO_ONE.get(r.name, "X") for r in top.residues if r.is_protein or r.name in THREE_TO_ONE)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _cif_to_pdb_text(path: str) -> str:
    """Read an mmCIF (single or multi-model) with biotite, return PDB text."""
    import biotite.structure.io as bsio

    arr = bsio.load_structure(path)
    # Protenix/AF3-style CIFs use multi-character chain ids ("A0") that PDB cannot hold;
    # ensembles are single-chain monomers and harmonize_topology() restores the ATLAS chain.
    arr.set_annotation("chain_id", np.full(arr.array_length(), "A"))
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        bsio.save_structure(tmp_path, arr)
        with open(tmp_path) as f:
            return f.read()
    finally:
        os.remove(tmp_path)


def load_structure_file(path: str):
    """Load .pdb/.cif (single or multi-model) as an mdtraj Trajectory."""
    md = _md()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        text = _cif_to_pdb_text(path)
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        try:
            return md.load(tmp_path)
        finally:
            os.remove(tmp_path)
    return md.load(path)


def load_many(paths: Sequence[str], top_path: Optional[str] = None):
    """Load and concatenate structure files. ``.xtc``/``.dcd`` need ``top_path``."""
    md = _md()
    trajs = []
    for p in paths:
        if os.path.splitext(p)[1].lower() in (".xtc", ".dcd", ".nc", ".h5"):
            if top_path is None:
                raise ValueError(f"{p} requires a topology file")
            trajs.append(md.load(p, top=top_path))
        else:
            trajs.append(load_structure_file(p))
    if not trajs:
        raise FileNotFoundError("no structure files to load")
    heavy = [strip_hydrogens(t) for t in trajs]
    base = heavy[0]
    keys0 = [(a.residue.index, a.name) for a in base.top.atoms]
    xyz = [base.xyz]
    for t in heavy[1:]:
        keys = [(a.residue.index, a.name) for a in t.top.atoms]
        if keys != keys0:
            # reorder to the first file's atom order when the sets agree
            lookup = {k: i for i, k in enumerate(keys)}
            if set(lookup) != set(keys0):
                raise ValueError("inconsistent atom sets across ensemble members")
            t = t.atom_slice([lookup[k] for k in keys0])
        xyz.append(t.xyz)
    return md.Trajectory(np.concatenate(xyz), base.top)


def strip_hydrogens(traj):
    keep = [a.index for a in traj.top.atoms if a.element is None or a.element.symbol != "H"]
    return traj if len(keep) == traj.n_atoms else traj.atom_slice(keep)


# ----------------------------------------------------------------------------
# Topology harmonisation
# ----------------------------------------------------------------------------

def _protein_residues(top) -> List:
    return [r for r in top.residues if r.name in THREE_TO_ONE]


def residue_mapping(pred_top, ref_top) -> List[int]:
    """Map predicted residues onto reference residues by sequence.

    Returns, for each predicted protein residue, the index of the reference
    residue it corresponds to. Handles identical sequences and contiguous
    sub-sequences (prediction covering a window of the reference).
    """
    pred_seq = "".join(THREE_TO_ONE[r.name] for r in _protein_residues(pred_top))
    ref_seq = "".join(THREE_TO_ONE[r.name] for r in _protein_residues(ref_top))
    if pred_seq == ref_seq:
        return list(range(len(ref_seq)))
    off = ref_seq.find(pred_seq)
    if off >= 0:
        return list(range(off, off + len(pred_seq)))
    if len(pred_seq) == len(ref_seq):
        mism = sum(a != b for a, b in zip(pred_seq, ref_seq))
        if mism <= max(1, len(ref_seq) // 50):  # tolerate rare X/unknown mismatches
            return list(range(len(ref_seq)))
    raise ValueError(
        f"predicted sequence ({len(pred_seq)} aa) does not match reference ({len(ref_seq)} aa)\n"
        f"pred: {pred_seq}\nref:  {ref_seq}"
    )


def harmonize_topology(traj, ref_top):
    """Rebuild ``traj``'s topology with residue names/numbers/chain from ``ref_top``.

    Atom names are normalised (``OT1``->``O``, ``OT2``/``OC2``->``OXT``) and
    non-protein residues are dropped.
    """
    md = _md()
    traj = strip_hydrogens(traj)
    pred_res = _protein_residues(traj.top)
    ref_res = _protein_residues(ref_top)
    mapping = residue_mapping(traj.top, ref_top)
    ref_chain = ref_res[0].chain
    chain_id = getattr(ref_chain, "chain_id", None) or "A"

    new_top = md.Topology()
    chain = new_top.add_chain(chain_id) if _supports_chain_id(new_top) else new_top.add_chain()
    keep = []
    rename = {"OT1": "O", "OC1": "O", "OT2": "OXT", "OC2": "OXT", "O1": "O", "O2": "OXT"}
    for res, ref_idx in zip(pred_res, mapping):
        rr = ref_res[ref_idx]
        new_res = new_top.add_residue(rr.name, chain, resSeq=rr.resSeq)
        seen = set()
        for atom in res.atoms:
            name = rename.get(atom.name, atom.name)
            if name in seen:
                continue
            seen.add(name)
            new_top.add_atom(name, atom.element, new_res)
            keep.append(atom.index)
    xyz = traj.xyz[:, keep]
    return md.Trajectory(xyz, new_top)


def _supports_chain_id(top) -> bool:
    import inspect

    return "chain_id" in inspect.signature(top.add_chain).parameters


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------

def subsample(traj, n: Optional[int], seed: Optional[int] = None):
    """Keep ``n`` frames: the first ``n`` (seed None) or a seeded random subset."""
    if n is None or traj.n_frames <= n:
        return traj
    if seed is None:
        return traj[:n]
    idx = np.random.RandomState(seed).permutation(traj.n_frames)[:n]
    return traj[np.sort(idx)]


def write_ensemble(traj, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    traj.save_pdb(out_path)
    return out_path


def build_ensemble(
    files: Sequence[str],
    out_path: str,
    ref_pdb: Optional[str] = None,
    top_path: Optional[str] = None,
    n_frames: Optional[int] = None,
    seed: Optional[int] = None,
    shuffle_files_seed: Optional[int] = None,
    max_files: Optional[int] = None,
):
    """Merge structure files into one standardized multi-model PDB.

    ``shuffle_files_seed``/``max_files`` reproduce protocols that pick a random
    subset of generated files before merging (BioKinema/BioMD: seed 66, 300).
    """
    files = sorted(files, key=natural_key)
    if shuffle_files_seed is not None:
        import random

        rnd = random.Random(shuffle_files_seed)
        rnd.shuffle(files)
    if max_files is not None:
        files = files[:max_files]
    traj = load_many(files, top_path=top_path)
    if ref_pdb is not None:
        traj = harmonize_topology(traj, _md().load(ref_pdb).top)
    traj = subsample(traj, n_frames, seed)
    return write_ensemble(traj, out_path), traj.n_frames


def natural_key(path: str):
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def glob_files(patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        out.extend(glob.glob(p))
    return sorted(set(out), key=natural_key)
