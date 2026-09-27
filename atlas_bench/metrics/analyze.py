"""Per-target analysis of a generated ensemble against the ATLAS MD reference.

Port of AlphaFlow ``scripts/analyze_ensembles.py::main`` (MIT, see ``core.py``).
The procedure (and therefore Table 1 of AnewSampling / BioMD) is:

1. Load the ATLAS starting structure ``{name}.pdb`` ("crystal"), the three
   100 ns production replicas ``{name}_prod_R{1,2,3}_fit.xtc`` (pooled) and the
   predicted ensemble ``{pdbdir}/{name}.pdb`` (multi-model PDB).
2. Strip hydrogens; restrict all three to the intersection of atoms (matched by
   ``RESNAMEresSeq-ATOMNAME``).
3. Superpose everything on the starting structure (all heavy atoms), then take
   the C-alpha subsets and superpose those again.
4. Compute RMSF, pairwise RMSD, Gaussian W2 (per heavy atom), PCA-space W2,
   contacts, and side-chain SASA statistics. MD is sub-sampled with a fixed
   seed (137): ``RAND1``/``RAND2`` match the number of predicted frames and
   ``RAND1K`` draws the 1,000 MD frames used for W2 and SASA.
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import core

DEFAULT_SEED = 137


@dataclass
class AnalysisOptions:
    bb_only: bool = False
    ca_only: bool = False
    seed: int = DEFAULT_SEED
    n_md_subsample: int = 1000  # RAND1K in AlphaFlow
    replicas: Sequence[int] = (1, 2, 3)
    compute_sasa: bool = True


def _mdtraj():
    import mdtraj  # local import keeps the metrics core importable without mdtraj

    return mdtraj


def align_tops(top1, top2):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]
    names2_set = set(names2)
    intersection = [nam for nam in names1 if nam in names2_set]
    index1 = {nam: i for i, nam in reversed(list(enumerate(names1)))}
    index2 = {nam: i for i, nam in reversed(list(enumerate(names2)))}
    mask1 = [index1[nam] for nam in intersection]
    mask2 = [index2[nam] for nam in intersection]
    return mask1, mask2


def _heavy(traj):
    return traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != "H"])


def load_md_reference(atlas_dir: str, name: str, replicas: Sequence[int] = (1, 2, 3)):
    md = _mdtraj()
    topfile = os.path.join(atlas_dir, name, f"{name}.pdb")
    ref_aa = md.load(topfile)
    trajs = [
        md.load(os.path.join(atlas_dir, name, f"{name}_prod_R{i}_fit.xtc"), top=topfile) for i in replicas
    ]
    traj_aa = trajs[0]
    for t in trajs[1:]:
        traj_aa = traj_aa + t
    return ref_aa, traj_aa


def _residue_arrays(top):
    atom_names = [a.name for a in top.atoms]
    residue_index = np.array([a.residue.index for a in top.atoms])
    return atom_names, residue_index


def _sidechain_sasa(traj):
    md = _mdtraj()
    sasa = md.shrake_rupley(traj, probe_radius=core.SASA_PROBE_RADIUS_NM)
    atom_names, residue_index = _residue_arrays(traj.top)
    return core.condense_sidechain_sasas(sasa, atom_names, residue_index)


def analyze_trajectories(ref_aa, traj_aa, aftraj_aa, opts: Optional[AnalysisOptions] = None) -> Dict:
    """Analyze one target given loaded mdtraj objects (crystal, MD, prediction)."""
    md = _mdtraj()
    opts = opts or AnalysisOptions()
    out: Dict = {}

    traj_aa = _heavy(traj_aa)
    ref_aa = _heavy(ref_aa)
    aftraj_aa = _heavy(aftraj_aa)

    if opts.bb_only:
        aftraj_aa = aftraj_aa.atom_slice(
            [a.index for a in aftraj_aa.top.atoms if a.name in core.BACKBONE_ATOM_NAMES]
        )
    elif opts.ca_only:
        aftraj_aa = aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name == "CA"])

    refmask, afmask = align_tops(traj_aa.top, aftraj_aa.top)
    if len(refmask) == 0:
        raise ValueError("No common atoms between MD topology and predicted ensemble")
    traj_aa = traj_aa.atom_slice(refmask)
    ref_aa = ref_aa.atom_slice(refmask)
    aftraj_aa = aftraj_aa.atom_slice(afmask)
    out["n_aligned_atoms"] = aftraj_aa.n_atoms
    out["n_pred_frames"] = aftraj_aa.n_frames
    out["n_md_frames"] = traj_aa.n_frames
    names = [a.name for a in aftraj_aa.top.atoms]
    # Backbone-frame models (BioEmu) add CB only; SASA metrics then reflect CB-level side chains.
    out["has_cb"] = "CB" in names
    out["has_sidechains"] = any(n not in core.BACKBONE_ATOM_NAMES + ("CB",) for n in names)

    rng = np.random.RandomState(opts.seed)
    # Same draw order as AlphaFlow (np.random.seed(137); randint x3).
    RAND1 = rng.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND2 = rng.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND1K = rng.randint(0, traj_aa.n_frames, opts.n_md_subsample)

    traj_aa.superpose(ref_aa)
    aftraj_aa.superpose(ref_aa)

    out["ca_mask"] = ca_mask = [a.index for a in traj_aa.top.atoms if a.name == "CA"]
    traj = traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    aftraj = aftraj_aa.atom_slice(ca_mask, False)
    traj.superpose(ref)
    aftraj.superpose(ref)
    n_atoms = aftraj.n_atoms

    # ---------------- PCA ----------------
    ref_pca, ref_coords = core.get_pca(traj.xyz)
    af_coords_ref_pca = ref_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    seed_coords_ref_pca = ref_pca.transform(ref.xyz.reshape(1, -1))

    af_pca, af_coords = core.get_pca(aftraj.xyz)
    ref_coords_af_pca = af_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_af_pca = af_pca.transform(ref.xyz.reshape(1, -1))

    joint_pca, _ = core.get_pca(np.concatenate([traj.xyz[RAND1], aftraj.xyz]))
    af_coords_joint_pca = joint_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    ref_coords_joint_pca = joint_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_joint_pca = joint_pca.transform(ref.xyz.reshape(1, -1))

    out["ref_variance"] = ref_pca.explained_variance_ / n_atoms * 100
    out["af_variance"] = af_pca.explained_variance_ / n_atoms * 100
    out["joint_variance"] = joint_pca.explained_variance_ / n_atoms * 100

    # ---------------- RMSF (all aligned heavy atoms, as in AlphaFlow) ----------------
    out["af_rmsf"] = md.rmsf(aftraj_aa, ref_aa) * 10
    out["ref_rmsf"] = md.rmsf(traj_aa, ref_aa) * 10
    out["af_ca_rmsf"] = md.rmsf(aftraj, ref) * 10
    out["ref_ca_rmsf"] = md.rmsf(traj, ref) * 10

    # ---------------- Gaussian W2 per heavy atom ----------------
    out["emd_mean"], out["emd_var"] = core.gaussian_w2_terms(traj_aa.xyz[RAND1K], aftraj_aa.xyz)

    # ---------------- Side-chain SASA ----------------
    if opts.compute_sasa:
        sasa_thresh = core.SASA_EXPOSED_THRESH_NM2
        af_sasa = _sidechain_sasa(aftraj_aa)
        ref_sasa = _sidechain_sasa(traj_aa[RAND1K])
        out["crystal_sasa"] = _sidechain_sasa(ref_aa)
        out["ref_sa_prob"] = (ref_sasa > sasa_thresh).mean(0)
        out["af_sa_prob"] = (af_sasa > sasa_thresh).mean(0)
        out["ref_mi_mat"] = core.sasa_mi(ref_sasa > sasa_thresh)
        out["af_mi_mat"] = core.sasa_mi(af_sasa > sasa_thresh)

    # ---------------- Contacts ----------------
    out["ref_contact_prob"] = core.contact_prob(traj.xyz[RAND1])
    out["af_contact_prob"] = core.contact_prob(aftraj.xyz)
    out["crystal_distmat"] = core.pairwise_distmat(ref.xyz[:1])[0]

    # ---------------- Pairwise RMSD ----------------
    ref_x1, ref_x2 = traj.xyz[RAND1], traj.xyz[RAND2]
    out["ref_mean_pairwise_rmsd"] = core.get_rmsds(ref_x1, ref_x2, broadcast=True).mean()
    out["af_mean_pairwise_rmsd"] = core.get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True).mean()
    out["ref_rms_pairwise_rmsd"] = np.square(core.get_rmsds(ref_x1, ref_x2, broadcast=True)).mean() ** 0.5
    out["af_rms_pairwise_rmsd"] = np.square(core.get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True)).mean() ** 0.5
    out["ref_self_mean_pairwise_rmsd"] = core.get_rmsds(ref_x1, ref_x1, broadcast=True).mean()
    out["ref_self_rms_pairwise_rmsd"] = np.square(core.get_rmsds(ref_x1, ref_x1, broadcast=True)).mean() ** 0.5

    out["cosine_sim"] = (ref_pca.components_[0] * af_pca.components_[0]).sum()

    K = core.PCA_N_COMPONENTS_W2
    out["EMD,ref"] = core.get_emd(ref_coords[RAND1], ref_coords[RAND2], af_coords_ref_pca, seed_coords_ref_pca, n_atoms, K=K)
    out["EMD,af2"] = core.get_emd(ref_coords_af_pca[RAND1], ref_coords_af_pca[RAND2], af_coords, seed_coords_af_pca, n_atoms, K=K)
    out["EMD,joint"] = core.get_emd(
        ref_coords_joint_pca[RAND1], ref_coords_joint_pca[RAND2], af_coords_joint_pca, seed_coords_joint_pca, n_atoms, K=K
    )
    return out


def analyze_target(name: str, atlas_dir: str, pdbdir: str, opts: Optional[AnalysisOptions] = None) -> Dict:
    md = _mdtraj()
    opts = opts or AnalysisOptions()
    ref_aa, traj_aa = load_md_reference(atlas_dir, name, opts.replicas)
    aftraj_aa = md.load(os.path.join(pdbdir, f"{name}.pdb"))
    return analyze_trajectories(ref_aa, traj_aa, aftraj_aa, opts)


def list_targets(pdbdir: str) -> List[str]:
    return sorted(f[: -len(".pdb")] for f in os.listdir(pdbdir) if f.endswith(".pdb"))


class _Worker:
    def __init__(self, atlas_dir, pdbdir, opts):
        self.atlas_dir, self.pdbdir, self.opts = atlas_dir, pdbdir, opts

    def __call__(self, name):
        try:
            return name, analyze_target(name, self.atlas_dir, self.pdbdir, self.opts), None
        except Exception as e:  # keep going; failures are reported in the summary
            return name, None, f"{type(e).__name__}: {e}"


def analyze_directory(
    atlas_dir: str,
    pdbdir: str,
    names: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    opts: Optional[AnalysisOptions] = None,
    out_path: Optional[str] = None,
):
    """Analyze every ``{name}.pdb`` in ``pdbdir``; write ``out.pkl`` (AlphaFlow format)."""
    import tqdm

    names = list(names) if names else list_targets(pdbdir)
    worker = _Worker(atlas_dir, pdbdir, opts or AnalysisOptions())
    results, failures = {}, {}
    if num_workers > 1:
        with Pool(num_workers) as pool:
            it = pool.imap_unordered(worker, names)
            for name, out, err in tqdm.tqdm(it, total=len(names)):
                (results.__setitem__(name, out) if err is None else failures.__setitem__(name, err))
    else:
        for name in tqdm.tqdm(names):
            name, out, err = worker(name)
            (results.__setitem__(name, out) if err is None else failures.__setitem__(name, err))
    out_path = out_path or os.path.join(pdbdir, "out.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    if failures:
        with open(os.path.splitext(out_path)[0] + ".failures.txt", "w") as f:
            for k, v in sorted(failures.items()):
                f.write(f"{k}\t{v}\n")
    print(f"Analyzed {len(results)}/{len(names)} targets -> {out_path}")
    for k, v in sorted(failures.items()):
        print(f"  FAILED {k}: {v}")
    return results, failures


def build_argparser(parser: Optional[argparse.ArgumentParser] = None):
    parser = parser or argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--atlas_dir", required=True, help="ATLAS root: {name}/{name}.pdb, {name}_prod_R*_fit.xtc")
    parser.add_argument("--pdbdir", required=True, help="Directory of {name}.pdb multi-model ensembles")
    parser.add_argument("--pdb_id", nargs="*", default=[])
    parser.add_argument("--bb_only", action="store_true")
    parser.add_argument("--ca_only", action="store_true")
    parser.add_argument("--no_sasa", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--out", default=None, help="Output pickle (default: {pdbdir}/out.pkl)")
    return parser


def main(args=None):
    args = build_argparser().parse_args(args) if not isinstance(args, argparse.Namespace) else args
    opts = AnalysisOptions(bb_only=args.bb_only, ca_only=args.ca_only, seed=args.seed, compute_sasa=not args.no_sasa)
    analyze_directory(args.atlas_dir, args.pdbdir, args.pdb_id, args.num_workers, opts, args.out)


if __name__ == "__main__":
    main()
