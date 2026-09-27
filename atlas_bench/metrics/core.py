"""Array-level building blocks of the ATLAS ensemble benchmark.

Ported from AlphaFlow's ``scripts/analyze_ensembles.py`` (MIT License,
Copyright (c) 2024 Bowen Jing, Bonnie Berger, Tommi Jaakkola,
https://github.com/bjing2016/alphaflow @ 0408d7c). The same code is reused
verbatim by ConfDiff, EBA, MDGen and BioMD/BioKinema, and Table 1 of the
AnewSampling paper copies the baseline numbers produced with it, so every
function here keeps the original numerics. Deviations are limited to numerical
guards and are marked with ``NOTE``.

All coordinates are in nanometers (mdtraj convention); distances reported by
the benchmark are converted to Angstrom (``* 10``) exactly where the original
code does it.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

BACKBONE_ATOM_NAMES = ("CA", "C", "N", "O", "OXT")

# Thresholds used by the AlphaFlow benchmark.
CONTACT_CUTOFF_NM = 0.8  # Calpha-Calpha contact cutoff (8 A)
SASA_PROBE_RADIUS_NM = 0.28  # Shrake-Rupley probe radius (2.8 A)
SASA_EXPOSED_THRESH_NM2 = 0.02  # side-chain SASA > 2.0 A^2 => exposed
PROB_THRESH = 0.1  # "in at least 10% of the ensemble"
WEAK_PROB_THRESH = 0.9  # crystal contact broken in >= 10% of the ensemble
PCA_N_COMPONENTS_W2 = 2  # PCA W2 distances use the top-2 PCs


def get_pca(xyz: np.ndarray):
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords


def get_rmsds(traj1: np.ndarray, traj2: np.ndarray, broadcast: bool = False) -> np.ndarray:
    """RMSD (Angstrom) between pre-aligned frames; no additional superposition."""
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:, None], traj2[None]
    distmat = np.square(traj1 - traj2).sum(-1) ** 0.5 / n_atoms ** 0.5 * 10
    return distmat


def sidechain_mask(atom_names) -> np.ndarray:
    return np.array([name not in BACKBONE_ATOM_NAMES for name in atom_names])


def condense_sidechain_sasas(sasas: np.ndarray, atom_names, residue_index) -> np.ndarray:
    """Sum atom-level SASA over side-chain atoms of each residue.

    ``atom_names``/``residue_index`` describe the atoms of the topology the
    SASA was computed on (``residue_index`` is 0-based and contiguous).
    """
    residue_index = np.asarray(residue_index)
    n_residues = int(residue_index.max()) + 1
    assert n_residues > 1
    if sasas.shape[1] != len(residue_index):
        raise ValueError(
            f"Number of atoms in topology ({len(residue_index)}) does not match "
            f"number of SASA values ({sasas.shape[1]}); compute atom-level SASA."
        )
    sc_mask = sidechain_mask(atom_names)
    rsd_sasas = np.zeros((sasas.shape[0], n_residues), dtype="float32")
    for i in range(n_residues):
        rsd_sasas[:, i] = sasas[:, sc_mask & (residue_index == i)].sum(1)
    return rsd_sasas


def sasa_mi(sasa: np.ndarray) -> np.ndarray:
    """Mutual information matrix between binary per-residue exposure indicators."""
    sasa = np.asarray(sasa, dtype=bool)
    N, L = sasa.shape
    joint_probs = np.zeros((L, L, 2, 2))

    joint_probs[:, :, 1, 1] = (sasa[:, :, None] & sasa[:, None, :]).mean(0)
    joint_probs[:, :, 1, 0] = (sasa[:, :, None] & ~sasa[:, None, :]).mean(0)
    joint_probs[:, :, 0, 1] = (~sasa[:, :, None] & sasa[:, None, :]).mean(0)
    joint_probs[:, :, 0, 0] = (~sasa[:, :, None] & ~sasa[:, None, :]).mean(0)

    marginal_probs = np.stack([1 - sasa.mean(0), sasa.mean(0)], -1)
    indep_probs = marginal_probs[None, :, None, :] * marginal_probs[:, None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(joint_probs * np.log(joint_probs / indep_probs), (-1, -2))
    mi[np.arange(L), np.arange(L)] = 0
    return mi


def get_mean_covar(xyz: np.ndarray):
    """Per-atom mean (n_atoms, 3) and 3x3 covariance (n_atoms, 3, 3)."""
    mean = xyz.mean(0)
    xyz = xyz - mean
    covar = (xyz[..., None] * xyz[..., None, :]).mean(0)
    return mean, covar


def sqrtm(M: np.ndarray) -> np.ndarray:
    """Batched matrix square root via eigendecomposition (as in AlphaFlow).

    NOTE: ``M`` is a product of two PSD matrices, so its eigenvalues are real
    and non-negative in exact arithmetic. The original code can return complex
    values / NaN from round-off; we drop the imaginary round-off and clip tiny
    negative eigenvalues, which only changes degenerate cases.
    """
    D, P = np.linalg.eig(M)
    D = np.clip(np.real(D), 0.0, None)
    out = (P * np.sqrt(D)[..., None, :]) @ np.linalg.inv(P)
    return np.real(out)


def gaussian_w2_terms(ref_xyz: np.ndarray, pred_xyz: np.ndarray):
    """Per-atom translation and variance terms of the Gaussian W2 distance (Angstrom).

    W2^2(N1, N2) = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2})  (paper Eq. 1).
    Returns ``emd_mean`` = ||mu1 - mu2|| and ``emd_var`` = sqrt(Tr(...)) per atom.
    """
    ref_mean, ref_covar = get_mean_covar(ref_xyz)
    af_mean, af_covar = get_mean_covar(pred_xyz)
    emd_mean = (np.square(ref_mean - af_mean).sum(-1) ** 0.5) * 10
    try:
        tr = np.trace(ref_covar + af_covar - 2 * sqrtm(ref_covar @ af_covar), axis1=1, axis2=2)
        emd_var = (np.clip(tr, 0.0, None) ** 0.5) * 10
    except np.linalg.LinAlgError:
        emd_var = np.trace(ref_covar, axis1=1, axis2=2) ** 0.5 * 10
    return emd_mean, emd_var


def rmwd_decomposition(emd_mean: np.ndarray, emd_var: np.ndarray) -> Dict[str, float]:
    """Root-mean W2 distance and its translation / variance contributions (paper Eq. 2)."""
    trans = float(np.square(emd_mean).mean() ** 0.5)
    var = float(np.square(emd_var).mean() ** 0.5)
    return {"rmwd": float(np.sqrt(trans ** 2 + var ** 2)), "rmwd_trans": trans, "rmwd_var": var}


def get_wasserstein(distmat: np.ndarray, p: int = 2) -> float:
    """Empirical W_p between two equally sized point clouds via optimal assignment."""
    assert distmat.shape[0] == distmat.shape[1]
    distmat = distmat ** p
    row_ind, col_ind = linear_sum_assignment(distmat)
    return float(distmat[row_ind, col_ind].mean() ** (1 / p))


def get_emd(ref_coords1, ref_coords2, af_coords, seed_coords, n_atoms: int, K: Optional[int] = None):
    """W2 statistics in a (PCA) coordinate space, normalised to RMSD units (Angstrom)."""
    if len(ref_coords1.shape) == 3:
        ref_coords1 = ref_coords1.reshape(ref_coords1.shape[0], -1)
        ref_coords2 = ref_coords2.reshape(ref_coords2.shape[0], -1)
        af_coords = af_coords.reshape(af_coords.shape[0], -1)
        seed_coords = seed_coords.reshape(seed_coords.shape[0], -1)
    if K is not None:
        ref_coords1 = ref_coords1[:, :K]
        ref_coords2 = ref_coords2[:, :K]
        af_coords = af_coords[:, :K]
        seed_coords = seed_coords[:, :K]
    scale = 10 / n_atoms ** 0.5
    emd = {}
    emd["ref|ref mean"] = (np.square(ref_coords1 - ref_coords1.mean(0)).sum(-1)).mean() ** 0.5 * scale

    distmat = np.square(ref_coords1[:, None] - ref_coords2[None]).sum(-1) ** 0.5 * scale
    emd["ref|ref2"] = get_wasserstein(distmat)
    emd["ref mean|ref2 mean"] = np.square(ref_coords1.mean(0) - ref_coords2.mean(0)).sum() ** 0.5 * scale

    distmat = np.square(ref_coords1[:, None] - af_coords[None]).sum(-1) ** 0.5 * scale
    emd["ref|af"] = get_wasserstein(distmat)
    emd["ref mean|af mean"] = np.square(ref_coords1.mean(0) - af_coords.mean(0)).sum() ** 0.5 * scale

    emd["ref|seed"] = (np.square(ref_coords1 - seed_coords).sum(-1)).mean() ** 0.5 * scale
    emd["ref mean|seed"] = (np.square(ref_coords1.mean(0) - seed_coords).sum(-1)).mean() ** 0.5 * scale

    emd["af|seed"] = (np.square(af_coords - seed_coords).sum(-1)).mean() ** 0.5 * scale
    emd["af|af mean"] = (np.square(af_coords - af_coords.mean(0)).sum(-1)).mean() ** 0.5 * scale
    emd["af mean|seed"] = (np.square(af_coords.mean(0) - seed_coords).sum(-1)).mean() ** 0.5 * scale
    return emd


def pairwise_distmat(ca_xyz: np.ndarray) -> np.ndarray:
    """(frames, L, L) Calpha distance matrices in nm."""
    return np.linalg.norm(ca_xyz[:, None, :] - ca_xyz[:, :, None], axis=-1)


def contact_prob(ca_xyz: np.ndarray, cutoff: float = CONTACT_CUTOFF_NM, chunk: int = 32) -> np.ndarray:
    """Fraction of frames with each Calpha pair closer than ``cutoff`` (chunked for memory)."""
    counts = np.zeros((ca_xyz.shape[1], ca_xyz.shape[1]))
    for i in range(0, ca_xyz.shape[0], chunk):
        counts += (pairwise_distmat(ca_xyz[i : i + chunk]) < cutoff).sum(0)
    return counts / ca_xyz.shape[0]


def rmsf(xyz: np.ndarray) -> np.ndarray:
    """Per-atom RMSF (Angstrom) of pre-superposed frames around their mean."""
    mean = xyz.mean(0)
    return np.sqrt(np.square(xyz - mean).sum(-1).mean(0)) * 10


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = (mask_a | mask_b).sum()
    if union == 0:
        return float("nan")
    return float((mask_a & mask_b).sum() / union)
