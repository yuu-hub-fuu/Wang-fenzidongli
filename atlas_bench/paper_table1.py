"""Table 1 of "Learning the All-Atom Equilibrium Distribution of Biomolecular
Interactions at Scale" (AnewSampling, 2026): ATLAS test set, monomer ensembles.

The paper states that all baseline numbers are sourced from the BioMD paper
(ICLR 2026), which in turn collects them from the original publications
(AlphaFlow/ESMFlow, ConfDiff, MDGen, EBA, ...). ``None`` marks entries reported
as "-" in the paper.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    category: str
    label: str  # row label in Table 1
    key: str  # column name produced by metrics.report.summarize (AlphaFlow naming)
    higher_is_better: bool


TABLE1_METRICS = (
    Metric("Predicting flexibility", "Pairwise RMSD r", "Pairwise RMSD r", True),
    Metric("Predicting flexibility", "Global RMSF r", "Global RMSF r", True),
    Metric("Predicting flexibility", "Per-target RMSF r", "Per target RMSF r", True),
    Metric("Distributional accuracy", "Root mean W2-dist.", "RMWD", False),
    Metric("Distributional accuracy", "Trans. contrib.", "RMWD trans", False),
    Metric("Distributional accuracy", "Var. contrib.", "RMWD var", False),
    Metric("Distributional accuracy", "MD PCA W2-dist.", "MD PCA W2", False),
    Metric("Distributional accuracy", "Joint PCA W2-dist.", "Joint PCA W2", False),
    Metric("Distributional accuracy", "% PC-sim > 0.5", "PC sim > 0.5 %", True),
    Metric("Ensemble observables", "Weak contacts J", "Weak contacts J", True),
    Metric("Ensemble observables", "Transient contacts J", "Transient contacts J", True),
    Metric("Ensemble observables", "Exposed residue J", "Exposed residue J", True),
    Metric("Ensemble observables", "Exposed MI matrix rs", "Exposed MI matrix rho", True),
)

# Column order of Table 1 -> display name.
METHOD_DISPLAY_NAMES = OrderedDict(
    [
        ("esmflow_md_full", "ESMFlow-MD (Full)"),
        ("esmflow_md_distilled", "ESMFlow-MD (Distilled)"),
        ("confdiff", "ConfDiff"),
        ("bioemu", "BioEmu"),
        ("str2str", "Str2Str"),
        ("mdgen", "MDGen"),
        ("eba", "EBA"),
        ("biomd", "BioMD"),
        ("anewsampling", "AnewSampling"),
    ]
)

_ROWS = [m.label for m in TABLE1_METRICS]
_VALUES = {
    #                       PwRMSD GlbRMSF PerRMSF RMWD  Trans  Var   MDPCA JntPCA PCsim  Weak  Trans  Expo  MI
    "esmflow_md_full":      [0.19, 0.31, 0.76, 3.60, 3.13, 1.74, 1.51, 3.19, 26, 0.55, 0.34, 0.49, 0.20],
    "esmflow_md_distilled": [0.19, 0.33, 0.74, 4.23, 3.75, 1.90, 1.87, 3.79, 33, 0.48, 0.30, 0.43, 0.16],
    "confdiff":             [0.59, 0.67, 0.85, 2.76, 2.23, 1.40, 1.44, 2.25, 35, 0.59, 0.36, 0.50, 0.24],
    "bioemu":               [0.46, 0.57, 0.71, 4.32, 4.04, 1.77, 1.97, 3.98, 51, 0.33, None, None, 0.07],
    "str2str":              [0.23, 0.38, 0.57, 4.05, 3.73, 1.43, 2.04, 3.55, 12, 0.43, None, None, 0.21],
    "mdgen":                [0.48, 0.50, 0.71, 2.69, None, None, 1.89, None, None, 0.51, None, 0.29, None],
    "eba":                  [0.62, 0.71, 0.90, 2.43, 2.03, 1.20, 1.19, 2.04, 44, 0.65, 0.41, 0.70, 0.36],
    "biomd":                [0.70, 0.76, 0.91, 2.18, 1.89, 1.10, 1.24, 1.82, 46, 0.61, 0.46, 0.70, 0.34],
    "anewsampling":         [0.85, 0.89, 0.93, 1.88, 1.63, 0.89, 1.16, 1.44, 51, 0.70, 0.52, 0.75, 0.41],
}

PAPER_TABLE1 = OrderedDict(
    (method, OrderedDict((row, (float("nan") if v is None else float(v))) for row, v in zip(_ROWS, vals)))
    for method, vals in _VALUES.items()
)
