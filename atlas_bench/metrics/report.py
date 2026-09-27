"""Aggregate per-target analyses (``out.pkl``) into the Table 1 metrics.

Port of AlphaFlow ``scripts/print_analysis.py`` (MIT, see ``core.py``); the
column names of the original script are kept and mapped onto the row labels of
Table 1 of the AnewSampling paper by :data:`TABLE1_METRICS`.
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from . import core
from ..paper_table1 import METHOD_DISPLAY_NAMES, PAPER_TABLE1, TABLE1_METRICS


def correlations(a, b, prefix=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            prefix + "pearson": scipy.stats.pearsonr(a, b)[0],
            prefix + "spearman": scipy.stats.spearmanr(a, b)[0],
            prefix + "kendall": scipy.stats.kendalltau(a, b)[0],
        }


def per_target_table(data: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-target statistics (AlphaFlow ``analyze_data``)."""
    rows = []
    for name, out in data.items():
        item = {
            "name": name,
            "md_pairwise": out["ref_mean_pairwise_rmsd"],
            "af_pairwise": out["af_mean_pairwise_rmsd"],
            "cosine_sim": abs(out["cosine_sim"]),
            "emd_mean": np.square(out["emd_mean"]).mean() ** 0.5,
            "emd_var": np.square(out["emd_var"]).mean() ** 0.5,
            "has_sidechains": out.get("has_sidechains", True),
            "n_pred_frames": out.get("n_pred_frames", np.nan),
        }
        item.update(correlations(out["af_rmsf"], out["ref_rmsf"], prefix="rmsf_"))
        for key in ("EMD,ref", "EMD,joint"):
            emd = out[key]
            tag = key.split(",")[1]
            item[tag + "emd"] = emd["ref|af"]
            item[tag + "emd_tr"] = emd["ref mean|af mean"]
            item[tag + "emd_int"] = max(emd["ref|af"] ** 2 - emd["ref mean|af mean"] ** 2, 0.0) ** 0.5

        crystal_contact_mask = out["crystal_distmat"] < core.CONTACT_CUTOFF_NM
        ref_transient = (~crystal_contact_mask) & (out["ref_contact_prob"] > core.PROB_THRESH)
        af_transient = (~crystal_contact_mask) & (out["af_contact_prob"] > core.PROB_THRESH)
        ref_weak = crystal_contact_mask & (out["ref_contact_prob"] < core.WEAK_PROB_THRESH)
        af_weak = crystal_contact_mask & (out["af_contact_prob"] < core.WEAK_PROB_THRESH)
        item["weak_contacts_iou"] = core.jaccard(ref_weak, af_weak)
        item["transient_contacts_iou"] = core.jaccard(ref_transient, af_transient)

        if "crystal_sasa" in out:
            buried_mask = out["crystal_sasa"][0] < core.SASA_EXPOSED_THRESH_NM2
            ref_sa_mask = (out["ref_sa_prob"] > core.PROB_THRESH) & buried_mask
            af_sa_mask = (out["af_sa_prob"] > core.PROB_THRESH) & buried_mask
            item["num_sasa"] = int(ref_sa_mask.sum())
            item["sasa_iou"] = core.jaccard(ref_sa_mask, af_sa_mask)
            item.update(
                correlations(out["ref_mi_mat"].flatten(), out["af_mi_mat"].flatten(), prefix="exposon_mi_")
            )
        else:
            item.update({"num_sasa": np.nan, "sasa_iou": np.nan, "exposon_mi_spearman": np.nan})
        rows.append(item)

    df = pd.DataFrame(rows).set_index("name")
    all_ref_rmsf = np.concatenate([data[name]["ref_rmsf"] for name in df.index])
    all_af_rmsf = np.concatenate([data[name]["af_rmsf"] for name in df.index])
    return all_ref_rmsf, all_af_rmsf, df


def summarize(data: Dict[str, Dict]) -> "OrderedDict[str, float]":
    """Benchmark-level summary with the column names of AlphaFlow's print_analysis."""
    ref_rmsf, af_rmsf, df = per_target_table(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = OrderedDict()
        s["count"] = len(df)
        s["MD pairwise RMSD"] = df.md_pairwise.median()
        s["Pairwise RMSD"] = df.af_pairwise.median()
        s["Pairwise RMSD r"] = scipy.stats.pearsonr(df.md_pairwise, df.af_pairwise)[0] if len(df) > 1 else np.nan
        s["MD RMSF"] = np.median(ref_rmsf)
        s["RMSF"] = np.median(af_rmsf)
        s["Global RMSF r"] = scipy.stats.pearsonr(ref_rmsf, af_rmsf)[0]
        s["Per target RMSF r"] = df.rmsf_pearson.median()
        s["RMWD"] = np.sqrt(df.emd_mean ** 2 + df.emd_var ** 2).median()
        s["RMWD trans"] = df.emd_mean.median()
        s["RMWD var"] = df.emd_var.median()
        s["MD PCA W2"] = df.refemd.median()
        s["Joint PCA W2"] = df.jointemd.median()
        s["PC sim > 0.5 %"] = (df.cosine_sim > 0.5).mean() * 100
        s["Weak contacts J"] = df.weak_contacts_iou.median()
        s["Weak contacts nans"] = df.weak_contacts_iou.isna().mean()
        s["Transient contacts J"] = df.transient_contacts_iou.median()
        s["Transient contacts nans"] = df.transient_contacts_iou.isna().mean()
        s["Exposed residue J"] = df.sasa_iou.median()
        s["Exposed MI matrix rho"] = df.exposon_mi_spearman.median()
        s["Side-chain coverage %"] = df.has_sidechains.mean() * 100
        s["Median #frames"] = df.n_pred_frames.median()
    return s


def load_pickle(path: str) -> Dict[str, Dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {k: v for k, v in data.items() if v is not None}


def restrict(data: Dict[str, Dict], names: Optional[Sequence[str]]) -> Dict[str, Dict]:
    if not names:
        return data
    keep = set(names)
    return {k: v for k, v in data.items() if k in keep}


def table1_frame(
    summaries: "OrderedDict[str, Dict[str, float]]",
    with_paper: bool = True,
) -> pd.DataFrame:
    """Table-1 shaped frame: rows = metrics, columns = methods (+ paper values)."""
    index = pd.MultiIndex.from_tuples([(m.category, m.label) for m in TABLE1_METRICS], names=["Category", "Metric"])
    cols = OrderedDict()
    for method, s in summaries.items():
        label = METHOD_DISPLAY_NAMES.get(method, method)
        cols[label] = [s.get(m.key, np.nan) for m in TABLE1_METRICS]
        if with_paper:
            paper = PAPER_TABLE1.get(method)
            if paper is not None:
                cols[f"{label} [paper]"] = [paper.get(m.label, np.nan) for m in TABLE1_METRICS]
    return pd.DataFrame(cols, index=index)


def format_table(df: pd.DataFrame, fmt: str = "markdown") -> str:
    def cell(v, pct):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{v:.0f}" if pct else f"{v:.2f}"

    pct_rows = {m.label for m in TABLE1_METRICS if m.key.endswith("%")}
    arrows = {m.label: ("↑" if m.higher_is_better else "↓") for m in TABLE1_METRICS}
    header = ["Category", "Metric"] + list(df.columns)
    lines = []
    if fmt == "markdown":
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
    for (cat, metric), row in df.iterrows():
        cells = [cat, f"{metric} ({arrows[metric]})"] + [cell(v, metric in pct_rows) for v in row.values]
        if fmt == "markdown":
            lines.append("| " + " | ".join(cells) + " |")
        elif fmt == "latex":
            lines.append(" & ".join(cells) + r" \\")
        else:
            lines.append("\t".join(cells))
    if fmt == "tsv":
        lines.insert(0, "\t".join(header))
    return "\n".join(lines)


def parse_run_specs(specs: Sequence[str]) -> "OrderedDict[str, str]":
    """``METHOD=path/to/out.pkl`` pairs (a bare path uses its parent directory name)."""
    import os

    runs = OrderedDict()
    for spec in specs:
        if "=" in spec:
            method, path = spec.split("=", 1)
        else:
            path = spec
            method = os.path.basename(os.path.dirname(os.path.abspath(path)))
        runs[method] = path
    return runs


def build_argparser(parser: Optional[argparse.ArgumentParser] = None):
    parser = parser or argparse.ArgumentParser(description="Print Table 1 from out.pkl files")
    parser.add_argument("runs", nargs="+", help="METHOD=path/to/out.pkl (METHOD should match a Table 1 column)")
    parser.add_argument("--targets", default=None, help="Optional file with target names to restrict to")
    parser.add_argument("--common_targets", action="store_true", help="Restrict all runs to their common targets")
    parser.add_argument("--no_paper", action="store_true", help="Do not print the paper-reported values")
    parser.add_argument("--format", choices=["markdown", "latex", "tsv"], default="markdown")
    parser.add_argument("--csv", default=None, help="Also save the full summary to CSV")
    return parser


def main(args=None):
    args = build_argparser().parse_args(args) if not isinstance(args, argparse.Namespace) else args
    runs = parse_run_specs(args.runs)
    datas = OrderedDict((m, load_pickle(p)) for m, p in runs.items())
    names: Optional[List[str]] = None
    if args.targets:
        with open(args.targets) as f:
            names = [l.strip().split(",")[0] for l in f if l.strip() and not l.startswith("name")]
    if args.common_targets:
        common = set.intersection(*(set(d) for d in datas.values()))
        names = sorted(common if names is None else common & set(names))
    summaries = OrderedDict((m, summarize(restrict(d, names))) for m, d in datas.items())
    full = pd.DataFrame(summaries)
    print(full.round(3).to_string())
    print()
    print(format_table(table1_frame(summaries, with_paper=not args.no_paper), args.format))
    if args.csv:
        full.to_csv(args.csv)
    return summaries


if __name__ == "__main__":
    main()
