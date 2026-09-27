"""End-to-end: native raw outputs of every baseline -> collect -> evaluate -> Table 1."""
import os

import mdtraj
import numpy as np
import pytest

import synthetic
from atlas_bench import atlas_data
from atlas_bench.baselines import REGISTRY, RunContext, get_baseline
from atlas_bench.metrics import analyze, report

N = 12  # frames per mock ensemble (250 in the real protocol)


def _ctx(atlas, tmp_path, key, **options):
    return RunContext(
        atlas_dir=atlas["atlas_dir"],
        run_dir=str(tmp_path / key),
        targets=atlas_data.read_split(atlas["split"]),
        repo_dir=str(tmp_path / "repo"),
        n_samples=N,
        options=options,
    )


def _pred(backbone_only=False, first_resseq=1, n=N, seed=7, with_cb=False):
    base = synthetic.build_protein(backbone_only=backbone_only, first_resseq=first_resseq, with_cb=with_cb)
    return synthetic.perturbed(base, n, amp=0.05, noise=0.02, seed=seed)


def write_mock_outputs(key, ctx, name):
    raw = ctx.raw_dir
    os.makedirs(raw, exist_ok=True)
    if key in ("esmflow_md_full", "esmflow_md_distilled", "mdgen"):
        _pred().save_pdb(os.path.join(raw, f"{name}.pdb"))
    elif key == "confdiff":
        d = os.path.join(raw, "test_gen", name)
        os.makedirs(d)
        for i, frame in enumerate(_pred()):
            frame.save_pdb(os.path.join(d, f"{name}_sample{i}.pdb"))
    elif key == "bioemu":
        d = os.path.join(raw, name)
        os.makedirs(d)
        t = _pred(backbone_only=True, first_resseq=0, with_cb=True)  # BioEmu writes N, CA, C, CB, O
        t[0].save_pdb(os.path.join(d, "topology.pdb"))
        t.save_xtc(os.path.join(d, "samples.xtc"))
    elif key == "str2str":
        d = os.path.join(raw, "all_delta")
        os.makedirs(d)
        _pred(backbone_only=True, first_resseq=0).save_pdb(os.path.join(d, f"{name}.pdb"))
    elif key == "eba":
        d = os.path.join(raw, "eba_atlas_test", "dumps", "atlas_test", f"{name}_0", "seed_42", "predictions")
        os.makedirs(d)
        for i, frame in enumerate(_pred()):
            synthetic.save_cif(frame, os.path.join(d, f"{name}_0_seed_42_sample_{i}.cif"))
    elif key == "biomd":
        for r in (1, 2, 3):
            d = os.path.join(raw, f"{name}_R{r}_0", "seed_101", "predictions")
            os.makedirs(d)
            for i, frame in enumerate(_pred(n=6, seed=r)):
                synthetic.save_cif(frame, os.path.join(d, f"{name}_R{r}_0_f{i}.cif"))
    elif key == "external":
        d = os.path.join(raw, "ext", name)
        os.makedirs(d)
        for i, frame in enumerate(_pred()):
            frame.save_pdb(os.path.join(d, f"s{i}.pdb"))
    else:
        raise KeyError(key)


EXPECTED_FRAMES = {"biomd": 15}  # 3 replicas x (6 - conditioning frame f0)


@pytest.mark.parametrize("key", list(REGISTRY))
def test_collect_and_evaluate(atlas, tmp_path, key):
    name = atlas["name"]
    opts = {}
    if key == "external":
        opts = {"patterns": [str(tmp_path / key / "raw" / "ext" / "{name}" / "*.pdb")]}
    if key == "biomd":
        opts = {"n_frames": 300}
    ctx = _ctx(atlas, tmp_path, key, **opts)
    baseline = get_baseline(key)
    write_mock_outputs(key, ctx, name)

    rep = baseline.run_collect(ctx)
    assert rep[name] == EXPECTED_FRAMES.get(key, N), rep

    ens = mdtraj.load(os.path.join(ctx.ensemble_dir, f"{name}.pdb"))
    ref = mdtraj.load(ctx.ref_pdb(name))
    ref_res = [(r.name, r.resSeq) for r in ref.top.residues]
    assert [(r.name, r.resSeq) for r in ens.top.residues] == ref_res
    assert all(a.element.symbol != "H" for a in ens.top.atoms)

    results, failures = analyze.analyze_directory(ctx.atlas_dir, ctx.ensemble_dir, [name],
                                                  out_path=os.path.join(ctx.run_dir, "out.pkl"))
    assert not failures
    out = results[name]
    assert out["has_sidechains"] == baseline.sidechains
    # every predicted atom was matched to the reference
    assert out["n_aligned_atoms"] == ens.n_atoms

    s = report.summarize(results)
    for m in ("RMWD", "RMWD trans", "RMWD var", "MD PCA W2", "Joint PCA W2", "Weak contacts J", "Per target RMSF r"):
        assert np.isfinite(s[m]), m
    if key == "str2str":  # N, CA, C, O only: no side-chain SASA at all
        assert np.isnan(s["Exposed residue J"])
    if key == "bioemu":
        assert out["has_cb"]


def test_table_output(atlas, tmp_path):
    ctx = _ctx(atlas, tmp_path, "esmflow_md_full")
    write_mock_outputs("esmflow_md_full", ctx, atlas["name"])
    get_baseline("esmflow_md_full").run_collect(ctx)
    pkl = os.path.join(ctx.run_dir, "out.pkl")
    analyze.analyze_directory(ctx.atlas_dir, ctx.ensemble_dir, out_path=pkl)
    summaries = report.main([f"esmflow_md_full={pkl}", "--format", "markdown"])
    df = report.table1_frame(summaries)
    assert list(df.columns) == ["ESMFlow-MD (Full)", "ESMFlow-MD (Full) [paper]"]
    assert df.loc[("Distributional accuracy", "Root mean W2-dist."), "ESMFlow-MD (Full) [paper]"] == 3.60
    assert len(df) == 13


def test_perfect_ensemble_scores(atlas, tmp_path):
    """Sampling the MD itself should score near-perfectly on the Table-1 metrics."""
    name = atlas["name"]
    ref_aa, traj_aa = analyze.load_md_reference(atlas["atlas_dir"], name)
    idx = np.random.RandomState(0).randint(0, traj_aa.n_frames, 250)
    out = analyze.analyze_trajectories(ref_aa, traj_aa, traj_aa[idx])
    s = report.summarize({name: out})
    assert s["Per target RMSF r"] > 0.95
    assert s["RMWD"] < 0.2
    assert s["PC sim > 0.5 %"] == 100
