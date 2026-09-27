"""The inference commands encode each baseline's published ATLAS protocol."""
import os

import pytest

from atlas_bench import atlas_data, config
from atlas_bench.__main__ import main as cli
from atlas_bench.baselines import BASELINES, RunContext, get_baseline

TARGETS = atlas_data.read_split(names=["6o2v_A", "7ead_A", "6uof_A", "7aex_A"])


def ctx_for(key, tmp_path, **options):
    return RunContext(atlas_dir="/atlas", run_dir=str(tmp_path / key), targets=TARGETS, repo_dir="/repo",
                      python="PY", gpus=["0", "1"], options=options)


def rendered(key, tmp_path, **options):
    b = get_baseline(key)
    return [[c.render() for c in stage] for stage in b.inference_stages(ctx_for(key, tmp_path, **options))]


def flat(stages):
    return "\n".join(c for s in stages for c in s)


def test_split_is_alphaflow_atlas_test():
    targets = atlas_data.read_split()
    assert len(targets) == 82
    assert targets[0].name == "6o2v_A"
    assert all(set(t.seqres) <= set("ACDEFGHIKLMNPQRSTVWYX") for t in targets)


def test_esmflow(tmp_path):
    full = flat(rendered("esmflow_md_full", tmp_path))
    dist = flat(rendered("esmflow_md_distilled", tmp_path))
    assert "--mode esmfold" in full and "--samples 250" in full
    assert "esmflow_md_base_202402.pt" in full and "--noisy_first" not in full
    assert "esmflow_md_distilled_202402.pt" in dist and "--noisy_first --no_diffusion" in dist
    # targets are sharded over the two GPUs
    assert "CUDA_VISIBLE_DEVICES=0" in full and "CUDA_VISIBLE_DEVICES=1" in full


def test_confdiff_stages(tmp_path):
    stages = rendered("confdiff", tmp_path, ckpt="/ck/OF-r3.ckpt")
    assert len(stages) == 4  # msa -> repr workers -> index bridge -> sampling
    assert "mmseqs_query_colabfold" in stages[0][0]
    assert len(stages[1]) == 2 and "--num-recycles 3" in stages[1][0]
    assert "confdiff_repr_index.py" in stages[2][0]
    s = stages[3][0]
    for part in ("experiment=full_atom", "data/dataset=atlas", "data/repr_loader=openfold",
                 "data.repr_loader.num_recycles=3", "test_gen_dataset.num_samples=250", "ckpt_path=/ck/OF-r3.ckpt"):
        assert part in s


def test_bioemu(tmp_path):
    s = flat(rendered("bioemu", tmp_path))
    assert s.count("bioemu.sample") == 4 and "bioemu-v1.1" in s and "--num_samples $n" in s
    s = flat(rendered("bioemu", tmp_path, filter_samples=False))
    assert "--num_samples 250" in s and "--filter_samples False" in s


def test_str2str_uses_250_samples(tmp_path):
    s = flat(rendered("str2str", tmp_path))
    assert "model.inference.n_replica=25" in s  # x 10 deltas
    assert "target_dir=null" in s


def test_mdgen(tmp_path):
    stages = rendered("mdgen", tmp_path)
    assert "mdgen_prep_atlas.py" in stages[0][0]
    s = flat(stages[1:])
    assert "--num_frames 250" in s and "--num_rollouts 1" in s and "--suffix _R1" in s


def test_eba(tmp_path):
    stages = rendered("eba", tmp_path)
    s = flat(stages)
    for part in ("--sample_diffusion.N_sample 250", "--sample_diffusion.N_step 20", "noise_scale_lambda 1.75",
                 "step_scale_eta 1.25", "--model.N_cycle 4", "--predict_only true", "--data.test_sets atlas_test"):
        assert part in s
    assert "ATLAS_DATA_ROOT_DIR" in stages[1][0]


def test_biomd(tmp_path):
    s = flat(rendered("biomd", tmp_path))
    assert "run_reproduce.sh" in s and "--stage inference" in s and "--gpus '0 1'" in s
    assert get_baseline("biomd").n_frames(ctx_for("biomd", tmp_path)) == 300


def test_biomd_config_excludes_7aex(tmp_path):
    bench = config.load_benchmark_config()
    bcfg = config.load_baseline_config("biomd", bench)
    ctx = config.make_context("biomd", bench, bcfg)
    names = [t.name for t in ctx.targets]
    assert len(names) == 81 and "7aex_A" not in names


@pytest.mark.parametrize("key", list(BASELINES))
def test_cli_dry_run(key, capsys):
    assert cli(["run", key, "--dry_run", "--targets", "6o2v_A"]) == 0
    out = capsys.readouterr().out
    assert "6o2v_A" in out or key in ("confdiff", "eba", "biomd")


def test_all_configs_parse():
    bench = config.load_benchmark_config()
    assert bench["n_samples"] == 250 and bench["baselines"] == list(BASELINES)
    for key in BASELINES:
        bcfg = config.load_baseline_config(key, bench)
        assert "python" in bcfg
        if bcfg.get("repo_dir"):
            assert os.path.isabs(bcfg["repo_dir"])
