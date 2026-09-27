"""ConfDiff (ConfDiff-OF-r3-MD) — Wang et al., ICML 2024 (bytedance/ConfDiff).

Table 1's "ConfDiff" column equals the ConfDiff README row
``ConfDiff-OF-r3-MD`` (Pairwise RMSD r 0.59, RMWD 2.76, ...): the full-atom
model conditioned on OpenFold representations with 3 recycles, fine-tuned on
ATLAS MD. Protocol: 250 samples / target (``configs/data/dataset/atlas.yaml``).

Pipeline (official scripts):
1. ColabFold MSAs  ``pretrain_repr.openfold.mmseqs_query_colabfold``
2. OpenFold node/edge repr  ``pretrain_repr.openfold.make_openfold_repr --num-recycles 3``
3. Sampling  ``src/eval.py experiment=full_atom data/dataset=atlas data/repr_loader=openfold``
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

from .. import atlas_data
from .base import TOOLS_DIR, Baseline, Command, RunContext, q


class ConfDiff(Baseline):
    key = "confdiff"
    display_name = "ConfDiff"
    citation = "Wang et al. Protein Conformation Generation via Force-Guided SE(3) Diffusion Models. ICML 2024."
    repo_url = "https://github.com/bytedance/ConfDiff"
    repo_commit = "9cfae1c14121e423d8d455d03506c7e8ee580e48"
    setup_script = "confdiff.sh"
    sidechains = True
    protocol = "ConfDiff-OF-r3-MD, OpenFold repr (3 recycles), 250 samples/target, gen_batch_size 20"

    def metadata_csv(self, ctx: RunContext) -> str:
        return os.path.join(ctx.inputs_dir, "atlas_test_confdiff.csv")

    def prepare(self, ctx: RunContext):
        atlas_data.write_csv(
            ctx.targets, self.metadata_csv(ctx), name_col="chain_name", extra={"seqlen": lambda t: len(t.seqres)}
        )

    def inference_stages(self, ctx: RunContext) -> List[List[Command]]:
        meta = self.metadata_csv(ctx)
        msa_dir = ctx.opt("msa_dir", os.path.join(ctx.run_dir, "msa"))
        repr_dir = ctx.opt("openfold_repr_dir", os.path.join(ctx.run_dir, "openfold_repr"))
        n_rec = int(ctx.opt("num_recycles", 3))
        of_ckpt = ctx.opt("openfold_ckpt", ctx.repo("pretrain_repr", "openfold", "openfold_params", "finetuning_no_templ_ptm_1.pt"))
        ckpt = ctx.opt("ckpt", None) or _find_ckpt(ctx.repo("checkpoints", "ConfDiff-MD"), "OF-r3")
        stages: List[List[Command]] = []

        if not ctx.opt("skip_msa", False):
            stages.append([Command(
                f"{ctx.python} -m pretrain_repr.openfold.mmseqs_query_colabfold {q(meta)} --outdir {q(msa_dir)}",
                cwd=ctx.repo_dir,
            )])
        if not ctx.opt("skip_repr", False):
            n = len(ctx.gpus)
            stages.append([])
            for wid, gpu in enumerate(ctx.gpus):
                stages[-1].append(Command(
                    f"{ctx.python} -m pretrain_repr.openfold.make_openfold_repr"
                    f" --input-csv-path {q(meta)} --msa-dir {q(msa_dir)} --output-dir {q(repr_dir)}"
                    f" --openfold-ckpt-fpath {q(of_ckpt)} --num-recycles {n_rec}"
                    f" --num-workers {n} --worker-id {wid}",
                    cwd=ctx.repo_dir, gpu=gpu, log=os.path.join(ctx.logs_dir, f"repr_gpu{gpu}.log"),
                ))
            # Writer/loader layouts differ in the official code; bridge them.
            stages.append([Command(f"{ctx.python} {q(TOOLS_DIR + '/confdiff_repr_index.py')} {q(repr_dir)} {n_rec}")])
        overrides = [
            f"task_name=atlas_bench_{self.key}",
            "experiment=full_atom",
            "data/dataset=atlas",
            "data/repr_loader=openfold",
            f"data.repr_loader.num_recycles={n_rec}",
            f"paths.openfold_repr.data_root={repr_dir}",
            f"data.dataset.test_gen_dataset.csv_path={meta}",
            f"data.dataset.test_gen_dataset.num_samples={ctx.n_samples}",
            f"data.gen_batch_size={ctx.opt('gen_batch_size', 20)}",
            f"ckpt_path={ckpt}",
            f"paths.output_dir={ctx.raw_dir}",
            f"seed={ctx.seed}",
            f"trainer.devices={len(ctx.gpus)}",
        ]
        overrides += list(ctx.opt("extra_overrides", []))
        stages.append([Command(
            f"{ctx.python} src/eval.py " + " ".join(q(o) for o in overrides),
            cwd=ctx.repo_dir,
            env={"PROJECT_ROOT": ctx.repo_dir},
            gpu=",".join(ctx.gpus),
        )])
        return stages

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        return [c for stage in self.inference_stages(ctx) for c in stage]

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        # module.py: {output_dir}/test_gen/{chain_name}/{chain_name}_sample{i}.pdb
        files = glob.glob(os.path.join(ctx.raw_dir, "**", name, f"{name}_sample*.pdb"), recursive=True)
        return {"files": files}


def _find_ckpt(root: str, tag: str) -> str:
    hits = sorted(glob.glob(os.path.join(root, "**", "*.ckpt"), recursive=True))
    hits = [h for h in hits if tag.lower() in h.lower()] or hits
    return hits[0] if hits else os.path.join(root, f"ConfDiff-{tag}-MD.ckpt")

