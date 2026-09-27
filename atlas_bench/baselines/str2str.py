"""Str2Str — Lu et al., ICLR 2024 (lujiarui/Str2Str).

Zero-shot perturb-and-denoise sampler trained on PDB only. It needs an input
structure; we use the ATLAS starting structure (heavy atoms), as done for the
ATLAS comparisons in ConfDiff/EBA. The default sampler sweeps 10 forward
diffusion depths (``delta`` = 0.25..0.70 step 0.05) with ``n_replica`` samples
each; ``n_replica = 25`` gives the 250 samples of the ATLAS protocol, merged by
Str2Str into ``{output_dir}/all_delta/{name}.pdb``. Backbone-only output.
"""
from __future__ import annotations

import os
from typing import Dict, List

from .. import atlas_data
from .base import Baseline, Command, RunContext, q

N_DELTAS = 10  # len(arange(0.25, 0.70 + 1e-5, 0.05))


class Str2Str(Baseline):
    key = "str2str"
    display_name = "Str2Str"
    citation = "Lu et al. Str2Str: A Score-based Framework for Zero-shot Protein Conformation Sampling. ICLR 2024."
    repo_url = "https://github.com/lujiarui/Str2Str"
    repo_commit = "0b690e990e2a95c73766db9e053a0e0ae0b8b181"
    setup_script = "str2str.sh"
    sidechains = False
    protocol = "src/eval.py task_name=inference, input = ATLAS start structure, 10 deltas x 25 replicas = 250 samples"

    def prepare(self, ctx: RunContext):
        atlas_data.export_start_structures(ctx.atlas_dir, ctx.targets, os.path.join(ctx.inputs_dir, "pdb"))

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        n_replica = int(ctx.opt("n_replica", max(1, ctx.n_samples // N_DELTAS)))
        ckpt = ctx.opt("ckpt", ctx.repo("data", "ckpt", "pretrain.pth"))
        cmds = []
        names = [t.name for t in ctx.targets]
        for gpu, shard in zip(ctx.gpus, self.shard(names, len(ctx.gpus))):
            overrides = [
                "task_name=inference",
                "target_dir=null",  # sampling only; evaluation is done by atlas_bench
                f"ckpt_path={ckpt}",
                f"model.inference.n_replica={n_replica}",
                f"model.inference.replica_per_batch={ctx.opt('replica_per_batch', 64)}",
                f"model.inference.output_dir={ctx.raw_dir}",
                f"hydra.run.dir={os.path.join(ctx.run_dir, 'hydra', 'gpu' + str(gpu))}",
                "data.dataset.accession_code_fillter=[" + ",".join(shard) + "]",
            ] + list(ctx.opt("extra_overrides", []))
            env = {
                "PROJECT_ROOT": ctx.repo_dir,
                "TEST_DATA": os.path.join(ctx.inputs_dir, "pdb"),
                "CACHE_DIR": ctx.opt("cache_dir", os.path.join(ctx.run_dir, "cache")),
                "TRAIN_DATA": ctx.opt("train_data", os.path.join(ctx.inputs_dir, "pdb")),
                "REFERENCE_DATA": "",
            }
            cmds.append(Command(
                f"{ctx.python} src/eval.py " + " ".join(q(o) for o in overrides),
                cwd=ctx.repo_dir, env=env, gpu=gpu, log=os.path.join(ctx.logs_dir, f"infer_gpu{gpu}.log"),
            ))
        return cmds

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        p = os.path.join(ctx.raw_dir, "all_delta", f"{name}.pdb")
        return {"files": [p] if os.path.exists(p) else []}
