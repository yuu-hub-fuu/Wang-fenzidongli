"""ESMFlow-MD (Full / Distilled) — Jing, Berger & Jaakkola, ICML 2024 (AlphaFlow repo).

Protocol (AlphaFlow README, "Evaluation scripts"): 250 samples per ATLAS test
target from sequence only, default schedule ``--tmax 1.0 --steps 10``;
distilled weights additionally use ``--noisy_first --no_diffusion``.
The authors released the exact ensembles used in their paper, which
``fetch`` downloads.
"""
from __future__ import annotations

import os
from typing import Dict, List

from .. import atlas_data
from .base import Baseline, Command, RunContext, q

HF = "https://huggingface.co/bjing-mit/alphaflow/resolve/main"


class _ESMFlowMD(Baseline):
    repo_url = "https://github.com/bjing2016/alphaflow"
    repo_commit = "0408d7c89dac444a43a9089d7427ce470b0a5e67"
    setup_script = "alphaflow.sh"
    citation = "Jing, Berger, Jaakkola. AlphaFold Meets Flow Matching for Generating Protein Ensembles. ICML 2024."
    sidechains = True
    weights_file = ""
    samples_file = ""
    extra_args: List[str] = []

    @property
    def protocol(self) -> str:  # type: ignore[override]
        extra = " ".join(self.extra_args)
        return f"predict.py --mode esmfold --samples 250 {extra}".strip()

    def prepare(self, ctx: RunContext):
        atlas_data.write_csv(ctx.targets, os.path.join(ctx.inputs_dir, "atlas_test.csv"))

    def weights(self, ctx: RunContext) -> str:
        return ctx.opt("weights", ctx.repo("params", self.weights_file))

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        csv = os.path.join(ctx.inputs_dir, "atlas_test.csv")
        cmds = []
        names = [t.name for t in ctx.targets]
        for gpu, shard in zip(ctx.gpus, self.shard(names, len(ctx.gpus))):
            args = [
                ctx.python, "predict.py", "--mode esmfold",
                "--input_csv", q(csv),
                "--weights", q(self.weights(ctx)),
                "--samples", str(ctx.n_samples),
                "--steps", str(ctx.opt("steps", 10)),
                "--tmax", str(ctx.opt("tmax", 1.0)),
                "--outpdb", q(ctx.raw_dir),
                "--no_overwrite",
                *self.extra_args,
                "--pdb_id", *shard,
            ]
            cmds.append(Command(" ".join(args), cwd=ctx.repo_dir, gpu=gpu,
                                log=os.path.join(ctx.logs_dir, f"infer_gpu{gpu}.log")))
        return cmds

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        p = os.path.join(ctx.raw_dir, f"{name}.pdb")
        return {"files": [p] if os.path.exists(p) else []}

    # released ensembles --------------------------------------------------
    def fetch_commands(self, ctx: RunContext) -> List[Command]:
        url = f"{HF}/samples/{self.samples_file}"
        dest = os.path.join(ctx.raw_dir, "official")
        return [
            Command(
                f"mkdir -p {q(dest)} && curl -fL --retry 3 -o {q(dest + '/samples.zip')} {q(url)} "
                f"&& unzip -o -q {q(dest + '/samples.zip')} -d {q(dest)} && rm {q(dest + '/samples.zip')}"
            )
        ]

    def fetched_files(self, ctx: RunContext, name: str) -> Dict:
        root = os.path.join(ctx.raw_dir, "official")
        for dirpath, _, files in os.walk(root):
            if f"{name}.pdb" in files:
                return {"files": [os.path.join(dirpath, f"{name}.pdb")]}
        return {"files": []}


class ESMFlowMDFull(_ESMFlowMD):
    key = "esmflow_md_full"
    display_name = "ESMFlow-MD (Full)"
    weights_file = "esmflow_md_base_202402.pt"
    samples_file = "esmflow_md_base_202402.zip"
    extra_args: List[str] = []


class ESMFlowMDDistilled(_ESMFlowMD):
    key = "esmflow_md_distilled"
    display_name = "ESMFlow-MD (Distilled)"
    weights_file = "esmflow_md_distilled_202402.pt"
    samples_file = "esmflow_md_distilled_202402.zip"
    extra_args = ["--noisy_first", "--no_diffusion"]
