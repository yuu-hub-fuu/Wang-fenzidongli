"""MDGen — Jing, Stärk, Jaakkola & Berger, NeurIPS 2024 (bjing2016/mdgen).

Trajectory generator. ATLAS protocol from the MDGen README:
``sim_inference.py --sim_ckpt atlas.ckpt --num_frames 250 --num_rollouts 1
--suffix _R1`` i.e. one 250-frame (400 ps/frame, 100 ns) rollout conditioned on
frame 0 of replica R1; the 250 frames form the ensemble evaluated with the
AlphaFlow scripts. Output: ``{out_dir}/{name}.pdb`` (full heavy atoms from
frames + torsions).
"""
from __future__ import annotations

import os
from typing import Dict, List

from .. import atlas_data
from .base import TOOLS_DIR, Baseline, Command, RunContext, q


class MDGen(Baseline):
    key = "mdgen"
    display_name = "MDGen"
    citation = "Jing, Stärk, Jaakkola, Berger. Generative Modeling of Molecular Dynamics Trajectories. NeurIPS 2024."
    repo_url = "https://github.com/bjing2016/mdgen"
    repo_commit = "81482a403b91c5a8437046da1d8d321ba97089cc"
    setup_script = "mdgen.sh"
    sidechains = True
    protocol = "sim_inference.py atlas.ckpt, 1 rollout x 250 frames from R1 frame 0 (no --xtc)"

    def prepare(self, ctx: RunContext):
        atlas_data.write_csv(ctx.targets, os.path.join(ctx.inputs_dir, "atlas_test.csv"))

    def inference_stages(self, ctx: RunContext) -> List[List[Command]]:
        split = os.path.join(ctx.inputs_dir, "atlas_test.csv")
        data_dir = os.path.join(ctx.inputs_dir, "data_atlas")
        replica = int(ctx.opt("replica", 1))
        ckpt = ctx.opt("ckpt", ctx.repo("weights", "atlas.ckpt"))
        prep = Command(
            f"{ctx.python} {q(TOOLS_DIR + '/mdgen_prep_atlas.py')} --split {q(split)}"
            f" --sim_dir {q(ctx.atlas_dir)} --outdir {q(data_dir)} --replicas {replica}",
            cwd=ctx.repo_dir, env={"PYTHONPATH": ctx.repo_dir},
        )
        rollouts = []
        names = [t.name for t in ctx.targets]
        for gpu, shard in zip(ctx.gpus, self.shard(names, len(ctx.gpus))):
            args = [
                ctx.python, "sim_inference.py",
                "--sim_ckpt", q(ckpt),
                "--data_dir", q(data_dir),
                "--num_frames", str(ctx.opt("num_frames", ctx.n_samples)),
                "--num_rollouts", str(ctx.opt("num_rollouts", 1)),
                "--split", q(split),
                "--suffix", f"_R{replica}",
                "--out_dir", q(ctx.raw_dir),
                "--pdb_id", *shard,
            ]
            rollouts.append(Command(" ".join(args), cwd=ctx.repo_dir, env={"PYTHONPATH": ctx.repo_dir}, gpu=gpu,
                                    log=os.path.join(ctx.logs_dir, f"infer_gpu{gpu}.log")))
        return [[prep], rollouts]

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        return [c for stage in self.inference_stages(ctx) for c in stage]

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        p = os.path.join(ctx.raw_dir, f"{name}.pdb")
        return {"files": [p] if os.path.exists(p) else []}
