"""BioEmu — Lewis et al., Science 2025 (microsoft/bioemu, ``pip install bioemu``).

Sequence-only sampler of backbone frames (written as N, CA, C, CB, O). Protocol here:
``python -m bioemu.sample --sequence SEQ --num_samples M --model_name bioemu-v1.1``
per target; BioEmu filters unphysical samples by default, so we request an
over-sampled ``M`` and keep the first 250 accepted frames (set
``filter_samples: false`` to keep all raw samples instead).

Side chains beyond CB are absent unless ``sidechains: hpacker`` runs the
official ``bioemu.sidechain_relax --no-md-equil`` reconstruction (or
``sidechains: faspr`` packs them during collect); with ``none`` the SASA
metrics are computed at CB level.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List

from .. import atlas_data
from .base import Baseline, Command, RunContext, q


class BioEmu(Baseline):
    key = "bioemu"
    display_name = "BioEmu"
    citation = "Lewis et al. Scalable emulation of protein equilibrium ensembles with generative deep learning. Science 2025."
    repo_url = "https://github.com/microsoft/bioemu"
    repo_commit = "8babb71563e97dc1feb9e3e2fbc00a1224b6a3a7"
    setup_script = "bioemu.sh"
    sidechains = False
    protocol = "bioemu.sample (bioemu-v1.1, DPM denoiser), filtered, first 250 accepted samples/target"

    def prepare(self, ctx: RunContext):
        atlas_data.write_fasta(ctx.targets, os.path.join(ctx.inputs_dir, "fasta"))

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        filt = bool(ctx.opt("filter_samples", True))
        oversample = float(ctx.opt("oversample", 1.5)) if filt else 1.0
        n_req = int(math.ceil(ctx.n_samples * oversample))
        model = ctx.opt("model_name", "bioemu-v1.1")
        cmds = []
        names = [t.name for t in ctx.targets]
        for gpu, shard in zip(ctx.gpus, self.shard(names, len(ctx.gpus))):
            for name in shard:
                out = os.path.join(ctx.raw_dir, name)
                fasta = os.path.join(ctx.inputs_dir, "fasta", f"{name}.fasta")
                args = [
                    ctx.python, "-m bioemu.sample",
                    "--sequence", q(fasta),
                    "--num_samples", str(n_req),
                    "--output_dir", q(out),
                    "--model_name", q(model),
                    "--batch_size_100", str(ctx.opt("batch_size_100", 10)),
                    "--filter_samples", str(filt),
                    "--base_seed", str(ctx.seed),
                ]
                if ctx.opt("denoiser_config"):
                    args += ["--denoiser_config", q(ctx.opt("denoiser_config"))]
                if ctx.opt("cache_embeds_dir"):
                    args += ["--cache_embeds_dir", q(ctx.opt("cache_embeds_dir"))]
                cmd = " ".join(args)
                if filt:
                    # Top up until enough samples survive filtering (bioemu resumes from
                    # the .npz batches already in output_dir).
                    cmd = _topup_loop(cmd, ctx.python, out, ctx.n_samples, n_req, int(ctx.opt("max_topups", 4)))
                if ctx.opt("sidechains", "none") == "hpacker":
                    cmd += (
                        f" && {ctx.python} -m bioemu.sidechain_relax --pdb-path {q(out + '/topology.pdb')}"
                        f" --xtc-path {q(out + '/samples.xtc')} --no-md-equil"
                    )
                cmds.append(Command(cmd, gpu=gpu, log=os.path.join(ctx.logs_dir, f"infer_gpu{gpu}.log")))
        return cmds

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        out = os.path.join(ctx.raw_dir, name)
        stem = "samples_sidechain_rec" if ctx.opt("sidechains", "none") == "hpacker" else "samples"
        xtc = os.path.join(out, f"{stem}.xtc")
        top = os.path.join(out, f"{stem}.pdb") if stem != "samples" else os.path.join(out, "topology.pdb")
        if not (os.path.exists(xtc) and os.path.exists(top)):
            return {"files": []}
        return {"files": [xtc], "top": top}


def _topup_loop(cmd: str, python: str, out: str, n_needed: int, n_req: int, max_topups: int) -> str:
    """Shell loop: rerun bioemu.sample with more requested samples until enough pass the filter."""
    count = (
        f"$({python} -c \"import mdtraj,sys; "
        f"print(mdtraj.load(sys.argv[1], top=sys.argv[2]).n_frames)\" "
        f"{q(out + '/samples.xtc')} {q(out + '/topology.pdb')} 2>/dev/null || echo 0)"
    )
    return (
        f"n={n_req}; for i in $(seq 0 {max_topups}); do "
        f"{cmd.replace('--num_samples ' + str(n_req), '--num_samples $n')} || exit 1; "
        f"c={count}; [ \"$c\" -ge {n_needed} ] && break; "
        f"n=$(( n + ({n_needed} - c) * 2 )); done"
    )
