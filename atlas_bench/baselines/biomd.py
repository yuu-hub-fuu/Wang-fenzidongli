"""BioMD — Feng et al., ICLR 2026, via its public successor BioKinema.

BioMD ("All-atom generative model for biomolecular dynamics simulation") has no
public code or weights at the time of writing. The same authors released
BioKinema (IDEA-XL/BioKinema, bioRxiv 2026, reference [24] of the AnewSampling
paper): the same Protenix-based hierarchical forecasting + interpolation
trajectory generator, shipped with a complete reproduction package for this
exact ATLAS benchmark (``experiments/atlas_benchmark``). We therefore run that
package with the ``sqrt`` checkpoint as the closest public BioMD
implementation; set ``ckpt`` to a BioMD checkpoint if you have one.

Protocol (``run_reproduce.sh``): for each target, three 100 ns rollouts at
1 ns/frame from frame 0 of MD replicas R1/R2/R3 (``coarse_frame_num 101``,
``W_H 1``, ``W_G 100``, ``N_step 20``, ``N_cycle 10``, ``lambda 1.75``,
``eta 1.5``, ``beta 0.5``, seed 101, with MSA). The ensemble is the union of
generated frames minus the conditioning frame (``*f0.cif``), shuffled with
seed 66 and truncated to 300 frames. The package scores 81 targets
(``7aex_A`` excluded).
"""
from __future__ import annotations

import glob
import os
import subprocess
from typing import Dict, List

from .. import atlas_data
from .base import Baseline, Command, Raw, RunContext, q


class BioMD(Baseline):
    key = "biomd"
    display_name = "BioMD"
    citation = (
        "Feng et al. BioMD: All-atom Generative Model for Biomolecular Dynamics Simulation. ICLR 2026; "
        "Feng et al. Physically Grounded Generative Modeling of All-Atom Biomolecular Dynamics (BioKinema). bioRxiv 2026."
    )
    repo_url = "https://github.com/IDEA-XL/BioKinema"
    repo_commit = "e03508e341b71d1a5f0fa533e46665943fd21791"
    setup_script = "biomd_biokinema.sh"
    sidechains = True
    default_n_frames = 300
    protocol = "BioKinema atlas_benchmark/run_reproduce.sh: 3 x 100 ns @ 1 ns from R1-3 frame 0, 300 frames (seed 66)"

    def init_frames_dir(self, ctx: RunContext) -> str:
        return ctx.opt("init_frames_dir", os.path.join(ctx.inputs_dir, "init_frames"))

    def prepare(self, ctx: RunContext):
        out = self.init_frames_dir(ctx)
        if ctx.opt("init_frames", "atlas") == "shipped":
            # the package's own frame-0 CIFs (reproduces its expected_metrics.txt)
            pkg = ctx.repo("experiments", "atlas_benchmark")
            tmp = os.path.join(ctx.inputs_dir, "_shipped")
            os.makedirs(tmp, exist_ok=True)
            subprocess.run(["tar", "-xzf", os.path.join(pkg, "init_frames.tar.gz"), "-C", tmp], check=True)
            os.makedirs(out, exist_ok=True)
            keep = {t.name for t in ctx.targets}
            for f in glob.glob(os.path.join(tmp, "init_frames", "*_R[123]_0.cif")):
                if os.path.basename(f).rsplit("_R", 1)[0] in keep:
                    os.replace(f, os.path.join(out, os.path.basename(f)))
            return
        atlas_data.export_replica_first_frames(ctx.atlas_dir, ctx.targets, out, fmt="cif", stem="{name}_R{r}_0")

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        ckpt = ctx.opt("ckpt", ctx.repo("checkpoints", "BioKinema_atlas+misato+mdposit_sqrt.pt"))
        py = ctx.opt("python_path", ctx.python)
        if " " in py:  # e.g. "conda run -n biokinema python": resolve the real interpreter
            py = Raw(f"$({py} -c 'import sys; print(sys.executable)')")
        env = {"BIOKINEMA_PY": py}
        for k in ("CUTLASS_PATH", "CUDA_HOME"):
            if ctx.opt(k.lower()):
                env[k] = ctx.opt(k.lower())
        args = [
            "bash", "experiments/atlas_benchmark/run_reproduce.sh",
            "--checkpoint_path", q(ckpt),
            "--md_dir", q(ctx.atlas_dir),
            "--output_dir", q(ctx.raw_dir),
            "--init_frames_dir", q(self.init_frames_dir(ctx)),
            "--gpus", q(" ".join(ctx.gpus)),
            "--msa_cache_dir", q(ctx.opt("msa_cache_dir", os.path.join(ctx.run_dir, "msa"))),
            "--stage", "inference",
        ]
        return [Command(" ".join(args), cwd=ctx.repo_dir, env=env)]

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        files: List[str] = []
        for r in (1, 2, 3):
            base = os.path.join(ctx.raw_dir, f"{name}_R{r}_0")
            d = os.path.join(base, "seed_101", "predictions")
            if not os.path.isdir(d):
                d = os.path.join(base, "predictions")
            files += glob.glob(os.path.join(d, "*.cif"))
        files = [f for f in files if not f.endswith("f0.cif")]
        return {"files": files}

    def collect_kwargs(self, ctx: RunContext) -> Dict:
        return {"shuffle_files_seed": int(ctx.opt("shuffle_seed", 66)), "max_files": int(ctx.opt("max_files", 300))}
