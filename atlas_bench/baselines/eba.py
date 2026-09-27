"""EBA (Energy-based Alignment) — Lu et al., ICML 2025 (lujiarui/eba).

Protenix (AF3-style) model fine-tuned on ATLAS MD then aligned with energy
feedback. Protocol of ``sh/sample_demo.sh``: 250 samples / target
(``N_sample 250``, ``N_step 20``, ``noise_scale_lambda 1.75``,
``step_scale_eta 1.25``, ``N_cycle 4``, seed 42) predicted with
``runner/train.py --predict_only true``. The authors also released these 250
ATLAS test samples (``artifacts.tar.gz``), which ``fetch`` downloads.

The Protenix data pipeline needs every test target as an mmCIF + bioassembly
pickle + index CSV under ``ATLAS_DATA_ROOT_DIR`` (EBA README, "Data
preparation"); only the sequence of the input structure is used by the model.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

from .. import atlas_data
from .base import Baseline, Command, RunContext, q

GCS = "https://storage.googleapis.com/project_icml25_eba"


class EBA(Baseline):
    key = "eba"
    display_name = "EBA"
    citation = "Lu et al. Aligning Protein Conformation Ensemble Generation with Physical Feedback. ICML 2025."
    repo_url = "https://github.com/lujiarui/eba"
    repo_commit = "c1f314506f313c5b4b9913fc43f1071fe70d70ec"
    setup_script = "eba.sh"
    sidechains = True
    protocol = "runner/train.py --predict_only, release.pt (EMA), N_sample 250, N_step 20, lambda 1.75, eta 1.25, N_cycle 4, seed 42"

    def data_root(self, ctx: RunContext) -> str:
        return ctx.opt("atlas_data_root", os.path.join(ctx.inputs_dir, "eba_atlas"))

    def prepare(self, ctx: RunContext):
        # {name}_0.cif mirrors scripts/convert_xtc_to_cif.py naming ({pc_name}_{idx}.cif)
        root = self.data_root(ctx)
        paths = atlas_data.export_start_structures(ctx.atlas_dir, ctx.targets, os.path.join(root, "mmcif"), fmt="cif")
        for name, p in paths.items():
            os.replace(p, os.path.join(os.path.dirname(p), f"{name}_0.cif"))

    def inference_stages(self, ctx: RunContext) -> List[List[Command]]:
        root = self.data_root(ctx)
        ckpt = ctx.opt("ckpt", ctx.repo("release.pt"))
        protenix_root = ctx.opt("protenix_data_root", os.path.expanduser("~/scratch/data/pdb/protenix/"))
        env = {
            "LAYERNORM_TYPE": "fast_layernorm",
            "USE_DEEPSPEED_EVO_ATTTENTION": "true",
            "CUTLASS_PATH": ctx.opt("cutlass_path", ctx.repo("cutlass")),
        }
        index = Command(
            f"{ctx.python} scripts/prepare_training_data.py -i {q(root + '/mmcif')} -o {q(root + '/indices_test.csv')}"
            f" -b {q(root + '/mmcif_bioassembly')} -d Atlas -n {int(ctx.opt('num_workers', 8))}",
            cwd=ctx.repo_dir,
        )
        # configs/configs_data.py hard-codes the two data roots; point them at our paths.
        sed_data = f's#^DATA_ROOT_DIR = .*#DATA_ROOT_DIR = "{protenix_root}"#'
        sed_atlas = f's#^ATLAS_DATA_ROOT_DIR = .*#ATLAS_DATA_ROOT_DIR = "{root}"#'
        patch = Command(f"sed -i -e {q(sed_data)} -e {q(sed_atlas)} configs/configs_data.py", cwd=ctx.repo_dir)
        n = len(ctx.gpus)
        entry = f"{ctx.python} -m torch.distributed.run --nproc_per_node={n}" if n > 1 else ctx.python
        args = [
            entry, "./runner/train.py",
            "--run_name", q(ctx.opt("run_name", "eba_atlas_test")),
            "--seed", str(ctx.opt("seed", 42)),
            "--base_dir", q(ctx.raw_dir),
            "--dtype bf16 --project protenix --use_wandb false",
            "--diffusion_batch_size 8 --eval_interval 400 --log_interval 50 --ema_decay 0.999 --max_steps 100000",
            "--sample_diffusion.noise_scale_lambda", str(ctx.opt("noise_scale_lambda", 1.75)),
            "--sample_diffusion.step_scale_eta", str(ctx.opt("step_scale_eta", 1.25)),
            "--sample_diffusion.N_step", str(ctx.opt("N_step", 20)),
            "--sample_diffusion.N_sample", str(ctx.n_samples),
            "--load_checkpoint_path", q(ckpt),
            "--load_ema_checkpoint_path", q(ckpt),
            "--model.N_cycle", str(ctx.opt("N_cycle", 4)),
            "--data.train_sets atlas_test --data.test_sets atlas_test",
            "--predict_only true",
        ]
        run = Command(" ".join(args), cwd=ctx.repo_dir, env=env, gpu=",".join(ctx.gpus))
        return [[index], [patch], [run]]

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        return [c for stage in self.inference_stages(ctx) for c in stage]

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        return {"files": _find_cifs(ctx.raw_dir, name)}

    # released artifacts ---------------------------------------------------
    def fetch_commands(self, ctx: RunContext) -> List[Command]:
        dest = os.path.join(ctx.raw_dir, "official")
        return [
            Command(
                f"mkdir -p {q(dest)} && curl -fL --retry 3 -o {q(dest + '/artifacts.tar.gz')} {q(GCS + '/artifacts.tar.gz')}"
                f" && tar -xzf {q(dest + '/artifacts.tar.gz')} -C {q(dest)} && rm {q(dest + '/artifacts.tar.gz')}"
            )
        ]

    def fetched_files(self, ctx: RunContext, name: str) -> Dict:
        return {"files": _find_cifs(os.path.join(ctx.raw_dir, "official"), name)}


def _find_cifs(root: str, name: str) -> List[str]:
    """Protenix dumper layout: .../{pdb_id}/seed_{s}/predictions/{pdb_id}_seed_{s}_sample_{k}.cif."""
    hits = []
    for pattern in (f"{name}", f"{name}_0", f"{name}_*"):
        hits = glob.glob(os.path.join(root, "**", pattern, "seed_*", "predictions", "*.cif"), recursive=True)
        hits = [h for h in hits if "_summary_" not in os.path.basename(h)]
        if hits:
            break
    return hits
