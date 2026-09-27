"""Common machinery for running a Table-1 baseline end to end.

A baseline goes through four stages, each idempotent:

``prepare``  write model inputs derived from the ATLAS test set,
``infer``    run the *official* inference code in the baseline's own env,
``collect``  convert raw outputs into ``{run_dir}/ensembles/{name}.pdb``,
``evaluate`` compute the AlphaFlow/BioMD metrics (``out.pkl``).

``fetch`` is an alternative to ``infer`` for baselines whose authors released
the exact ATLAS ensembles behind their paper numbers (ESMFlow-MD, EBA).
"""
from __future__ import annotations

import datetime
import json
import os
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .. import atlas_data, ensemble_io
from ..atlas_data import Target

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")


class Raw(str):
    """Env value inserted verbatim (double-quoted) so shell substitutions expand."""


def _env_assign(k: str, v) -> str:
    return f'{k}="{v}"' if isinstance(v, Raw) else f"{k}={shlex.quote(str(v))}"


@dataclass
class Command:
    """A shell command executed with ``bash -c`` inside ``cwd``."""

    cmd: str
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    gpu: Optional[str] = None  # sets CUDA_VISIBLE_DEVICES
    log: Optional[str] = None
    allow_fail: bool = False

    def render(self) -> str:
        assigns = []
        if self.gpu is not None:
            assigns.append(_env_assign("CUDA_VISIBLE_DEVICES", self.gpu))
        assigns += [_env_assign(k, v) for k, v in self.env.items()]
        parts = []
        if self.cwd:
            parts.append(f"cd {shlex.quote(self.cwd)}")
        if assigns:
            parts.append("export " + " ".join(assigns))
        if not parts:
            return self.cmd
        # subshell: exports and cd reach every part of compound commands, but not the caller
        return "( " + " && ".join(parts) + f" && {self.cmd} )"


@dataclass
class RunContext:
    atlas_dir: str
    run_dir: str  # everything for this baseline lives here
    targets: List[Target]
    repo_dir: Optional[str] = None  # checkout of the official repository
    python: str = "python"  # interpreter of the baseline env (may be "conda run -n X python")
    gpus: Sequence[str] = ("0",)
    n_samples: int = 250
    seed: int = 42
    dry_run: bool = False
    options: Dict = field(default_factory=dict)

    @property
    def inputs_dir(self) -> str:
        return os.path.join(self.run_dir, "inputs")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.run_dir, "raw")

    @property
    def ensemble_dir(self) -> str:
        return os.path.join(self.run_dir, "ensembles")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.run_dir, "logs")

    def opt(self, key, default=None):
        return self.options.get(key, default)

    def repo(self, *parts) -> str:
        if not self.repo_dir:
            raise ValueError("repo_dir is not configured for this baseline")
        return os.path.join(self.repo_dir, *parts)

    def ref_pdb(self, name: str) -> str:
        return atlas_data.target_paths(self.atlas_dir, name)["pdb"]


class Baseline:
    """Base class; subclasses fill in the class attributes and stage hooks."""

    key: str = ""
    display_name: str = ""
    citation: str = ""
    repo_url: str = ""
    repo_commit: str = ""  # commit the harness was written against
    setup_script: str = ""  # scripts/envs/<file>
    sidechains: bool = True  # whether raw outputs contain side-chain atoms
    default_sidechain_packer: str = "none"  # "faspr": pack backbone-only outputs during collect
    protocol: str = ""  # one-line description of the sampling protocol
    default_n_frames: Optional[int] = None  # None -> ctx.n_samples (250, AlphaFlow protocol)

    # ------------------------------------------------------------------ hooks
    def prepare(self, ctx: RunContext) -> None:
        """Write inputs into ``ctx.inputs_dir``."""

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        raise NotImplementedError

    def inference_stages(self, ctx: RunContext) -> List[List[Command]]:
        """Sequential stages; commands inside a stage run in parallel per-GPU lanes."""
        return [self.inference_commands(ctx)]

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        """Return ``{"files": [...], "top": optional topology}`` for one target."""
        raise NotImplementedError

    def fetch_commands(self, ctx: RunContext) -> List[Command]:
        raise NotImplementedError(f"{self.display_name} has no released ATLAS ensembles; use infer")

    def fetched_files(self, ctx: RunContext, name: str) -> Dict:
        return self.raw_files(ctx, name)

    def collect_kwargs(self, ctx: RunContext) -> Dict:
        return {}

    # ------------------------------------------------------------------ helpers
    def sidechain_packer(self, ctx: RunContext) -> str:
        return str(ctx.opt("sidechains", self.default_sidechain_packer))

    def expects_sidechains(self, ctx: RunContext) -> bool:
        return self.sidechains or self.sidechain_packer(ctx) in ("faspr", "hpacker")

    def n_frames(self, ctx: RunContext) -> Optional[int]:
        return ctx.opt("n_frames", self.default_n_frames or ctx.n_samples)

    @staticmethod
    def shard(items: Sequence, n: int) -> List[List]:
        n = max(1, n)
        return [list(items[i::n]) for i in range(n) if items[i::n]]

    # ------------------------------------------------------------------ stages
    def run_prepare(self, ctx: RunContext):
        os.makedirs(ctx.inputs_dir, exist_ok=True)
        self.prepare(ctx)

    def run_commands(self, ctx: RunContext, commands: List[Command]):
        """Run commands; commands with distinct ``gpu`` values run in parallel lanes."""
        os.makedirs(ctx.logs_dir, exist_ok=True)
        lanes: Dict[Optional[str], List[Command]] = {}
        for c in commands:
            lanes.setdefault(c.gpu, []).append(c)
        if ctx.dry_run:
            for c in commands:
                print(c.render())
            return []
        errors: List[str] = []

        def run_lane(cmds: List[Command]):
            for c in cmds:
                log_path = c.log or os.path.join(ctx.logs_dir, f"{self.key}.log")
                with open(log_path, "a") as log:
                    log.write(f"\n$ {c.render()}\n")
                    log.flush()
                    rc = subprocess.run(["bash", "-c", c.render()], stdout=log, stderr=subprocess.STDOUT).returncode
                if rc != 0 and not c.allow_fail:
                    errors.append(f"exit {rc}: {c.render()} (log: {log_path})")

        threads = [threading.Thread(target=run_lane, args=(cmds,)) for cmds in lanes.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise RuntimeError("\n".join(errors))
        return commands

    def run_infer(self, ctx: RunContext):
        os.makedirs(ctx.raw_dir, exist_ok=True)
        done = []
        for stage in self.inference_stages(ctx):
            done += self.run_commands(ctx, stage) or []
        return done

    def run_fetch(self, ctx: RunContext):
        os.makedirs(ctx.raw_dir, exist_ok=True)
        return self.run_commands(ctx, self.fetch_commands(ctx))

    def run_collect(self, ctx: RunContext, fetched: bool = False) -> Dict[str, object]:
        os.makedirs(ctx.ensemble_dir, exist_ok=True)
        report: Dict[str, object] = {}
        for t in ctx.targets:
            try:
                spec = (self.fetched_files if fetched else self.raw_files)(ctx, t.name)
                files = spec.get("files", [])
                if not files:
                    report[t.name] = "missing"
                    continue
                kwargs = dict(self.collect_kwargs(ctx))
                kwargs.update(spec.get("kwargs", {}))
                out_pdb = os.path.join(ctx.ensemble_dir, f"{t.name}.pdb")
                faspr = self.sidechain_packer(ctx) == "faspr"
                merged = os.path.join(ctx.run_dir, "ensembles_backbone", f"{t.name}.pdb") if faspr else out_pdb
                _, n = ensemble_io.build_ensemble(
                    files,
                    merged,
                    ref_pdb=ctx.ref_pdb(t.name),
                    top_path=spec.get("top"),
                    n_frames=self.n_frames(ctx),
                    **kwargs,
                )
                if faspr:
                    from .. import sidechain_pack

                    n = sidechain_pack.faspr_pack_ensemble(
                        merged, out_pdb, ctx.opt("faspr_bin"), int(ctx.opt("faspr_workers", 8))
                    )
                report[t.name] = n
            except Exception as e:  # keep collecting other targets
                report[t.name] = f"error: {type(e).__name__}: {e}"
        self.write_manifest(ctx, {"collect": report, "fetched": fetched})
        return report

    def write_manifest(self, ctx: RunContext, extra: Dict):
        path = os.path.join(ctx.run_dir, "manifest.json")
        manifest = {}
        if os.path.exists(path):
            with open(path) as f:
                manifest = json.load(f)
        manifest.update(
            {
                "baseline": self.key,
                "display_name": self.display_name,
                "repo_url": self.repo_url,
                "repo_commit_reference": self.repo_commit,
                "repo_commit_used": _git_head(ctx.repo_dir),
                "protocol": self.protocol,
                "sidechain_packer": self.sidechain_packer(ctx),
                "n_samples": ctx.n_samples,
                "seed": ctx.seed,
                "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        )
        manifest.update(extra)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)


def _git_head(repo_dir: Optional[str]) -> Optional[str]:
    if not repo_dir or not os.path.isdir(os.path.join(repo_dir, ".git")):
        return None
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def q(x) -> str:
    return shlex.quote(str(x))
