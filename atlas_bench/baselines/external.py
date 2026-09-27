"""Evaluate any pre-generated ensembles (e.g. AnewSampling) with the same pipeline.

Options (``configs/baselines/external.yaml``)::

    patterns: ["/path/to/samples/{name}/*.pdb"]   # globs, {name} = ATLAS target
    top: null                                      # topology for .xtc inputs, may use {name}
    n_frames: 250
"""
from __future__ import annotations

from typing import Dict, List

from .. import ensemble_io
from .base import Baseline, Command, RunContext


class External(Baseline):
    key = "external"
    display_name = "External ensembles"
    protocol = "user-provided samples, merged per target"

    def inference_commands(self, ctx: RunContext) -> List[Command]:
        return []

    def raw_files(self, ctx: RunContext, name: str) -> Dict:
        patterns = ctx.opt("patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        files = ensemble_io.glob_files([p.format(name=name) for p in patterns])
        top = ctx.opt("top")
        return {"files": files, "top": top.format(name=name) if top else None}
