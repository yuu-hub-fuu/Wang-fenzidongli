"""Runners for the baselines of Table 1 (ATLAS, monomer ensembles)."""
from collections import OrderedDict

from .base import Baseline, Command, RunContext
from .bioemu import BioEmu
from .biomd import BioMD
from .confdiff import ConfDiff
from .eba import EBA
from .esmflow import ESMFlowMDDistilled, ESMFlowMDFull
from .external import External
from .mdgen import MDGen
from .str2str import Str2Str

# Table 1 column order.
BASELINES = OrderedDict(
    (cls.key, cls)
    for cls in (ESMFlowMDFull, ESMFlowMDDistilled, ConfDiff, BioEmu, Str2Str, MDGen, EBA, BioMD)
)
REGISTRY = OrderedDict(BASELINES)
REGISTRY[External.key] = External


def get_baseline(key: str) -> Baseline:
    try:
        return REGISTRY[key]()
    except KeyError:
        raise KeyError(f"unknown baseline {key!r}; choose from {list(REGISTRY)}") from None


__all__ = ["BASELINES", "REGISTRY", "Baseline", "Command", "RunContext", "get_baseline"]
