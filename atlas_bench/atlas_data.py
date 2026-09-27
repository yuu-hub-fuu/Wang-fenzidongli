"""ATLAS test-set handling: split, download, verification and model inputs.

Layout expected by the evaluator (identical to AlphaFlow / MDGen / ConfDiff /
EBA / BioKinema)::

    {atlas_dir}/{name}/{name}.pdb                  # starting structure (reference)
    {atlas_dir}/{name}/{name}_prod_R{1,2,3}_fit.xtc  # 3 x 100 ns production replicas
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLIT = os.path.join(PACKAGE_DIR, "data", "splits", "atlas_test.csv")

# ATLAS moved from static zips to an API; try the API first, then the legacy URL
# used by AlphaFlow's scripts/download_atlas.sh.
ATLAS_URLS = (
    "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{name}",
    "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{name}/{name}_protein.zip",
)
# BioKinema (BioMD) evaluates 81 targets: 7aex_A is dropped from the 82-target split.
BIOMD_EXCLUDED_TARGETS = ("7aex_A",)


@dataclass
class Target:
    name: str
    seqres: str

    @property
    def pdb_id(self) -> str:
        return self.name.split("_")[0]

    @property
    def chain(self) -> str:
        return self.name.split("_")[1] if "_" in self.name else "A"


def read_split(path: str = DEFAULT_SPLIT, names: Optional[Sequence[str]] = None) -> List[Target]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    targets = [Target(r["name"], r["seqres"]) for r in rows]
    if names:
        keep = set(names)
        targets = [t for t in targets if t.name in keep]
        missing = keep - {t.name for t in targets}
        if missing:
            raise KeyError(f"targets not in split: {sorted(missing)}")
    return targets


def target_paths(atlas_dir: str, name: str) -> Dict[str, str]:
    d = os.path.join(atlas_dir, name)
    out = {"pdb": os.path.join(d, f"{name}.pdb")}
    for r in (1, 2, 3):
        out[f"R{r}"] = os.path.join(d, f"{name}_prod_R{r}_fit.xtc")
    return out


def verify_target(atlas_dir: str, name: str) -> List[str]:
    """Return the names of missing reference files for one target.

    Only the PBC-corrected, fitted ``*_prod_R{i}_fit.xtc`` trajectories are
    accepted: raw GROMACS trajectories can contain molecules broken across
    periodic boundaries, which would corrupt every metric.
    """
    return [os.path.basename(p) for p in target_paths(atlas_dir, name).values() if not os.path.exists(p)]


def _download(url: str, dest: str) -> bool:
    cmd = ["curl", "-fL", "--retry", "3", "--retry-delay", "5", "-o", dest, url]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def download_target(atlas_dir: str, name: str) -> str:
    d = os.path.join(atlas_dir, name)
    os.makedirs(d, exist_ok=True)
    if not verify_target(atlas_dir, name):
        return f"{name}: present"
    zpath = os.path.join(d, f"{name}_protein.zip")
    for tmpl in ATLAS_URLS:
        if _download(tmpl.format(name=name), zpath) and zipfile.is_zipfile(zpath):
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(d)
            os.remove(zpath)
            _flatten(d)
            missing = verify_target(atlas_dir, name)
            return f"{name}: ok" if not missing else f"{name}: incomplete, missing {missing}"
    if os.path.exists(zpath):
        os.remove(zpath)
    return f"{name}: DOWNLOAD FAILED"


def _flatten(d: str):
    """Move files out of a single nested folder produced by some archives."""
    entries = [e for e in os.listdir(d) if not e.startswith(".")]
    subdirs = [e for e in entries if os.path.isdir(os.path.join(d, e))]
    if len(subdirs) == 1 and len(entries) == 1:
        sub = os.path.join(d, subdirs[0])
        for f in os.listdir(sub):
            shutil.move(os.path.join(sub, f), os.path.join(d, f))
        os.rmdir(sub)


def download_atlas(atlas_dir: str, targets: Sequence[Target], workers: int = 8) -> List[str]:
    os.makedirs(atlas_dir, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: download_target(atlas_dir, t.name), targets))


# ----------------------------------------------------------------------------
# Model inputs derived from ATLAS
# ----------------------------------------------------------------------------

def write_fasta(targets: Sequence[Target], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for t in targets:
        p = os.path.join(out_dir, f"{t.name}.fasta")
        with open(p, "w") as f:
            f.write(f">{t.name}\n{t.seqres}\n")
        paths[t.name] = p
    return paths


def write_csv(targets: Sequence[Target], path: str, name_col: str = "name", extra: Optional[Dict] = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    extra = extra or {}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([name_col, "seqres", *extra.keys()])
        for t in targets:
            w.writerow([t.name, t.seqres, *[v(t) if callable(v) else v for v in extra.values()]])
    return path


def export_start_structures(atlas_dir: str, targets: Sequence[Target], out_dir: str, fmt: str = "pdb") -> Dict[str, str]:
    """Heavy-atom copies of the ATLAS starting structures (``{name}.pdb``)."""
    from . import ensemble_io

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for t in targets:
        traj = ensemble_io.strip_hydrogens(ensemble_io.load_structure_file(target_paths(atlas_dir, t.name)["pdb"]))
        paths[t.name] = _save(traj, os.path.join(out_dir, f"{t.name}.{fmt}"))
    return paths


def export_replica_first_frames(
    atlas_dir: str, targets: Sequence[Target], out_dir: str, replicas=(1, 2, 3), fmt: str = "cif", stem: str = "{name}_R{r}_0"
) -> Dict[str, str]:
    """Frame 0 of each production replica (MDGen / BioMD initial conditions)."""
    import mdtraj

    from . import ensemble_io

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for t in targets:
        tp = target_paths(atlas_dir, t.name)
        for r in replicas:
            frame = mdtraj.load_frame(tp[f"R{r}"], 0, top=tp["pdb"])
            frame = ensemble_io.strip_hydrogens(frame)
            key = stem.format(name=t.name, r=r)
            paths[key] = _save(frame, os.path.join(out_dir, f"{key}.{fmt}"))
    return paths


def _save(traj, path: str) -> str:
    if path.endswith(".cif"):
        import biotite.structure.io as bsio
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            traj.save_pdb(tmp_path)
            bsio.save_structure(path, bsio.load_structure(tmp_path))
        finally:
            os.remove(tmp_path)
    else:
        traj.save_pdb(path)
    return path
