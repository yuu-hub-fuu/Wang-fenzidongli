"""Command line entry point: ``python -m atlas_bench <command>``.

Commands
--------
list                     show the Table-1 baselines and their protocols
data                     download / verify the ATLAS test set
run BASELINE             prepare -> infer|fetch -> collect -> evaluate for one baseline
all                      run every baseline listed in configs/benchmark.yaml, then print Table 1
evaluate                 analyze a directory of {name}.pdb ensembles (writes out.pkl)
table                    print Table 1 from out.pkl files (with the paper's numbers)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

from . import atlas_data, config
from .baselines import BASELINES, REGISTRY, get_baseline
from .metrics import analyze, report

STAGES = ("prepare", "infer", "collect", "evaluate")


def cmd_list(args):
    for key, cls in REGISTRY.items():
        b = cls()
        print(f"{key:22s} {b.display_name:24s} sidechains={'yes' if b.sidechains else 'no ':3s}  {b.protocol}")
        if b.repo_url:
            print(f"{'':22s} {b.repo_url} @ {b.repo_commit[:10]}  (env: scripts/envs/{b.setup_script})")


def cmd_data(args):
    bench = config.load_benchmark_config(args.config)
    atlas_dir = args.atlas_dir or bench["atlas_dir"]
    targets = atlas_data.read_split(bench["split"], args.targets or None)
    if args.download:
        for line in atlas_data.download_atlas(atlas_dir, targets, args.workers):
            print(line)
    bad = {t.name: atlas_data.verify_target(atlas_dir, t.name) for t in targets}
    bad = {k: v for k, v in bad.items() if v}
    print(f"{len(targets) - len(bad)}/{len(targets)} ATLAS targets complete in {atlas_dir}")
    for k, v in bad.items():
        print(f"  {k}: missing {v}")
    return 1 if bad else 0


def run_baseline(key, bench, args, stages):
    baseline = get_baseline(key)
    bcfg = config.load_baseline_config(key, bench, getattr(args, "baseline_config", None))
    ctx = config.make_context(key, bench, bcfg, args.targets or None, args.gpus or None, args.dry_run)
    fetch = args.fetch or bcfg.get("mode") == "fetch"
    print(f"== {baseline.display_name} ({key}) -> {ctx.run_dir}  [{len(ctx.targets)} targets, "
          f"{'fetch released samples' if fetch else 'official inference'}]")
    if "prepare" in stages and not fetch:
        if ctx.dry_run:
            print(f"[dry-run] prepare inputs in {ctx.inputs_dir}")
        else:
            baseline.run_prepare(ctx)
    if "infer" in stages:
        (baseline.run_fetch if fetch else baseline.run_infer)(ctx)
    if ctx.dry_run:
        return None
    if "collect" in stages:
        rep = baseline.run_collect(ctx, fetched=fetch)
        ok = sum(isinstance(v, int) for v in rep.values())
        print(f"collected {ok}/{len(rep)} ensembles -> {ctx.ensemble_dir}")
        for k, v in rep.items():
            if not isinstance(v, int):
                print(f"  {k}: {v}")
    out_pkl = os.path.join(ctx.run_dir, "out.pkl")
    if "evaluate" in stages:
        names = [t.name for t in ctx.targets if os.path.exists(os.path.join(ctx.ensemble_dir, f"{t.name}.pdb"))]
        analyze.analyze_directory(
            ctx.atlas_dir, ctx.ensemble_dir, names, int(args.eval_workers or bench["eval_workers"]),
            analyze.AnalysisOptions(), out_pkl,
        )
    return out_pkl


def cmd_run(args):
    bench = config.load_benchmark_config(args.config)
    stages = args.stages.split(",") if args.stages else list(STAGES)
    bad = set(stages) - set(STAGES)
    if bad:
        raise SystemExit(f"unknown stages {bad}; choose from {STAGES}")
    run_baseline(args.baseline, bench, args, stages)


def cmd_all(args):
    bench = config.load_benchmark_config(args.config)
    keys = args.baselines or bench.get("baselines") or list(BASELINES)
    stages = args.stages.split(",") if args.stages else list(STAGES)
    pkls = OrderedDict()
    failures = {}
    for key in keys:
        try:
            pkl = run_baseline(key, bench, args, stages)
            if pkl and os.path.exists(pkl):
                pkls[key] = pkl
        except Exception as e:
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failures[key]}", file=sys.stderr)
    if pkls:
        report.main([f"{k}={v}" for k, v in pkls.items()] + (["--common_targets"] if args.common_targets else []))
    return 1 if failures else 0


def cmd_evaluate(args):
    analyze.main(args)


def cmd_table(args):
    report.main(args)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m atlas_bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list baselines")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("data", help="download / verify ATLAS test trajectories")
    s.add_argument("--config", default=None)
    s.add_argument("--atlas_dir", default=None)
    s.add_argument("--download", action="store_true")
    s.add_argument("--targets", nargs="*")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(func=cmd_data)

    def run_args(s):
        s.add_argument("--config", default=None, help="benchmark config (default configs/benchmark.yaml)")
        s.add_argument("--targets", nargs="*", help="subset of ATLAS targets")
        s.add_argument("--gpus", nargs="*", help="override GPU ids")
        s.add_argument("--stages", default=None, help=f"comma separated subset of {','.join(STAGES)}")
        s.add_argument("--fetch", action="store_true", help="download the authors' released ensembles instead of inference")
        s.add_argument("--dry_run", "--dry-run", action="store_true", help="print commands only")
        s.add_argument("--eval_workers", type=int, default=None)

    s = sub.add_parser("run", help="run one baseline")
    s.add_argument("baseline", choices=list(REGISTRY))
    s.add_argument("--baseline_config", default=None, help="override configs/baselines/<baseline>.yaml")
    run_args(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("all", help="run all baselines then print Table 1")
    s.add_argument("--baselines", nargs="*", choices=list(REGISTRY))
    s.add_argument("--common_targets", action="store_true")
    run_args(s)
    s.set_defaults(func=cmd_all)

    s = sub.add_parser("evaluate", help="analyze {name}.pdb ensembles against ATLAS MD")
    analyze.build_argparser(s)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("table", help="print Table 1 from out.pkl files")
    report.build_argparser(s)
    s.set_defaults(func=cmd_table)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
