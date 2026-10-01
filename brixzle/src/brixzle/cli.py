"""Command-line workflow; explicit commands distinguish pilots from campaigns."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from .assembly import benchmark_suite, plan_assembly, target_from_dict
from .evaluation import evaluate_design, provenance, trial_seeds
from .geometry import generate_design
from .models import RunConfig, load_config, load_parameters, write_json
from .optimization import run_search
from .reporting import build_report
from .simulation import run_trial


def _parser():
    parser = argparse.ArgumentParser(description="Brixzle CAD, contact trials, and geometry optimization")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "plan", "trial", "evaluate", "optimize", "report", "defaults"):
        p = sub.add_parser(name)
        if name not in ("report", "defaults"):
            p.add_argument("--config", type=Path, help="JSON overrides of default profiles")
            p.add_argument("--mode", choices=("floating", "fork"), help="Floating release proxy (default) or passive fork")
        if name in ("generate", "plan", "trial", "evaluate"):
            p.add_argument("--parameters", type=Path, help="JSON brick design parameters")
            p.add_argument("--family", choices=("pegs", "rails", "combined"))
        if name in ("plan", "trial", "evaluate"):
            p.add_argument("--target", type=Path, help="Custom voxel target JSON")
            p.add_argument("--benchmark", choices=[t.name for t in benchmark_suite()])
        if name == "trial":
            p.add_argument("--seed", type=int, default=1234)
        if name in ("trial", "evaluate"):
            p.add_argument("--nominal", action="store_true", help="Disable error perturbations, not physical contacts")
        if name in ("evaluate", "optimize"):
            p.add_argument("--trials", type=int)
            p.add_argument("--seed", type=int)
        if name == "evaluate":
            p.add_argument("--diagnostics", action="store_true", help="Include arch and dome benchmarks")
            p.add_argument("--phase", choices=("search", "reevaluate", "validate", "diagnostic"), default="search")
        if name == "optimize":
            p.add_argument("--families", nargs="+", choices=("pegs", "rails", "combined"), default=["pegs", "rails", "combined"])
            p.add_argument("--reinforcements", nargs="+", choices=("web", "ibeam"), default=["web", "ibeam"])
            p.add_argument("--population", type=int)
            p.add_argument("--generations", type=int)
            p.add_argument("--workers", type=int)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--search-only", action="store_true", help="Defer shortlist and held-out validation")
        if name == "report":
            p.add_argument("run_dir", type=Path)
            p.add_argument("--out", type=Path)
        else:
            p.add_argument("--out", type=Path, default=Path("runs") / name)
    return parser


def _targets(args, config):
    if args.target is not None:
        return [target_from_dict(json.loads(args.target.read_text()))]
    targets = benchmark_suite(getattr(args, "diagnostics", False), config.simulation.manipulation_mode)
    if args.benchmark:
        return [t for t in benchmark_suite() if t.name == args.benchmark]
    return targets


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            print(build_report(args.run_dir, args.out))
            return 0
        if args.command == "defaults":
            write_json(args.out / "config.json", RunConfig())
            from .models import DesignParameters
            write_json(args.out / "parameters.json", DesignParameters())
            for t in benchmark_suite():
                write_json(args.out / f"{t.name}.json", t)
            print(args.out)
            return 0
        config = load_config(args.config)
        overrides = {key: getattr(args, key) for key in ("trials", "seed", "population", "generations", "workers")
                     if getattr(args, key, None) is not None}
        config = replace(config, search=replace(config.search, **overrides))
        if args.mode:
            config = replace(config, simulation=replace(config.simulation, manipulation_mode=args.mode))
        if getattr(args, "nominal", False):
            config = replace(config, uncertainty=replace(config.uncertainty, lateral_mm=0, vertical_mm=0,
                yaw_deg=0, tilt_deg=0, clearance_error_mm=0, mass_fraction=0, tracking_mm=0,
                friction_min=(config.uncertainty.friction_min+config.uncertainty.friction_max)/2,
                friction_max=(config.uncertainty.friction_min+config.uncertainty.friction_max)/2))
        config.validate()
        if args.command == "optimize":
            result = run_search(config, args.out, args.families, args.reinforcements,
                                args.resume, not args.search_only)
            report = build_report(args.out)
            print(f"Candidates: {result['candidate_count']}; reliability target met: {result['target_met']}; report: {report}")
            return 0
        parameters = load_parameters(args.parameters, args.family)
        if args.command == "evaluate":
            evaluation = evaluate_design(parameters, _targets(args, config), config,
                trial_seeds(config.search.seed, config.search.trials, args.phase), args.out)
            print(json.dumps(asdict(evaluation), indent=2))
            return 0 if evaluation.feasible else 2
        design = generate_design(parameters, config, args.out / "geometry" if args.command != "generate" else args.out)
        write_json(args.out / "config.json", config)
        write_json(args.out / "provenance.json", provenance())
        if args.command == "generate":
            print(f"{design.mass_g:.3f} g; {len(design.collision_pieces)} contact pieces; feasible: {design.feasible}")
            for diagnostic in design.diagnostics:
                print(diagnostic)
            return 0 if design.feasible else 2
        targets = _targets(args, config)
        if args.command == "trial" and args.benchmark is None and args.target is None:
            targets = [benchmark_suite(False, config.simulation.manipulation_mode)[0]]
        failed = False
        for target in targets:
            plan = plan_assembly(design, target, config)
            write_json(args.out / target.name / "plan.json", plan)
            if args.command == "plan":
                print(f"{target.name}: {'feasible' if plan.feasible else plan.reason}; placements: {plan.order}")
                failed |= not plan.feasible
            else:
                trial = run_trial(design, plan, seed=args.seed, config=config, output_dir=args.out / target.name)
                print(json.dumps(asdict(trial), indent=2))
                failed |= not trial.success
        return 2 if failed else 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"brixzle: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
