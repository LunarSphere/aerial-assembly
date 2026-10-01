"""Seeded complete-trial aggregation, uncertainty, and reliability estimates."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from importlib.metadata import version
import json
import platform

import numpy as np
from scipy.stats import beta

from .assembly import benchmark_suite, plan_assembly
from .geometry import generate_design
from .models import DesignParameters, Evaluation, RunConfig, Target, digest, write_json
from .simulation import run_trial, sample_disturbance


def reliability_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided exact Clopper–Pearson lower confidence bound."""
    if not 0 <= successes <= trials or trials < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    return 0.0 if successes == 0 else float(beta.ppf(1-confidence, successes, trials-successes+1))


def provenance() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "source_hash": digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}),
            "packages": {name: version(name) for name in ("brixzle", "cadquery", "mujoco", "numpy", "scipy", "networkx", "pymoo", "cloudpickle", "matplotlib")}}


def trial_seeds(seed: int, count: int, phase: str = "search") -> list[int]:
    offsets = {"search": 0, "reevaluate": 1, "validate": 2, "diagnostic": 3}
    if phase not in offsets:
        raise ValueError("Unknown seed phase")
    rng = np.random.default_rng(np.random.SeedSequence([seed, offsets[phase]]))
    return [int(x) for x in rng.integers(0, 2**31-1, count)]


def evaluate_design(parameters: DesignParameters, benchmark_targets: list[Target] | None = None,
                    config: RunConfig | None = None, seeds: list[int] | None = None,
                    output_dir: Path | None = None) -> Evaluation:
    config = config or RunConfig()
    config.validate()
    targets = benchmark_targets if benchmark_targets is not None else benchmark_suite(False, config.simulation.manipulation_mode)
    if not targets or not any(t.mandatory for t in targets):
        raise ValueError("At least one mandatory benchmark is required")
    seeds = seeds if seeds is not None else trial_seeds(config.search.seed, config.search.trials)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Trial seeds must be nonempty and unique")
    fingerprint = digest({"parameters": asdict(parameters), "config": asdict(config),
                          "targets": [asdict(t) for t in targets], "seeds": seeds, "provenance": provenance()})
    folder = None if output_dir is None else output_dir / fingerprint
    if folder is not None and (folder / "evaluation.json").exists():
        return Evaluation(**json.loads((folder / "evaluation.json").read_text()))
    diagnostics = []
    try:
        design = generate_design(parameters, config, None if folder is None else folder / "geometry")
    except (ValueError, RuntimeError) as exc:
        result = Evaluation(fingerprint, asdict(parameters), [1, 1000, 1e6, 40, 1000],
                            [1, 0, 0], {}, False, [f"CAD generation failed: {exc}"])
        if folder is not None:
            write_json(folder / "evaluation.json", result)
            write_json(folder / "config.json", config)
            write_json(folder / "provenance.json", provenance())
        return result
    diagnostics.extend(design.diagnostics)
    summaries = {}
    mandatory_results = []
    bad_plans, warnings, printed_mass_violations = 0, 0, 0
    manufactured = {}
    plans = [(t, plan_assembly(design, t, config)) for t in targets]
    for target, plan in plans:
        if target.mandatory and not plan.feasible:
            bad_plans += 1
        if folder is not None:
            write_json(folder / target.name / "plan.json", plan)
        results = []
        for seed in seeds:
            trial_folder = None if folder is None else folder / target.name / str(seed)
            if trial_folder is not None and (trial_folder / "trial.json").exists():
                from .models import TrialResult
                trial = TrialResult(**json.loads((trial_folder / "trial.json").read_text()))
            else:
                actual = None
                if plan.feasible and design.feasible:
                    if seed not in manufactured:
                        error = sample_disturbance(config.uncertainty, seed, 0)["clearance_error_mm"]
                        try:
                            manufactured[seed] = generate_design(replace(parameters,
                                clearance_mm=parameters.clearance_mm+error), config) if error else design
                        except (ValueError, RuntimeError):
                            manufactured[seed] = None
                    actual = manufactured[seed]
                trial = run_trial(design, plan, seed=seed, config=config, output_dir=trial_folder,
                                  manufactured_design=actual)
            results.append(trial)
        successes = sum(r.success for r in results)
        lower = reliability_lower_bound(successes, len(results), config.search.confidence)
        caps = config.simulation.placement_timeout_s*max(1, sum(not v.anchored for v in target.voxels))
        # Failed trials receive the full benchmark timeout; early failures must
        # never appear faster than a successful complete assembly.
        durations = [min(r.assembly_time_s, caps) if r.success else caps for r in results]
        summaries[target.name] = {
            "manipulation_mode": config.simulation.manipulation_mode,
            "time_scope": "approach_release_seating_hold" if config.simulation.manipulation_mode == "floating" else "pickup_transport_seating_hold",
            "mandatory": target.mandatory, "plan_feasible": plan.feasible, "plan_reason": plan.reason,
            "trials": len(results), "successes": successes, "success_rate": successes/len(results),
            "success_lower_bound": lower, "meets_reliability_target": lower >= config.search.reliability,
            "force_p95_n": float(np.quantile([r.peak_force_n for r in results], 0.95)),
            "mean_capped_time_s": float(np.mean(durations)),
            "mean_collisions": float(np.mean([r.collisions for r in results])),
            "failures": dict(Counter(r.reason for r in results if not r.success)),
            "numerical_warnings": sum(r.numerical_warning for r in results),
        }
        if target.mandatory:
            mandatory_results.extend(results)
            warnings += summaries[target.name]["numerical_warnings"]
            printed_mass_violations += sum(r.reason == "printed_mass_limit" for r in results)
    mandatory = [value for value in summaries.values() if value["mandatory"]]
    objectives = [1-min(v["success_rate"] for v in mandatory),
                  max(v["force_p95_n"] for v in mandatory),
                  float(np.mean([v["mean_capped_time_s"] for v in mandatory])),
                  design.mass_g, float(np.mean([v["mean_collisions"] for v in mandatory]))]
    constraints = [float(len(design.diagnostics)+printed_mass_violations), float(bad_plans), float(warnings)]
    result = Evaluation(design.geometry_hash, asdict(parameters), objectives, constraints,
                        summaries, not any(x > 0 for x in constraints), diagnostics)
    if folder is not None:
        write_json(folder / "evaluation.json", result)
        write_json(folder / "config.json", config)
        write_json(folder / "seeds.json", seeds)
        write_json(folder / "provenance.json", provenance())
    return result
