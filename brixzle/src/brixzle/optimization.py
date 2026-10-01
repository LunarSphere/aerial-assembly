"""NSGA-III search, checkpoint resume, and held-out candidate validation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import asdict, replace
from pathlib import Path
import json
import cloudpickle
import random

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .assembly import benchmark_suite
from .evaluation import evaluate_design, provenance, trial_seeds
from .geometry import generate_design
from .models import DesignParameters, Evaluation, RunConfig, digest, write_json

# Continuous genes; topology and reinforcement are independent campaigns.
GENES = {
    "pitch_mm": (24.0, 40.0), "wall_mm": (1.2, 2.0), "rib_mm": (1.2, 2.0),
    "peg_width_mm": (4.5, 7.0), "peg_height_mm": (1.5, 3.0),
    "rail_depth_mm": (1.0, 3.0), "rail_neck_mm": (3.5, 5.0),
    "rail_head_mm": (5.5, 8.0), "clearance_mm": (0.18, 0.6),
    "guide_angle_deg": (55.0, 78.0), "guide_height_mm": (1.0, 3.0),
    "approach_bias_mm": (-1.0, 1.0),
}


def decode(x, family="combined", reinforcement="ibeam") -> DesignParameters:
    return DesignParameters(family=family, reinforcement=reinforcement,
                            **{name: float(val) for name, val in zip(GENES, x)})


def _evaluate_job(job):
    parameters, config, seeds, folder = job
    return evaluate_design(parameters, config=config, seeds=seeds, output_dir=folder)


class BrickProblem(Problem):
    def __init__(self, config: RunConfig, family: str, reinforcement: str, folder: Path, seeds: list[int]):
        self.config, self.family, self.reinforcement, self.folder, self.seeds = config, family, reinforcement, folder, seeds
        super().__init__(n_var=len(GENES), n_obj=5, n_ieq_constr=3,
                         xl=[b[0] for b in GENES.values()], xu=[b[1] for b in GENES.values()])

    def _evaluate(self, X, out, *args, **kwargs):
        jobs = [(decode(x, self.family, self.reinforcement), self.config, self.seeds, self.folder) for x in X]
        if self.config.search.workers > 1:
            with ProcessPoolExecutor(max_workers=self.config.search.workers, mp_context=get_context("spawn")) as pool:
                values = list(pool.map(_evaluate_job, jobs))
        else:
            values = [_evaluate_job(job) for job in jobs]
        out["F"] = np.asarray([v.objectives for v in values])
        out["G"] = np.asarray([v.constraints for v in values])


def pareto_indices(evaluations: list[Evaluation]) -> list[int]:
    eligible = [i for i, e in enumerate(evaluations) if e.feasible]
    if not eligible:
        return []
    f = np.array([evaluations[i].objectives for i in eligible])
    return [eligible[int(i)] for i in NonDominatedSorting().do(f, only_non_dominated_front=True)]


def weighted_order(evaluations: list[Evaluation], weights) -> list[int]:
    """Normalize each minimization objective over the supplied candidate set."""
    if not evaluations:
        return []
    f = np.array([e.objectives for e in evaluations])
    low, high = f.min(axis=0), f.max(axis=0)
    normalized = (f-low)/np.where(high > low, high-low, 1)
    score = normalized @ np.array(weights)/sum(weights)
    return np.argsort(score, kind="stable").tolist()


def _atomic_pickle(path: Path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        cloudpickle.dump(value, stream)
    temporary.replace(path)


def run_search(config: RunConfig, output_dir: Path, families=("pegs", "rails", "combined"),
               reinforcements=("web", "ibeam"), resume: bool = False,
               validate: bool = True) -> dict:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = digest({"config": asdict(config), "families": list(families),
                        "reinforcements": list(reinforcements), "provenance": provenance(), "search_version": 1})
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if not resume:
            raise ValueError("Run directory already exists; use --resume or a new directory")
        if old["signature"] != signature:
            raise ValueError("Resume configuration, campaigns, or dependency versions differ")
    elif resume:
        raise ValueError("No run manifest to resume")
    write_json(manifest, {"signature": signature, "config": asdict(config), "families": list(families),
                          "reinforcements": list(reinforcements), "provenance": provenance()})
    seeds = trial_seeds(config.search.seed, config.search.trials)
    all_candidates = []
    for family in families:
        if family not in ("pegs", "rails", "combined"):
            raise ValueError("Unknown geometry family")
        for reinforcement in reinforcements:
            if reinforcement not in ("web", "ibeam"):
                raise ValueError("Unknown reinforcement")
            campaign = output_dir / f"{family}-{reinforcement}"
            campaign.mkdir(parents=True, exist_ok=True)
            baseline = evaluate_design(DesignParameters(family=family, reinforcement=reinforcement),
                                       config=config, seeds=seeds, output_dir=campaign / "evaluations")
            all_candidates.append(baseline)
            print(f"{family}/{reinforcement}: baseline success objective={baseline.objectives[0]:.3f}; "
                  f"constraints={baseline.constraints}", flush=True)
            # Connectivity cannot be changed by continuous genes within a family.
            # Preserve the baseline evidence instead of wasting a whole campaign.
            if baseline.constraints[1] > 0:
                write_json(campaign / "skipped.json", {"reason": "Topology cannot connect mandatory targets",
                                                       "evaluation": asdict(baseline)})
                continue
            checkpoint = campaign / "checkpoint.pkl"
            if checkpoint.exists() and resume:
                # Local checkpoints are trusted executable Python objects. Never
                # load checkpoint files supplied by another person or service.
                with checkpoint.open("rb") as stream:
                    saved = cloudpickle.load(stream)
                algorithm = saved["algorithm"]
                np.random.set_state(saved["numpy_random_state"])
                random.setstate(saved["python_random_state"])
            else:
                directions = get_reference_directions("energy", 5, config.search.population,
                                                       seed=config.search.seed)
                algorithm = NSGA3(pop_size=config.search.population, ref_dirs=directions)
                problem = BrickProblem(config, family, reinforcement, campaign / "evaluations", seeds)
                algorithm.setup(problem, termination=get_termination("n_gen", config.search.generations),
                                seed=config.search.seed, verbose=False)
            while algorithm.has_next():
                algorithm.next()
                _atomic_pickle(checkpoint, {"algorithm": algorithm, "numpy_random_state": np.random.get_state(),
                                            "python_random_state": random.getstate()})
                write_json(campaign / "progress.json", {"generation": algorithm.n_gen-1,
                    "evaluations": algorithm.evaluator.n_eval,
                    "population_objectives": algorithm.pop.get("F").tolist(),
                    "population_constraints": algorithm.pop.get("G").tolist()})
                print(f"{family}/{reinforcement}: generation {algorithm.n_gen-1}, "
                      f"{algorithm.evaluator.n_eval} candidates evaluated", flush=True)
            for x in algorithm.pop.get("X"):
                all_candidates.append(evaluate_design(decode(x, family, reinforcement), config=config,
                                                       seeds=seeds, output_dir=campaign / "evaluations"))
    # Include every historical feasible candidate rather than only the last
    # population; the evaluation archive is also the interruption-safe cache.
    for path in output_dir.glob("*/evaluations/*/evaluation.json"):
        all_candidates.append(Evaluation(**json.loads(path.read_text())))
    unique = {digest(e.parameters): e for e in all_candidates}
    all_candidates = list(unique.values())
    front = [all_candidates[i] for i in pareto_indices(all_candidates)]
    shortlist = [front[i] for i in weighted_order(front, config.search.weights)[:config.search.shortlist]]
    reevaluated, validated, numerical_checks, numerical_comparisons = [], [], [], []
    if validate:
        reevaluation_seeds = trial_seeds(config.search.seed, config.search.reevaluate_trials, "reevaluate")
        for e in shortlist:
            reevaluated.append(evaluate_design(DesignParameters(**e.parameters), config=config,
                                               seeds=reevaluation_seeds, output_dir=output_dir / "reevaluate"))
        # Reliability comes before weighted preferences when choosing finalists.
        indices = weighted_order(reevaluated, config.search.weights)
        indices.sort(key=lambda i: (not reevaluated[i].feasible,
                                    -min((b["success_lower_bound"] for b in reevaluated[i].benchmarks.values()), default=0)))
        indices = [i for i in indices if reevaluated[i].feasible]
        validation_seeds = trial_seeds(config.search.seed, config.search.validation_trials, "validate")
        for i in indices[:config.search.finalists]:
            parameters = DesignParameters(**reevaluated[i].parameters)
            v = evaluate_design(parameters, config=config, seeds=validation_seeds, output_dir=output_dir / "validate")
            validated.append(v)
            tighter = replace(config, simulation=replace(config.simulation,
                timestep_s=config.simulation.timestep_s/2,
                solver_iterations=config.simulation.solver_iterations*2))
            # Reuse a small held-out subset for paired numerical sensitivity.
            paired_seeds = validation_seeds[:min(8, len(validation_seeds))]
            standard = evaluate_design(parameters, config=config, seeds=paired_seeds,
                                       output_dir=output_dir / "numerical-standard")
            numerical = evaluate_design(parameters, config=tighter, seeds=paired_seeds,
                                       output_dir=output_dir / "numerical")
            numerical_checks.append(numerical)
            comparison = {"design_hash": v.design_hash, "benchmarks": {}, "passed": numerical.feasible}
            for name, summary in standard.benchmarks.items():
                tighter_summary = numerical.benchmarks[name]
                force_delta = abs(summary["force_p95_n"]-tighter_summary["force_p95_n"])
                stable = (summary["successes"] == tighter_summary["successes"] and
                          force_delta <= max(0.05, 0.25*summary["force_p95_n"]))
                comparison["benchmarks"][name] = {"successes_standard": summary["successes"],
                    "successes_tighter": tighter_summary["successes"], "force_p95_delta_n": force_delta,
                    "passed": stable}
                comparison["passed"] &= stable
            numerical_comparisons.append(comparison)
            generate_design(parameters, config, output_dir / "finalists" / v.design_hash)
            evaluate_design(parameters, benchmark_suite(True, config.simulation.manipulation_mode), config,
                            trial_seeds(config.search.seed, min(8, config.search.validation_trials), "diagnostic"),
                            output_dir / "diagnostic")
    report = {"candidate_count": len(all_candidates), "pareto": [asdict(e) for e in front],
              "shortlist": [asdict(e) for e in shortlist], "reevaluated": [asdict(e) for e in reevaluated],
              "validated": [asdict(e) for e in validated], "numerical_checks": [asdict(e) for e in numerical_checks],
              "numerical_comparisons": numerical_comparisons,
              "target_met": any(e.feasible and numerical_comparisons[i]["passed"] and
                                all(b["meets_reliability_target"] for b in e.benchmarks.values() if b["mandatory"])
                                for i, e in enumerate(validated))}
    write_json(output_dir / "search.json", report)
    return report
