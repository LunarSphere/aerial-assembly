"""NSGA-III search over brick parameters.

Objectives (all minimised): -P(success), F_max, T_assembly, mass, collisions.
P(success) pools coarse-placement drops over a friction sweep with structure
trials (overhang, wall, bridge). Analytic rules are inequality constraints
and short-circuit simulation for infeasible designs. The final design is the
Pareto point with the best weighted score on normalised objectives.
"""
from dataclasses import asdict, dataclass, field, replace
import json
import os
from multiprocessing import Pool
from pathlib import Path
import time
import traceback

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.parallelization import StarmapParallelization
from pymoo.util.ref_dirs import get_reference_directions

from aerial_assembly.config import Physics

from . import cad, rules as R, structures as S, trials as T
from .params import BOUNDS, V0, BrickParams, with_vector

OBJECTIVES = ['neg_p_success', 'f_max_N', 't_assembly_s', 'mass_g', 'collisions']
# Rules whose margins are uninformative as constraints (always satisfied by bounds).
SKIP_RULES = {'tooth_strength', 'mass_min'}


@dataclass
class EvalConfig:
    drop_samples: int = 24
    frictions: tuple = (0.2, 0.35, 0.5)
    overhang_n: int = 10
    wall: tuple = (3, 3)
    bridge_arms: int = 3
    structure_seeds: tuple = (1,)
    weights: dict = field(default_factory=lambda: {'p': 1.0, 'f': 0.15, 't': 0.15, 'm': 0.3, 'c': 0.15})
    error: dict = field(default_factory=dict)  # overrides for trials.ErrorModel

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ('frictions', 'wall', 'structure_seeds'):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def decode(x, names):
    """Continuous vector -> BrickParams; the last gene picks the lean mode."""
    p = with_vector(V0, names, x[:-1])
    return replace(p, lean_mode='uniform' if x[-1] >= 0.5 else 'alternate')


def evaluate(p: BrickParams, ec: EvalConfig):
    """Return (objectives dict, rule margins, details). Never raises."""
    t0 = time.time()
    worst = {'neg_p_success': 0., 'f_max_N': 200., 't_assembly_s': 5., 'mass_g': 50., 'collisions': 50.}
    try:
        brick = cad.build_brick(p)
    except Exception as exc:  # geometry failed to build: infeasible
        return worst, {'geometry_builds': -1.}, {'error': repr(exc)}
    margins = {k: v for k, v in R.rules(p, brick).items() if k not in SKIP_RULES}
    margins['geometry_builds'] = 1.
    if not R.feasible(margins):
        return dict(worst, mass_g=brick['mass_g']), margins, {'skipped': 'rules', 'seconds': time.time() - t0}
    err = T.ErrorModel(**ec.error)
    drops, details = [], {}
    try:
        for mu in ec.frictions:
            cfg = T.TrialConfig(physics=Physics(timestep=0.001, friction=mu, contact_timeconst=0.004,
                                                iterations=50), error=err)
            rows, _ = T.drop_trials(p, brick, cfg, samples=ec.drop_samples, seed=int(mu*1000))
            drops += rows
        ds = T.summarize_drops(drops)
        cfg = T.TrialConfig(error=err)
        ideal = T.TrialConfig(error=T.ErrorModel(ideal=True, aim_bias=err.aim_bias))
        ovh, _ = T.assemble(p, brick, S.overhang(ec.overhang_n, +1), ideal)
        scores = {'drop': ds['p_success'], 'overhang': ovh['stable_bricks']/ovh['total']}
        times = [r['time'] for r in drops]
        for seed in ec.structure_seeds:
            wall, _ = T.assemble(p, brick, S.wall(*ec.wall), cfg, seed=seed)
            bridge, _ = T.assemble(p, brick, S.bridge(ec.bridge_arms), cfg, seed=seed)
            scores[f'wall{seed}'] = wall['stable_bricks']/wall['total']
            scores[f'bridge{seed}'] = bridge['stable_bricks']/bridge['total']
            times += [s['time'] for s in wall['stages'] + bridge['stages']]
        p_success = float(np.mean(list(scores.values())))
        obj = {'neg_p_success': -p_success, 'f_max_N': ds['f_max_N'],
               't_assembly_s': float(np.mean(times)), 'mass_g': brick['mass_g'],
               'collisions': ds['collisions']}
        details = {'scores': scores, 'drop_outcomes': ds['outcomes'], 'overhang_stable': ovh['stable_bricks'],
                   'overhang_sag_mm': ovh['max_sag_mm'], 'seconds': time.time() - t0}
        return obj, margins, details
    except Exception:
        return dict(worst, mass_g=brick['mass_g']), margins, {'error': traceback.format_exc(limit=3)}


def _eval_job(x, names, ec_dict):
    p = decode(np.asarray(x), names)
    obj, margins, details = evaluate(p, EvalConfig.from_dict(ec_dict))
    return p.to_dict(), obj, margins, details


class BrickProblem(ElementwiseProblem):
    def __init__(self, names, ec: EvalConfig, constraint_names, log, **kwargs):
        lo = [BOUNDS[n][0] for n in names] + [0.]
        hi = [BOUNDS[n][1] for n in names] + [1.]
        super().__init__(n_var=len(lo), n_obj=len(OBJECTIVES), n_ieq_constr=len(constraint_names),
                         xl=np.array(lo), xu=np.array(hi), **kwargs)
        self.names, self.ec, self.constraint_names, self.log = names, ec, constraint_names, log

    def _evaluate(self, x, out, *args, **kwargs):
        params, obj, margins, details = _eval_job(x, self.names, asdict(self.ec))
        out['F'] = [obj[k] for k in OBJECTIVES]
        out['G'] = [-margins.get(k, -1.) for k in self.constraint_names]
        # One file per worker process: appends never interleave.
        with open(f'{self.log}.{os.getpid()}.jsonl', 'a') as fh:
            fh.write(json.dumps({'x': list(map(float, x)), 'params': params, 'objectives': obj,
                                 'margins': margins, 'details': details}, default=float) + '\n')


class Checkpoint(Callback):
    def __init__(self, out):
        super().__init__()
        self.out = Path(out)

    def notify(self, algorithm):
        pop = algorithm.pop
        gen = algorithm.n_gen
        payload = {'generation': gen, 'X': pop.get('X').tolist(), 'F': pop.get('F').tolist(),
                   'G': pop.get('G').tolist() if pop.get('G') is not None else None}
        (self.out/f'gen_{gen:03d}.json').write_text(json.dumps(payload))
        F = pop.get('F')
        print(f'gen {gen}: best P={-F[:, 0].min():.3f}  min mass={F[:, 3].min():.1f} g', flush=True)


def constraint_names():
    margins = R.rules(V0, cad.build_brick(V0, solid=False))
    return [k for k in margins if k not in SKIP_RULES] + ['geometry_builds']


def select(F, weights):
    """Weighted score on min-max normalised objectives; returns best index and scores."""
    F = np.asarray(F, dtype=float)
    lo, hi = F.min(0), F.max(0)
    N = (F - lo)/np.where(hi > lo, hi - lo, 1.)
    w = np.array([weights['p'], weights['f'], weights['t'], weights['m'], weights['c']])
    # neg_p_success is minimised, so (1 - N[:,0]) is the normalised success.
    score = w[0]*(1 - N[:, 0]) - (N[:, 1:]*w[1:]).sum(1)
    return int(np.argmax(score)), score


def seed_population(names, size, rng, spread=0.15):
    """V0 (both lean modes) plus Gaussian perturbations: start near the educated guess."""
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])
    base = np.array([getattr(V0, n) for n in names])
    rows = [np.r_[base, 0.25], np.r_[base, 0.75]]
    while len(rows) < size:
        x = np.clip(base + rng.normal(0, spread, len(names))*(hi - lo), lo, hi)
        rows.append(np.r_[x, rng.uniform()])
    return np.array(rows)


def run(out, pop_size=24, generations=10, partitions=3, workers=8, seed=0, ec: EvalConfig = EvalConfig(),
        names=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(names or BOUNDS)
    cons = constraint_names()
    ref_dirs = get_reference_directions('das-dennis', len(OBJECTIVES), n_partitions=partitions)
    pop_size = max(pop_size, len(ref_dirs))
    (out/'config.json').write_text(json.dumps({'names': names, 'objectives': OBJECTIVES, 'constraints': cons,
                                               'pop_size': pop_size, 'generations': generations,
                                               'partitions': partitions, 'seed': seed,
                                               'eval': asdict(ec)}, indent=2))
    log = out/'evaluations'
    with Pool(workers) as pool:
        problem = BrickProblem(names, ec, cons, str(log), elementwise_runner=StarmapParallelization(pool.starmap))
        initial = seed_population(names, pop_size, np.random.default_rng(seed))
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=initial)
        res = minimize(problem, algorithm, ('n_gen', generations), seed=seed, callback=Checkpoint(out),
                       verbose=False)
    if res.F is None or len(np.atleast_2d(res.F)) == 0:
        result = {'status': 'no_feasible_solution'}
        (out/'result.json').write_text(json.dumps(result, indent=2))
        return result
    X, F = np.atleast_2d(res.X), np.atleast_2d(res.F)
    best, scores = select(F, ec.weights)
    front = [{'params': decode(x, names).to_dict(), 'objectives': dict(zip(OBJECTIVES, map(float, f))),
              'score': float(s)} for x, f, s in zip(X, F, scores)]
    result = {'status': 'ok', 'chosen': front[best], 'pareto_front': front}
    (out/'result.json').write_text(json.dumps(result, indent=2))
    chosen = BrickParams.from_dict(front[best]['params'])
    cad.export(cad.build_brick(chosen), out/'chosen')
    (out/'chosen'/'params.json').write_text(json.dumps(chosen.to_dict(), indent=2))
    return result
