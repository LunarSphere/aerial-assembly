"""NSGA-III over polycube brick parameters.

Objectives (minimised): -P(success), F_max, T_assembly, mass, collisions.
P(success) pools coarse drops over a friction sweep with structure trials.
Rules are inequality constraints and skip simulation when violated.
"""
from dataclasses import asdict, dataclass, field
import json
from multiprocessing import get_context
import os
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

from . import cad, rules as R, structures as S, trials as T
from .params import BOUNDS, V0, BrickParams, with_vector
from .scene import Physics

OBJECTIVES = ['neg_p_success', 'f_max_N', 't_assembly_s', 'mass_g', 'collisions']
WORST = {'neg_p_success': 0., 'f_max_N': 50., 't_assembly_s': 5., 'mass_g': 50., 'collisions': 50.}


@dataclass
class EvalConfig:
    drop_samples: int = 24
    frictions: tuple = (0.2, 0.35, 0.5)
    structures: tuple = ('pillar', 'column', 'overhang', 'wall')
    noisy: tuple = ('column', 'wall')
    seeds: tuple = (1,)
    weights: dict = field(default_factory=lambda: {'p': 1.0, 'f': 0.1, 't': 0.1, 'm': 0.3, 'c': 0.1})

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ('frictions', 'structures', 'noisy', 'seeds'):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def evaluate(p: BrickParams, ec: EvalConfig):
    t0 = time.time()
    try:
        brick = cad.build_brick(p)
    except Exception as exc:
        return dict(WORST), {'geometry_builds': -1.}, {'error': repr(exc)}
    margins = R.rules(p, brick)
    margins['geometry_builds'] = 1.
    if not R.feasible(margins):
        return dict(WORST, mass_g=brick['mass_g']), margins, {'skipped': 'rules'}
    try:
        drops = []
        for mu in ec.frictions:
            rows, _ = T.drop_trials(p, brick, T.TrialConfig(physics=Physics(friction=mu)),
                                    samples=ec.drop_samples, seed=int(mu*1000))
            drops += rows
        ds = T.summarize_drops(drops)
        scores, times = {'drop': ds['p_success']}, [r['time'] for r in drops]
        ideal = T.TrialConfig(error=T.ErrorModel(ideal=True))
        for name in ec.structures:
            r, _ = T.assemble(p, brick, S.CATALOG[name](), ideal)
            scores[f'ideal_{name}'] = r['stable_bricks']/r['total']
        for name in ec.noisy:
            for seed in ec.seeds:
                r, _ = T.assemble(p, brick, S.CATALOG[name](), T.TrialConfig(), seed=seed, stop_on_fail=False)
                scores[f'noisy_{name}{seed}'] = sum(s['outcome'] == 'seated' for s in r['stages'])/r['total']
                times += [s['time'] for s in r['stages']]
        obj = {'neg_p_success': -float(np.mean(list(scores.values()))), 'f_max_N': ds['f_max_N'],
               't_assembly_s': float(np.mean(times)), 'mass_g': brick['mass_g'], 'collisions': ds['collisions']}
        return obj, margins, {'scores': scores, 'seconds': time.time() - t0}
    except Exception:
        return dict(WORST, mass_g=brick['mass_g']), margins, {'error': traceback.format_exc(limit=3)}


def _job(x, names, ec_dict, base_dict):
    p = with_vector(BrickParams.from_dict(base_dict), names, x)
    obj, margins, details = evaluate(p, EvalConfig.from_dict(ec_dict))
    return p.to_dict(), obj, margins, details


class BrickProblem(ElementwiseProblem):
    def __init__(self, names, ec, base, cons, log, **kw):
        super().__init__(n_var=len(names), n_obj=len(OBJECTIVES), n_ieq_constr=len(cons),
                         xl=np.array([BOUNDS[n][0] for n in names]),
                         xu=np.array([BOUNDS[n][1] for n in names]), **kw)
        self.names, self.ec, self.base, self.cons, self.log = names, ec, base, cons, log

    def _evaluate(self, x, out, *args, **kwargs):
        params, obj, margins, details = _job(x, self.names, asdict(self.ec), self.base.to_dict())
        out['F'] = [obj[k] for k in OBJECTIVES]
        out['G'] = [-margins.get(k, -1.) for k in self.cons]
        with open(f'{self.log}.{os.getpid()}.jsonl', 'a') as fh:
            fh.write(json.dumps({'params': params, 'objectives': obj, 'margins': margins,
                                 'details': details}, default=float) + '\n')


class Checkpoint(Callback):
    def __init__(self, out):
        super().__init__()
        self.out = Path(out)

    def notify(self, algorithm):
        pop, gen = algorithm.pop, algorithm.n_gen
        (self.out/f'gen_{gen:03d}.json').write_text(json.dumps(
            {'generation': gen, 'X': pop.get('X').tolist(), 'F': pop.get('F').tolist()}))
        F = pop.get('F')
        print(f'gen {gen}: best P={-F[:, 0].min():.3f}  min mass={F[:, 3].min():.1f} g', flush=True)


def select(F, weights):
    F = np.asarray(F, dtype=float)
    lo, hi = F.min(0), F.max(0)
    N = (F - lo)/np.where(hi > lo, hi - lo, 1.)
    w = np.array([weights[k] for k in ('p', 'f', 't', 'm', 'c')])
    score = w[0]*(1 - N[:, 0]) - (N[:, 1:]*w[1:]).sum(1)
    return int(np.argmax(score)), score


def seed_population(names, size, rng, base, spread=0.12):
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])
    x0 = np.clip([getattr(base, n) for n in names], lo, hi)
    rows = [x0]
    while len(rows) < size:
        rows.append(np.clip(x0 + rng.normal(0, spread, len(names))*(hi - lo), lo, hi))
    return np.array(rows)


def run(out, pop_size=24, generations=10, partitions=3, workers=8, seed=0, ec=EvalConfig(), base=V0, names=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(names or BOUNDS)
    cons = list(R.rules(base, cad.build_brick(base))) + ['geometry_builds']
    ref_dirs = get_reference_directions('das-dennis', len(OBJECTIVES), n_partitions=partitions)
    pop_size = max(pop_size, len(ref_dirs))
    (out/'config.json').write_text(json.dumps({'names': names, 'constraints': cons, 'pop_size': pop_size,
                                               'generations': generations, 'base': base.to_dict(),
                                               'eval': asdict(ec)}, indent=2))
    with get_context('forkserver').Pool(workers) as pool:
        problem = BrickProblem(names, ec, base, cons, str(out/'evaluations'),
                               elementwise_runner=StarmapParallelization(pool.starmap))
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size,
                          sampling=seed_population(names, pop_size, np.random.default_rng(seed), base))
        res = minimize(problem, algorithm, ('n_gen', generations), seed=seed, callback=Checkpoint(out))
    if res.F is None:
        result = {'status': 'no_feasible_solution'}
    else:
        X, F = np.atleast_2d(res.X), np.atleast_2d(res.F)
        best, scores = select(F, ec.weights)
        front = [{'params': with_vector(base, names, x).to_dict(), 'objectives': dict(zip(OBJECTIVES, map(float, f))),
                  'score': float(s)} for x, f, s in zip(X, F, scores)]
        result = {'status': 'ok', 'chosen': front[best], 'pareto_front': front}
        chosen = BrickParams.from_dict(front[best]['params'])
        cad.export(cad.build_brick(chosen), out/'chosen')
        (out/'chosen'/'params.json').write_text(json.dumps(chosen.to_dict(), indent=2))
    (out/'result.json').write_text(json.dumps(result, indent=2))
    return result
