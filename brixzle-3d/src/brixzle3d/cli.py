"""Command line: export, rules, drop, assemble, push, optimize."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from . import cad, rules as R, structures as S, trials as T
from .params import V0, BrickParams
from .scene import Physics


def load_params(path):
    if not path:
        return V0
    values = json.loads(Path(path).read_text())
    values = values.get('params', values)
    return BrickParams.from_dict({**V0.to_dict(), **values})


def _new_dir(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'Output directory {out} is not empty; choose a new one')
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=float) + '\n')


def _cfg(args):
    err = T.ErrorModel(ideal=getattr(args, 'ideal', False))
    if args.sigma is not None:
        err = replace(err, sigma_xy=args.sigma)
    return T.TrialConfig(physics=Physics(friction=args.mu), error=err)


def cmd_export(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    b = cad.build_brick(p)
    cad.export(b, out, step=not args.no_step)
    margins = R.rules(p, b)
    _write(out/'params.json', p.to_dict())
    _write(out/'rules.json', margins)
    _write(out/'geometry.json', {'mass_g': b['mass_g'], 'solid_volume_mm3': b['volume'],
                                 'printed_volume_mm3': b['printed_volume'], 'com_mm': b['com'],
                                 'tine_bores': b['bores'], 'collision_pieces': len(b['collision']),
                                 'bounds_mm': b['mesh'].bounds.tolist()})
    print(f'{b["mass_g"]:.2f} g printed, feasible={R.feasible(margins)}; wrote {out}')


def cmd_rules(args):
    p = load_params(args.params)
    for k, v in R.rules(p, cad.build_brick(p)).items():
        print(f'{"ok " if v >= 0 else "FAIL"} {k:20s} {v:8.2f}')


def cmd_drop(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    b = cad.build_brick(p)
    cfg = _cfg(args)
    rows, xml = T.drop_trials(p, b, cfg, samples=args.samples, seed=args.seed, record=args.video > 0)
    grid = np.arange(-12, 12.1, 2.0)
    sweep, _ = T.drop_trials(p, b, cfg, offsets=[(x, y) for x in grid for y in grid])
    summary = T.summarize_drops(rows)
    summary['capture_grid'] = {f'{r["dx"]:+.0f},{r["dy"]:+.0f}': r['outcome'] for r in sweep}
    summary['capture_square_mm'] = max([abs(r['dx']) for r in sweep
                                        if all(s['outcome'] == 'seated' for s in sweep
                                               if max(abs(s['dx']), abs(s['dy'])) <= abs(r['dx']))], default=0.)
    summary['mass_g'] = b['mass_g']
    summary['config'] = T.as_dict(cfg)
    _write(out/'summary.json', summary)
    _write(out/'drops.json', [{k: v for k, v in r.items() if k != 'frames'} for r in rows])
    (out/'scene.xml').write_text(xml)
    if args.video:
        import mujoco
        from .render import render
        model = mujoco.MjModel.from_xml_string(xml)
        picks = [i for i, r in enumerate(rows) if r['outcome'] != 'seated'][:args.video] + [0]
        for i in picks:
            render(model, rows[i]['frames'], out/f'drop_{i:03d}_{rows[i]["outcome"]}.mp4', azimuth=135, elevation=-25)
    print(json.dumps({k: summary[k] for k in ('p_success', 'outcomes', 'f_max_N', 't_settle_s',
                                              'collisions', 'capture_square_mm')}, indent=1))


def cmd_assemble(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    if args.structure not in S.CATALOG:
        sys.exit(f'Unknown structure; choose from {sorted(S.CATALOG)}')
    b = cad.build_brick(p)
    frames = [] if args.video else None
    result, (model, data, info, xml) = T.assemble(p, b, S.CATALOG[args.structure](), _cfg(args), seed=args.seed,
                                                  stop_on_fail=not args.keep_going, record=frames)
    result['config'] = T.as_dict(_cfg(args))
    _write(out/'summary.json', result)
    (out/'scene.xml').write_text(xml)
    if args.video:
        from .render import render
        render(model, frames, out/'assembly.mp4', azimuth=135, elevation=-20)
        render(model, frames, out/'final.png', azimuth=135, elevation=-20)
    seated = sum(s['outcome'] == 'seated' for s in result['stages'])
    print(f'{result["structure"]}: {result["status"]}, {result["stable_bricks"]}/{result["total"]} stable, '
          f'{seated}/{len(result["stages"])} placements seated, max sag {result["max_sag_mm"]:.2f} mm')


def cmd_push(args):
    p = load_params(args.params)
    print(json.dumps(T.push_test(p, cad.build_brick(p), S.CATALOG[args.structure](), _cfg(args))))


def cmd_optimize(args):
    from . import optimize as O
    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    run = cfg.get('run', {})
    for k in ('pop', 'gens', 'workers', 'partitions', 'seed'):
        if getattr(args, k) is not None:
            run[k] = getattr(args, k)
    _new_dir(args.out)
    result = O.run(args.out, pop_size=run.get('pop', 24), generations=run.get('gens', 10),
                   partitions=run.get('partitions', 3), workers=run.get('workers', 8), seed=run.get('seed', 0),
                   ec=O.EvalConfig.from_dict(cfg.get('eval', {})), base=load_params(args.params))
    print(json.dumps(result.get('chosen', result), indent=1))


def parser():
    ap = argparse.ArgumentParser(prog='brixzle3d')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(sp, out=True):
        sp.add_argument('--params')
        if out:
            sp.add_argument('--out', required=True)
        sp.add_argument('--mu', type=float, default=0.35)
        sp.add_argument('--seed', type=int, default=0)
        sp.add_argument('--sigma', type=float, help='placement sigma (mm) in X and Y')

    sp = sub.add_parser('export', help='STL/STEP, mass model, rule margins')
    sp.add_argument('--params')
    sp.add_argument('--out', required=True)
    sp.add_argument('--no-step', action='store_true')
    sp.set_defaults(func=cmd_export)
    sp = sub.add_parser('rules')
    sp.add_argument('--params')
    sp.set_defaults(func=cmd_rules)
    sp = sub.add_parser('drop', help='coarse-placement Monte Carlo + XY capture grid')
    common(sp)
    sp.add_argument('--samples', type=int, default=64)
    sp.add_argument('--video', type=int, default=0)
    sp.set_defaults(func=cmd_drop)
    sp = sub.add_parser('assemble')
    sp.add_argument('structure')
    common(sp)
    sp.add_argument('--ideal', action='store_true')
    sp.add_argument('--keep-going', action='store_true')
    sp.add_argument('--video', action='store_true')
    sp.set_defaults(func=cmd_assemble)
    sp = sub.add_parser('push')
    sp.add_argument('structure', nargs='?', default='column')
    common(sp, out=False)
    sp.set_defaults(func=cmd_push)
    sp = sub.add_parser('optimize')
    sp.add_argument('--config')
    sp.add_argument('--params', help='base design (non-optimised fields)')
    sp.add_argument('--out', required=True)
    for k in ('pop', 'gens', 'workers', 'partitions', 'seed'):
        sp.add_argument(f'--{k}', type=int)
    sp.set_defaults(func=cmd_optimize)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
