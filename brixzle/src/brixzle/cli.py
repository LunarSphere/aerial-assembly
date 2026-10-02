"""Command line: export, rules, drop, assemble, push, optimize."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

from aerial_assembly.config import Physics, read_json, write_json

from . import cad, profile as pr, rules as R, structures as S, trials as T
from .params import V0, BrickParams


def load_params(path):
    """V0 overridden by a JSON object (partial dicts allowed; a 'params' key is unwrapped)."""
    if not path:
        return V0
    values = read_json(path)
    values = values.get('params', values)
    return replace(V0, **BrickParams.from_dict({**V0.to_dict(), **values}).to_dict())


def _new_dir(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'Output directory {out} is not empty; choose a new one')
    out.mkdir(parents=True, exist_ok=True)
    return out


def _trial_config(args):
    physics = Physics(timestep=args.timestep, friction=args.mu, contact_timeconst=args.timeconst, iterations=50)
    error = T.ErrorModel(ideal=getattr(args, 'ideal', False))
    if getattr(args, 'sigma', None) is not None:
        error = replace(error, sigma_xy=args.sigma)
    return T.TrialConfig(physics=physics, error=error)


def _plot_profile(p, bundle, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely import affinity
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))

    def draw(a, g, **kw):
        for q in ([g] if g.geom_type == 'Polygon' else g.geoms):
            x, y = q.exterior.xy
            a.fill(x, y, **kw)
            a.plot(x, y, 'k', lw=.5)
    draw(ax[0], bundle['outer'], color='tab:blue')
    for h in bundle['holes']:
        draw(ax[0], h, color='w')
    for c in bundle['channels']:
        draw(ax[0], pr.channel_footprint(p, c), color='tab:red', alpha=.6)
    ax[0].set_title(f'{p.lean_mode} brick profile at y=0 ({bundle["mass_g"]:.1f} g)')
    base, _ = pr.base_outer(p, 5, p.tooth_L + p.slot_extra + 4)
    draw(ax[1], base, color='0.6')
    for k, starts in enumerate([[0, 2], [1, 3], [0, 2], [1]]):
        flip = k % 2 and not p.uniform
        g = pr.mirror_x(bundle['outer']) if flip else bundle['outer']
        for i0 in starts:
            draw(ax[1], affinity.translate(g, (i0 + 1)*p.U, k*p.H0), color=['tab:blue', 'tab:orange'][k % 2], alpha=.85)
    ax[1].set_title('running bond on anchored base')
    for a in ax:
        a.set_aspect('equal')
        a.grid(True, alpha=.4)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def cmd_export(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    bundle = cad.build_brick(p)
    cad.export(bundle, out)
    margins = R.rules(p, bundle)
    write_json(out/'params.json', p.to_dict())
    write_json(out/'rules.json', margins)
    write_json(out/'geometry.json', {'mass_g': bundle['mass_g'], 'volume_mm3': bundle['volume'],
                                     'com_mm': bundle['com'], 'tines_mm': bundle['channels'],
                                     'collision_pieces': len(bundle['collision']),
                                     'collision_volume_mm3': cad.collision_volume(bundle),
                                     'outer_volume_mm3': bundle['profile_area']*p.D,
                                     'bounds_mm': bundle['mesh'].bounds.tolist(),
                                     'amplitude_mm': p.amplitude})
    _plot_profile(p, bundle, out/'profile.png')
    print(f'{bundle["mass_g"]:.2f} g, feasible={R.feasible(margins)}; wrote {out}')


def cmd_rules(args):
    p = load_params(args.params)
    margins = R.rules(p, cad.build_brick(p, solid=False))
    for k, v in margins.items():
        print(f'{"ok " if v >= 0 else "FAIL"} {k:24s} {v:9.2f}')


def cmd_drop(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    bundle = cad.build_brick(p)
    cfg = _trial_config(args)
    rows, xml = T.drop_trials(p, bundle, cfg, samples=args.samples, seed=args.seed, record=args.video > 0)
    sweep, _ = T.drop_trials(p, bundle, cfg, offsets=np.arange(-p.U/2 - 4, p.U/2 + 4.1, 1.0))
    summary = T.summarize_drops(rows)
    seated = [abs(r['dx']) for r in sweep if r['outcome'] == 'seated']
    ok = [r['dx'] for r in sweep if r['outcome'] == 'seated']
    summary['capture_sweep_mm'] = [min(ok), max(ok)] if ok else None
    summary['capture_sweep'] = [(r['dx'], r['outcome']) for r in sweep]
    summary['mass_g'] = bundle['mass_g']
    summary['config'] = T.as_dict(cfg)
    write_json(out/'summary.json', summary)
    write_json(out/'drops.json', [{k: v for k, v in r.items() if k != 'frames'} for r in rows])
    (out/'scene.xml').write_text(xml)
    if args.video:
        import mujoco
        from .render import render
        model = mujoco.MjModel.from_xml_string(xml)
        failures = [i for i, r in enumerate(rows) if r['outcome'] != 'seated'][:args.video]
        for i in failures + [0]:
            render(model, rows[i]['frames'], out/f'drop_{i:03d}_{rows[i]["outcome"]}.mp4', azimuth=90, elevation=-5)
    print(json.dumps({k: summary[k] for k in ('p_success', 'outcomes', 'f_max_N', 't_settle_s',
                                              'collisions', 'capture_sweep_mm')}, indent=1))


def cmd_assemble(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    bundle = cad.build_brick(p)
    if args.structure not in S.CATALOG:
        sys.exit(f'Unknown structure; choose from {sorted(S.CATALOG)}')
    structure = S.CATALOG[args.structure]()
    frames = [] if args.video else None
    result, (model, data, info, xml) = T.assemble(p, bundle, structure, _trial_config(args), seed=args.seed,
                                                  stop_on_fail=not args.keep_going, record=frames)
    result['config'] = T.as_dict(_trial_config(args))
    write_json(out/'summary.json', result)
    (out/'scene.xml').write_text(xml)
    if args.video:
        from .render import render
        render(model, frames, out/'assembly.mp4', azimuth=90, elevation=-5)
        render(model, frames, out/'final.png', azimuth=120, elevation=-15)
    print(f'{result["structure"]}: {result["status"]}, {result["stable_bricks"]}/{result["total"]} stable, '
          f'max sag {result["max_sag_mm"]:.2f} mm')


def cmd_push(args):
    p = load_params(args.params)
    bundle = cad.build_brick(p)
    res = T.push_test(p, bundle, S.tower(args.height), _trial_config(args))
    print(json.dumps(res))


def cmd_optimize(args):
    from . import optimize as O
    cfg = read_json(args.config) if args.config else {}
    ec = O.EvalConfig.from_dict(cfg.get('eval', {}))
    run = cfg.get('run', {})
    for k in ('pop', 'gens', 'workers', 'partitions', 'seed'):
        if getattr(args, k) is not None:
            run[k] = getattr(args, k)
    _new_dir(args.out)
    result = O.run(args.out, pop_size=run.get('pop', 24), generations=run.get('gens', 10),
                   partitions=run.get('partitions', 3), workers=run.get('workers', 8),
                   seed=run.get('seed', 0), ec=ec)
    if result['status'] == 'ok':
        print(json.dumps(result['chosen'], indent=1))
    else:
        print(result['status'])


def parser():
    ap = argparse.ArgumentParser(prog='brixzle')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(sp, out=True):
        sp.add_argument('--params', help='JSON of BrickParams overrides (default: V0)')
        if out:
            sp.add_argument('--out', required=True)
        sp.add_argument('--mu', type=float, default=0.35)
        sp.add_argument('--timestep', type=float, default=0.001)
        sp.add_argument('--timeconst', type=float, default=0.004)
        sp.add_argument('--seed', type=int, default=0)
        sp.add_argument('--sigma', type=float, help='placement sigma (mm) in X and Y')

    sp = sub.add_parser('export', help='CAD solid, STL/STEP, profile plot, rule margins')
    sp.add_argument('--params')
    sp.add_argument('--out', required=True)
    sp.set_defaults(func=cmd_export)
    sp = sub.add_parser('rules', help='print analytic design-rule margins')
    sp.add_argument('--params')
    sp.set_defaults(func=cmd_rules)
    sp = sub.add_parser('drop', help='Monte Carlo coarse-placement drops + capture sweep')
    common(sp)
    sp.add_argument('--samples', type=int, default=64)
    sp.add_argument('--ideal', action='store_true')
    sp.add_argument('--video', type=int, default=0, help='render up to N failed drops (plus drop 0)')
    sp.set_defaults(func=cmd_drop)
    sp = sub.add_parser('assemble', help='sequential assembly of a catalog structure')
    sp.add_argument('structure')
    common(sp)
    sp.add_argument('--ideal', action='store_true', help='no placement error (aim bias only)')
    sp.add_argument('--keep-going', action='store_true')
    sp.add_argument('--video', action='store_true')
    sp.set_defaults(func=cmd_assemble)
    sp = sub.add_parser('push', help='ramped lateral push on an ideal tower')
    common(sp, out=False)
    sp.add_argument('--height', type=int, default=6)
    sp.set_defaults(func=cmd_push)
    sp = sub.add_parser('optimize', help='NSGA-III over brick parameters')
    sp.add_argument('--config')
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
