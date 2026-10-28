"""Command line: export, rules, size, animate, drop (with capture sweeps)."""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

from brixzle import approach_error as AE
from brixzle.config import Physics, read_json, write_json
from brixzle.trials import ErrorModel, TrialConfig, as_dict, summarize_drops

from . import cad, geometry as G, rules as R
from .params import DEFAULT, BBParams


def load_params(path, overrides=()):
    """DEFAULT overridden by a JSON object (partial dicts allowed; a 'params' key is unwrapped), then by
    ``key=value`` strings."""
    values = {}
    if path:
        values = read_json(path)
        values = values.get('params', values)
    types = {f.name: f.type for f in fields(BBParams)}
    for item in overrides or ():
        key, _, raw = item.partition('=')
        if key not in types:
            sys.exit(f'unknown parameter {key!r}')
        values[key] = raw if types[key] in (str, 'str') else float(raw)
    return BBParams.from_dict({**DEFAULT.to_dict(), **{k: v for k, v in values.items() if k != 'derived'}})


def _new_dir(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'Output directory {out} is not empty; choose a new one')
    out.mkdir(parents=True, exist_ok=True)
    return out


def _trial_config(args):
    physics = Physics(timestep=args.timestep, friction=args.mu, contact_timeconst=args.timeconst, iterations=50)
    # aim_bias belongs to A/B's aim; the biasbrick aim is geometry.aim, added in trials.
    error = ErrorModel(ideal=getattr(args, 'ideal', False), aim_bias=0.)
    if getattr(args, 'clearance', None):
        error = replace(error, clearance=tuple(args.clearance))
    if getattr(args, 'approach_error', None):
        error = AE.error_model(AE.load(args.approach_error), args.error_stat, base=error, config=args.approach_error)
    return TrialConfig(physics=physics, error=error)


def _settings(args, p):
    """Rules check the measured error if given, else the error the design was sized for."""
    if getattr(args, 'approach_error', None):
        return R.settings_for_tolerance(AE.required_tolerance(AE.load(args.approach_error), args.error_stat))
    return R.settings_for_tolerance(p.aim_r)


def _report(p, bundle, settings):
    landings = {below: dict(zip(('near', 'far'), R.landing(p, settings.placement_error, below))) for below in ('A', 'B')}
    return {'params': p.to_dict(), 'derived': p.derived(), 'mass_g': bundle['mass_g'], 'com_mm': bundle['com'],
            'aim_mm': {below: G.aim(p, below) for below in ('A', 'B')}, 'landing_mm': landings,
            'x_band_mm': {below: G.x_band(p, below) for below in ('A', 'B')},
            'tines_mm': bundle['channels'], 'sleeve_y_mm': bundle['sleeve_y'],
            'rules': R.rules(p, bundle, settings), 'rule_settings': settings.__dict__}


def cmd_export(args):
    p = load_params(args.params)
    if args.part:
        p = replace(p, part=args.part)
    out = _new_dir(args.out)
    bundle = cad.build_brick(p)
    cad.export(bundle, out)
    report = _report(p, bundle, _settings(args, p))
    report.update(watertight=bool(bundle['mesh'].is_watertight), fragments=bundle['fragments'],
                  volume_mm3=bundle['volume'], collision_volume_mm3=cad.collision_volume(bundle),
                  bounds_mm=bundle['mesh'].bounds.tolist())
    write_json(out/'geometry.json', report)
    write_json(out/'params.json', p.to_dict())
    from . import render
    render.views(p, bundle, out/'views.png')
    render.stack(p, bundle, out/'stack.png')
    render.sections(p, bundle, out/'sections.png')
    print(f'{bundle["mass_g"]:.2f} g, feasible={R.feasible(report["rules"])}; wrote {out}')


def cmd_rules(args):
    p = load_params(args.params)
    margins = R.rules(p, cad.build_brick(p), _settings(args, p))
    for k, v in margins.items():
        print(f'{"ok " if v >= 0 else "FAIL"} {k:24s} {v:9.2f}')


def cmd_size(args):
    tol = AE.required_tolerance(AE.load(args.approach_error), args.error_stat)
    p = R.size_for_tolerance(tol, load_params(args.params, args.set))
    report = {'params': p.to_dict(), 'derived': p.derived(),
              'sized_for': {'statistic': args.error_stat, 'tolerance_mm': tol, 'config': args.approach_error}}
    if args.out:
        out = Path(args.out)
        if out.exists():
            sys.exit(f'{out} exists; choose a new file')
        write_json(out, report)
    print(f'U = {p.U} mm, flange {p.flange} mm, aim y {p.aim_y:.2f} mm (rod {p.rod_L} mm) '
          f'for {args.error_stat} {tol:.2f} mm')


def cmd_animate(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    from .animate import animate
    report = animate(p, out, step=args.step)
    write_json(out/'animation.json', {'params': p.to_dict(), 'kind': 'kinematic, not simulated', **report})
    print(f'{report["frames"]} frames; wrote {out}')


def cmd_drop(args):
    from . import trials as T
    p = load_params(args.params, args.set)
    out = _new_dir(args.out)
    bricks = T.bundles(p)
    cfg = _trial_config(args)
    summary = {'aim_mm': G.aim(p, 'A'), 'on_brick': args.on_brick}
    xml = None
    if args.samples:
        rows, xml = T.drop_trials(p, bricks, cfg, samples=args.samples, seed=args.seed, record=args.video > 0,
                                  on_brick=args.on_brick)
        summary.update(summarize_drops(rows))
        write_json(out/'drops.json', [{k: v for k, v in r.items() if k != 'frames'} for r in rows])
    if args.sweep:
        ax, ay = G.aim(p, 'A')
        for axis, around in (('dx', ax), ('dy', ay), ('yaw', 0.)):
            sweep, _ = T.drop_trials(p, bricks, cfg, offsets=T.sweep_offsets(p, axis, step=args.sweep_step),
                                     on_brick=args.on_brick)
            summary[f'capture_sweep_{axis}'] = T.capture_band(sweep, axis, around)
            summary[f'sweep_{axis}'] = [(r[axis], r['outcome']) for r in sweep]
    if args.map:
        dxs = [float(v) for v in __import__('numpy').arange(-16., 26. + 1e-9, args.map_step)]
        dys = [float(v) for v in __import__('numpy').arange(-4., p.D/2 + p.flange + 1e-9, args.map_step)]
        cells, _ = T.drop_trials(p, bricks, cfg, offsets=T.map_offsets(dxs, dys), on_brick=args.on_brick)
        write_json(out/'map.json', [{k: r[k] for k in ('dx', 'dy', 'outcome', 'err_mm', 'angle_deg')} for r in cells])
        summary['map_seated'] = sum(r['outcome'] == 'seated' for r in cells)/len(cells)
        summary['best_aim'] = {}
        for radius in args.map_radius:
            aim, score = T.best_aim(cells, radius)
            summary['best_aim'][str(radius)] = {'aim_mm': aim, 'ring_score': score,
                                                'current_aim_score': T.ring_score(cells, G.aim(p, 'A'), radius)}
        from .render import capture_map
        capture_map(p, cells, summary['best_aim'], out/'map.png')
    summary.update(mass_g={k: b['mass_g'] for k, b in bricks.items()}, params=p.to_dict(), config=as_dict(cfg),
                   seed=args.seed)
    write_json(out/'summary.json', summary)
    if xml:
        (out/'scene.xml').write_text(xml)
    if args.video and args.samples:
        import mujoco
        from brixzle.render import render
        model = mujoco.MjModel.from_xml_string(xml)
        failures = [i for i, r in enumerate(rows) if r['outcome'] != 'seated'][:args.video]
        for i in failures + [0]:
            render(model, rows[i]['frames'], out/f'drop_{i:03d}_{rows[i]["outcome"]}.mp4', azimuth=-120, elevation=-20)
    keys = ('p_success', 'outcomes', 'f_max_N', 'capture_sweep_dx', 'capture_sweep_dy', 'capture_sweep_yaw',
            'map_seated', 'best_aim')
    print(json.dumps({k: summary[k] for k in keys if k in summary}, indent=1))


def cmd_assemble(args):
    from . import structures as S, trials as T
    p = load_params(args.params, args.set)
    out = _new_dir(args.out)
    catalog = S.catalog()
    if args.structure not in catalog:
        sys.exit(f'Unknown structure; choose from {sorted(catalog)}')
    structure = catalog[args.structure]()
    bricks = T.bundles(p)
    frames = [] if args.video else None
    cfg = _trial_config(args)
    result, (model, data, info, xml) = T.assemble(p, bricks, structure, cfg, seed=args.seed,
                                                  stop_on_fail=not args.keep_going, record=frames, place=args.place)
    result.update(config=as_dict(cfg), params=p.to_dict(), seed=args.seed, placement='placed' if args.place else 'dropped',
                  mass_g={k: b['mass_g'] for k, b in bricks.items()})
    write_json(out/'summary.json', result)
    (out/'scene.xml').write_text(xml)
    if args.video:
        from brixzle.render import render
        render(model, frames, out/'assembly.mp4', azimuth=-120, elevation=-20)
        render(model, frames, out/'final.png', azimuth=-120, elevation=-20)
    print(f'{result["structure"]}: {result["status"]}, {result["stable_bricks"]}/{result["total"]} stable, '
          f'max sag {result["max_sag_mm"]:.2f} mm')


def cmd_push(args):
    from . import structures as S, trials as T
    p = load_params(args.params, args.set)
    out = _new_dir(args.out)
    res = {d: T.push_test(p, T.bundles(p), S.tower(args.height), _trial_config(args), direction=v)
           for d, v in (('+x', (1, 0, 0)), ('-x', (-1, 0, 0)), ('+y', (0, 1, 0)), ('-y', (0, -1, 0)))}
    write_json(out/'summary.json', {'push': res, 'height': args.height, 'params': p.to_dict()})
    print(json.dumps(res))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='biasbrick')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def error_args(sp, required=False):
        sp.add_argument('--approach-error', required=required,
                        help='measured X-Y error: Exp 3 metrics.json or ../brixzle/configs/error-*.json')
        sp.add_argument('--error-stat', choices=AE.STATISTICS, default='mean')

    sp = sub.add_parser('export', help='STL, STEP, views, stack, sections and rule margins')
    sp.add_argument('--params')
    sp.add_argument('--part', choices=['A', 'B'])
    sp.add_argument('--out', required=True)
    error_args(sp)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser('rules', help='analytic rule margins (>= 0 is OK)')
    sp.add_argument('--params')
    error_args(sp)
    sp.set_defaults(func=cmd_rules)

    sp = sub.add_parser('size', help='smallest U whose aim bias and catchment cover the measured error')
    sp.add_argument('--params')
    sp.add_argument('--set', action='append', default=[], help='override a parameter before sizing: key=value')
    sp.add_argument('--out')
    error_args(sp, required=True)
    sp.set_defaults(func=cmd_size)

    sp = sub.add_parser('animate', help='kinematic video of one brick stacking onto another (no physics)')
    sp.add_argument('--params')
    sp.add_argument('--step', type=float, default=0.4, help='mm per frame along the slide')
    sp.add_argument('--out', required=True)
    sp.set_defaults(func=cmd_animate)

    sp = sub.add_parser('drop', help='drop Monte Carlo onto the base (A) or a seated A (B), and capture sweeps')
    sp.add_argument('--params')
    sp.add_argument('--set', action='append', default=[], help='override a parameter: key=value')
    sp.add_argument('--samples', type=int, default=96, help='Monte Carlo drops (0: sweeps only)')
    sp.add_argument('--seed', type=int, default=0)
    sp.add_argument('--on-brick', action='store_true', help='drop a B onto an ideally seated A')
    sp.add_argument('--ideal', action='store_true', help='no placement error: drop at the aim point')
    sp.add_argument('--sweep', action='store_true', help='also run capture sweeps in dx, dy and yaw around the aim')
    sp.add_argument('--sweep-step', type=float, default=1.0)
    sp.add_argument('--map', action='store_true', help='2D dx x dy capture map (ideal otherwise) and the best aim')
    sp.add_argument('--map-step', type=float, default=2.0)
    sp.add_argument('--map-radius', type=float, action='append', default=[],
                    help='error radius (mm) to pick the best aim for; repeatable')
    sp.add_argument('--mu', type=float, default=0.35)
    sp.add_argument('--clearance', type=float, nargs=2, metavar=('LO', 'HI'),
                    help='release height range (mm) above the highest reach (default: brixzle 5 30, unmeasured)')
    sp.add_argument('--timestep', type=float, default=0.001)
    sp.add_argument('--timeconst', type=float, default=0.004)
    sp.add_argument('--video', type=int, default=0, help='render up to N failed drops (plus drop 0)')
    sp.add_argument('--out', required=True)
    error_args(sp)
    sp.set_defaults(func=cmd_drop)

    def trial_args(sp):
        sp.add_argument('--params')
        sp.add_argument('--set', action='append', default=[], help='override a parameter: key=value')
        sp.add_argument('--mu', type=float, default=0.35)
        sp.add_argument('--clearance', type=float, nargs=2, metavar=('LO', 'HI'))
        sp.add_argument('--timestep', type=float, default=0.001)
        sp.add_argument('--timeconst', type=float, default=0.004)
        sp.add_argument('--ideal', action='store_true', help='no placement error: drop at the aim point')
        sp.add_argument('--out', required=True)
        error_args(sp)

    sp = sub.add_parser('assemble', help='build a structure brick by brick (dropped, or --place for carried)')
    sp.add_argument('structure')
    sp.add_argument('--seed', type=int, default=0)
    sp.add_argument('--keep-going', action='store_true', help='do not stop at the first failed placement')
    sp.add_argument('--place', action='store_true', help='set each brick at its seat at rest (ideal carried placement)')
    sp.add_argument('--video', action='store_true')
    trial_args(sp)
    sp.set_defaults(func=cmd_assemble)

    sp = sub.add_parser('push', help='ideal tower, then a ramped lateral force on its top brick (+-x, +-y)')
    sp.add_argument('--height', type=int, default=6)
    trial_args(sp)
    sp.set_defaults(func=cmd_push)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
