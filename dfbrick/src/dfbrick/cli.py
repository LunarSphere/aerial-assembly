"""Command line: export, rules, size, drop, assemble, push, fly-assemble (as brixzle's, for DF bricks)."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from brixzle import approach_error as AE
from brixzle.config import Physics, read_json, write_json
from brixzle.trials import ErrorModel, TrialConfig, as_dict, summarize_drops

from . import cad, rules as R, structures as S, trials as T
from .params import DEFAULT, DFParams


def load_params(path):
    """DEFAULT overridden by a JSON object (partial dicts allowed; a 'params' key is unwrapped)."""
    if not path:
        return DEFAULT
    values = read_json(path)
    values = values.get('params', values)
    return DFParams.from_dict({**DEFAULT.to_dict(), **{k: v for k, v in values.items() if k != 'derived'}})


def _new_dir(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'Output directory {out} is not empty; choose a new one')
    out.mkdir(parents=True, exist_ok=True)
    return out


def _trial_config(args):
    physics = Physics(timestep=args.timestep, friction=args.mu, contact_timeconst=args.timeconst, iterations=50)
    # aim_bias exists for A/B's tooth stroke; DF insertion is vertical, so it aims at the seat.
    error = ErrorModel(ideal=getattr(args, 'ideal', False), track_supporter=getattr(args, 'track', False),
                       exact_courses=getattr(args, 'exact_courses', 0), aim_bias=0.)
    if getattr(args, 'sigma', None) is not None:
        error = replace(error, sigma_xy=args.sigma)
    if getattr(args, 'approach_error', None):
        error = AE.error_model(AE.load(args.approach_error), args.error_stat, base=error,
                               config=args.approach_error)
    return TrialConfig(physics=physics, error=error)


def _settings(args):
    if getattr(args, 'approach_error', None):
        return R.settings_for_tolerance(AE.required_tolerance(AE.load(args.approach_error), args.error_stat))
    return R.RuleSettings()


def _report(p, bundle, settings):
    return {'params': p.to_dict(), 'derived': p.derived(), 'mass_g': bundle['mass_g'], 'com_mm': bundle['com'],
            'capture_mm': R.capture(p), 'tines_mm': bundle['channels'], 'rules': R.rules(p, bundle, settings),
            'rule_settings': settings.__dict__}


def cmd_export(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    bundle = cad.build_brick(p)
    cad.export(bundle, out)
    report = _report(p, bundle, _settings(args))
    report.update(watertight=bool(bundle['mesh'].is_watertight), fragments=bundle['fragments'],
                  volume_mm3=bundle['volume'], collision_volume_mm3=cad.collision_volume(bundle),
                  bounds_mm=bundle['mesh'].bounds.tolist())
    write_json(out/'geometry.json', report)
    write_json(out/'params.json', p.to_dict())
    from .render import views
    views(p, bundle, out/'views.png')
    print(f'{bundle["mass_g"]:.2f} g, feasible={R.feasible(report["rules"])}; wrote {out}')


def cmd_rules(args):
    p = load_params(args.params)
    margins = R.rules(p, cad.build_brick(p), _settings(args))
    for k, v in margins.items():
        print(f'{"ok " if v >= 0 else "FAIL"} {k:24s} {v:9.2f}')


def cmd_size(args):
    summary = AE.load(args.approach_error)
    tol = AE.required_tolerance(summary, args.error_stat)
    p = R.size_for_tolerance(tol, load_params(args.params), margin=args.margin, yaw_deg=args.yaw)
    bundle = cad.build_brick(p)
    report = _report(p, bundle, R.settings_for_tolerance(tol))
    report['sized_for'] = {'statistic': args.error_stat, 'tolerance_mm': tol, 'margin_mm': args.margin,
                           'yaw_deg': args.yaw,
                           'approach_error': args.approach_error, 'source': summary['source']}
    report['feasible'] = R.feasible(report['rules'])
    write_json(args.out, report)
    failing = {k: round(float(v), 2) for k, v in report['rules'].items() if v < 0}
    print(f'U = {p.U} mm, {bundle["mass_g"]:.1f} g, capture {R.capture(p):.1f} mm for {tol:.1f} mm; '
          f'failing rules: {failing or "none"}; wrote {args.out}')


def cmd_drop(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    brick = cad.build_brick(p)
    cfg = _trial_config(args)
    rows, xml = T.drop_trials(p, brick, cfg, samples=args.samples, seed=args.seed, record=args.video > 0,
                              on_brick=args.on_brick)
    summary = summarize_drops(rows)
    span = R.capture(p) + p.U/2
    for axis, rng in (('dx', span), ('dy', span), ('yaw', 30.)):
        sweep, _ = T.drop_trials(p, brick, cfg, offsets=T.sweep_offsets(axis, rng, 1.0), on_brick=args.on_brick)
        summary[f'capture_sweep_{axis}'] = T.capture_band(sweep, axis)
        summary[f'sweep_{axis}'] = [(r[axis], r['outcome']) for r in sweep]
    summary['mass_g'] = brick['mass_g']
    summary['params'] = p.to_dict()
    summary['config'] = as_dict(cfg)
    write_json(out/'summary.json', summary)
    write_json(out/'drops.json', [{k: v for k, v in r.items() if k != 'frames'} for r in rows])
    (out/'scene.xml').write_text(xml)
    if args.video:
        import mujoco
        from brixzle.render import render
        model = mujoco.MjModel.from_xml_string(xml)
        failures = [i for i, r in enumerate(rows) if r['outcome'] != 'seated'][:args.video]
        for i in failures + [0]:
            render(model, rows[i]['frames'], out/f'drop_{i:03d}_{rows[i]["outcome"]}.mp4', azimuth=60, elevation=-20)
    print(json.dumps({k: summary[k] for k in ('p_success', 'outcomes', 'f_max_N', 'capture_sweep_dx',
                                              'capture_sweep_dy', 'capture_sweep_yaw')}, indent=1))


def cmd_assemble(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    catalog = S.catalog(p)
    if args.structure not in catalog:
        sys.exit(f'Unknown structure; choose from {sorted(catalog)}')
    structure = catalog[args.structure]()
    brick = cad.build_brick(p)
    frames = [] if args.video else None
    cfg = _trial_config(args)
    result, (model, data, info, xml) = T.assemble(p, brick, structure, cfg, seed=args.seed,
                                                  stop_on_fail=not args.keep_going, record=frames)
    result['config'] = as_dict(cfg)
    result['params'] = p.to_dict()
    result['mass_g'] = brick['mass_g']
    write_json(out/'summary.json', result)
    (out/'scene.xml').write_text(xml)
    if args.video:
        from brixzle.render import render
        render(model, frames, out/'assembly.mp4', azimuth=135, elevation=-25)
        render(model, frames, out/'final.png', azimuth=135, elevation=-25)
    print(f'{result["structure"]}: {result["status"]}, {result["stable_bricks"]}/{result["total"]} stable, '
          f'max sag {result["max_sag_mm"]:.2f} mm')


def cmd_push(args):
    p = load_params(args.params)
    res = T.push_test(p, cad.build_brick(p), S.tower(p, args.height), _trial_config(args))
    print(json.dumps(res))


def cmd_fly_assemble(args):
    from brixzle import gripper as Gr
    from brixzle.flyassemble import FlyConfig, config_from_dict
    from . import fly as F
    p = load_params(args.params)
    out = _new_dir(args.out)
    if args.structure not in S.catalog(p):
        sys.exit(f'Unknown structure; choose from {sorted(S.catalog(p))}')
    gp = Gr.params_from_dict(read_json(args.gripper)) if args.gripper else Gr.ForkParams()
    cfg = config_from_dict(read_json(args.config)) if args.config else FlyConfig()
    if args.lighthouse:
        from brixzle.estimator import LighthouseModel
        cfg.lighthouse = LighthouseModel.from_dict(read_json(args.lighthouse))
    cfg.physics = Physics(timestep=args.timestep, friction=args.mu, contact_timeconst=args.timeconst, iterations=50)
    cfg.keep_going = args.keep_going
    error = _trial_config(args).error if args.approach_error else None
    seeds = list(range(args.seed, args.seed + args.seeds))
    plan, layout, brick, gripper, structure = F.make(p, gp, args.structure, cfg, args.bricks)
    write_json(out/'plan.json', plan)
    write_json(out/'config.json', {'params': p.to_dict(), 'gripper': gp.to_dict(), 'fly': cfg.to_dict(),
                                   'seeds': seeds, 'bricks': args.bricks, 'structure': args.structure,
                                   'placement_error': as_dict(TrialConfig(error=error))['error'] if error else None,
                                   'gripper_margins': F.gripper_margins(p, gp, brick, gripper, cfg.drone),
                                   'note': 'simulated; uncalibrated contact and aerodynamics; not hardware rates'})
    if args.video:
        h = F.DFHarness(p, plan, brick, gripper, structure, cfg, seed=seeds[0], record_fps=30, error=error)
        first = h.run()
        from brixzle.render import render
        render(h.model, h.world.frames, out/'flight.mp4', fps=30, azimuth=60, elevation=-20)
        render(h.model, h.world.frames, out/'final.png', azimuth=60, elevation=-20)
        first.pop('events', None)
        first['seed'] = seeds[0]
        results = [first] + F.run_seeds(p, gp, args.structure, cfg, seeds[1:], args.bricks, args.workers, error)
    else:
        results = F.run_seeds(p, gp, args.structure, cfg, seeds, args.bricks, args.workers, error)
    with open(out/'bricks.jsonl', 'w') as f:
        for r in results:
            for b in r['bricks']:
                f.write(json.dumps({'seed': r['seed'], **b}, default=float) + '\n')
    summary = F.summarize(results)
    write_json(out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('p_brick_success', 'p_structure_complete', 'outcomes')}, indent=1))


def parser():
    ap = argparse.ArgumentParser(prog='dfbrick')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def error_args(sp):
        sp.add_argument('--approach-error', help='measured X-Y error: Exp 3 metrics.json or brixzle configs/error-*.json')
        sp.add_argument('--error-stat', choices=['mean', 'max', 'empirical'], default='mean')

    def common(sp, out=True):
        sp.add_argument('--params', help='JSON of DFParams overrides (default: DEFAULT)')
        if out:
            sp.add_argument('--out', required=True)
        sp.add_argument('--mu', type=float, default=0.35)
        sp.add_argument('--timestep', type=float, default=0.001)
        sp.add_argument('--timeconst', type=float, default=0.004)
        sp.add_argument('--seed', type=int, default=0)
        sp.add_argument('--sigma', type=float, help='placement sigma (mm) in X and Y (legacy gauss model)')
        error_args(sp)

    sp = sub.add_parser('export', help='CAD solid, STL/STEP, views, rule margins')
    sp.add_argument('--params')
    sp.add_argument('--out', required=True)
    error_args(sp)
    sp.set_defaults(func=cmd_export)
    sp = sub.add_parser('rules', help='print analytic design-rule margins')
    sp.add_argument('--params')
    error_args(sp)
    sp.set_defaults(func=cmd_rules)
    sp = sub.add_parser('size', help='smallest voxel pitch whose capture covers the measured error')
    sp.add_argument('--params', help='base params (everything but U)')
    sp.add_argument('--approach-error', required=True)
    sp.add_argument('--error-stat', choices=['mean', 'max', 'empirical'], default='mean')
    sp.add_argument('--margin', type=float, default=0., help='capture beyond the tolerance and yaw (mm)')
    sp.add_argument('--yaw', type=float, default=5., help='yaw (deg) the capture must also absorb at the feet')
    sp.add_argument('--out', required=True, help='JSON report (params + rules)')
    sp.set_defaults(func=cmd_size)
    sp = sub.add_parser('drop', help='Monte Carlo drops + X, Y and yaw capture sweeps')
    common(sp)
    sp.add_argument('--samples', type=int, default=64)
    sp.add_argument('--ideal', action='store_true')
    sp.add_argument('--on-brick', action='store_true', help='drop onto a seated brick instead of the base')
    sp.add_argument('--video', type=int, default=0, help='render up to N failed drops (plus drop 0)')
    sp.set_defaults(func=cmd_drop)
    sp = sub.add_parser('assemble', help='sequential assembly: tower, wall, bridge, overhang+/-, ring')
    sp.add_argument('structure')
    common(sp)
    sp.add_argument('--ideal', action='store_true')
    sp.add_argument('--track', action='store_true', help='aim at the measured supporter pose (global camera)')
    sp.add_argument('--exact-courses', type=int, default=0)
    sp.add_argument('--keep-going', action='store_true')
    sp.add_argument('--video', action='store_true')
    sp.set_defaults(func=cmd_assemble)
    sp = sub.add_parser('push', help='ramped lateral push on an ideal tower')
    common(sp, out=False)
    sp.add_argument('--height', type=int, default=6)
    sp.set_defaults(func=cmd_push)
    sp = sub.add_parser('fly-assemble', help='closed-loop simulated Crazyflie picks, flies and places DF bricks')
    sp.add_argument('structure')
    common(sp)
    sp.add_argument('--gripper', help='gripper params JSON (default: fork)')
    sp.add_argument('--config', help='brixzle FlyConfig JSON (drone, lighthouse, camera, fixture, planner, ...)')
    sp.add_argument('--lighthouse', help='Lighthouse model JSON (overrides --config)')
    sp.add_argument('--bricks', type=int, help='only the first N bricks of the order')
    sp.add_argument('--seeds', type=int, default=1)
    sp.add_argument('--workers', type=int, default=1)
    sp.add_argument('--keep-going', action='store_true')
    sp.add_argument('--video', action='store_true', help='MP4 of the first seed')
    sp.set_defaults(func=cmd_fly_assemble)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
