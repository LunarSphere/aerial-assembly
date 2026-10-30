"""Command line: export, rules, size, animate (trial commands follow once the geometry is reviewed)."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

from brixzle import approach_error as AE
from brixzle.config import read_json, write_json

from . import cad, rules as R
from .params import DEFAULT, BBParams


def load_params(path):
    """DEFAULT overridden by a JSON object (partial dicts allowed; a 'params' key is unwrapped)."""
    if not path:
        return DEFAULT
    values = read_json(path)
    values = values.get('params', values)
    return BBParams.from_dict({**DEFAULT.to_dict(), **{k: v for k, v in values.items() if k != 'derived'}})


def _new_dir(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'Output directory {out} is not empty; choose a new one')
    out.mkdir(parents=True, exist_ok=True)
    return out


def _settings(args, p):
    """Rules check the measured error if given, else the error the design was sized for."""
    if getattr(args, 'approach_error', None):
        return R.settings_for_tolerance(AE.required_tolerance(AE.load(args.approach_error), args.error_stat))
    return R.settings_for_tolerance(p.aim_r)


def _report(p, bundle, settings):
    near, far = R.landing(p, settings.placement_error)
    return {'params': p.to_dict(), 'derived': p.derived(), 'mass_g': bundle['mass_g'], 'com_mm': bundle['com'],
            'capture_mm': R.capture(p), 'crease_deg': R.crease_angle(p), 'landing_mm': {'near': near, 'far': far},
            'tines_mm': bundle['channels'], 'cavity_y_mm': bundle['cavity_y'],
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
    p = R.size_for_tolerance(tol, load_params(args.params))
    report = {'params': p.to_dict(), 'derived': p.derived(), 'capture_mm': R.capture(p),
              'sized_for': {'statistic': args.error_stat, 'tolerance_mm': tol, 'config': args.approach_error}}
    if args.out:
        out = Path(args.out)
        if out.exists():
            sys.exit(f'{out} exists; choose a new file')
        write_json(out, report)
    print(f'U = {p.U} mm, aim ({p.aim[0]:.2f}, {p.aim[1]:.2f}) mm, capture {R.capture(p):.2f} mm '
          f'for {args.error_stat} {tol:.2f} mm')


def cmd_animate(args):
    p = load_params(args.params)
    out = _new_dir(args.out)
    from .animate import animate
    report = animate(p, out, step=args.step)
    write_json(out/'animation.json', {'params': p.to_dict(), 'kind': 'kinematic, not simulated', **report})
    print(f'{report["frames"]} frames; wrote {out}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='biasbrick3d')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def error_args(sp, required=False):
        sp.add_argument('--approach-error', required=required,
                        help='measured X-Y error: Exp 3 metrics.json or ../brixzle/configs/error-*.json')
        sp.add_argument('--error-stat', choices=AE.STATISTICS, default='mean')

    sp = sub.add_parser('export', help='STL, STEP, views, stack, sections and rule margins')
    sp.add_argument('--params')
    sp.add_argument('--part', choices=['X', 'Y'])
    sp.add_argument('--out', required=True)
    error_args(sp)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser('rules', help='analytic rule margins (>= 0 is OK)')
    sp.add_argument('--params')
    error_args(sp)
    sp.set_defaults(func=cmd_rules)

    sp = sub.add_parser('size', help='smallest U whose aim bias and catchment cover the measured error')
    sp.add_argument('--params')
    sp.add_argument('--out')
    error_args(sp, required=True)
    sp.set_defaults(func=cmd_size)

    sp = sub.add_parser('animate', help='kinematic video of one brick stacking onto another (no physics)')
    sp.add_argument('--params')
    sp.add_argument('--step', type=float, default=0.4, help='mm per frame along the slide')
    sp.add_argument('--out', required=True)
    sp.set_defaults(func=cmd_animate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
