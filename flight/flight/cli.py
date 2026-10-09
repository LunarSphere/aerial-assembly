"""``fly plan.json --sim`` or ``fly plan.json --uri radio://... [--check-only | --arm]``.

Without ``--sim`` and without ``--check-only``/``--arm`` this is a dry run: no radio link is
opened. ``--sim`` needs the brixzle package (MuJoCo); it is imported only then.
"""
import argparse
import json
from pathlib import Path
import sys


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan', help='compiled plan.json (brixzle fly-assemble writes one)')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sim', action='store_true', help='fly the plan in the MuJoCo simulator')
    mode.add_argument('--uri', help='real Crazyflie, e.g. radio://0/80/2M/E7E7E7E7E7')
    ap.add_argument('--check-only', action='store_true', help='hardware: connect and run the preflight, never arm')
    ap.add_argument('--arm', action='store_true', help='hardware: arm and fly (supervised flights only)')
    ap.add_argument('--poses', help='hardware: brick poses JSON (manual camera fallback)')
    ap.add_argument('--realsense', metavar='RIG', help='hardware: sense poses with the overhead RealSense '
                    '(brixzle/src/testflight/rig.json)')
    ap.add_argument('--replay', nargs='*', help='with --realsense: replay saved captures instead of the camera')
    ap.add_argument('--keep-going', action='store_true')
    ap.add_argument('--seed', type=int, default=0, help='sim: noise seed')
    ap.add_argument('--out', help='sim: new directory for the log, summary and video')
    ap.add_argument('--video', action='store_true', help='sim: render flight_close.mp4')
    ap.add_argument('--show-calls', action='store_true', help='dry run: print every cflib call it would send')
    args = ap.parse_args(argv)
    if args.sim and (args.arm or args.check_only):
        ap.error('--arm/--check-only apply to real hardware only')
    if args.arm and args.check_only:
        ap.error('choose one of --check-only and --arm')
    if args.poses and args.realsense:
        ap.error('choose one of --poses and --realsense')
    return args


def run_sim(plan, args):
    from brixzle import flyassemble as FA
    res, h = FA.run_plan(plan, seed=args.seed, record_fps=30 if args.video else 0)
    events = res.pop('events')
    print(json.dumps({k: res[k] for k in ('mission_status', 'placed', 'total', 'successes', 'sim_time_s')}))
    if args.out:
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            sys.exit(f'Output directory {out} is not empty; choose a new one')
        out.mkdir(parents=True, exist_ok=True)
        (out/'summary.json').write_text(json.dumps(res, indent=1, default=float))
        (out/'events.json').write_text(json.dumps(events, indent=1, default=float))
        if args.video:
            from brixzle.render import render
            render(h.model, h.world.frames, out/'flight_close.mp4', fps=30, azimuth=120, elevation=-15, track='cf',
                   distance=0.3)
    return 0 if res['mission_status'] == 'complete' else 1


def main(argv=None):
    args = parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    try:
        if args.sim:
            return run_sim(plan, args)
        from . import hardware
        if not (args.arm or args.check_only):
            print(f'DRY RUN for {args.uri}: no radio link opened. Use --check-only or --arm to connect.')
            _, calls = hardware.dry_run(plan)
            if args.show_calls:
                for c in calls:
                    print(' '.join(json.dumps(x) if not isinstance(x, str) else x for x in c))
            return 0
        if args.arm and not (args.poses or args.realsense):
            sys.exit('--arm needs --realsense rig.json (or --poses poses.json)')
        from .sensing import JsonPoseSource, RealSensePoseSource
        if args.realsense:
            poses = RealSensePoseSource(args.realsense, args.replay)
            print(f'camera sees: {json.dumps({k: v.to_dict() for k, v in poses.poses().items()})}')
        else:
            poses = JsonPoseSource(args.poses) if args.poses else None
        try:
            hardware.fly(plan, args.uri, poses, arm=args.arm, keep_going=args.keep_going)
        finally:
            if hasattr(poses, 'close'):
                poses.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f'fly: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
