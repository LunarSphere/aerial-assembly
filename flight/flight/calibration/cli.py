"""flight-cal: survey, Lighthouse geometry, camera registration and supervised hover."""
import argparse
import json
from pathlib import Path
import sys

from . import artifacts, rig


def camera_arguments(parser):
    parser.add_argument('--device', help='RealSense serial number')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, choices=(15, 30), default=30, help='RealSense stream rate; use 15 on USB 2')
    parser.add_argument('--depth-width', type=int, choices=(640, 848), default=848, help='RealSense depth width; use 640 on USB 2')


def flight_arguments(p):
    p.add_argument('--demo', action='store_true', help='refresh camera pose, use PID and 10 cm tracking tolerance')
    p.add_argument('--controller', choices=('pid', 'mellinger'), help='default: PID for demo, Mellinger otherwise')
    p.add_argument('--setup', help='completed compact setup directory; fills all calibration and camera settings')
    p.add_argument('--rig')
    p.add_argument('--lighthouse')
    p.add_argument('--calibration', help='accepted camera session directory')
    p.add_argument('--out', help='run directory, replacing a previous run there (default runs/<command>)')
    p.add_argument('--uri')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--camera-check-only', action='store_true', help='check camera registration only; no radio connection or arming')
    mode.add_argument('--arm', action='store_true')
    camera_arguments(p)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    p = commands.add_parser('setup', aliases=['survey'], help='four cfclient-style placements; camera/radio/stations detected automatically')
    p.add_argument('--out', default='runs/setup', help='setup directory, replacing the previous setup there (or rig.json filename)')
    camera_arguments(p)
    p.add_argument('--uri', help='optional Crazyflie URI; defaults to Crazyradio scan')
    p.add_argument('--detector-model', help='optional prompted drone ONNX model; prefers prepared YOLOE-L, otherwise nano')
    p.add_argument('--max-camera-rms-px', type=float, default=1., help='camera fit RMS limit in pixels; default 1.0, use 2.0 to allow a 1.88 px fit')
    p.add_argument('--advanced', action='store_true', help='use the earlier detailed manual measurement questionnaire')
    p.add_argument('--layout', help='camera-estimated layout.json to prefill tag inventory, positions and rotations')
    p.add_argument('--tag-ids', type=int, nargs='+', help='manual tag IDs; default 0–5, at least 0,1,2 plus two check tags')
    p.add_argument('--measure-check-tags', action='store_true', help='independently measure check positions and sizes when importing a layout')
    p = commands.add_parser('discover', help='read-only Crazyradio scan and Lighthouse station discovery')
    p.add_argument('--uri')
    p = commands.add_parser('prepare-detector', help='one-time visual-prompt model export; no devices opened')
    p.add_argument('--out', help='optional ONNX output path')
    p.add_argument('--size', choices=('n', 's', 'm', 'l', 'x'), default='l', help='YOLOE capacity; default l (large)')
    p.add_argument('--image-size', type=int, help='square detector input; default 640 for larger models, 960 for nano')
    p = commands.add_parser('estimate-tags', help='capture camera data and estimate an initial floor-tag layout')
    camera_arguments(p)
    p.add_argument('--out', required=True, help='new directory for layout.json, images and camera evidence')
    p = commands.add_parser('lighthouse', help='human-guided calibration; rejects verification error >0.4 mm')
    p.add_argument('--rig', required=True)
    p.add_argument('--uri', required=True)
    p.add_argument('--out', required=True, help='new session directory')
    p = commands.add_parser('camera', help='register the camera to accepted Lighthouse geometry')
    p.add_argument('--rig', required=True)
    p.add_argument('--lighthouse', required=True, help='accepted Lighthouse session directory')
    p.add_argument('--out', required=True)
    camera_arguments(p)
    p = commands.add_parser('hover-check', help='take off, hold over launch, land; dry run unless --arm')
    flight_arguments(p)
    p.add_argument('--hover-seconds', type=float, default=10., help='stationary hold duration, 0–60 seconds (default 10)')
    p = commands.add_parser('tag-visit', help='visit beside every tag toward the centre; dry run unless --arm')
    flight_arguments(p)
    p.add_argument('--tag-offset-m', type=float, default=.15, help='minimum inward offset; automatically extends up to 0.40 m to clear tags in camera view (default 0.15)')
    p = commands.add_parser('devices', help='list connected RealSense cameras')
    p = commands.add_parser('replay-lighthouse', help='offline re-solve saved samples; never uploads or opens a radio')
    p.add_argument('--samples', required=True)
    p.add_argument('--out', required=True)
    return root


def devices(args):
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise RuntimeError('Install flight[realsense] to enumerate RealSense devices') from e
    return [dict(name=d.get_info(rs.camera_info.name), serial=d.get_info(rs.camera_info.serial_number))
            for d in rs.context().query_devices()]


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ('hover-check', 'tag-visit') and not args.out:
        args.out = f'runs/{args.command}'
    try:
        if args.command in ('survey', 'setup'):
            if args.advanced:
                if args.max_camera_rms_px != 1.:
                    raise ValueError('--max-camera-rms-px applies to compact setup, not --advanced')
                rig.survey(args.out, layout_path=args.layout, tag_ids=args.tag_ids, measure_checks=args.measure_check_tags)
                print(f'Survey saved: {args.out}')
            else:
                from .setup import run
                print(run(args))
        elif args.command == 'discover':
            from .discovery import inspect
            print(json.dumps(inspect(args.uri), indent=2))
        elif args.command == 'prepare-detector':
            from .drone import prepare
            print(prepare(args.out, size=args.size, image_size=args.image_size))
        elif args.command == 'estimate-tags':
            from .layout import capture
            print(capture(args))
        elif args.command == 'devices':
            print(json.dumps(devices(args), indent=2))
        elif args.command == 'replay-lighthouse':
            from .lighthouse import solve_bounded
            out = artifacts.session(args.out, 'lighthouse-replay', {})
            result = solve_bounded(Path(args.samples).resolve(), out)
            artifacts.report(out, replay_only=True, quality_passed=result['accepted'],
                             accepted=False, solution=result)
            print(f'Offline quality check: {result["accepted"]}; no hardware upload. Report: {out}')
            return 0 if result['accepted'] else 1
        else:
            if args.command in ('hover-check', 'tag-visit'):
                if args.setup:
                    setup = Path(args.setup)
                    status = artifacts.read(setup/'report.json')
                    if status.get('kind') != 'setup' or not (status.get('accepted') and status.get('completed')):
                        raise ValueError('Select a completed setup')
                    args.rig, args.lighthouse, args.calibration = (str(setup/'rig.json'), str(setup/'lighthouse'), str(setup/'camera'))
                    args.uri = args.uri or status['uri']
                    calibration = artifacts.read(setup/'camera/report.json')
                    profile = calibration['profile']
                    args.device = profile['device']
                    args.width, args.height, args.fps = profile['width'], profile['height'], profile.get('fps', 30)
                    args.depth_width = 640 if args.fps == 15 else 848
                if not (args.rig and args.lighthouse and args.calibration):
                    raise ValueError('Use --setup DIRECTORY, or provide --rig, --lighthouse and --calibration')
            survey = rig.load(args.rig)
            if args.command == 'lighthouse':
                from .lighthouse import calibrate
                print(calibrate(args, survey))
            elif args.command == 'camera':
                from .vision import calibrate
                print(calibrate(args, survey))
            elif args.command in ('hover-check', 'tag-visit'):
                from .hover import run
                output = run(args, survey)
                print(output)
                return 0 if artifacts.read(output / 'report.json')['accepted'] else 1
        return 0
    except KeyboardInterrupt:
        print('flight-cal: interrupted; inspect the session report before retrying', file=sys.stderr)
        return 130
    except Exception as e:
        print(f'flight-cal: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
