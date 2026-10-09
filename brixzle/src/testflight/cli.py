"""``testflight <command>``: calibration, sensing and the field plan for the real test flight.

  model       regenerate model.json from the brixzle CAD
  lighthouse  solve testflight_lighthouse.yaml with cflib and list what to improve
  survey      Lighthouse readings with the drone placed by hand (tags | home | fence | point); never arms
  capture     save a RealSense burst (colour.png, depth.npy, intrinsics.json) for offline work
  calibrate   camera pose from the surveyed tags -> calib/camera_pose.json
  sense       detect the brick/platform/fixture and print their poses (live or from a capture)
  plan        sense the layout and write the field plan (+ offline checks and a dry run)
  demo        everything above on a synthetic camera, with the real mission and a fake Crazyflie

Fly the field plan with:  uv run fly <plan.json> --uri radio://... --arm --realsense src/testflight/rig.json
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


def _rig(args):
    from .rig import Rig
    return Rig(args.rig)


def cmd_model(args):
    from .model import MODEL_PATH, build_model
    d = build_model(*([args.params] if args.params else []))
    MODEL_PATH.write_text(json.dumps(d))
    print(f'wrote {MODEL_PATH} (brick {d["brick"]["size"]} mm, part {d["part"]} of {d["params"]})')


def cmd_lighthouse(args):
    from . import lighthouse as Lh
    r = Lh.solve(args.yaml or Lh.YAML_PATH)
    for k, b in r['base_stations'].items():
        print(f'base station {k}: {np.round(np.asarray(b["pos_m"]), 3).tolist()} m, yaw {b["yaw_deg"]:.1f} deg')
    print(f'samples {r["samples"]}; solver error {r["error_mm"]} mm; '
          f'stations {Lh.angle_between_stations(r):.0f} deg apart seen from the origin')
    for w in r['warnings']:
        print(f'  - {w}')
    rig = _rig(args)
    print(f'wrote {rig.save_calib("lighthouse_solution.json", r)}')
    return 0 if r['ok'] else 1


def cmd_survey(args):
    from . import survey as Sv
    rig = _rig(args)
    reader = Sv.Reader(args.uri)
    try:
        if args.what == 'tags':
            ids = args.ids or rig['tags']['ids']
            d = Sv.survey_tags(reader, ids, args.size, args.drone_height)
            d['dictionary'] = rig['tags']['dictionary']
            print(f'wrote {rig.save_calib("tag_survey.json", d)}')
        elif args.what in ('home', 'fence'):
            field = rig.load_calib('field.json', required=False) or {}
            field.update(Sv.survey_home(reader) if args.what == 'home' else Sv.survey_fence(reader))
            print(f'wrote {rig.save_calib("field.json", field)}')
        else:
            p, yaw, std = reader.read()
            print(json.dumps({'pos_mm': np.round(p*1e3, 1).tolist(), 'yaw_deg': round(yaw, 2), 'spread_mm': round(std, 2)}))
    finally:
        reader.close()


def cmd_capture(args):
    from .camera import RealSenseCamera, save_frame
    rig = _rig(args)
    cam = RealSenseCamera(rig['camera'])
    try:
        print(f'wrote {save_frame(cam.capture(args.frames), args.out)}')
    finally:
        cam.close()


def cmd_calibrate(args):
    from . import calib as Ca
    from .camera import open_camera
    rig = _rig(args)
    survey = rig.load_calib('tag_survey.json')
    cam = open_camera(rig, args.capture)
    try:
        rep = Ca.calibrate(cam, survey, rig['tags']['dictionary'], rig['tags']['ids'], frames=args.frames)
    finally:
        cam.close()
    print(f'wrote {rig.save_calib("camera_pose.json", rep)}')
    print(f'camera at {np.round(np.asarray(rep["camera_position_m"])*1e3, 1).tolist()} mm (Lighthouse)')
    ok = True
    for name, (passed, msg) in rep['checks'].items():
        print(f'  {"ok  " if passed else "FAIL"} {name}: {msg}')
        ok &= passed
    return 0 if ok else 1


def cmd_sense(args):
    from .rig import Rig
    from .sense import make_source
    rig = Rig(args.rig, overrides={'log_dir': str(Path(args.debug).resolve())} if args.debug else None)
    src = make_source(rig, args.capture)
    rows = {}
    try:
        for _ in range(args.repeat):
            poses = src.poses()
            for k, p in poses.items():
                rows.setdefault(k, []).append([*np.asarray(p.pos)*1e3, math.degrees(p.yaw)])
            print(json.dumps({k: {'pos_mm': np.round(np.asarray(p.pos)*1e3, 1).tolist(),
                                  'yaw_deg': round(math.degrees(p.yaw), 2)} for k, p in poses.items()}))
            if src.last.issues:
                print('  notes: ' + '; '.join(src.last.issues))
    finally:
        src.close()
    if args.repeat > 1:
        for k, v in rows.items():
            v = np.array(v)
            print(f'{k}: n={len(v)} mean {np.round(v.mean(0), 1).tolist()} std {np.round(v.std(0), 2).tolist()} '
                  '(x, y, z mm, yaw deg)')


def cmd_plan(args):
    from flight.checks import check_plan
    from flight.hardware import dry_run
    from . import fieldplan as FP
    from .model import load_model
    from .sense import make_source
    rig = _rig(args)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'{out} is not empty; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    field = {**rig['field'], **(rig.load_calib('field.json', required=False) or {})}
    src = make_source(rig, args.capture)
    try:
        sensed = src.poses()
    finally:
        src.close()
    f = rig['field']
    base = (json.loads(Path(args.from_plan).read_text()) if args.from_plan else
            FP.compile_plan(args.params, args.gripper, args.fly, f['camera_sigma_mm'], f['camera_sigma_deg']))
    plan = FP.adapt_plan(base, sensed, field, src.cam.floor_z, load_model().platform_origin_height, rig['names'])
    (out/'plan.json').write_text(json.dumps(plan, indent=1, default=float))
    for msg in FP.field_problems(plan, sensed, rig['names'], field):
        print(f'warning: {msg}')
    check_plan(plan)
    lines = []
    summary, calls = dry_run(plan, out=lines.append)
    (out/'dry_run.txt').write_text('\n'.join(lines) + '\n' + '\n'.join(' '.join(map(str, c)) for c in calls) + '\n')
    print(f'wrote {out/"plan.json"}: dry run {summary["dry_run_status"]}, {summary["calls"]} cflib calls, '
          f'{summary["segments_s"]:.0f} s of segments')
    print(f'next: uv run fly {out/"plan.json"} --uri <radio> --check-only --realsense {rig.path}')


def cmd_demo(args):
    from .demo import run_demo
    s = run_demo(args.out, seed=args.seed)
    print(json.dumps({k: s.get(k) for k in ('mission_status', 'bricks', 'final_brick_in_seat_frame_mm')},
                     indent=1, default=float))
    return 0 if s['mission_status'] == 'complete' else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog='testflight', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rig', help='rig.json (default: src/testflight/rig.json)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sp = sub.add_parser('model')
    sp.add_argument('--params', help='BrickParams JSON (default: configs/ab.json, the printed A/B pair)')
    sp.set_defaults(func=cmd_model)
    sp = sub.add_parser('lighthouse')
    sp.add_argument('yaml', nargs='?', help='cfclient geometry input (default: src/testflight/testflight_lighthouse.yaml)')
    sp.set_defaults(func=cmd_lighthouse)
    sp = sub.add_parser('survey')
    sp.add_argument('what', choices=['tags', 'home', 'fence', 'point'])
    sp.add_argument('--uri', required=True)
    sp.add_argument('--size', type=float, help='tags: black-square edge (m), measured')
    sp.add_argument('--drone-height', type=float, help='tags: drone origin above the floor when on a tag (m)')
    sp.add_argument('--ids', type=int, nargs='*')
    sp.set_defaults(func=cmd_survey)
    sp = sub.add_parser('capture')
    sp.add_argument('--out', required=True)
    sp.add_argument('--frames', type=int, default=15)
    sp.set_defaults(func=cmd_capture)
    sp = sub.add_parser('calibrate')
    sp.add_argument('--capture', nargs='*', help='saved captures instead of the live camera')
    sp.add_argument('--frames', type=int, default=20)
    sp.set_defaults(func=cmd_calibrate)
    sp = sub.add_parser('sense')
    sp.add_argument('--capture', nargs='*')
    sp.add_argument('--repeat', type=int, default=1)
    sp.add_argument('--debug', help='directory for overlay images and detections')
    sp.set_defaults(func=cmd_sense)
    sp = sub.add_parser('plan')
    sp.add_argument('--out', required=True)
    sp.add_argument('--capture', nargs='*')
    sp.add_argument('--from-plan', help='adapt this compiled plan.json instead of compiling one')
    sp.add_argument('--params')
    sp.add_argument('--gripper')
    sp.add_argument('--fly')
    sp.set_defaults(func=cmd_plan)
    sp = sub.add_parser('demo')
    sp.add_argument('--out')
    sp.add_argument('--seed', type=int, default=0)
    sp.set_defaults(func=cmd_demo)
    args = ap.parse_args(argv)
    if args.cmd == 'survey' and args.what == 'tags' and (args.size is None or args.drone_height is None):
        ap.error('survey tags needs --size and --drone-height (both measured)')
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
