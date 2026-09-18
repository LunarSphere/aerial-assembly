"""Run and view a single local CAD drop."""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from .config import Physics, Release, TrialSettings, read_json


def configuration(path):
    values = read_json(path) if path else {}
    unknown = set(values)-{'physics', 'trial', 'release'}
    if unknown:
        raise ValueError(f'Unknown drop configuration sections: {unknown}')
    return (Physics(**values.get('physics', {})),
            TrialSettings(**({'duration': 1., 'dwell': .1} | values.get('trial', {}))),
            Release(**values.get('release', {})))


def replay(directory, trial_index, collisions):
    # Viewer uses recorded states only. User interaction cannot alter scored runs.
    import mujoco.viewer
    directory = Path(directory)
    model = mujoco.MjModel.from_xml_path(str(directory/'scene.xml'))
    trajectory = np.load(directory/f'trial_{trial_index:05d}'/'trajectory.npz')['state']
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        from .render import frame_camera
        frame_camera(viewer.cam, directory, trajectory)
        viewer.opt.geomgroup[2] = not collisions
        viewer.opt.geomgroup[3] = collisions or not any(model.geom_group == 2)
        start = time.monotonic()
        for row in trajectory:
            if not viewer.is_running():
                break
            time.sleep(max(0, start+row[0]-time.monotonic()))
            data.time = row[0]
            data.qpos[:] = row[1:8]
            data.qvel[:] = row[8:14]
            mujoco.mj_forward(model, data)
            viewer.sync()


def parser():
    p = argparse.ArgumentParser(description='One local two-block CAD drop (SI units).')
    sub = p.add_subparsers(dest='command', required=True)
    cad = sub.add_parser('cad-drop', help='Prepare a local GOAT export and record one drop')
    cad.add_argument('directory', help='Local directory containing robot.xml and assets')
    cad.add_argument('--out', required=True, help='New output directory')
    cad.add_argument('--config', help='JSON with physics/trial/release sections')
    cad.add_argument('--density', type=float, default=600., help='Provisional effective density in kg/m^3')
    cad.add_argument('--mass-grams', type=float, help='Measured complete block mass; assumes uniform distribution')
    cad.add_argument('--cache', default='assets/cad-cache')
    cad.add_argument('--video', action='store_true', help='Also render an MP4')
    render = sub.add_parser('render', help='Render an existing recorded drop to MP4')
    render.add_argument('directory')
    render.add_argument('--out', required=True)
    render.add_argument('--trial', type=int, default=0, help='Trial index for older recordings')
    view = sub.add_parser('replay', help='View a recorded drop interactively')
    view.add_argument('directory')
    view.add_argument('--trial', type=int, default=0, help='Trial index for older recordings')
    view.add_argument('--collisions', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'replay':
            replay(args.directory, args.trial, args.collisions)
            return 0
        if args.command == 'render':
            from .render import render_video
            render_video(args.directory, args.out, args.trial)
            return 0
        from .cad_workflow import cad_drop
        physics, settings, release = configuration(args.config)
        result = cad_drop(args.directory, args.out, density=args.density, mass_grams=args.mass_grams,
                          physics=physics, settings=settings, release=release, cache=args.cache,
                          progress=lambda message: print(message, flush=True))
        if args.video:
            from .render import render_video
            render_video(args.out, Path(args.out)/'drop.mp4')
        print(json.dumps(result, indent=2))
        return 0 if result['valid'] else 2
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as e:
        print(f'Error: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
