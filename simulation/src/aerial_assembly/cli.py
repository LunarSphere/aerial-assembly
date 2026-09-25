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
    cad = sub.add_parser('cad-drop', help='Prepare a local block export and record one drop')
    cad.add_argument('directory', help='Local directory containing robot.xml and its mesh assets')
    cad.add_argument('--out', required=True, help='New output directory')
    cad.add_argument('--config', help='JSON with physics/trial/release sections')
    cad.add_argument('--density', type=float, default=600., help='Provisional effective density in kg/m^3')
    cad.add_argument('--mass-grams', type=float, help='Measured complete block mass; assumes uniform distribution')
    cad.add_argument('--cache', default='assets/cad-cache')
    cad.add_argument('--video', action='store_true', help='Also render an MP4')
    chain = sub.add_parser('cad-chain', help='Add blocks until a stepping block touches the floor')
    chain.add_argument('complement', help='Local complement export containing robot.xml and mesh assets')
    chain.add_argument('stepper', help='Local two-peg block export containing robot.xml and mesh assets')
    chain.add_argument('--config', help='JSON with physics, settle_seconds, and insertion_margin')
    chain.add_argument('--out', required=True, help='New output directory')
    chain.add_argument('--max-blocks', type=int, default=20)
    chain.add_argument('--base-mass-grams', type=float, default=480.)
    chain.add_argument('--block-mass-grams', type=float, default=26.)
    chain.add_argument('--cache', default='assets/cad-cache')
    grid = sub.add_parser('cad-grid', help='Search a discrete six-dimensional drop grid')
    grid.add_argument('directory', help='Local export containing robot.xml and its mesh assets')
    grid.add_argument('--config', required=True)
    grid.add_argument('--out', required=True)
    grid.add_argument('--density', type=float, default=600.)
    grid.add_argument('--mass-grams', type=float)
    grid.add_argument('--cache', default='assets/cad-cache')
    grid.add_argument('--dry-run', action='store_true', help='Count states without preparing CAD or simulating')
    grid.add_argument('--resume', action='store_true')
    grid.add_argument('--max-states', type=int, help='Maximum additional states this invocation')
    grid.add_argument('--record-successes', action='store_true')
    grid.add_argument('--record-trial', type=int, action='append', default=[], help='Save this grid index for replay; repeatable')
    grid.add_argument('--progress-every', type=int, default=10)
    render = sub.add_parser('render', help='Render an existing recorded drop to MP4')
    render.add_argument('directory')
    render.add_argument('--out', required=True)
    render.add_argument('--trial', type=int, default=0, help='Trial index for older recordings')
    chain_render = sub.add_parser('render-chain', help='Render a recorded incremental-chain stage to MP4')
    chain_render.add_argument('directory')
    chain_render.add_argument('--out', required=True)
    chain_render.add_argument('--stage', type=int, help='Stage number; default is the last recorded stage')
    view = sub.add_parser('replay', help='View a recorded drop interactively')
    view.add_argument('directory')
    view.add_argument('--trial', type=int, default=0, help='Trial index for older recordings')
    view.add_argument('--collisions', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'cad-chain':
            from .chain import cad_chain
            result = cad_chain(args.complement, args.stepper, args.out,
                               max_blocks=args.max_blocks,
                               base_mass_grams=args.base_mass_grams,
                               block_mass_grams=args.block_mass_grams,
                               config=args.config, cache=args.cache,
                               progress=lambda message: print(message, flush=True))
            print(json.dumps(result, indent=2))
            return 0
        if args.command == 'cad-grid':
            from .grid import cad_grid, configuration as grid_configuration
            if args.dry_run:
                grid, _, _, _, _ = grid_configuration(args.config)
                print(json.dumps(grid.describe(), indent=2))
                return 0
            result = cad_grid(args.directory, args.out, config=args.config,
                              density=args.density, mass_grams=args.mass_grams, cache=args.cache,
                              resume=args.resume, max_states=args.max_states,
                              record_successes=args.record_successes, record_trials=args.record_trial,
                              progress_every=args.progress_every,
                              progress=lambda message: print(message, flush=True))
            print(json.dumps(result, indent=2))
            return 2 if result['invalid_states'] else 0
        if args.command == 'replay':
            replay(args.directory, args.trial, args.collisions)
            return 0
        if args.command == 'render':
            from .render import render_video
            render_video(args.directory, args.out, args.trial)
            return 0
        if args.command == 'render-chain':
            from .render import render_chain_video
            render_chain_video(args.directory, args.out, args.stage)
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
    except KeyboardInterrupt:
        print('Interrupted. Completed grid states are checkpointed; use --resume.', file=sys.stderr)
        return 130
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as e:
        print(f'Error: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
