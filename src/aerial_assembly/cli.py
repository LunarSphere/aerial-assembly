import argparse
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from .config import Envelope, Physics, Release, TrialSettings, read_json, write_json
from .experiments import batch, compare_candidates, convergence
from .geometry import demo_geometry, load_bundle, prepare_geometry
from .model import build_model
from .simulation import validate_geometry


def configuration(path):
    values = read_json(path) if path else {}
    unknown = set(values)-{'physics','trial','release','envelope'}
    if unknown:
        raise ValueError(f'Unknown experiment sections: {unknown}')
    return (Physics(**values.get('physics', {})), TrialSettings(**values.get('trial', {})),
            Release(**values.get('release', {})), Envelope(**values.get('envelope', {})))


def progress(index, result):
    print(f"Trial {index}: {result['status']} ({result['invalid_reason'] or 'valid'}); "
          f"max penetration {result['max_penetration']*1e6:.2f} um", flush=True)


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
    p = argparse.ArgumentParser(description='Passive two-block assembly experiments (SI units).')
    sub = p.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Generate a synthetic test fixture; not the Onshape model')
    demo.add_argument('--out', default='assets/demo.json')
    demo.add_argument('--ramp-angle', type=float, default=50)
    demo.add_argument('--lead-angle', type=float, default=30)
    demo.add_argument('--facets', type=int, default=48)
    prepare = sub.add_parser('prepare', help='Import a local CAD manifest containing convex collision pieces')
    prepare.add_argument('manifest')
    prepare.add_argument('--out', required=True)
    inspect = sub.add_parser('inspect-mjcf', help='Check an onshape-to-robot export before preparation')
    inspect.add_argument('xml')
    inspect.add_argument('--out', help='Optional JSON inspection report')
    mjcf = sub.add_parser('prepare-mjcf', help='Import a rigid onshape-to-robot block with seating metadata')
    mjcf.add_argument('xml')
    mjcf.add_argument('--metadata', required=True, help='SI feature metadata in the exported root body frame')
    mjcf.add_argument('--body', help='Optional assertion of the exported root body name')
    mjcf.add_argument('--out', required=True)
    cad = sub.add_parser('cad-drop', help='Prepare a downloaded GOAT export and record a drop; no CAD edits')
    cad.add_argument('directory', help='Downloaded directory containing robot.xml and assets')
    cad.add_argument('--out', required=True)
    cad.add_argument('--config', help='Shared physics/trial/release-envelope configuration')
    cad.add_argument('--density', type=float, default=600., help='Provisional effective density in kg/m^3')
    cad.add_argument('--mass-grams', type=float, help='Measured complete block mass; assumes uniform distribution')
    cad.add_argument('--count', type=int, default=1, help='1: aligned preview; >1: seeded random objective evaluation')
    cad.add_argument('--seed', type=int, default=42)
    cad.add_argument('--cache', default='assets/cad-cache')
    cad.add_argument('--video', action='store_true', help='Also render an MP4 headlessly')
    rank = sub.add_parser('cad-rank', help='Rank downloaded CAD runs with identical experiment settings')
    rank.add_argument('directories', nargs='+')
    render = sub.add_parser('render', help='Render an existing recorded drop to MP4 without a display')
    render.add_argument('directory')
    render.add_argument('--out', required=True)
    render.add_argument('--trial', type=int, default=0)
    for name in ['validate','drop','batch','sweep','converge']:
        q = sub.add_parser(name)
        q.add_argument('bundle')
        q.add_argument('--config', help='JSON with physics/trial/release/envelope sections')
        q.add_argument('--out', required=True, help='New output directory (JSON file for validate)')
        if name in ['batch']:
            q.add_argument('--count', type=int, default=100)
            q.add_argument('--seed', type=int, default=42)
            q.add_argument('--record', action='store_true')
            q.add_argument('--releases', help='JSON list of explicit Release objects; overrides random envelope')
        if name == 'sweep':
            q.add_argument('--axis', choices=['x','y','roll','pitch','yaw','height'], default='x')
            q.add_argument('--values', type=float, nargs='+', required=True, help='meters for offsets/height; degrees for angles')
    for name in ['compare','demo-grid']:
        q = sub.add_parser(name)
        q.add_argument('--config')
        q.add_argument('--out', required=True)
        q.add_argument('--count', type=int, default=100)
        q.add_argument('--seed', type=int, default=42)
        if name == 'compare':
            q.add_argument('bundles', nargs='+')
        else:
            q.add_argument('--ramp-angles', nargs='+', type=float, default=[45,50,55])
            q.add_argument('--lead-angles', nargs='+', type=float, default=[25,30,35])
    view = sub.add_parser('replay')
    view.add_argument('directory')
    view.add_argument('--trial', type=int, default=0)
    view.add_argument('--collisions', action='store_true')
    snapshot = sub.add_parser('onshape-export', help='Read-only revision-pinned API export into a local cache')
    snapshot.add_argument('config')
    snapshot.add_argument('--cache', default='assets/onshape')
    update = sub.add_parser('onshape-variables', help='Preview Variable Studio edits; --apply explicitly writes them')
    update.add_argument('config')
    update.add_argument('changes', help='JSON object mapping independent variable names to expressions')
    update.add_argument('--apply', action='store_true')
    update.add_argument('--audit', default='runs/onshape-update')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'demo':
            b = demo_geometry(args.out, args.ramp_angle, args.lead_angle, args.facets)
            print(f"Created SYNTHETIC fixture: {args.out}\n{b['asset_hash']}")
            return 0
        if args.command == 'prepare':
            b = prepare_geometry(args.manifest, args.out)
            print(f"Prepared {args.out}: {b['asset_hash']}")
            return 0
        if args.command == 'inspect-mjcf':
            from .mjcf import inspect_mjcf
            report = inspect_mjcf(args.xml)
            if args.out:
                write_json(args.out, report)
            print(json.dumps(report, indent=2))
            return 0 if report['ready_for_metadata'] else 2
        if args.command == 'prepare-mjcf':
            from .mjcf import prepare_mjcf
            b = prepare_mjcf(args.xml, args.metadata, args.out, args.body)
            print(f"Prepared {args.out}: {b['asset_hash']}")
            return 0
        if args.command == 'cad-drop':
            from .cad_workflow import cad_drop
            physics, settings, release, envelope = configuration(args.config)
            if not args.config:
                settings = TrialSettings(duration=1., dwell=.1)
            result = cad_drop(args.directory, args.out, density=args.density, mass_grams=args.mass_grams,
                              count=args.count, seed=args.seed, physics=physics, settings=settings,
                              envelope=envelope, release=release, cache=args.cache,
                              progress=lambda message: print(message, flush=True))
            if args.video:
                from .render import render_video
                render_video(args.out, Path(args.out)/'drop.mp4')
            print(json.dumps(result, indent=2))
            return 2 if result['objective_value'] is None else 0
        if args.command == 'cad-rank':
            from .cad_workflow import rank_downloads
            print(json.dumps(rank_downloads(args.directories), indent=2))
            return 0
        if args.command == 'render':
            from .render import render_video
            render_video(args.directory, args.out, args.trial)
            return 0
        if args.command == 'replay':
            replay(args.directory, args.trial, args.collisions)
            return 0
        if args.command.startswith('onshape-'):
            from .onshape import OnshapeClient, merge_variables
            config = read_json(args.config)
            client = OnshapeClient(config.get('base_url', 'https://cad.onshape.com'), config.get('api_version','v16'))
            if args.command == 'onshape-export':
                print(client.snapshot(config, args.cache))
            else:
                changes = read_json(args.changes)
                if args.apply:
                    client.update_variables(config, changes, args.audit)
                    print(f'Updated experiment workspace; audit: {args.audit}')
                else:
                    print(json.dumps(merge_variables(client.variables(config), changes, config['independent_variables']), indent=2))
            return 0
        physics, settings, release, envelope = configuration(args.config)
        if args.command in ['compare','demo-grid']:
            if args.command == 'demo-grid':
                generated = Path(args.out).with_name(Path(args.out).name+'-assets')
                generated.mkdir(parents=True, exist_ok=False)
                candidates = [demo_geometry(generated/f'{i:03d}.json', a, b)
                              for i,(a,b) in enumerate(product(args.ramp_angles, args.lead_angles))]
            else:
                candidates = [load_bundle(p) for p in args.bundles]
            result = compare_candidates(candidates, envelope.sample(args.count,args.seed), args.out,
                                        physics, settings, args.seed, progress)
        else:
            bundle = load_bundle(args.bundle)
            if args.command == 'validate':
                model, _ = build_model(bundle, physics)
                result = validate_geometry(model, bundle)
                write_json(args.out, result)
                print(json.dumps(result, indent=2))
                return 0 if result['passed'] else 2
            if args.command == 'converge':
                result = convergence(bundle, args.out, physics, settings)
                print(json.dumps(result, indent=2))
                return 0 if result['passed'] else 2
            if args.command == 'drop':
                releases, seed, record = [release], None, True
            elif args.command == 'batch':
                releases = ([Release(**r) for r in read_json(args.releases)] if args.releases
                            else envelope.sample(args.count, args.seed))
                if not releases:
                    raise ValueError('Release list is empty')
                seed, record = (None if args.releases else args.seed), args.record
            else:
                releases = []
                for value in args.values:
                    if args.axis in ['x','y']:
                        v = list(release.offset)
                        v[['x','y'].index(args.axis)] = value
                        releases.append(replace(release, offset=tuple(v)))
                    elif args.axis == 'height':
                        releases.append(replace(release, height=value))
                    else:
                        v = list(release.rpy_deg)
                        v[['roll','pitch','yaw'].index(args.axis)] = value
                        releases.append(replace(release, rpy_deg=tuple(v)))
                seed, record = None, False
            result = batch(bundle, releases, args.out, physics, settings, seed, record, progress)
        print(json.dumps(result, indent=2))
        # A valid physical failure is a completed experiment; invalid numerics are not.
        return 2 if result.get('invalid_fraction',0) or result.get('excluded_invalid_candidates') else 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as e:
        print(f'Error: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
