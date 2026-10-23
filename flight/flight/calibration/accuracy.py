"""Exp 3: X-Y approach accuracy over surveyed AprilTag centres (issue #24).

Positioning error is the camera-measured deck centre minus the surveyed tag centre;
control error is the Lighthouse-reported deck position minus the commanded centre.
Everything is computed from saved run files, so metrics can be regenerated offline.
"""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from ..frames import Pose
from . import artifacts
from .camera import pixel_rays

SETTLE_S = .5  # hover.execute starts its settled window 0.5 s after each segment
SURVEY_WARNING_M = .01
DEFAULT_TRIALS = 5
ERROR_TYPES = ('positioning', 'control')
LIMITATIONS = [
    'Drone tilt is ignored: the detection centre is treated as the point straight above the deck centre.',
    'The camera measures the visual centre of the detection box, not the Lighthouse deck sensor centre.',
    'The camera pose is refit to the floor tags every frame, so surveyed centres match rig.json by construction '
    'and camera motion does not bias positioning error.',
    'Only X-Y is evaluated; Z is out of scope.',
]


def survey_tags(observations, T_lighthouse_camera, intrinsics, rig):
    """Tag centres in the Lighthouse frame: corner rays intersected with each tag's plane."""
    missing = sorted(t['id'] for t in rig['tags'] if t['id'] not in observations)
    if missing:
        raise ValueError(f'Tag survey needs every tag visible; missing {missing}. Clear the view and retry.')
    pose = Pose.from_dict(T_lighthouse_camera)
    origin = np.asarray(pose.pos)
    tags = []
    for tag in rig['tags']:
        directions = pose.rotate(pixel_rays(observations[tag['id']], intrinsics))
        rig_centre = np.asarray(tag['pose']['pos'], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = (rig_centre[2]-origin[2])/directions[:, 2]
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f'Tag {tag["id"]}: camera rays do not meet the tag plane')
        centre = np.mean(origin+directions*scale[:, None], axis=0)
        delta = centre[:2]-rig_centre[:2]
        tags.append(dict(id=tag['id'], centre_m=centre.tolist(), rig_centre_m=rig_centre.tolist(),
                         delta_xy_m=delta.tolist(), delta_xy_norm_m=float(np.linalg.norm(delta))))
    warnings = [f'Tag {t["id"]}: surveyed centre is {t["delta_xy_norm_m"]*100:.1f} cm from rig.json'
                for t in tags if t['delta_xy_norm_m'] > SURVEY_WARNING_M]
    return dict(version=1, kind='tag-survey', method='median AprilTag corners; rays intersected with the rig tag plane',
                T_lighthouse_camera=T_lighthouse_camera, tags=tags,
                max_delta_xy_m=max(t['delta_xy_norm_m'] for t in tags), warnings=warnings)


def centres(survey):
    return {t['id']: t['centre_m'] for t in survey['tags']}


def deck_xy(frames):
    """Deck X-Y from detection rays meeting a horizontal plane at the median depth height."""
    origins = np.array([f['drone_ray_world_m']['origin'] for f in frames], float)
    directions = np.array([f['drone_ray_world_m']['direction'] for f in frames], float)
    points = np.array([f['drone_world_m'] for f in frames], float)
    height = float(np.median(points[:, 2]))
    scale = (height-origins[:, 2])/directions[:, 2]
    xy = origins[:, :2]+directions[:, :2]*scale[:, None]
    return dict(xy_m=xy.mean(axis=0).tolist(), depth_xy_m=points[:, :2].mean(axis=0).tolist(),
                deck_height_m=height, frames=len(frames))


def recentre(events, centroids, intrinsics, shift=True):
    """Keep camera events whose saved image has a centroid (image name -> pixel); optionally aim their rays there.

    The log holds each frame's box-centre ray but not its refit rotation, so the new ray adds the pixel
    offset rotated by the registration pose. Exp 3 refits differ from it by at most 0.35 degrees, which on
    a centroid offset of a few pixels moves the deck point by about 0.1 mm.
    """
    out = []
    for event in events:
        if event.get('event') != 'camera':
            out.append(event)
        elif event.get('image') in centroids and 'drone_ray_world_m' in event:
            if shift:
                rays = pixel_rays([centroids[event['image']], event['drone']['center_px']], intrinsics)
                pose = Pose.from_dict(event['registration']['T_lighthouse_camera'])
                ray = dict(event['drone_ray_world_m'])
                ray['direction'] = (np.asarray(ray['direction'])+pose.rotate(rays[0]-rays[1])).tolist()
                event = dict(event, drone_ray_world_m=ray)
            out.append(event)
    return out


def approaches(events, centres, trial):
    """One row per tag dwell that completed: Lighthouse and camera X-Y over the settled window."""
    centres = {int(k): v for k, v in centres.items()}
    targets = [e for e in events if e.get('event') == 'target']
    telemetry = [(e['host_time'], e['values']) for e in events
                 if e.get('event') == 'telemetry' and 'stateEstimate.x' in e['values']]
    # The monitor may re-emit the latest camera status; keep one copy per capture time.
    cameras = sorted({e['time']: e for e in events if e.get('event') == 'camera'
                      and 'drone_ray_world_m' in e and 'drone_world_m' in e}.items())
    rows = []
    for i, target in enumerate(targets):
        if 'tag_id' not in target:
            continue
        start = target['t']+target['duration_s']+SETTLE_S
        end = start+target['dwell_s']
        # A dwell is complete only if the flight went on to command the next segment.
        if i+1 >= len(targets) or targets[i+1]['t'] < end:
            continue
        tag = int(target['tag_id'])
        centre = np.asarray(centres[tag][:2], float)
        commanded = np.asarray(target['target_m'][:2], float)
        lighthouse = np.array([[v['stateEstimate.x'], v['stateEstimate.y']] for t, v in telemetry
                               if start <= t <= end], float).reshape(-1, 2)
        frames = [e for t, e in cameras if start <= t <= end]
        row = dict(trial=trial, tag_id=tag, tag_centre_xy_m=centre.tolist(), commanded_xy_m=commanded.tolist(),
                   lighthouse_samples=len(lighthouse), camera_frames=len(frames),
                   lighthouse_xy_m=None, camera_xy_m=None, camera_depth_xy_m=None, deck_height_m=None,
                   camera_alignment_rms_px=None, positioning_error_xy_m=None, positioning_error_m=None,
                   control_error_xy_m=None, control_error_m=None)
        if len(lighthouse):
            reported = lighthouse.mean(axis=0)
            error = reported-commanded
            row.update(lighthouse_xy_m=reported.tolist(), control_error_xy_m=error.tolist(),
                       control_error_m=float(np.linalg.norm(error)))
        if frames:
            deck = deck_xy(frames)
            error = np.asarray(deck['xy_m'])-centre
            alignment = [f['alignment_rms_px'] for f in frames if f.get('alignment_rms_px') is not None]
            row.update(camera_xy_m=deck['xy_m'], camera_depth_xy_m=deck['depth_xy_m'],
                       deck_height_m=deck['deck_height_m'],
                       camera_alignment_rms_px=float(np.median(alignment)) if alignment else None,
                       positioning_error_xy_m=error.tolist(), positioning_error_m=float(np.linalg.norm(error)))
        rows.append(row)
    return rows


def launch_offset(events, trial):
    """Before takeoff: Lighthouse pad position minus the camera's (tag-frame) pad position.

    The camera is refit to the floor tags, so a change in this offset between trials means the
    Lighthouse stations moved relative to the floor (or the drone was reset off its mark).
    """
    targets = [e for e in events if e.get('event') == 'target']
    if not targets:
        return None
    takeoff = targets[0]['t']
    lighthouse = np.array([[e['values']['stateEstimate.x'], e['values']['stateEstimate.y']] for e in events
                           if e.get('event') == 'telemetry' and 'stateEstimate.x' in e['values']
                           and e['host_time'] < takeoff], float).reshape(-1, 2)
    frames = list({e['time']: e for e in events if e.get('event') == 'camera' and e['time'] < takeoff
                   and 'drone_ray_world_m' in e and 'drone_world_m' in e}.values())
    if not len(lighthouse) or not frames:
        return dict(trial=trial, lighthouse_xy_m=None, camera_xy_m=None, offset_xy_m=None, offset_m=None)
    camera = np.asarray(deck_xy(frames)['xy_m'])
    offset = lighthouse.mean(axis=0)-camera
    return dict(trial=trial, lighthouse_xy_m=lighthouse.mean(axis=0).tolist(), camera_xy_m=camera.tolist(),
                offset_xy_m=offset.tolist(), offset_m=float(np.linalg.norm(offset)), camera_frames=len(frames))


def statistics(vectors):
    vectors = np.asarray(vectors, float).reshape(-1, 2)
    if not len(vectors):
        return dict(n=0)
    norms = np.linalg.norm(vectors, axis=1)
    return dict(n=len(norms), mean_error_m=float(norms.mean()), rmse_m=float(np.sqrt(np.mean(norms**2))),
                std_error_m=float(norms.std(ddof=1)) if len(norms) > 1 else 0.,
                max_error_m=float(norms.max()), bias_xy_m=vectors.mean(axis=0).tolist())


def summarize(rows):
    """Mean, RMSE, sample standard deviation and bias of X-Y error, overall and per tag."""
    summary = {}
    for kind in ERROR_TYPES:
        key = f'{kind}_error_xy_m'
        measured = [r for r in rows if r[key] is not None]
        summary[kind] = dict(overall=statistics([r[key] for r in measured]),
                             per_tag={str(tag): statistics([r[key] for r in measured if r['tag_id'] == tag])
                                      for tag in sorted({r['tag_id'] for r in rows})},
                             missing=len(rows)-len(measured))
    return summary


def plot(metrics, path):
    """Per-tag scatter of X-Y error vectors around each tag centre, one panel per error type."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = metrics['approaches']
    errors = [np.linalg.norm(r[f'{k}_error_xy_m']) for r in rows for k in ERROR_TYPES
              if r[f'{k}_error_xy_m'] is not None]
    # Errors are centimetres around tags a metre apart; magnify them to a readable radius.
    scale = min(10., .12/max(errors)) if errors and max(errors) > 0 else 10.
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    tags = {r['tag_id']: r['tag_centre_xy_m'] for r in rows}
    for axis, kind in zip(axes, ERROR_TYPES):
        summary = metrics['summary'][kind]
        for tag, centre in sorted(tags.items()):
            vectors = np.array([r[f'{kind}_error_xy_m'] for r in rows if r['tag_id'] == tag
                                and r[f'{kind}_error_xy_m'] is not None], float).reshape(-1, 2)
            axis.plot(*centre, marker='+', color='black', markersize=14)
            if not len(vectors):
                continue
            points = np.asarray(centre)+vectors*scale
            axis.scatter(points[:, 0], points[:, 1], s=12, alpha=.6)
            stats = summary['per_tag'][str(tag)]
            mean = np.asarray(centre)+np.asarray(stats['bias_xy_m'])*scale
            axis.plot(*mean, marker='x', color='red', markersize=9)
            axis.add_patch(plt.Circle(centre, stats['rmse_m']*scale, fill=False, linestyle='--', color='grey'))
            axis.annotate(f'tag {tag}\nmean {stats["mean_error_m"]*100:.1f} cm\nRMSE {stats["rmse_m"]*100:.1f} cm',
                          centre, xytext=(8, 8), textcoords='offset points', fontsize=8)
        overall = summary['overall']
        text = (f'mean {overall["mean_error_m"]*100:.1f} cm, RMSE {overall["rmse_m"]*100:.1f} cm, n={overall["n"]}'
                if overall['n'] else 'no measurements')
        axis.set_title(f'{kind.capitalize()} error ({text})\nvectors magnified {scale:.1f}x; '
                       'x = mean, dashed circle = RMSE', fontsize=10)
        axis.set_xlabel('Lighthouse X (m)')
        axis.set_ylabel('Lighthouse Y (m)')
        axis.set_aspect('equal', adjustable='datalim')
        axis.grid(alpha=.3)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_metrics(session_dir):
    """Rebuild metrics.json and approach-errors.png from a session's saved trial runs."""
    root = Path(session_dir)
    session = artifacts.read(root/'session.json')
    if session.get('kind') != 'approach-accuracy':
        raise ValueError(f'{root} is not an approach-accuracy session')
    surveys = {c['index']: artifacts.read(root/c['survey']) for c in calibrations(session)}
    rows, included, excluded, launches = [], [], [], []
    for trial in session['trials']:
        if not trial.get('accepted'):
            excluded.append(trial['dir'])
            continue
        events = [json.loads(line) for line in (root/trial['dir']/'events.jsonl').read_text().splitlines()]
        calibration = trial.get('calibration', 1)
        trial_rows = approaches(events, centres(surveys[calibration]), trial['index'])
        rows.extend(dict(row, calibration=calibration) for row in trial_rows)
        launch = launch_offset(events, trial['index'])
        launches.append(None if launch is None else dict(launch, calibration=calibration))
        included.append(trial['dir'])
    metrics = dict(version=1, kind='approach-accuracy-metrics', units='metres, Lighthouse frame',
                   trials_included=included, trials_excluded=excluded,
                   error_definitions=dict(
                       positioning='camera deck centre (ray meeting the median depth height) minus surveyed tag centre',
                       control='mean Lighthouse deck position minus commanded deck centre'),
                   settled_window='from 0.5 s after each segment, for the dwell',
                   known_limitations=LIMITATIONS, approaches=rows, summary=summarize(rows),
                   calibrations=calibrations(session),
                   summary_by_calibration={str(c): summarize([r for r in rows if r['calibration'] == c])
                                           for c in sorted({r['calibration'] for r in rows})},
                   launch_offsets=[l for l in launches if l is not None],
                   launch_offset_note='Lighthouse minus camera pad position before takeoff; drift between trials '
                                      'means the stations moved relative to the floor tags or the reset was off its mark')
    artifacts.write(root/'metrics.json', metrics)
    plot(metrics, root/'approach-errors.png')
    return metrics


def capture_survey(args, rig, out, image_name='survey.png'):
    """Survey the tag centres once per session, with the drone resting on its launch pad."""
    from .camera import cv, open_camera, intrinsics_compatibility
    from .vision import detect
    calibration = artifacts.read(Path(args.calibration)/'report.json')
    saved = artifacts.read(Path(args.calibration)/'intrinsics.json')
    camera = open_camera(args)
    try:
        if camera.profile != calibration['profile']:
            raise ValueError('Camera profile differs from accepted calibration')
        compatibility = intrinsics_compatibility(saved, camera.intrinsics)
        if not compatibility['compatible'] and not args.demo:
            raise ValueError(f'Camera lens mapping changed by {compatibility["max_pixel_drift"]:.3f} px; recalibrate')
        ids = {t['id'] for t in rig['tags']}
        for _ in range(15):  # let exposure and depth settle
            camera.frame()
        observations, deadline, frame = [], time.monotonic()+30, None
        while len(observations) < 20 and time.monotonic() < deadline:
            frame = camera.frame()
            found = detect(frame.color)
            if ids.issubset(found):
                observations.append({k: found[k] for k in ids})
        if len(observations) < 20:
            raise ValueError(f'Could not collect 20 frames showing every tag {sorted(ids)}; '
                             'keep the drone on its pad and the tags clear')
        quads = {k: np.median([o[k] for o in observations], axis=0) for k in ids}
        # The cage carrying the camera can shift, so always refit the camera to the floor tags.
        from .registration import refresh_pose
        intrinsics = camera.intrinsics if args.demo else saved
        registration = refresh_pose(rig, quads, intrinsics, calibration['T_lighthouse_camera'])
        pose = registration['T_lighthouse_camera']
        result = survey_tags(quads, pose, intrinsics, rig)
        picture = frame.color.copy()
        for quad in quads.values():
            cv().polylines(picture, [np.rint(quad).astype(np.int32)], True, (0, 255, 0), 2)
        cv().imwrite(str(Path(out)/image_name), picture)
        result.update(frames=len(observations), corners_px={str(k): v.tolist() for k, v in quads.items()},
                      camera_pose_source='refit_to_floor_tags',
                      registration=registration, intrinsics=intrinsics, intrinsics_check=compatibility)
        return result
    finally:
        camera.close()


def calibrations(session):
    """The setups a session has flown with, in order; sessions before recalibration support had one."""
    return session.get('calibrations') or [dict(index=1, setup=session.get('setup'), from_trial=1,
                                                rig_fingerprint=session.get('rig_fingerprint'),
                                                rig='rig.json', survey='survey.json')]


def recalibrate(args, rig, root, session):
    """Continue a session on a new setup: same tags, new survey; later trials use it."""
    first = artifacts.read(root/'rig.json')
    if sorted(t['id'] for t in first['tags']) != sorted(t['id'] for t in rig['tags']):
        raise ValueError('The new setup uses different tags; start a new session')
    history = calibrations(session)
    index = history[-1]['index']+1
    print(f'New setup {args.setup}: re-surveying tag centres for calibration {index}; '
          'keep the drone on its launch pad and the tags clear.', flush=True)
    survey = capture_survey(args, rig, root, image_name=f'survey-{index:02d}.png')
    artifacts.write(root/f'survey-{index:02d}.json', survey)
    artifacts.write(root/f'rig-{index:02d}.json', rig)
    history.append(dict(index=index, setup=args.setup, rig_fingerprint=artifacts.fingerprint(rig),
                        rig=f'rig-{index:02d}.json', survey=f'survey-{index:02d}.json',
                        from_trial=1+max((t['index'] for t in session['trials']), default=0),
                        created_utc=datetime.now(timezone.utc).isoformat()))
    session['calibrations'] = history
    session['rig_fingerprint'] = history[-1]['rig_fingerprint']
    artifacts.write(root/'session.json', session)
    for warning in survey['warnings']:
        print(f'Warning: {warning}', flush=True)
    print(f'Continuing the session on calibration {index}; earlier trials keep their own calibration.', flush=True)
    return survey


def session_settings(args):
    return dict(demo=bool(args.demo), controller=args.controller or ('pid' if args.demo else 'mellinger'),
                dwell_s=float(args.dwell_s))


def run_session(args, rig, ask=input):
    """Dry run by default; with --arm, fly trials into trial-NN/ until the session's target is accepted.

    The target is stored in session.json. ``--resume DIR --trials N`` changes it, alone
    (no devices opened) or together with ``--arm``.
    """
    from . import hover
    if args.trials is not None and not 1 <= args.trials <= 100:
        raise ValueError('--trials must be between 1 and 100')
    live = args.arm or args.check_only or args.camera_check_only
    if args.resume:
        root = Path(args.resume)
        session = artifacts.read(root/'session.json')
        if session.get('kind') != 'approach-accuracy' or session.get('dry_run'):
            raise ValueError(f'{root} is not a live approach-accuracy session')
        previous = session.get('target_trials', DEFAULT_TRIALS)
        if args.trials is not None and args.trials != previous:
            session['target_trials'] = args.trials
            artifacts.write(root/'session.json', session)
            print(f'Session target changed from {previous} to {args.trials} accepted trials.')
        if not live:
            if args.trials is None:
                raise ValueError('--resume continues live trials; add --arm or --check-only, or --trials N to change the target')
            return root
        if session['settings'] != session_settings(args):
            raise ValueError(f'Session settings {session["settings"]} differ; resume with the same --demo, '
                             '--controller and --dwell-s')
        if calibrations(session)[-1]['rig_fingerprint'] != artifacts.fingerprint(rig):
            survey = recalibrate(args, rig, root, session)
        else:
            survey = artifacts.read(root/calibrations(session)[-1]['survey'])
    else:
        root = Path(args.out)
        if root.exists():
            raise ValueError(f'{root} already exists; choose a new --out (sessions are never overwritten), '
                             'or continue it with --resume')
        root.mkdir(parents=True)
        session = dict(version=1, kind='approach-accuracy', created_utc=datetime.now(timezone.utc).isoformat(),
                       dry_run=not live, rig_fingerprint=artifacts.fingerprint(rig), settings=session_settings(args),
                       setup=args.setup, calibration=args.calibration, lighthouse=args.lighthouse,
                       target_trials=args.trials or DEFAULT_TRIALS, trials=[])
        artifacts.write(root/'rig.json', rig)
        artifacts.write(root/'session.json', session)
        if not live:
            trial = copy.copy(args)
            trial.out = str(root/'preview')
            hover.run(trial, rig, ask=ask)
            print(f'DRY RUN: route from rig.json tag centres; no devices opened. Preview: {root/"preview"}')
            return root
        print('Surveying tag centres; keep the drone on its launch pad and the tags clear.', flush=True)
        survey = capture_survey(args, rig, root)
        artifacts.write(root/'survey.json', survey)
        for warning in survey['warnings']:
            print(f'Warning: {warning}', flush=True)
    trial_args = copy.copy(args)
    trial_args.tag_centres = centres(survey)
    if not args.arm:
        # Preflight and camera checks share the session survey but are not trials.
        index = 1+sum(1 for _ in root.glob('preflight-*'))
        trial_args.out = str(root/f'preflight-{index:02d}')
        return hover.run(trial_args, rig, ask=ask)

    def save():
        artifacts.write(root/'session.json', session)
        if args.save_metrics:
            write_metrics(root)

    target = session.get('target_trials', DEFAULT_TRIALS)
    while True:
        accepted = sum(1 for t in session['trials'] if t['accepted'])
        if accepted >= target:
            print(f'{accepted} accepted trials recorded; session complete.')
            break
        index = 1+max((t['index'] for t in session['trials']), default=0)
        answer = ask(f'Reset the drone on its launch pad and clear the area. Enter to fly trial {index} '
                     f'({accepted}/{target} accepted), or q to stop: ').strip().lower()
        if answer in ('q', 'quit'):
            print(f'Stopped after {accepted} accepted trials; continue with --resume {root}')
            break
        trial_args.out = str(root/f'trial-{index:02d}')
        record = dict(index=index, dir=f'trial-{index:02d}', accepted=False,
                      calibration=calibrations(session)[-1]['index'])
        try:
            output = hover.run(trial_args, rig, ask=ask)
            report = artifacts.read(output/'report.json')
            record.update(accepted=bool(report.get('accepted')), flight_complete=bool(report.get('flight_complete')))
        except KeyboardInterrupt:
            record.update(reason='interrupted')
            session['trials'].append(record)
            save()
            raise
        except Exception as e:
            record.update(reason=str(e))
            print(f'Trial {index} failed: {e}', flush=True)
        session['trials'].append(record)
        save()
    if args.save_metrics:
        write_metrics(root)
    return root
