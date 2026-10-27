"""Exp 3: X-Y approach accuracy over surveyed AprilTag centres (issue #24).

Positioning error is the camera-measured drone position minus the surveyed tag centre;
control error is the Lighthouse-reported deck position minus the commanded centre.
The camera measures the drone by the white square of the small tag on it when the session sets
one (issue #40), otherwise, for earlier sessions, by the detection box centre.
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
DRONE_TAG_GATE_M = .05  # white-square candidates farther than this from Lighthouse are not the drone
SHARED_LIMITATIONS = [
    'The camera pose is refit to the floor tags every frame, so surveyed centres match rig.json by construction '
    'and camera motion does not bias positioning error.',
    'Only X-Y is evaluated; Z is out of scope.',
]
LIMITATIONS = dict(
    detection=['Drone tilt is ignored: the detection centre is treated as the point straight above the deck centre.',
               'The camera measures the visual centre of the detection box, not the Lighthouse deck sensor centre.',
               *SHARED_LIMITATIONS],
    drone_tag=['The drone tag is about 10 px across and cannot be decoded; it is located as a bright square '
               'blob, so prop blur, shadows or uneven lighting on the white square can shift its centroid.',
               'The tag height is Lighthouse Z plus the measured tag height above the deck; a height error moves '
               'X-Y along the camera ray (by the error times the ray slope).',
               'Drone tilt shifts the tag X-Y by its height above the deck centre times sin(tilt).',
               'Any X-Y offset between the mounted tag centre and the deck centre appears as a constant bias.',
               f'Lighthouse picks which white-square candidate is the drone: the nearest within '
               f'{DRONE_TAG_GATE_M*100:.0f} cm. A larger camera-Lighthouse disagreement counts as gated out, not '
               'as an error.',
               *SHARED_LIMITATIONS])
POSITIONING = dict(
    detection='camera deck centre (ray meeting the median depth height) minus surveyed tag centre',
    drone_tag='white tag square centroid ray meeting the plane at Lighthouse Z plus tag height '
              '(candidate nearest Lighthouse), minus surveyed tag centre')


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


def plane_xy(origins, directions, height):
    """X-Y where camera rays meet a horizontal plane at ``height``."""
    origins, directions = np.asarray(origins, float), np.asarray(directions, float)
    return origins[..., :2]+directions[..., :2]*((height-origins[..., 2])/directions[..., 2])[..., None]


def ray_xy(frames, key, height):
    """Mean X-Y where the frames' camera rays meet a horizontal plane at ``height``."""
    return plane_xy([f[key]['origin'] for f in frames], [f[key]['direction'] for f in frames], height).mean(axis=0)


def deck_xy(frames):
    """Deck X-Y from detection rays meeting a horizontal plane at the median depth height."""
    points = np.array([f['drone_world_m'] for f in frames], float)
    height = float(np.median(points[:, 2]))
    return dict(xy_m=ray_xy(frames, 'drone_ray_world_m', height).tolist(),
                depth_xy_m=points[:, :2].mean(axis=0).tolist(), deck_height_m=height, frames=len(frames))


def drone_tag_point(frame, height, near_xy, gate_m=DRONE_TAG_GATE_M):
    """X-Y of the frame's white-square candidate nearest ``near_xy`` at ``height``, within the gate."""
    best = None
    for candidate in frame['drone_tag_candidates']:
        xy = plane_xy(frame['drone_tag_origin_m'], candidate['direction'], height)
        distance = np.linalg.norm(xy-near_xy)
        if distance <= gate_m and (best is None or distance < best[0]):
            best = (distance, xy)
    return None if best is None else best[1]


def drone_tag_xy(frames, tag, lighthouse_xy, lighthouse_z):
    """Drone X-Y from white-square rays meeting a plane at the tag's height.

    The tag is too small for its own range, so its height is the median Lighthouse Z plus the
    tag's measured height above the deck. Lighthouse also picks which candidate is the drone
    (the nearest within the gate); the X-Y itself is the camera's.
    """
    if not len(lighthouse_z) or not len(lighthouse_xy):
        return None
    height = float(np.median(lighthouse_z))+tag['height_m']
    near = np.mean(lighthouse_xy, axis=0)
    points = [p for p in (drone_tag_point(f, height, near) for f in frames) if p is not None]
    return dict(xy_m=np.mean(points, axis=0).tolist() if points else None, depth_xy_m=None,
                deck_height_m=height, frames=len(points), gated_out=len(frames)-len(points))


def camera_frames(events, tag=None):
    """Camera statuses that measured the drone, one per capture time (the monitor may re-emit one)."""
    keys = ('drone_tag_candidates',) if tag else ('drone_ray_world_m', 'drone_world_m')
    return sorted({e['time']: e for e in events if e.get('event') == 'camera'
                   and all(k in e for k in keys)}.items())


def drone_xy(frames, tag, lighthouse_xy, lighthouse_z):
    return drone_tag_xy(frames, tag, lighthouse_xy, lighthouse_z) if tag else deck_xy(frames)


def approaches(events, centres, trial, drone_tag=None):
    """One row per tag dwell that completed: Lighthouse and camera X-Y over the settled window.

    With ``drone_tag`` (the session setting) the camera measures the drone by the white tag
    square on it; without, as in sessions before issue #40, by the detection box centre.
    """
    centres = {int(k): v for k, v in centres.items()}
    targets = [e for e in events if e.get('event') == 'target']
    telemetry = [(e['host_time'], e['values']) for e in events
                 if e.get('event') == 'telemetry' and 'stateEstimate.x' in e['values']]
    cameras = camera_frames(events, drone_tag)
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
        window = [v for t, v in telemetry if start <= t <= end]
        lighthouse = np.array([[v['stateEstimate.x'], v['stateEstimate.y']] for v in window], float).reshape(-1, 2)
        lighthouse_z = [v['stateEstimate.z'] for v in window if 'stateEstimate.z' in v]
        frames = [e for t, e in cameras if start <= t <= end]
        row = dict(trial=trial, tag_id=tag, positioning_source='drone_tag' if drone_tag else 'detection',
                   tag_centre_xy_m=centre.tolist(), commanded_xy_m=commanded.tolist(),
                   lighthouse_samples=len(lighthouse), camera_frames=len(frames), camera_gated_out=None,
                   lighthouse_xy_m=None, camera_xy_m=None, camera_depth_xy_m=None, deck_height_m=None,
                   camera_alignment_rms_px=None, positioning_error_xy_m=None, positioning_error_m=None,
                   control_error_xy_m=None, control_error_m=None)
        if len(lighthouse):
            reported = lighthouse.mean(axis=0)
            error = reported-commanded
            row.update(lighthouse_xy_m=reported.tolist(), control_error_xy_m=error.tolist(),
                       control_error_m=float(np.linalg.norm(error)))
        deck = drone_xy(frames, drone_tag, lighthouse, lighthouse_z) if frames else None
        if deck and drone_tag:
            row.update(camera_frames=deck['frames'], camera_gated_out=deck['gated_out'])
        if deck and deck['xy_m'] is not None:
            # sqrt((x_drone - x_tag)^2 + (y_drone - y_tag)^2), with the drone at its mean over the dwell.
            error = np.asarray(deck['xy_m'])-centre
            alignment = [f['alignment_rms_px'] for f in frames if f.get('alignment_rms_px') is not None]
            row.update(camera_xy_m=deck['xy_m'], camera_depth_xy_m=deck['depth_xy_m'],
                       deck_height_m=deck['deck_height_m'],
                       camera_alignment_rms_px=float(np.median(alignment)) if alignment else None,
                       positioning_error_xy_m=error.tolist(), positioning_error_m=float(np.linalg.norm(error)))
        rows.append(row)
    return rows


def launch_offset(events, trial, drone_tag=None):
    """Before takeoff: Lighthouse pad position minus the camera's (tag-frame) pad position.

    The camera is refit to the floor tags, so a change in this offset between trials means the
    Lighthouse stations moved relative to the floor (or the drone was reset off its mark).
    """
    targets = [e for e in events if e.get('event') == 'target']
    if not targets:
        return None
    takeoff = targets[0]['t']
    pad = [e['values'] for e in events if e.get('event') == 'telemetry' and 'stateEstimate.x' in e['values']
           and e['host_time'] < takeoff]
    lighthouse = np.array([[v['stateEstimate.x'], v['stateEstimate.y']] for v in pad], float).reshape(-1, 2)
    frames = [e for t, e in camera_frames(events, drone_tag) if t < takeoff]
    deck = drone_xy(frames, drone_tag, lighthouse, [v['stateEstimate.z'] for v in pad if 'stateEstimate.z' in v]) \
        if frames else None
    if not len(lighthouse) or not deck or deck['xy_m'] is None:
        return dict(trial=trial, lighthouse_xy_m=None, camera_xy_m=None, offset_xy_m=None, offset_m=None)
    camera = np.asarray(deck['xy_m'])
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
    tag = session.get('settings', {}).get('drone_tag')
    source = 'drone_tag' if tag else 'detection'
    rows, included, excluded, launches = [], [], [], []
    for trial in session['trials']:
        if not trial.get('accepted'):
            excluded.append(trial['dir'])
            continue
        events = [json.loads(line) for line in (root/trial['dir']/'events.jsonl').read_text().splitlines()]
        calibration = trial.get('calibration', 1)
        trial_rows = approaches(events, centres(surveys[calibration]), trial['index'], tag)
        rows.extend(dict(row, calibration=calibration) for row in trial_rows)
        launch = launch_offset(events, trial['index'], tag)
        launches.append(None if launch is None else dict(launch, calibration=calibration))
        included.append(trial['dir'])
    metrics = dict(version=1, kind='approach-accuracy-metrics', units='metres, Lighthouse frame',
                   trials_included=included, trials_excluded=excluded, positioning_source=source,
                   drone_tag=tag,
                   error_definitions=dict(
                       positioning=POSITIONING[source],
                       control='mean Lighthouse deck position minus commanded deck centre'),
                   settled_window='from 0.5 s after each segment, for the dwell',
                   known_limitations=LIMITATIONS[source], approaches=rows, summary=summarize(rows),
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


def drone_tag_check(args, rig):
    """Camera only, never the radio: how reliably the overhead camera finds the drone tag (issue #40).

    Place or hold the drone under the camera with its tag ``--height-m`` above the floor tags.
    Every frame refits the camera to the visible floor tags and lists the white-square candidates;
    the drone is the one nearest ``--near-px`` (within ``--gate-px``). Without ``--near-px`` the
    check only circles every candidate in drone-tag-check.png so you can pick the drone's.
    """
    from .camera import cv, open_camera, intrinsics_compatibility
    from .vision import detect, drone_tag
    from .hover import WARMUP_FRAMES, refit_frame
    if not (np.isfinite(args.drone_tag_size_m) and 0 < args.drone_tag_size_m <= .1):
        raise ValueError('--drone-tag-size-m must be between 0 and 0.1 m')
    if not (np.isfinite(args.height_m) and 0 <= args.height_m <= 1.):
        raise ValueError('--height-m must be between 0 and 1 m')
    if not 1 <= args.frames <= 3000:
        raise ValueError('--frames must be between 1 and 3000')
    out = Path(args.out)
    if out.exists():
        raise ValueError(f'{out} already exists; choose a new --out (checks are never overwritten)')
    tag = dict(size_m=float(args.drone_tag_size_m))
    near = None if args.near_px is None else np.asarray(args.near_px, float)
    floor_z = float(np.mean([t['pose']['pos'][2] for t in rig['tags']]))
    calibration = artifacts.read(Path(args.calibration)/'report.json')
    saved = artifacts.read(Path(args.calibration)/'intrinsics.json')
    camera = open_camera(args)
    try:
        if camera.profile != calibration['profile']:
            raise ValueError('Camera profile differs from accepted calibration')
        compatibility = intrinsics_compatibility(saved, camera.intrinsics)
        if not compatibility['compatible'] and not args.demo:
            raise ValueError(f'Camera lens mapping changed by {compatibility["max_pixel_drift"]:.3f} px; recalibrate')
        intrinsics = camera.intrinsics if args.demo else saved
        out.mkdir(parents=True)
        for _ in range(WARMUP_FRAMES):
            camera.frame()
        start = pose = Pose.from_dict(calibration['T_lighthouse_camera'])
        records, picture = [], None
        for _ in range(args.frames):
            frame = camera.frame()
            found = detect(frame.color)
            pose, registration = refit_frame(rig, found, intrinsics, pose, start)
            record = dict(time=frame.timestamp, visible_ids=sorted(found), frame_registration=registration,
                          **drone_tag(frame.color, found, tag, intrinsics, pose.to_dict(), floor_z))
            candidates = record['drone_tag_candidates']
            if near is not None and candidates:
                distances = [np.linalg.norm(np.subtract(c['center_px'], near)) for c in candidates]
                if min(distances) <= args.gate_px:
                    record['drone_tag'] = candidates[int(np.argmin(distances))]
            records.append(record)
            if picture is None:  # first frame: every candidate circled, the drone's in green
                picture = frame.color.copy()
                for c in candidates:
                    centre = tuple(int(v) for v in np.rint(c['center_px']))
                    colour = (0, 255, 0) if c is record.get('drone_tag') else (0, 0, 255)
                    cv().circle(picture, centre, 12, colour, 2)
                    cv().putText(picture, f'{centre[0]},{centre[1]}', (centre[0]+14, centre[1]),
                                 cv().FONT_HERSHEY_SIMPLEX, .4, colour, 1)
        cv().imwrite(str(out/'drone-tag-check.png'), picture)
    finally:
        camera.close()
    measured = [r for r in records if 'drone_tag' in r]
    height = floor_z+args.height_m
    xy = np.array([plane_xy(r['drone_tag_origin_m'], r['drone_tag']['direction'], height) for r in measured],
                  float).reshape(-1, 2)
    pixels = np.array([r['drone_tag']['center_px'] for r in measured], float).reshape(-1, 2)

    def median(key):
        return np.median([r['drone_tag'][key] for r in measured], axis=0).tolist() if measured else None
    report = dict(version=1, kind='drone-tag-check', drone_tag=tag, radio_connected=False,
                  calibration=str(args.calibration), intrinsics_check=compatibility,
                  floor_z_m=floor_z, tag_height_above_floor_m=args.height_m,
                  near_px=args.near_px, gate_px=args.gate_px, frames=len(records),
                  candidates_per_frame=float(np.median([len(r['drone_tag_candidates']) for r in records])),
                  detected=len(measured), detection_rate=len(measured)/len(records),
                  center_px=pixels.mean(axis=0).tolist() if measured else None,
                  center_std_px=pixels.std(axis=0).tolist() if measured else None,
                  xy_m=xy.mean(axis=0).tolist() if measured else None,
                  # RMS X-Y distance from the mean position: the frame-to-frame spread of a still drone.
                  xy_std_m=float(np.sqrt(np.sum(xy.var(axis=0)))) if measured else None,
                  size_px=median('size_px'), contrast=median('contrast'), records=records)
    artifacts.write(out/'drone-tag-check.json', report)
    return report


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


def drone_tag_setting(args):
    """The white tag square on the drone (--drone-tag-size-m, --drone-tag-height-m), or None."""
    size, height = getattr(args, 'drone_tag_size_m', None), getattr(args, 'drone_tag_height_m', None)
    if size is None and height is None:
        return None
    if size is None or height is None:
        raise ValueError('Give both --drone-tag-size-m (white square edge) and --drone-tag-height-m '
                         '(tag surface above the Lighthouse deck position), in metres')
    if not (np.isfinite(size) and 0 < size <= .1):
        raise ValueError('--drone-tag-size-m must be between 0 and 0.1 m')
    if not (np.isfinite(height) and -.1 <= height <= .1):
        raise ValueError('--drone-tag-height-m must be between -0.1 and 0.1 m')
    return dict(size_m=float(size), height_m=float(height))


def session_settings(args):
    settings = dict(demo=bool(args.demo), controller=args.controller or ('pid' if args.demo else 'mellinger'),
                    dwell_s=float(args.dwell_s))
    # Only sessions that use a drone tag record it, so earlier sessions still resume.
    tag = drone_tag_setting(args)
    return settings if tag is None else dict(settings, drone_tag=tag)


def run_session(args, rig, ask=input):
    """Dry run by default; with --arm, fly trials into trial-NN/ until the session's target is accepted.

    The target is stored in session.json. ``--resume DIR --trials N`` changes it, alone
    (no devices opened) or together with ``--arm``.
    """
    from . import hover
    if args.trials is not None and not 1 <= args.trials <= 100:
        raise ValueError('--trials must be between 1 and 100')
    live = args.arm or args.check_only or args.camera_check_only
    drone_tag_setting(args)
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
