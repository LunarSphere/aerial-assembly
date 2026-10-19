"""Compact cfclient-style setup: four placement prompts, automatic radio and camera survey."""
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from ..frames import Pose, xyzw_to_wxyz
from ..clock import RealClock
from ..preflight import wait_for_parameters
from . import artifacts, lighthouse, layout
from .camera import cv, RealSenseCamera, project_points
from .discovery import scan_uri, discover_stations
from .drone import NanoDetector, observe
from .presets import default_hardware
from .vision import detect

PROMPTS = (
    '1/4 Place the drone LEVEL on its support where you want the origin. Face its nose along your intended +X direction. Press Enter.',
    '2/4 Move the SAME support and drone exactly 1 metre along +X. Keep the same height and nose direction. Press Enter.',
    '3/4 Put the drone and support elsewhere on the SAME level plane, away from the X-axis. Keep the same height and nose direction. Press Enter.',
    '4/4 Return the drone to the ORIGINAL launch support and nose direction. Press Enter.',
)


def automatic_camera(args):
    """The one D455; USB 2 uses the profiles observed on this actual camera."""
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise RuntimeError('Setup needs flight[realsense] for the D455.') from e
    devices = [d for d in rs.context().query_devices() if 'D455' in d.get_info(rs.camera_info.name)]
    if args.device:
        devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == args.device]
    if len(devices) != 1:
        raise RuntimeError('Connect one D455, or select it with --device SERIAL')
    device = devices[0]
    args.device = device.get_info(rs.camera_info.serial_number)
    usb2 = device.get_info(rs.camera_info.usb_type_descriptor).startswith('2')
    args.fps, args.depth_width = (15, 640) if usb2 else (30, 848)
    if usb2 and (args.width, args.height) == (1280, 720):
        args.width, args.height = 640, 480
        print('D455 is on USB 2: using 640×480 color/depth at 15 fps to reduce USB traffic.')
    return RealSenseCamera(args.device, args.width, args.height, args.fps, args.depth_width)


def environment(camera, output):
    frames, quads = [], []
    print(f'Camera profile: {camera.profile}. Warming up camera, then looking for AprilTags 25h9...')
    for _ in range(15):
        camera.frame()
    deadline, next_status = time.monotonic()+30, 0.
    seen_ids, seen_counts, attempts = set(), {}, 0
    frame, found = None, {}
    while len(frames) < 20 and time.monotonic() < deadline:
        frame = camera.frame()
        found = detect(frame.color)
        attempts += 1
        seen_ids.update(found)
        for tag_id in found:
            seen_counts[tag_id] = seen_counts.get(tag_id, 0)+1
        if {0, 1, 2}.issubset(found) and len(found) >= 5:
            frames.append(frame)
            quads.append(found)
        if time.monotonic() >= next_status:
            print(f'Tags visible: {sorted(found)}; usable frames {len(frames)}/20. Need 0, 1, 2 plus two other tags.')
            cv().imwrite(str(output/'tag-visibility.png'), frame.color)
            next_status = time.monotonic()+2
    artifacts.write(output/'tag-visibility.json', dict(version=1, profile=camera.profile,
        frames_examined=attempts, usable_frames=len(frames), ids_seen=sorted(seen_ids),
        counts={str(k): v for k, v in seen_counts.items()}, last_ids=sorted(found),
        last_frame_metadata={} if frame is None else frame.metadata))
    if len(frames) < 20:
        if frame is not None:
            cv().imwrite(str(output/'tag-visibility.png'), frame.color)
        raise RuntimeError(f'Insufficient tag observations: {len(frames)}/20 usable frames; '
                           f'last visible IDs {sorted(found)}, ever seen {sorted(seen_ids)}. '
                           f'Need 0,1,2 and two other tags together. Inspect {output/"tag-visibility.png"} '
                           f'and {output/"tag-visibility.json"}.')
    print(f'Tag layout capture complete: 20 usable frames; IDs {sorted(set.intersection(*(set(q) for q in quads)))}.')
    ids = set.intersection(*(set(v) for v in quads))
    observations = {i: np.median([q[i] for q in quads], axis=0) for i in ids}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        depth = np.nanmedian(np.array([f.depth_m for f in frames]), axis=0)
    np.save(output/'floor-depth-m.npy', depth)
    result = layout.estimate(observations, camera.intrinsics, depth)
    artifacts.write(output/'layout.json', result)
    artifacts.write(output/'intrinsics.json', camera.intrinsics)
    cv().imwrite(str(output/'floor.png'), layout.annotate(frames[-1].color, observations, result))
    return result


def stable_drone_observation(camera, detector, output, label, timeout=10.):
    """Wait for three consecutive visible, stationary detections at a placement."""
    deadline, stable, last_error, announced = time.monotonic()+timeout, [], '', False
    image_path = output/f'{label}-last.png'
    while time.monotonic() < deadline:
        frame = camera.frame()
        try:
            observation = observe(frame, camera.intrinsics, detector)
        except RuntimeError as error:
            stable.clear()
            last_error = str(error)
            cv().imwrite(str(image_path), frame.color)
            artifacts.write(output/f'{label}-visibility.json', dict(version=1, accepted=False,
                reason=last_error, image=image_path.name, frame_metadata=frame.metadata,
                best_confidence=getattr(detector, 'last_max_confidence', None),
                required_confidence=getattr(detector, 'confidence', None)))
            if not announced:
                print('Waiting for a stable drone detection. Move hands away and keep the drone still and visible from above.')
                announced = True
            continue
        if stable and np.linalg.norm(np.subtract(observation['center_px'], stable[0]['center_px'])) > 3:
            stable.clear()
            last_error = 'Drone position was moving between camera frames'
        stable.append(observation)
        if len(stable) >= 3:
            observation = dict(observation, stable_frames=len(stable),
                               minimum_confidence=min(s['confidence'] for s in stable))
            observation['tag_centers_px'] = {str(i): np.mean(q, axis=0).tolist()
                                              for i, q in detect(frame.color).items()}
            artifacts.write(output/f'{label}-visibility.json', dict(version=1, accepted=True, observation=observation))
            return frame, observation
    if 'frame' in locals():
        cv().imwrite(str(image_path), frame.color)
        artifacts.write(output/f'{label}-visibility.json', dict(version=1, accepted=False,
            reason=last_error or 'Not enough stationary detections before timeout', image=image_path.name,
            frame_metadata=frame.metadata, consecutive_detections=len(stable),
            best_confidence=getattr(detector, 'last_max_confidence', None),
            required_confidence=getattr(detector, 'confidence', None)))
    raise RuntimeError(f'Could not obtain three stable drone detections within {timeout:g} seconds: '
                       f'{last_error}. Inspect {image_path} and {output/(label+"-visibility.json")}.')


def paired_sample(cf, camera, detector, stations, role, output, index):
    before_frame, before = stable_drone_observation(camera, detector, output, f'drone-{index:02d}-before')
    sample = lighthouse.collect(cf, stations, diagnostics_path=output/f'drone-{index:02d}-lighthouse-stream.json')
    frame, after = stable_drone_observation(camera, detector, output, f'drone-{index:02d}-after')
    if np.linalg.norm(np.subtract(before['center_px'], after['center_px'])) > 3:
        raise RuntimeError('Drone moved during measurement; repeat with the drone held still')
    if np.linalg.norm(np.subtract(before['point_camera_m'], after['point_camera_m'])) > .01:
        raise RuntimeError('Drone depth changed during measurement; hold still and retry')
    point = np.mean([before['point_camera_m'], after['point_camera_m']], axis=0).tolist()
    observation = dict(center_px=np.mean([before['center_px'], after['center_px']], axis=0).tolist(),
                       point_camera_m=point, before=before, after=after,
                       synchronization='stationary bracketed capture', image=f'drone-{index:02d}.jpg')
    box = np.rint(after['bbox_xyxy']).astype(int)
    image = frame.color.copy()
    cv().rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
    cv().imwrite(str(output/observation['image']), image)
    return dict(role=role, **sample, camera=observation)


def collect_samples(cf, camera, detector, stations, output, ask=input):
    """Three reference placements: origin, +X and an off-axis point on the same plane."""
    raw = dict(version=1, base_stations=stations, samples=[], reference_only=True)
    def capture(role):
        sample = paired_sample(cf, camera, detector, stations, role, output, len(raw['samples']))
        if role in ('x', 'xy') and raw['samples']:
            origin = raw['samples'][0]['camera']
            current = sample['camera']
            original_tags = origin['before'].get('tag_centers_px', {})
            current_tags = current['before'].get('tag_centers_px', {})
            common = set(original_tags)&set(current_tags)
            if len(common) >= 2:
                drift = float(np.median([np.linalg.norm(np.subtract(original_tags[i], current_tags[i])) for i in common]))
                if drift > 3:
                    raise RuntimeError(f'Camera or floor tags moved between reference placements ({drift:.1f} px median shift). '
                                       'Secure the camera, leave the tags fixed, and restart setup; the reference views must share one camera pose.')
            if np.linalg.norm(np.subtract(current['center_px'], origin['center_px'])) < 10:
                raise RuntimeError('Camera still sees the drone at the origin after the reference move. Color may be frozen or the detector is tracking the wrong object; inspect drone images and reconnect the camera.')
        raw['samples'].append(sample)
        artifacts.write(output/'samples.json', raw)
        print(f'Recorded {role}: {len(raw["samples"])} total positions.')
    for prompt, role in zip(PROMPTS[:3], ('origin', 'x', 'xy')):
        ask(prompt+' ')
        capture(role)
    return raw


def align_reference_camera(raw, world, intrinsics, max_rms_px=1.):
    """Three RGB-D correspondences determine a provisional rigid transform."""
    points = [s['camera']['point_camera_m'] for s in raw['samples']]
    if len(points) != 3 or any(p is None for p in points):
        raise ValueError('Setup requires RGB-D depth at all three reference placements')
    camera = np.asarray(points, float)
    if camera.shape != (3, 3) or not np.all(np.isfinite(camera)):
        raise ValueError('Invalid reference depth observations')
    a, b = camera-camera.mean(0), world-world.mean(0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        observed, expected = np.linalg.norm(camera[i]-camera[j]), np.linalg.norm(world[i]-world[j])
        if abs(observed-expected) > max(.05, expected*.1):
            raise ValueError(f'Reference camera/Lighthouse distances disagree: {raw["samples"][i]["role"]} to '
                             f'{raw["samples"][j]["role"]}: camera {observed:.3f} m versus Lighthouse {expected:.3f} m. '
                             'Check for frozen color frames or an incorrect drone detection; repeat capture, not the fit tolerance.')
    if min(np.linalg.matrix_rank(a, tol=.01), np.linalg.matrix_rank(b, tol=.01)) < 2:
        raise ValueError('Reference placements must form a triangle; put the XY placement farther off the X-axis')
    u, _, vt = np.linalg.svd(a.T@b)
    correction = np.diag([1., 1., np.linalg.det(vt.T@u.T)])
    r = vt.T@correction@u.T
    t = world.mean(0)-r@camera.mean(0)
    transform = Pose(tuple(t), tuple(xyzw_to_wxyz(Rotation.from_matrix(r).as_quat())))
    inverse = transform.inverse()
    rv = Rotation.from_matrix(r.T).as_rotvec()
    projected = project_points(world, rv, np.asarray(inverse.pos), intrinsics)
    pixels = np.array([s['camera']['center_px'] for s in raw['samples']])
    rms = float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))
    if np.any(inverse.apply(world)[:, 2] <= 0):
        raise ValueError('Reference alignment places the drone behind the camera')
    pair_diagnostics = [dict(roles=[raw['samples'][i]['role'], raw['samples'][j]['role']],
                             camera_distance_m=float(np.linalg.norm(camera[i]-camera[j])),
                             lighthouse_distance_m=float(np.linalg.norm(world[i]-world[j])))
                        for i, j in ((0, 1), (0, 2), (1, 2))]
    sample_diagnostics = [dict(role=s['role'], residual_px=(projected[i]-pixels[i]).tolist(),
                               residual_lighthouse_m=(transform.apply(camera[i])-world[i]).tolist())
                          for i, s in enumerate(raw['samples'])]
    return dict(T_lighthouse_camera=transform.to_dict(), reprojection_rms_px=rms,
                reprojection_limit_px=max_rms_px, rms_limit_overridden=max_rms_px != 1.,
                reference_pair_diagnostics=pair_diagnostics, training_sample_diagnostics=sample_diagnostics,
                training_max_error_m=float(np.max(np.linalg.norm(transform.apply(camera)-world, axis=1))),
                fit_method='three_reference_rgbd_rigid_alignment', fit_valid=bool(rms <= max_rms_px),
                accepted=bool(rms <= max_rms_px))


def align_camera(raw, solution, intrinsics, max_rms_px=1.):
    if not np.isfinite(max_rms_px) or max_rms_px <= 0:
        raise ValueError('Camera RMS limit must be a finite positive number')
    if len(raw['samples']) != len(solution['samples']):
        raise ValueError('Camera/Lighthouse sample counts do not match')
    world = np.array([s['position_m'] for s in solution['samples']], float)
    if world.shape != (len(raw['samples']), 3) or not np.all(np.isfinite(world)):
        raise ValueError('Lighthouse solver did not produce every required pose')
    return align_reference_camera(raw, world, intrinsics, max_rms_px=max_rms_px)


def build_rig(floor, camera_result, stations, hardware):
    transform = Pose.from_dict(camera_result['T_lighthouse_camera'])
    floor_to_lh = transform.compose(Pose.from_dict(floor['T_tagplane_camera']).inverse())
    tags = []
    for tag in floor['tags']:
        tags.append(dict(tag, pose=floor_to_lh.compose(Pose.from_dict(tag['pose'])).to_dict(),
                         survey_source='camera_registered_by_drone', independently_verified=False))
    offset = np.array(hardware['body_to_sample_m'])
    home = -offset  # origin placement defines the deck position [0,0,0]
    points = np.array([t['pose']['pos'] for t in tags])
    pad = .18
    low = np.minimum(points.min(0), home)-[pad, pad, .1]
    high = np.maximum(points.max(0), home)+[pad, pad, .65]
    lower_z = hardware['envelope_body_m']['min'][2]
    emergency = float(np.median(points[:, 2])-lower_z)
    data = dict(version=1, family='25h9', tags=tags, reference_method='human_placements',
                sample_offset_m=(-points[0]).tolist(), base_stations=stations,
                body_to_sample_m=offset.tolist(), home=dict(pos=home.tolist(), yaw=0.),
                emergency_land_body_z_m=emergency, geofence=dict(min=low.tolist(), max=high.tolist()),
                envelope_body_m=hardware['envelope_body_m'], mass_kg=hardware['mass_kg'],
                hardware_preset=hardware, geofence_source='tag extent plus 18 cm; clear workspace assumed')
    from .rig import validate
    return validate(data)


def check_launch_return(origin, home):
    """Require the measured 3D return to be within 2 cm of the origin observation."""
    pixel_error = float(np.linalg.norm(np.subtract(home['center_px'], origin['center_px'])))
    error = float(np.linalg.norm(np.subtract(home['point_camera_m'], origin['point_camera_m'])))
    if not np.isfinite(error) or error > .02:
        raise ValueError(f'Drone is {error*100:.1f} cm from the original launch position/height; return within 2 cm')
    return dict(version=1, accepted=True, metric='depth_3d', error_m=error, limit_m=.02, pixel_error=pixel_error)


def save_completed_setup(root, rig, solution, alignment, intrinsics, profile,
                         detector_path, detector_name, uri, stations, prompts):
    """Persist a fitted setup after its launch return (and any required upload) passed."""
    lh_out, camera_out = root/'lighthouse', root/'camera'
    key = artifacts.fingerprint(solution['geometry'])
    for p in (root, lh_out, camera_out):
        artifacts.write(p/'rig.json', rig)
        artifacts.report(p, rig_fingerprint=artifacts.fingerprint(rig))
    artifacts.report(lh_out, accepted=True, upload_verified=False, geometry_fingerprint=key, uri=uri)
    artifacts.write(camera_out/'intrinsics.json', intrinsics)
    artifacts.report(camera_out, **alignment, lighthouse_fingerprint=key, profile=profile,
        intrinsics_fingerprint=artifacts.fingerprint(intrinsics),
        detector_model=detector_path, detector=detector_name.lower()+'-onnx')
    artifacts.report(root, accepted=True, completed=True, uri=uri, base_stations=stations,
                     camera_profile=profile, prompts=prompts, reason=None,
                     hardware_assumptions=rig['hardware_preset']['assumptions'])


def run(args, ask=input):
    from ..hardware import connect
    from cflib.crazyflie.mem import LighthouseMemHelper
    max_rms_px = getattr(args, 'max_camera_rms_px', 1.)
    if not np.isfinite(max_rms_px) or max_rms_px <= 0:
        raise ValueError('--max-camera-rms-px must be a finite positive number')
    target = Path(args.out)
    root = target.with_suffix('.setup') if target.suffix == '.json' else target
    root = artifacts.session(root, 'setup', {})
    artifacts.report(root, camera_reprojection_limit_px=max_rms_px, camera_rms_limit_overridden=max_rms_px != 1.)
    camera = None
    try:
        detector = NanoDetector(args.detector_model)
        print(f'Drone detector: {detector.architecture}, input {detector.size}px, CPU.')
        camera = automatic_camera(args)
        floor = environment(camera, root)
        print('Keep the camera fixed. Choose origin, +X and XY placements with the ENTIRE drone inside the camera view.')
        uri = args.uri or scan_uri()
        with connect(uri) as scf:
            wait_for_parameters(scf, RealClock())
            cf = scf.cf
            if int(float(cf.param.get_value('deck.bcLighthouse4'))) != 1:
                raise RuntimeError('Lighthouse deck not detected')
            if int(float(cf.param.get_value('lighthouse.systemType'))) != 2:
                raise RuntimeError('This preset expects Lighthouse v2; configure v2 in cfclient first')
            if cf.platform.get_protocol_version() >= 12 and cf.supervisor.is_armed:
                raise RuntimeError('Setup requires the drone resting with motors disarmed')
            # This workflow never requests arming, takeoff, or commander setpoints.
            station_history = []
            try:
                stations, status = discover_stations(cf, history=station_history)
            finally:
                artifacts.write(root/'station-discovery.json', dict(version=1, samples=station_history))
            print(f'Connected {uri}; detected Lighthouse channels {[i+1 for i in stations]}; tags {[t["id"] for t in floor["tags"]]}.')
            calibs = lighthouse.wait_callback(LighthouseMemHelper(cf).read_all_calibs)
            factory = {str(i): calibs[i].as_file_object() for i in stations if i in calibs and calibs[i].valid}
            if len(factory) != 2:
                raise RuntimeError('Both detected stations must have valid factory calibration')
            lh_out = artifacts.session(root/'lighthouse', 'lighthouse', {})
            artifacts.write(lh_out/'factory-calibration.json', dict(version=1, stations=factory))
            raw = collect_samples(cf, camera, detector, stations, lh_out, ask)
            solution = lighthouse.solve_bounded(lh_out/'samples.json', lh_out)
            artifacts.report(lh_out, solution=solution, station_status=status)
            if not solution.get('fit_valid'):
                raise ValueError('Lighthouse estimation did not converge with valid reference placements; repeat setup')
            alignment = align_camera(raw, solution, camera.intrinsics, max_rms_px=max_rms_px)
            camera_out = artifacts.session(root/'camera', 'camera', {})
            artifacts.report(camera_out, **dict(alignment, accepted=False))
            if not alignment['fit_valid']:
                details = ''
                pairs = alignment.get('reference_pair_diagnostics', [])
                if pairs:
                    details = (f'; origin-to-X distance: camera {pairs[0]["camera_distance_m"]*1000:.1f} mm, '
                               f'Lighthouse {pairs[0]["lighthouse_distance_m"]*1000:.1f} mm')
                raise ValueError(f'Camera fitting failed: RMS {alignment["reprojection_rms_px"]:.2f} px exceeds {max_rms_px:g} px'
                                 f'{details}. See {camera_out/"report.json"}')
            rig = build_rig(floor, alignment, stations, default_hardware())
            from .hover import route
            route(rig)
            ask(PROMPTS[3]+' ')
            _, home = stable_drone_observation(camera, detector, root, 'launch-return')
            artifacts.write(root/'launch-return-check.json', check_launch_return(raw['samples'][0]['camera'], home))
            save_completed_setup(root, rig, solution, alignment, camera.intrinsics, camera.profile,
                detector.path, detector.architecture, uri, stations, 4)
            if target.suffix == '.json':
                artifacts.write(target, rig)
        print(f'Setup saved: {root}. Motors remained disarmed; Lighthouse geometry is uploaded when a flight command starts.')
        return root
    except BaseException as e:
        artifacts.report(root, accepted=False, reason=str(e))
        raise
    finally:
        if camera is not None:
            camera.close()
