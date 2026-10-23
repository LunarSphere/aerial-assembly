"""Human-guided Lighthouse geometry; no arming operations in this module."""
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

import numpy as np

from . import artifacts

LIMIT_M = .0004
SOLVER_EVALUATIONS = 1000
# Calibrated stations explain a resting drone's sweep angles to about 0.1-0.5 mrad (set from setup data).
STATION_RESIDUAL_MRAD = 1.5


def wait_callback(start, timeout=30.):
    queue = Queue()
    start(lambda value: queue.put(value))
    try:
        return queue.get(timeout=timeout)
    except Empty as e:
        raise TimeoutError('Lighthouse operation timed out; no accepted calibration') from e


def geometry_objects(geometry):
    from cflib.crazyflie.mem import LighthouseBsGeometry
    result = {}
    for key, value in geometry.items():
        g = LighthouseBsGeometry()
        g.origin, g.rotation_matrix, g.valid = value['origin'], value['rotation'], True
        result[int(key)] = g
    return result


def wait_factory_calibration(cf, stations, timeout=30.):
    """Require actual v2 OOTX reception, rather than trusting potentially stale stored calibration."""
    from cflib.crazyflie.log import LogConfig
    from ..clock import RealClock
    from ..preflight import samples
    clock = RealClock()
    mask = sum(1 << int(k) for k in stations)
    deadline = clock.time() + timeout
    with samples(cf, LogConfig, clock, ('lighthouse.bsCalCon',), log_name='Cal OOTX') as queue:
        while clock.time() < deadline:
            if not cf.is_connected():
                raise ConnectionError('Radio disconnected while waiting for factory calibration')
            try:
                received, data = queue.get(timeout=.2)
            except Empty:
                continue
            if isinstance(data, Exception):
                raise data
            if clock.time() - received < 1 and int(data['lighthouse.bsCalCon']) & mask == mask:
                return
    raise TimeoutError('Both v2 stations must deliver factory calibration over the air; clear line of sight and retry')


def check_live_geometry(cf, saved, calibs_saved):
    """Float32 memory readback needs numeric tolerance, not a byte-for-byte hash."""
    from cflib.crazyflie.mem import LighthouseMemHelper
    helper = LighthouseMemHelper(cf)
    actual = wait_callback(helper.read_all_geos)
    valid = {str(k): v for k, v in actual.items() if v.valid}
    if set(valid) != set(saved):
        raise ValueError('Drone Lighthouse stations differ from the accepted calibration')
    for k, expected in saved.items():
        g = valid[k]
        if (not np.allclose(g.origin, expected['origin'], atol=2e-6, rtol=0)
                or not np.allclose(g.rotation_matrix, expected['rotation'], atol=2e-6, rtol=0)):
            raise ValueError('Drone Lighthouse geometry differs; repeat calibration')
    wait_factory_calibration(cf, saved)
    calibs = wait_callback(helper.read_all_calibs)
    for k, expected in calibs_saved.items():
        live = calibs.get(int(k))
        if live is None or not live.valid or artifacts.fingerprint(live.as_file_object()) != artifacts.fingerprint(expected):
            raise ValueError('Base-station factory calibration differs; repeat calibration')


def upload(cf, solution, calibs):
    from cflib.crazyflie.mem import LighthouseMemHelper
    from cflib.localization.lighthouse_config_manager import LighthouseConfigWriter
    # Firmware may support 2, 4 or 16 slots. cflib's writer defaults to 16 and
    # invalidates every unused slot; writes beyond firmware capacity fail even
    # when our two requested stations were successfully written.
    helper = LighthouseMemHelper(cf)
    geo_slots = set(wait_callback(helper.read_all_geos))
    calib_slots = set(wait_callback(helper.read_all_calibs))
    if (not geo_slots or geo_slots != calib_slots or len(geo_slots) > 16
            or geo_slots != set(range(len(geo_slots)))):
        raise RuntimeError('Could not determine Lighthouse memory capacity consistently; reconnect and retry')
    requested = {int(k) for k in solution['geometry']}
    if not requested or not requested.issubset(geo_slots):
        raise ValueError(f'Lighthouse channels {[k+1 for k in sorted(requested)]} exceed firmware capacity '
                         f'({len(geo_slots)} station slots)')
    writer = LighthouseConfigWriter(cf, nr_of_base_stations=len(geo_slots))
    print(f'Uploading Lighthouse channels {[k+1 for k in sorted(requested)]}; '
          f'firmware supports {len(geo_slots)} station slots.')
    # systemType was checked before sampling; do not change it here (that erases factory data).
    try:
        success = wait_callback(lambda cb: writer.write_and_store_config(cb, geos=geometry_objects(solution['geometry'])))
    finally:
        # cflib 0.1.33 registers this callback but does not remove it on completion.
        cf.loc.receivedLocationPacket.remove_callback(writer._received_location_packet)
    if not success:
        raise RuntimeError(f'Lighthouse geometry upload/persistence failed for {len(geo_slots)} supported slots')
    check_live_geometry(cf, solution['geometry'], calibs)
    return dict(supported_station_slots=len(geo_slots), written_station_ids=sorted(requested),
                persisted=True, readback_verified=True)


def collect(cf, stations, timeout=20., diagnostics_path=None):
    """Average 50 matched, two-station packets. cflib's ordinary average stops at either BS."""
    from cflib.localization.lighthouse_sweep_angle_reader import LighthouseMatchedSweepAngleReader
    from collections import Counter
    from ..preflight import set_parameter
    queue = Queue()
    # enLhMtchStm is a uint8 firmware parameter. Keep the stream budget within
    # 255, with headroom for rejected packets while collecting 50 usable pairs.
    deadline, packets = time.monotonic() + timeout, []
    counts, groups, reader = Counter(), Counter(), None
    restarts, last_group_at, last_progress = 0, time.monotonic(), time.monotonic()
    def count_packet(packet):
        if packet.type == cf.loc.LH_MATCHED_ANGLE_STREAM:
            counts[str(packet.data['basestation'])] += 1
    def start_reader():
        # cflib starts the stream before writing min_bs/max_time. Configure and
        # acknowledge these first, including clearing an abandoned finite stream.
        set_parameter(cf, 'locSrv.enLhMtchStm', 0)
        set_parameter(cf, 'locSrv.minBsLhMtchStm', 2)
        set_parameter(cf, 'locSrv.maxTimeLhMtchStm', 25)
        active = LighthouseMatchedSweepAngleReader(cf, lambda s: queue.put(s), sample_count=255, min_bs=2)
        try:
            active.start(timeout=max(.001, deadline-time.monotonic()))
        except BaseException:
            active.stop()
            raise
        return active
    cf.loc.receivedLocationPacket.add_callback(count_packet)
    try:
        # cflib 0.1.33 starts a timer even for its default timeout=0, which would
        # immediately stop the stream. Always pass our actual collection timeout.
        reader = start_reader()
        last_group_at = time.monotonic()
        while len(packets) < 50:
            if not cf.is_connected():
                raise ConnectionError('Radio disconnected during angle collection')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Lighthouse angle collection timed out: {len(packets)}/50 usable pairs; '
                                   f'raw matched packets by channel { {int(k)+1:v for k,v in counts.items()} }; '
                                   f'rejected groups {dict(groups)}. Station discovery and matched-angle streaming are separate checks.')
            try:
                sample = queue.get(timeout=min(.2, max(.001, deadline - time.monotonic())))
            except Empty:
                now = time.monotonic()
                if now-last_group_at > 2 and restarts == 0 and deadline-now > 3:
                    print('No matched Lighthouse groups for 2 s; restarting the angle stream once.')
                    reader.stop()
                    reader = start_reader()
                    restarts += 1
                    last_group_at = time.monotonic()
                if now-last_progress >= 2:
                    print(f'Lighthouse samples: {len(packets)}/50 usable pairs; received packets by channel '
                          f'{ {int(k)+1:v for k,v in counts.items()} }. Keep the drone level with its deck unobstructed.')
                    last_progress = now
                continue
            last_group_at = time.monotonic()
            if set(sample.angles_calibrated) != set(stations):
                groups['unexpected_station_set'] += 1
                continue
            angles = {str(k): [[v.lh_v1_horiz_angle, v.lh_v1_vert_angle] for v in vs]
                      for k, vs in sample.angles_calibrated.items()}
            if all(np.asarray(v).shape == (4, 2) and np.all(np.isfinite(v)) for v in angles.values()):
                packets.append(angles)
            else:
                groups['invalid_angles'] += 1
    finally:
        try:
            if reader is not None:
                reader.stop()
        finally:
            cf.loc.receivedLocationPacket.remove_callback(count_packet)
            if diagnostics_path is not None:
                artifacts.write(diagnostics_path, dict(version=1, usable_pairs=len(packets),
                    required_pairs=50, raw_packets_by_channel={str(int(k)+1):v for k,v in counts.items()},
                    rejected_groups=dict(groups), stream_restarts=restarts, complete=len(packets)==50))
    return dict(angles={str(k): np.mean([p[str(k)] for p in packets], axis=0).tolist() for k in stations},
                packets=packets, timestamp=time.monotonic())


def solve(raw):
    # Reference-only: just the origin, +X and XY placements, no extra or check samples.
    skip_verification = skip_geometry = raw.get('reference_only') is True
    from cflib.localization.lighthouse_bs_vector import LighthouseBsVector, LighthouseBsVectors
    from cflib.localization.lighthouse_cf_pose_sample import LhCfPoseSample
    from cflib.localization.lighthouse_geo_estimation_manager import LhGeoInputContainerData, LhGeoEstimationManager
    from cflib.localization.lighthouse_types import LhDeck4SensorPositions
    groups = {k: [] for k in ('origin', 'x', 'xy', 'xyz', 'verification')}
    for record in raw['samples']:
        angles = record['angles']
        if {int(k) for k in angles} != set(raw['base_stations']):
            raise ValueError('Every calibration sample must see the two configured stations')
        for v in angles.values():
            if np.asarray(v).shape != (4, 2) or not np.all(np.isfinite(v)):
                raise ValueError('Invalid sweep angles')
        sample = LhCfPoseSample({int(k): LighthouseBsVectors([LighthouseBsVector(*v) for v in values])
                                for k, values in angles.items()})
        sample.augment_with_ippe(LhDeck4SensorPositions.positions)
        groups[record['role']].append(sample)
    if len(groups['origin']) != 1 or len(groups['x']) != 1 or len(groups['xy']) < 1:
        raise ValueError('Origin, 1 m X and XY references are required')
    if skip_verification and groups['verification']:
        raise ValueError('Skipped verification must not contain check samples')
    if skip_geometry and groups['xyz']:
        raise ValueError('Skipped geometry sampling must not contain XYZ samples')
    if (not skip_geometry and len(groups['xyz']) < 6) or (not skip_verification and len(groups['verification']) < 5):
        raise ValueError('Need at least six XYZ and five independent verification samples')
    data = LhGeoInputContainerData(LhDeck4SensorPositions.positions)
    data.origin, data.x_axis, data.xy_plane = groups['origin'][0], groups['x'], groups['xy']
    data.xyz_space, data.verification = groups['xyz'], groups['verification']
    # cflib stops each least-squares pass after 100 evaluations and then reports "not converged".
    # With 16 samples one pass needs more; the final fit is the same, so allow more evaluations.
    from cflib.localization import lighthouse_geometry_solver as geometry_solver
    original_init = geometry_solver.SolverData.__init__
    def more_evaluations(self):
        original_init(self)
        self.max_nr_iter = SOLVER_EVALUATIONS
    geometry_solver.SolverData.__init__ = more_evaluations
    try:
        result = LhGeoEstimationManager.estimate_geometry(data)
    finally:
        geometry_solver.SolverData.__init__ = original_init
    samples = [dict(role=str(s.sample_type), valid=s.is_valid,
                    position_m=s.pose.translation.tolist() if s.has_pose else None,
                    error_m=float(s.error_distance) if np.isfinite(s.error_distance) else None,
                    status=str(s.status)) for s in result.samples]
    checks = [s for s in samples if s['role'] == 'verification']
    errors = [s['error_m'] for s in checks if s['valid'] and s['error_m'] is not None]
    # Independently ensure coverage; a handful of repeated points must not pass.
    positions = np.array([s['position_m'] for s in checks if s['position_m'] is not None])
    coverage = len(positions) >= 5 and all(np.ptp(positions, axis=0) >= [.2, .2, .1])
    maximum = max(errors) if errors else None
    fit_valid = bool(result.progress_is_ok and result.has_converged
                    and result.is_origin_sample_valid and result.is_x_axis_samples_valid
                    and result.is_xy_plane_samples_valid
                    and set(result.bs_poses) == set(raw['base_stations']))
    accepted = bool(fit_valid and not skip_verification
                    and len(errors) == len(checks) >= 5 and coverage and maximum <= LIMIT_M)
    geometry = {str(k): dict(origin=v.translation.tolist(), rotation=v.rot_matrix.tolist())
                for k, v in result.bs_poses.items()}
    def stats(s):
        return None if s is None else {k: float(v) if np.isfinite(v) else None for k, v in s._asdict().items()}
    return dict(version=1, accepted=accepted, converged=bool(result.has_converged),
                fit_valid=fit_valid, reference_only=skip_geometry,
                progress_ok=bool(result.progress_is_ok), progress=result.progress_info,
                verification_limit_m=LIMIT_M, max_verification_error_m=maximum,
                coverage_ok=bool(coverage), estimation_stats=stats(result.error_stats),
                verification_stats=stats(result.verification_stats), samples=samples, geometry=geometry)


def station_consistency(angles, geometry, start=(0., 0., 0.), limit_mrad=STATION_RESIDUAL_MRAD):
    """Fit one deck pose to every station's sweep angles through the stored geometry.

    If the stations still match their calibration, one pose explains all angles to the
    measurement noise. A station that moved since calibration cannot be fitted together with
    the others, and its residual stands out. Independent of exactly where the drone rests.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from cflib.localization.lighthouse_bs_vector import LighthouseBsVector
    from cflib.localization.lighthouse_types import LhDeck4SensorPositions
    sensors = np.asarray(LhDeck4SensorPositions.positions, float)
    stations = sorted(angles)
    measured = {k: np.asarray(angles[k], float) for k in stations}
    frames = {k: (np.asarray(geometry[k]['rotation'], float), np.asarray(geometry[k]['origin'], float))
              for k in stations}

    def predict(params, k):
        world = Rotation.from_rotvec(params[:3]).apply(sensors)+params[3:]
        rotation, origin = frames[k]
        return np.array([LighthouseBsVector.from_cart(rotation.T @ (p-origin)).lh_v1_angle_pair for p in world])

    def residual(params):
        return np.concatenate([(predict(params, k)-measured[k]).ravel() for k in stations])

    fit = least_squares(residual, np.r_[0., 0., 0., np.asarray(start, float)], x_scale='jac')
    per_station = {}
    for k in stations:
        r = (predict(fit.x, k)-measured[k]).ravel()
        distance = float(np.linalg.norm(fit.x[3:]-frames[k][1]))
        rms = float(np.sqrt(np.mean(r**2)))
        per_station[k] = dict(rms_mrad=rms*1000, rms_at_deck_mm=rms*distance*1000, distance_m=distance)
    worst = max(per_station, key=lambda k: per_station[k]['rms_mrad'])
    return dict(position_m=fit.x[3:].tolist(), stations=per_station, worst_station=worst,
                worst_rms_mrad=per_station[worst]['rms_mrad'], limit_mrad=limit_mrad,
                consistent=bool(per_station[worst]['rms_mrad'] <= limit_mrad))


def solve_bounded(raw_path, output, timeout=120.):
    """Run heavy cflib/IPPE work in a killable subprocess; never hang the radio command."""
    try:
        process = subprocess.run([sys.executable, '-m', __name__, str(raw_path), str(output / 'solution.json')],
                                 capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        (output / 'solver.log').write_text(str(e))
        raise TimeoutError('Geometry solver exceeded 120 s; repeat calibration with better sample coverage') from e
    (output / 'solver.log').write_text(process.stdout + process.stderr)
    if process.returncode:
        raise ValueError('Geometry solver failed; inspect solver.log and repeat calibration')
    return artifacts.read(output / 'solution.json')


def calibrate(args, rig, ask=input):
    from ..clock import RealClock
    from ..hardware import connect
    from ..preflight import wait_for_parameters
    from cflib.crazyflie.mem import LighthouseMemHelper
    from cflib.localization.lighthouse_config_manager import LighthouseConfigFileManager
    output = artifacts.session(args.out, 'lighthouse', rig)
    raw = dict(version=1, base_stations=rig['base_stations'], samples=[])
    try:
        with connect(args.uri) as scf:
            cf = scf.cf
            wait_for_parameters(scf, RealClock())
            if int(float(cf.param.get_value('deck.bcLighthouse4'))) != 1:
                raise ValueError('Lighthouse deck not detected')
            if int(float(cf.param.get_value('lighthouse.systemType'))) != 2:
                raise ValueError('Configure Lighthouse v2 in cfclient, then rerun')
            # Human manipulation only; explicitly disarm before collecting samples.
            cf.supervisor.send_arming_request(False)
            wait_factory_calibration(cf, rig['base_stations'])
            calibs = wait_callback(LighthouseMemHelper(cf).read_all_calibs)
            selected = {str(k): calibs[k].as_file_object() for k in rig['base_stations']
                        if k in calibs and calibs[k].valid}
            if len(selected) != 2:
                raise ValueError('Missing factory calibration; let both v2 stations deliver calibration data first')
            artifacts.write(output / 'factory-calibration.json', dict(version=1, stations=selected))
            tasks = [('origin', 'STEP 1 — Place the drone where you want the origin: centre its Lighthouse deck '
                      'above tag 0, level on the support, with the nose pointing from tag 0 toward tag 1.'),
                     ('x', 'STEP 2 — Move the drone and the same support to tag 1, exactly 1 metre along '
                      'the positive X-axis from tag 0. Keep the same height and nose direction.'),
                     ('xy', 'STEP 3 — Move the drone and support to tag 2, somewhere on the same level '
                      'surface but not on the line between tags 0 and 1. Keep the same height and nose direction.')]
            tasks += [('xyz', f'Space sample {i+1}/8 — Hold the drone still at a different spot in the flight area. '
                       'Vary the height and tilt between samples; keep both stations visible.') for i in range(8)]
            tasks += [('verification', f'Check sample {i+1}/5 — Hold the drone at a NEW spot to check calibration. '
                       'Spread these checks around the flight area and include the intended hover height.')
                      for i in range(5)]
            for role, prompt in tasks:
                tag = next((t for t in rig['tags'] if t['role'] == role), None)
                print(prompt + (f' (tag ID {tag["id"]})' if tag else ''))
                ask('Hold still with both stations visible; press Enter to record: ')
                record = collect(cf, rig['base_stations'])
                raw['samples'].append(dict(role=role, **record))
                artifacts.write(output / 'samples.json', raw)
                print(f'Recorded {role}; {len(raw["samples"])} positions saved.')
            solution = solve_bounded(output / 'samples.json', output)
            artifacts.report(output, **{k: v for k, v in solution.items() if k not in ('version', 'accepted')})
            if not solution['accepted']:
                raise ValueError(f'Calibration rejected: max verification error '
                                 f'{solution["max_verification_error_m"]} m (limit {LIMIT_M} m), '
                                 f'coverage={solution["coverage_ok"]}. Inspect report, improve samples, and redo calibration.')
            from .hover import route
            hover_reference_z = route(rig)['segments'][0]['target_m'][2]
            verification_z = [s['position_m'][2] for s in solution['samples']
                              if s['role'] == 'verification' and s['position_m'] is not None]
            if max(verification_z) < hover_reference_z - .03:
                raise ValueError('Verification samples do not reach planned hover height; redo with flight-volume coverage')
            upload(cf, solution, selected)
            LighthouseConfigFileManager.write(str(output / 'lighthouse.yaml'),
                                               geos=geometry_objects(solution['geometry']),
                                               calibs={k: calibs[k] for k in rig['base_stations']})
            artifacts.report(output, accepted=True, upload_verified=True, uri=args.uri,
                             geometry_fingerprint=artifacts.fingerprint(solution['geometry']))
    except BaseException as e:
        artifacts.report(output, accepted=False, reason=str(e))
        raise
    return output


if __name__ == '__main__':
    artifacts.write(sys.argv[2], solve(artifacts.read(sys.argv[1])))
