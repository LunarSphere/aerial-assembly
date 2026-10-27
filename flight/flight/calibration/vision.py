"""AprilTag geometry, independent check tags, and RGB-D diagnostics."""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..frames import Pose, xyzw_to_wxyz
from . import artifacts
from .camera import cv, open_camera, validate_intrinsics, ideal_pixels, pixel_rays, project_points
from .rig import corners


def detect(image):
    c = cv()
    params = c.aruco.DetectorParameters()
    params.cornerRefinementMethod = c.aruco.CORNER_REFINE_SUBPIX
    detector = c.aruco.ArucoDetector(c.aruco.getPredefinedDictionary(c.aruco.DICT_APRILTAG_25h9), params)
    quads, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    flat = [int(i) for i in ids.flatten()]
    if len(set(flat)) != len(flat):
        raise ValueError('Duplicate AprilTag ID in image')
    return {i: np.asarray(q, float).reshape(4, 2) for i, q in zip(flat, quads)}


def _candidate_poses(world, pixels, intrinsics):
    c = cv()
    k, d = np.asarray(intrinsics['matrix'], float), np.asarray(intrinsics['distortion'], float)
    world, pixels = np.ascontiguousarray(world, float), np.ascontiguousarray(pixels, float)
    if np.ptp(world[:, 2]) > 1e-7:
        raise ValueError('This milestone expects coplanar surveyed tags')
    # IPPE is most reliable with its plane exactly Z=0, including nonzero survey heights.
    anchor = np.mean(world, axis=0)
    planar = np.ascontiguousarray(world - anchor)
    planar[:, 2] = 0.
    ok, rs, ts, _ = c.solvePnPGeneric(planar, ideal_pixels(pixels, intrinsics), k, np.zeros(5), flags=c.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError('Camera pose estimation failed')
    candidates = []
    for rvec, tvec in zip(rs, ts):
        r, _ = c.Rodrigues(rvec)
        tvec = (tvec.reshape(3) - r @ anchor).reshape(3, 1)
        points = world @ r.T + tvec.reshape(3)
        center = -r.T @ tvec.reshape(3)
        # Camera must be above the printed side of the tag plane; all corners in front.
        if np.min(points[:, 2]) <= 0 or center[2] <= np.max(world[:, 2]):
            continue
        projected = project_points(world, rvec, tvec, intrinsics)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
        p = Pose(tuple(center), tuple(xyzw_to_wxyz(Rotation.from_matrix(r.T).as_quat())))
        candidates.append((rms, p, rvec, tvec))
    if not candidates:
        raise ValueError('Camera pose is physically inconsistent')
    candidates.sort(key=lambda v: v[0])
    if len(candidates) > 1:
        a, b = candidates[:2]
        if b[0] - a[0] < .05 and (np.linalg.norm(np.subtract(a[1].pos, b[1].pos)) > .005
                                  or a[1].angle_to(b[1]) > 1):
            raise ValueError('Unresolved planar camera-pose ambiguity; improve view/layout')
    return candidates[0]


def colour_blobs(image, hue, exclude=(), min_saturation=70, min_value=60, area_px=(20, 2000)):
    """Round blobs within an OpenCV hue range (0-179), such as a coloured sticker on the drone.

    Centres are saturation-weighted centroids, so they are sub-pixel even when the blob is
    only a few pixels across. Black-white tag edges show faint colour fringes on the D455: the
    minimum saturation and area reject most (the sticker is about 60 px at saturation ~100), and
    blobs inside ``exclude`` quads (decoded floor tags, grown by a quarter) are skipped.
    """
    c = cv()
    hsv = c.cvtColor(image, c.COLOR_BGR2HSV)
    h, saturation, value = hsv[..., 0], hsv[..., 1].astype(float), hsv[..., 2]
    mask = ((h >= hue[0]) & (h <= hue[1]) & (saturation >= min_saturation) & (value >= min_value)).astype(np.uint8)
    mask = c.morphologyEx(mask, c.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # rejoin a marker split by noise
    n, labels, stats, _ = c.connectedComponentsWithStats(mask, connectivity=8)
    grown = [(np.mean(q, axis=0)+1.25*(np.asarray(q, float)-np.mean(q, axis=0))).astype(np.float32) for q in exclude]
    blobs = []
    for i in range(1, n):
        x, y, w, height, area = stats[i]
        if not area_px[0] <= area <= area_px[1] or max(w, height) > 1.5*min(w, height) or area < .5*w*height:
            continue
        weights = (labels[y:y+height, x:x+w] == i)*saturation[y:y+height, x:x+w]
        yy, xx = np.mgrid[y:y+height, x:x+w]
        centre = np.array([(xx*weights).sum(), (yy*weights).sum()])/weights.sum()
        if any(c.pointPolygonTest(q, tuple(map(float, centre)), False) >= 0 for q in grown):
            continue
        blobs.append(dict(center_px=centre.tolist(), size_px=[int(w), int(height)], area_px=int(area),
                          hue=float(np.median(h[y:y+height, x:x+w][labels[y:y+height, x:x+w] == i]))))
    return blobs


def drone_marker(image, marker, intrinsics, transform, exclude=()):
    """Candidate rays through the coloured marker on the drone (issue #40).

    The marker is a few pixels across from the overhead camera, found by colour. Pass the decoded
    floor-tag quads as ``exclude``: only visible tags, since one the drone covers is not decoded.
    Anything else of that colour passes too, so every candidate is kept; offline, the one nearest
    Lighthouse is the drone.
    """
    pose = Pose.from_dict(transform)
    blobs = colour_blobs(image, marker['hue'], exclude)
    rays = pose.rotate(pixel_rays([b['center_px'] for b in blobs], intrinsics)) if blobs else []
    return dict(drone_marker_origin_m=list(pose.pos),
                drone_marker_candidates=[dict(b, direction=list(map(float, r))) for b, r in zip(blobs, rays)])


def solve(rig, observations, intrinsics):
    """Fit reference tags only; reconstruct check centres independently by calibrated PnP."""
    c = cv()
    validate_intrinsics(intrinsics)
    tags = rig['tags']
    missing = [t['id'] for t in tags if t['id'] not in observations]
    if missing:
        raise ValueError(f'Missing surveyed tags: {missing}')
    refs = [t for t in tags if t['role'] != 'check']
    world = np.concatenate([corners(t) for t in refs])
    pixels = np.concatenate([observations[t['id']] for t in refs])
    rms, transform, rvec, tvec = _candidate_poses(world, pixels, intrinsics)
    k, d = np.asarray(intrinsics['matrix'], float), np.asarray(intrinsics['distortion'], float)
    # Refine only after selecting a physically valid planar solution.
    rvec, tvec = c.solvePnPRefineLM(world, ideal_pixels(pixels, intrinsics), k, np.zeros(5), rvec.copy(), tvec.copy())
    r = c.Rodrigues(rvec)[0]
    transform = Pose(tuple(-r.T @ tvec.reshape(3)), tuple(xyzw_to_wxyz(Rotation.from_matrix(r.T).as_quat())))
    if transform.pos[2] <= np.max(world[:, 2]):
        raise ValueError('Refinement moved camera below the tag plane')
    projected = project_points(world, rvec, tvec, intrinsics)
    rms = float(np.sqrt(np.mean(np.sum((projected - pixels)**2, axis=1))))
    checks = []
    for tag in tags:
        if tag['role'] != 'check':
            continue
        # Independent single-tag pose: no use of the surveyed check position in fitting it.
        local = dict(tag, pose=dict(pos=[0., 0., 0.], quat=[1., 0., 0., 0.]))
        check_rms, camera_in_tag, _, _ = _candidate_poses(corners(local), observations[tag['id']], intrinsics)
        tag_in_camera = camera_in_tag.inverse()
        measured = transform.apply(tag_in_camera.pos)
        error = float(np.linalg.norm(measured - np.asarray(tag['pose']['pos'])))
        checks.append(dict(id=tag['id'], measured_center_m=measured.tolist(), error_m=error,
                           independently_verified=tag.get('independently_verified', True),
                           reprojection_rms_px=check_rms))
    maximum = max(t['error_m'] for t in checks)
    independent = all(t['independently_verified'] for t in checks)
    return dict(T_lighthouse_camera=transform.to_dict(), reprojection_rms_px=rms,
                check_tags=checks, max_check_error_m=maximum,
                independent_tag_verification=independent,
                accepted=bool(independent and rms <= 1. and maximum <= .005))


def depth_report(frame, observations, rig, intrinsics, transform):
    """Robust measured tag-surface points; black/invalid depth pixels are excluded."""
    c = cv()
    p = Pose.from_dict(transform)
    result = []
    for tag in rig['tags']:
        quad = observations[tag['id']]
        center = np.mean(quad, axis=0)
        # Interior inset avoids mixed depth along the tag boundary.
        polygon = np.rint(center + .6 * (quad - center)).astype(np.int32)
        mask = np.zeros(frame.depth_m.shape, np.uint8)
        c.fillConvexPoly(mask, polygon, 1)
        yy, xx = np.nonzero(mask)
        depths = frame.depth_m[yy, xx]
        valid = np.isfinite(depths) & (depths > 0)
        if np.count_nonzero(valid) < 20:
            result.append(dict(id=tag['id'], valid=False, reason='insufficient valid depth'))
            continue
        xy = np.column_stack([xx[valid], yy[valid]]).astype(float)
        xyz = pixel_rays(xy, intrinsics) * depths[valid, None]
        measured = np.array([p.apply(v) for v in xyz])
        result.append(dict(id=tag['id'], valid=True, valid_pixels=int(len(measured)),
                           median_surface_z_m=float(np.median(measured[:, 2])),
                           surface_z_error_m=float(np.median(measured[:, 2]) - tag['pose']['pos'][2])))
    return result


def calibrate(args, rig):
    lh = artifacts.accepted(args.lighthouse, 'lighthouse', rig)
    output = artifacts.session(args.out, 'camera', rig)
    try:
        camera = open_camera(args)
        try:
            observations, frames = [], []
            deadline = time.monotonic() + 30
            while len(frames) < 20 and time.monotonic() < deadline:
                frame = camera.frame()
                found = detect(frame.color)
                if all(t['id'] in found for t in rig['tags']):
                    observations.append(found)
                    frames.append(frame)
            if len(frames) < 20:
                raise ValueError(f'Need 20 frames showing all {len(rig["tags"])} configured tags; check visibility/lighting/tag family')
            # Median corners suppress frame noise without changing surveyed geometry.
            averaged = {t['id']: np.median([v[t['id']] for v in observations], axis=0) for t in rig['tags']}
            for i, frame in enumerate(frames):
                if not cv().imwrite(str(output / f'color-{i:02d}.png'), frame.color):
                    raise OSError('Could not save calibration image')
                if frame.depth_m is not None:
                    np.save(output / f'depth-{i:02d}.npy', frame.depth_m)
            artifacts.write(output / 'observations.json', dict(version=1, frames=[
                dict(timestamp=f.timestamp, corners={str(k): v.tolist() for k, v in obs.items()})
                for f, obs in zip(frames, observations)]))
            artifacts.write(output / 'intrinsics.json', camera.intrinsics)
            result = solve(rig, averaged, camera.intrinsics)
            result['depth_diagnostics'] = depth_report(frames[-1], observations[-1], rig, camera.intrinsics,
                                                        result['T_lighthouse_camera'])
            if not all(d['valid'] for d in result['depth_diagnostics']):
                result['accepted'] = False
                result['reason'] = 'Invalid depth on one or more surveyed tag surfaces'
            result.update(profile=camera.profile,
                          intrinsics_fingerprint=artifacts.fingerprint(camera.intrinsics),
                          lighthouse_fingerprint=lh['geometry_fingerprint'])
            artifacts.report(output, **result)
            if not result['accepted']:
                if not result['independent_tag_verification']:
                    raise ValueError('Camera layout estimates are not independent verification. Run survey with '
                                     '--layout ... --measure-check-tags and measure the check tags before acceptance.')
                raise ValueError('Camera calibration rejected; inspect report and repeat survey/capture')
        finally:
            camera.close()
    except BaseException as e:
        artifacts.report(output, accepted=False, reason=str(e))
        raise
    return output
