"""Initial tag survey estimates from RGB corners and a measured or depth-derived plane."""
from pathlib import Path
import math
import time
import warnings

import numpy as np

from . import artifacts
from .camera import cv, open_camera, pixel_rays, validate_intrinsics
from .vision import detect
from ..frames import Pose, xyzw_to_wxyz
from scipy.spatial.transform import Rotation


def fit_plane(points):
    points = np.asarray(points, float)
    if len(points) < 100 or not np.all(np.isfinite(points)):
        raise ValueError('Need at least 100 valid depth samples on the tag surfaces')
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(120):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs((points-a) @ normal) < .012
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < .7 * len(points):
        raise ValueError('Tag depths do not agree on one floor plane; check depth alignment and tag surfaces')
    kept = points[best]
    center = np.mean(kept, axis=0)
    normal = np.linalg.svd(kept-center, full_matrices=False)[2][-1]
    if np.dot(normal, center) > 0:
        normal = -normal  # upward, toward the overhead camera
    distance = float(np.dot(normal, center))
    rms = float(np.sqrt(np.mean(((kept-center) @ normal)**2)))
    return normal, distance, dict(inliers=int(best.sum()), points=len(points), rms_m=rms)


def estimate(observations, intrinsics, depth):
    validate_intrinsics(intrinsics)
    if not {0, 1, 2}.issubset(observations) or len(observations) < 5:
        raise ValueError('Need tags 0 (origin), 1 (+X), 2 (XY) and at least two check tags visible')
    depth = np.asarray(depth, float)
    expected = (intrinsics['profile']['height'], intrinsics['profile']['width'])
    if depth.shape != expected:
        raise ValueError('Depth must be aligned to the color image resolution')
    samples = []
    for quad in observations.values():
        center = np.mean(quad, axis=0)
        polygon = np.rint(center + .8 * (np.asarray(quad)-center)).astype(np.int32)
        mask = np.zeros(depth.shape, np.uint8)
        cv().fillConvexPoly(mask, polygon, 1)
        yy, xx = np.nonzero(mask)
        z = depth[yy, xx]
        valid = np.isfinite(z) & (z > 0)
        samples.extend((pixel_rays(np.column_stack([xx[valid], yy[valid]]), intrinsics) * z[valid, None]).tolist())
    normal, distance, plane_stats = fit_plane(samples)
    source = 'rgbd_estimate'
    vertices = {}
    for tag_id, quad in observations.items():
        rays = pixel_rays(quad, intrinsics)
        denominator = rays @ normal
        if np.any(np.abs(denominator) < 1e-5):
            raise ValueError('Tag ray is parallel to the estimated floor')
        xyz = rays * (distance / denominator)[:, None]
        if np.any(xyz[:, 2] <= 0):
            raise ValueError('Estimated tag plane lies behind the camera')
        vertices[tag_id] = xyz
    centers = {k: np.mean(v, axis=0) for k, v in vertices.items()}
    origin = centers[0]
    x_axis = centers[1]-origin
    baseline = float(np.linalg.norm(x_axis))
    if baseline < .1:
        raise ValueError('Origin and X-axis tags are too close')
    x_axis /= baseline
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.array([np.cross(y_axis, normal), y_axis, normal])
    tags = []
    for tag_id in sorted(vertices):
        points = (vertices[tag_id]-origin) @ rotation.T
        pos = np.mean(points, axis=0)
        pos[2] = 0.
        if tag_id == 0:
            pos[:] = 0.
        elif tag_id == 1:
            pos[:] = [baseline, 0., 0.]
        sizes = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
        if max(sizes)/min(sizes) > 1.2:
            raise ValueError(f'Tag {tag_id} is not square on the estimated plane')
        heading = points[1]-points[0]
        yaw = math.atan2(heading[1], heading[0])
        tags.append(dict(id=tag_id, role={0: 'origin', 1: 'x', 2: 'xy'}.get(tag_id, 'check'),
                         size_m=float(np.mean(sizes)),
                         pose=dict(pos=pos.tolist(), quat=[math.cos(yaw/2), 0., 0., math.sin(yaw/2)]),
                         survey_source=source, independently_verified=False))
    return dict(version=1, kind='tag-layout', family='25h9', coordinate_frame='tag_plane',
                source=source, accepted_for_flight=False, reference_distance_m=baseline,
                camera_height_above_tag_plane_m=abs(distance), plane_stats=plane_stats,
                plane_camera=dict(normal_camera=normal.tolist(), distance_m=distance),
                T_tagplane_camera=Pose(tuple(-rotation @ origin),
                    tuple(xyzw_to_wxyz(Rotation.from_matrix(rotation).as_quat()))).to_dict(),
                tags=tags, profile=intrinsics['profile'],
                note='Initial camera estimates, not independent ground truth. Confirm the 1 m baseline and survey check tags.')


def annotate(image, observations, result):
    image = image.copy()
    for tag in result['tags']:
        quad = np.asarray(observations[tag['id']])
        cv().polylines(image, [np.rint(quad).astype(np.int32)], True, (0, 255, 0), 2)
        x, y = tag['pose']['pos'][:2]
        p = np.mean(quad, axis=0).astype(int)
        cv().putText(image, f'{tag["id"]}: ({x:.2f}, {y:.2f}) m', tuple(p),
                    cv().FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    return image


def capture(args):
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    camera = None
    try:
        camera = open_camera(args)
        frames, observations, deadline = [], [], time.monotonic()+30
        # Let exposure and stereo depth settle before collecting the survey burst.
        for _ in range(15):
            camera.frame()
        while len(frames) < 20 and time.monotonic() < deadline:
            frame = camera.frame()
            seen = detect(frame.color)
            if {0, 1, 2}.issubset(seen) and len(seen) >= 5:
                frames.append(frame)
                observations.append(seen)
        if len(frames) < 20:
            raise ValueError('Could not collect 20 frames containing the three references and two check tags')
        ids = set.intersection(*(set(v) for v in observations))
        quads = {i: np.median([v[i] for v in observations], axis=0) for i in ids}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            depth = np.nanmedian(np.array([f.depth_m for f in frames]), axis=0)
        np.save(output / 'depth-m.npy', depth)
        result = estimate(quads, camera.intrinsics, depth)
        artifacts.write(output / 'layout.json', result)
        artifacts.write(output / 'intrinsics.json', camera.intrinsics)
        artifacts.write(output / 'observations.json', dict(version=1, corners={str(k): v.tolist() for k, v in quads.items()}))
        cv().imwrite(str(output / 'color.png'), frames[-1].color)
        cv().imwrite(str(output / 'layout.png'), annotate(frames[-1].color, quads, result))
        return output
    except BaseException as e:
        artifacts.write(output / 'error.json', dict(version=1, error=str(e)))
        raise
    finally:
        if camera is not None:
            camera.close()
