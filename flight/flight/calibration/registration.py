"""Refresh camera pose against fixed tags, without changing their Lighthouse positions."""
import numpy as np
from scipy.spatial.transform import Rotation

from ..frames import Pose, wxyz_to_xyzw, xyzw_to_wxyz
from .camera import cv, ideal_pixels, project_points
from .rig import corners


def refresh_pose(rig, observations, intrinsics, previous):
    """Multi-tag resection, with each tag also checked against a fit excluding it.

    The saved tag map must remain fixed relative to the Lighthouse stations.
    This is registration to that map, not independent calibration verification.
    """
    tags = [t for t in rig['tags'] if t['id'] in observations]
    if len(tags) < 3:
        raise ValueError('Camera refresh needs at least three fixed AprilTags visible')
    centers = np.array([t['pose']['pos'] for t in tags])
    if np.linalg.svd(centers-centers.mean(axis=0), compute_uv=False)[1] < .1:
        raise ValueError('Camera refresh needs tags spread across the floor, not in a line')
    world = np.ascontiguousarray(np.concatenate([corners(t) for t in tags]), dtype=float)
    pixels = np.ascontiguousarray(np.concatenate([observations[t['id']] for t in tags]), dtype=float)
    prior = Pose.from_dict(previous)
    inverse = prior.inverse()
    initial_r = Rotation.from_quat(wxyz_to_xyzw(inverse.quat)).as_rotvec()
    initial_t = np.array(inverse.pos)

    def fit(mask):
        ok, r, t = cv().solvePnP(world[mask], ideal_pixels(pixels[mask], intrinsics),
            np.array(intrinsics['matrix']), np.zeros(5), initial_r.copy(), initial_t.copy(),
            useExtrinsicGuess=True, flags=cv().SOLVEPNP_ITERATIVE)
        if not ok or not np.all(np.isfinite(np.r_[r.ravel(), t.ravel()])):
            raise ValueError('Could not fit the camera pose to the fixed tags')
        rotation = Rotation.from_rotvec(r.reshape(3))
        pose = Pose(tuple(-rotation.inv().apply(t.reshape(3))),
                    tuple(xyzw_to_wxyz(rotation.inv().as_quat())))
        if pose.pos[2] <= centers[:, 2].max() or np.min(pose.inverse().apply(world)[:, 2]) <= 0:
            raise ValueError('Camera refresh produced a pose below or behind the tag plane')
        errors = np.linalg.norm(project_points(world, r, t, intrinsics)-pixels, axis=1)
        return pose, errors

    mask = np.ones(len(world), dtype=bool)
    pose, errors = fit(mask)
    checks = []
    for i, tag in enumerate(tags):
        heldout = mask.copy()
        heldout[4*i:4*i+4] = False
        _, check_errors = fit(heldout)
        checks.append(dict(id=tag['id'], rms_px=float(np.sqrt(np.mean(errors[~heldout]**2))),
                           heldout_rms_px=float(np.sqrt(np.mean(check_errors[~heldout]**2)))))
    rms = float(np.sqrt(np.mean(errors**2)))
    maximum = max(c['heldout_rms_px'] for c in checks)
    if rms > 2. or maximum > 3.:
        raise ValueError(f'Fixed tag map is inconsistent: fit {rms:.2f} px, held-out tag {maximum:.2f} px; '
                         'check for a moved tag or repeat setup')
    before = project_points(world, initial_r, initial_t, intrinsics)-pixels
    return dict(version=1, T_lighthouse_camera=pose.to_dict(), previous_T_lighthouse_camera=previous,
                before_rms_px=float(np.sqrt(np.mean(np.sum(before**2, axis=1)))),
                reprojection_rms_px=rms, max_heldout_rms_px=maximum, tags=checks,
                translation_change_m=float(np.linalg.norm(np.array(pose.pos)-prior.pos)),
                rotation_change_deg=pose.angle_to(prior), independent_verification=False,
                method='fixed_tag_map_pnp', assumes_tags_and_lighthouses_fixed=True)
