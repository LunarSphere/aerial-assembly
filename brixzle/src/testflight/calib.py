"""Camera pose in the Lighthouse frame from surveyed AprilTags (PnP over every corner).

``tag_survey.json`` (from ``testflight survey tags``): each tag's centre (Lighthouse frame, on the
floor) and yaw, i.e. the direction of the tag's +x (rightwards when the tag reads upright), plus
the black-square edge length. Corners follow OpenCV's order: top-left, top-right, bottom-right,
bottom-left of the upright tag.

``camera_pose.json``: X_cam = R X_lh + t, the colour intrinsics it was solved with, the floor
plane z = a x + b y + c through the surveyed corners, and a depth scale that makes the depth of
the tag pixels land on that plane.
"""
import time

import numpy as np


# ---- survey ------------------------------------------------------------------------------------
def tag_corners_world(survey):
    """{id: (4, 3)} corner positions (m) in OpenCV order."""
    s = survey['size_m']/2
    local = np.array([[-s, s], [s, s], [s, -s], [-s, -s]])
    out = {}
    for k, t in survey['tags'].items():
        c = np.asarray(t['center'], float)
        a = np.radians(t['yaw_deg'])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        xy = local @ R.T + c[:2]
        out[int(k)] = np.column_stack([xy, np.full(4, c[2])])
    return out


def floor_plane(survey):
    """(a, b, c) of the floor z = a x + b y + c, least squares through every surveyed corner."""
    P = np.vstack(list(tag_corners_world(survey).values()))
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    if np.linalg.matrix_rank(A) < 3:
        return np.array([0., 0., float(np.mean(P[:, 2]))])
    return np.linalg.lstsq(A, P[:, 2], rcond=None)[0]


def floor_z(plane, x, y):
    a, b, c = plane
    return a*np.asarray(x) + b*np.asarray(y) + c


# ---- detection ---------------------------------------------------------------------------------
def detector(dictionary):
    import cv2
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)


def detect_tags(color, dictionary, ids=None):
    """{id: (4, 2) pixel corners}; ids outside ``ids`` are ignored."""
    corners, found, _ = detector(dictionary).detectMarkers(color)
    out = {}
    if found is None:
        return out
    for c, i in zip(corners, found.ravel()):
        if ids is None or int(i) in ids:
            out[int(i)] = c.reshape(4, 2).astype(float)
    return out


def average_detections(dets):
    """Mean corners of every tag seen in at least half the frames."""
    ids = {}
    for d in dets:
        for k, v in d.items():
            ids.setdefault(k, []).append(v)
    return {k: np.mean(v, axis=0) for k, v in ids.items() if len(v) >= max(1, len(dets)/2)}


# ---- geometry ----------------------------------------------------------------------------------
def solve_pose(obs, world, K, dist):
    """PnP over all corners of the tags in both ``obs`` and ``world``: (R, t, rms_px, {id: rms_px})."""
    import cv2
    ids = sorted(set(obs) & set(world))
    if len(ids) < 2:
        raise ValueError(f'need at least 2 surveyed tags in view, saw {ids}')
    obj = np.vstack([world[i] for i in ids]).astype(np.float64)
    img = np.vstack([obs[i] for i in ids]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError('solvePnP failed')
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    per = {i: float(np.sqrt(np.mean(err[4*k:4*k + 4]**2))) for k, i in enumerate(ids)}
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel(), float(np.sqrt(np.mean(err**2))), per


def undistort(uv, K, dist):
    """Normalised camera coordinates (N, 2) of pixels (N, 2)."""
    import cv2
    return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)


def rays_world(uv, R, t, K, dist):
    """(origin (3,), directions (N, 3)) with each direction's camera-z component equal to 1."""
    n = undistort(uv, K, dist)
    d_cam = np.column_stack([n, np.ones(len(n))])
    return -R.T @ t, d_cam @ R          # (R^T d)^T = d^T R


def backproject_to_z(uv, z, R, t, K, dist):
    C, d = rays_world(uv, R, t, K, dist)
    s = (np.asarray(z) - C[2])/d[:, 2]
    return C + s[:, None]*d


def leave_one_out(obs, world, K, dist):
    """{id: mean corner error (mm)}: re-solve without the tag, back-project its corners to its plane."""
    out = {}
    ids = sorted(set(obs) & set(world))
    if len(ids) < 3:
        return out
    for i in ids:
        rest = {k: obs[k] for k in ids if k != i}
        R, t, _, _ = solve_pose(rest, world, K, dist)
        P = backproject_to_z(obs[i], world[i][:, 2], R, t, K, dist)
        out[i] = float(np.mean(np.linalg.norm(P - world[i], axis=1))*1e3)
    return out


def tag_pixels(shape, quad, shrink=0.15):
    """(rows, cols) of pixels inside a tag quad shrunk towards its centre."""
    import cv2
    c = quad.mean(axis=0)
    q = c + (quad - c)*(1 - shrink)
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 1)
    return np.nonzero(mask)


def depth_scale(frame, obs, world, R, t, plane):
    """(scale, residual_mm, n): depth multiplier that puts the tag pixels on the floor plane."""
    scales, pts = [], []
    for i in sorted(set(obs) & set(world)):
        r, c = tag_pixels(frame.depth.shape, obs[i])
        d = frame.depth[r, c]
        ok = d > 0
        if ok.sum() < 10:
            continue
        uv = np.column_stack([c[ok], r[ok]]).astype(float)
        C, dirs = rays_world(uv, R, t, frame.K, frame.dist)
        P0 = C + d[ok, None]*dirs
        zf = floor_z(plane, P0[:, 0], P0[:, 1])
        scales.append((zf - C[2])/(d[ok]*dirs[:, 2]))
        pts.append((C, dirs, d[ok]))
    if not scales:
        return 1., float('nan'), 0
    s = float(np.median(np.concatenate(scales)))
    res = []
    for C, dirs, d in pts:
        P = C + s*d[:, None]*dirs
        res.append(P[:, 2] - floor_z(plane, P[:, 0], P[:, 1]))
    res = np.concatenate(res)
    return s, float(np.median(np.abs(res))*1e3), int(len(res))


# ---- the whole calibration ---------------------------------------------------------------------
def calibrate(camera, survey, dictionary, ids=None, frames=20, n_depth=None):
    """Detect tags over ``frames`` captures, solve the pose, run the checks; returns camera_pose dict."""
    world = tag_corners_world(survey)
    dets, last = [], None
    for _ in range(frames):
        last = camera.capture(n_depth)
        dets.append(detect_tags(last.color, dictionary, ids))
    obs = average_detections(dets)
    missing = sorted(set(world) - set(obs))
    R, t, rms, per = solve_pose(obs, world, last.K, last.dist)
    loo = leave_one_out(obs, world, last.K, last.dist)
    plane = floor_plane(survey)
    scale, depth_res, n = depth_scale(last, obs, world, R, t, plane)
    h, w = last.depth.shape
    allc = np.vstack(list(world.values()))[:, :2]
    report = {
        'tags_bbox': [allc.min(axis=0).tolist(), allc.max(axis=0).tolist()],
        'R': R.tolist(), 't': t.tolist(), 'K': last.K.tolist(), 'dist': list(map(float, last.dist)),
        'width': w, 'height': h, 'camera_position_m': (-R.T @ t).tolist(),
        'rms_px': rms, 'per_tag_rms_px': {str(k): v for k, v in per.items()},
        'loo_mm': {str(k): v for k, v in loo.items()}, 'tags_used': sorted(obs), 'tags_missing': missing,
        'floor': list(map(float, plane)), 'depth_scale': scale, 'depth_residual_mm': depth_res,
        'depth_pixels': n, 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    report['checks'] = calibration_checks(report)
    return report


def calibration_checks(r, rms_max=1.0, loo_max=3.0, scale_tol=0.03, depth_res_max=4.0):
    """{name: (ok, message)} for the acceptance criteria in README section 2."""
    loo = max(r['loo_mm'].values()) if r['loo_mm'] else float('nan')
    return {
        'reprojection': (r['rms_px'] < rms_max, f'RMS {r["rms_px"]:.2f} px (< {rms_max})'),
        'leave_one_out': (bool(loo < loo_max), f'worst left-out tag {loo:.1f} mm (< {loo_max})'),
        'all_tags': (not r['tags_missing'], f'missing tags {r["tags_missing"]}'),
        'depth_scale': (abs(r['depth_scale'] - 1) < scale_tol, f'depth scale {r["depth_scale"]:.4f} (1 +- {scale_tol})'),
        'depth_on_floor': (bool(r['depth_residual_mm'] < depth_res_max),
                           f'tag depth off the floor by {r["depth_residual_mm"]:.1f} mm median (< {depth_res_max})'),
    }


class CameraPose:
    """Loaded ``camera_pose.json``: pixel rays, back-projection and projection in the Lighthouse frame."""

    def __init__(self, d):
        self.raw = d
        self.R = np.asarray(d['R'], float)
        self.t = np.asarray(d['t'], float)
        self.K = np.asarray(d['K'], float)
        self.dist = np.asarray(d['dist'], float)
        self.plane = np.asarray(d['floor'], float)
        self.scale = float(d.get('depth_scale', 1.))
        self.C = -self.R.T @ self.t
        self._rays = None

    def check_frame(self, frame):
        if frame.depth.shape != (self.raw['height'], self.raw['width']) or not np.allclose(frame.K, self.K, atol=0.5):
            raise ValueError('frame intrinsics/resolution differ from the calibration; re-run testflight calibrate')

    def rays(self, shape):
        """(H, W, 3) world directions per pixel (camera-z component 1)."""
        if self._rays is None or self._rays.shape[:2] != shape:
            h, w = shape
            u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
            _, d = rays_world(np.column_stack([u.ravel(), v.ravel()]), self.R, self.t, self.K, self.dist)
            self._rays = d.reshape(h, w, 3)
        return self._rays

    def points(self, depth):
        """(H, W, 3) Lighthouse-frame points of a depth image (NaN where invalid)."""
        d = np.where(depth > 0, depth*self.scale, np.nan)
        return self.C + d[..., None]*self.rays(depth.shape)

    def floor_z(self, x, y):
        return floor_z(self.plane, x, y)

    def project(self, P):
        import cv2
        rvec, _ = cv2.Rodrigues(self.R)
        uv, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 3), rvec, self.t, self.K, self.dist)
        return uv.reshape(-1, 2)
