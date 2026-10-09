"""Colour + depth detection of the brick, the platform and the pick fixture, in the Lighthouse frame.

Per frame:
1. Every pixel's depth becomes a Lighthouse point (calibrated camera pose); h = its height above the
   surveyed floor plane; each pixel carries its floor-area footprint (depth^2 / fx fy).
2. HSV masks: pink (brick and platform), blue (fixture). Height windows split pink: the brick's top
   (24-55 mm above the floor when seated) from the low parts (platform, fixture: 3-32 mm). A carried
   brick (higher) and the floor/tags/clutter (no colour or wrong height) drop out.
3. Points are clustered on a 5 mm floor grid; each cluster is fitted with the object's CAD footprint
   rectangle by maximising the covered area over (yaw, x, y). The fit tolerates holes (sawtooth
   valleys below the window) and partial views (the platform's two ends when the brick sits on it).
4. Yaw is known modulo 180 deg from the rectangle; the configured prior picks the half-turn.
5. Brick height: the median of (measured height - CAD top surface at each point's brick-frame x, y)
   is the brick origin's height above the floor (11.5 mm when seated), so a jammed brick reads high.

Anything that could be either of two objects, or a partial brick, is reported as ``ambiguous``.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .model import Model


class SceneError(RuntimeError):
    """The scene is ambiguous; the mission must not act on it."""


def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi


def quat_wxyz(yaw, pitch=0., roll=0.):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_quat()
    return (float(w), float(x), float(y), float(z))


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


@dataclass
class Detection:
    kind: str
    centre: np.ndarray            # footprint centre (x, y), m
    yaw: float                    # rad, half-turn resolved
    pos: np.ndarray               # frame origin (Lighthouse), m
    coverage: float               # covered fraction of the CAD footprint
    outside: float                # cluster area outside the footprint / footprint area
    z_rel: float = float('nan')   # brick: origin height above the floor (m)
    z_mad: float = float('nan')   # brick: robust spread of the height residuals (m)
    roll: float = 0.
    pitch: float = 0.
    n: int = 0

    def quat(self):
        return quat_wxyz(self.yaw, self.pitch, self.roll)

    def pose(self):
        from flight.frames import Pose
        return Pose(tuple(float(v) for v in self.pos), self.quat())

    def to_dict(self):
        return {'kind': self.kind, 'pos': [float(v) for v in self.pos], 'yaw_deg': math.degrees(self.yaw),
                'roll_deg': math.degrees(self.roll), 'pitch_deg': math.degrees(self.pitch),
                'centre': [float(v) for v in self.centre], 'coverage': self.coverage, 'outside': self.outside,
                'z_rel_mm': self.z_rel*1e3, 'z_mad_mm': self.z_mad*1e3, 'n': self.n}


@dataclass
class SceneResult:
    brick: Detection = None
    platform: Detection = None
    fixture: Detection = None
    issues: list = field(default_factory=list)      # human-readable
    ambiguous: bool = False
    debug: dict = field(default_factory=dict)

    def flag(self, msg, ambiguous=True):
        self.issues.append(msg)
        self.ambiguous |= ambiguous

    def to_dict(self):
        return {k: (getattr(self, k).to_dict() if getattr(self, k) is not None else None)
                for k in ('brick', 'platform', 'fixture')} | {'issues': self.issues, 'ambiguous': self.ambiguous}


# ---- small pieces ------------------------------------------------------------------------------
def hsv_mask(color, ranges):
    import cv2
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    m = np.zeros(color.shape[:2], bool)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    return m


def grow_mask(m, k=3):
    """Dilate a colour mask: blended edge pixels fail HSV, and the height window decides them instead."""
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def pca_yaw(xy, w):
    c = np.average(xy, axis=0, weights=w)
    d = xy - c
    cov = (d*w[:, None]).T @ d/w.sum()
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    return math.atan2(v[1], v[0])


def clusters(xy, radius, cell=0.005):
    """Index arrays of points whose 5 mm cells connect within ``radius``."""
    import cv2
    if len(xy) == 0:
        return []
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo)/cell).astype(int)
    pad = int(math.ceil(radius/cell)) + 1
    grid = np.zeros((ij[:, 0].max() + 2*pad + 1, ij[:, 1].max() + 2*pad + 1), np.uint8)
    grid[ij[:, 0] + pad, ij[:, 1] + pad] = 1
    r = int(math.ceil(radius/cell))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r + 1, 2*r + 1))
    n, labels = cv2.connectedComponents(cv2.dilate(grid, kernel), connectivity=8)
    lab = labels[ij[:, 0] + pad, ij[:, 1] + pad]
    return [np.nonzero(lab == k)[0] for k in range(1, n) if np.any(lab == k)]


def fit_rect(xy, w, size, yaw0=None, yaw_range=math.radians(12), shift=0.015, cell=0.001):
    """Best placement of a fixed L x W rectangle over weighted points: (centre, yaw, covered area)."""
    L, W = size
    c0 = np.average(xy, axis=0, weights=w)
    yaw0 = pca_yaw(xy, w) if yaw0 is None else yaw0
    nu, nv = int(round((L + 2*shift)/cell)), int(round((W + 2*shift)/cell))
    Lc, Wc = int(round(L/cell)), int(round(W/cell))

    def score(a):
        uv = (xy - c0) @ rot2(a)                      # rows: points in the rectangle's axes
        H, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=(nu, nv),
                                 range=[[-L/2 - shift, L/2 + shift], [-W/2 - shift, W/2 + shift]], weights=w)
        S = np.zeros((nu + 1, nv + 1))
        S[1:, 1:] = H.cumsum(0).cumsum(1)
        sums = S[Lc:, Wc:] - S[:-Lc, Wc:] - S[Lc:, :-Wc] + S[:-Lc, :-Wc]
        best = sums.max()
        i, j = np.nonzero(sums >= best - 1e-3*max(best, 1e-12))     # centre of a plateau
        off = np.array([i.mean()*cell - shift, j.mean()*cell - shift])
        return best, off

    def search(angles):
        res = [(score(a), a) for a in angles]
        (s, off), a = max(res, key=lambda r: r[0][0])
        return s, off, a

    s, off, a = search(yaw0 + np.arange(-yaw_range, yaw_range + 1e-9, math.radians(1.0)))
    s, off, a = search(a + np.arange(-math.radians(1.0), math.radians(1.0) + 1e-9, math.radians(0.1)))
    return c0 + rot2(a) @ off, a, s


def refine_moments(xy, w, c, a, size, grow=0.003):
    """Centroid and principal axis of the points inside the fitted rectangle (grown a little).

    For a fully visible object the moments are unbiased and beat the rectangle search's plateau;
    the rectangle only decides which points belong to the object.
    """
    keep = inside_rect(xy, c, a, size, grow)
    if keep.sum() < 10:
        return c, a
    c2 = np.average(xy[keep], axis=0, weights=w[keep])
    a2 = pca_yaw(xy[keep], w[keep])
    a2 = a2 if abs(wrap(a2 - a)) <= math.pi/2 else wrap(a2 + math.pi)
    return c2, a2


def choose_yaw(a, prior, ambiguity):
    """The half-turn of ``a`` nearest ``prior``; (yaw, ambiguous)."""
    y = a if abs(wrap(a - prior)) <= math.pi/2 else wrap(a + math.pi)
    return wrap(y), abs(abs(wrap(a - prior)) - math.pi/2) < ambiguity


def inside_rect(xy, centre, yaw, size, grow=0.):
    uv = (xy - centre) @ rot2(yaw)
    return (np.abs(uv[:, 0]) <= size[0]/2 + grow) & (np.abs(uv[:, 1]) <= size[1]/2 + grow)


# ---- the detector ------------------------------------------------------------------------------
class Detector:
    def __init__(self, model: Model, rig, cam):
        self.m, self.rig, self.cam = model, rig, cam
        bbox = cam.raw.get('tags_bbox')
        margin = rig['workspace_margin_m']
        self.ws = None if bbox is None else (np.asarray(bbox[0]) - margin, np.asarray(bbox[1]) + margin)

    def measure(self, frame):
        self.cam.check_frame(frame)
        P = self.cam.points(frame.depth)
        valid = np.isfinite(P[..., 2])
        h = np.where(valid, P[..., 2] - self.cam.floor_z(P[..., 0], P[..., 1]), np.nan)
        if self.ws is not None:
            valid &= (P[..., 0] >= self.ws[0][0]) & (P[..., 0] <= self.ws[1][0]) & \
                     (P[..., 1] >= self.ws[0][1]) & (P[..., 1] <= self.ws[1][1])
        d = frame.depth*self.cam.scale
        area = d*d/(self.cam.K[0, 0]*self.cam.K[1, 1])
        colors = self.rig['colors']
        return {'P': P, 'h': h, 'valid': valid, 'area': area,
                'pink': grow_mask(hsv_mask(frame.color, colors['pink'])) & valid,
                'blue': grow_mask(hsv_mask(frame.color, colors['blue'])) & valid}

    def _window(self, h, name):
        lo, hi = (v*1e-3 for v in self.rig['heights_mm'][name])
        with np.errstate(invalid='ignore'):
            return (h >= lo) & (h <= hi)

    def _fit_all(self, xy, w, size, radius, min_cluster):
        """[(idx, centre, yaw_mod_pi, coverage, outside)] for clusters with enough area."""
        A = size[0]*size[1]
        out = []
        for idx in clusters(xy, radius):
            tot = w[idx].sum()
            if tot < min_cluster*A:
                continue
            c, a, s = fit_rect(xy[idx], w[idx], size)
            out.append((idx, c, a, s/A, (tot - s)/A))
        return out

    def detect(self, frame):
        acc, yp = self.rig['accept'], self.rig['yaw_prior_deg']
        amb = math.radians(self.rig['yaw_ambiguity_deg'])
        meas = self.measure(frame)
        P, h, area = meas['P'], meas['h'], meas['area']
        res = SceneResult(debug={'masks': {}})

        # Brick: pink, at brick-top height.
        sel = meas['pink'] & self._window(h, 'brick')
        res.debug['masks']['brick'] = sel
        xy, w, hh = P[sel][:, :2], area[sel], h[sel]
        xy_b, w_b = xy, w
        fits = self._fit_all(xy, w, self.m.brick_size, 0.010, acc['min_cluster'])
        good = [f for f in fits if f[3] >= acc['brick_coverage'] and f[4] <= acc['max_outside']]
        if len(good) > 1:
            res.flag(f'{len(good)} brick-like objects')
        elif not good and any(f[3] >= acc['partial_coverage'] for f in fits):
            res.flag('partial brick: best coverage %.2f' % max(f[3] for f in fits))

        # Platform: pink, low, outside the brick's footprint.
        sel = meas['pink'] & self._window(h, 'low')
        xy, w = P[sel][:, :2], area[sel]
        for f in good:
            keep = ~inside_rect(xy, f[1], f[2], self.m.brick_size, grow=0.003)
            xy, w = xy[keep], w[keep]
        res.debug['masks']['platform'] = sel
        fits = self._fit_all(xy, w, self.m.platform_size, 0.040, acc['min_cluster']*acc['platform_coverage'])
        good_p = [f for f in fits if f[3] >= acc['platform_coverage'] and f[4] <= acc['max_outside']]
        if len(good_p) > 1:
            res.flag(f'{len(good_p)} platform-like objects')
        elif good_p:
            idx, c, a, cov, out = good_p[0]
            yaw, unsure = choose_yaw(a, math.radians(yp['platform']), amb)
            if unsure:
                res.flag('platform yaw is near prior +-90 deg; cannot tell its +x end')
            origin = c - rot2(yaw) @ np.array([self.m.platform_centre_x, 0.])
            z = float(self.cam.floor_z(*origin)) + self.m.platform_origin_height
            res.platform = Detection('platform', c, yaw, np.array([*origin, z]), cov, out, n=len(idx))

        # Fixture: blue, low. Informational (the plan picks from the brick's own pose).
        sel = meas['blue'] & self._window(h, 'low')
        res.debug['masks']['fixture'] = sel
        xy, w = P[sel][:, :2], area[sel]
        fits = self._fit_all(xy, w, self.m.fixture_size, 0.040, acc['min_cluster']*acc['fixture_coverage'])
        good_f = [f for f in fits if f[3] >= acc['fixture_coverage']]
        if len(good_f) > 1:
            res.flag(f'{len(good_f)} fixture-like objects', ambiguous=False)
        if good_f:
            idx, c, a, cov, out = max(good_f, key=lambda f: f[3])
            yaw, _ = choose_yaw(a, math.radians(yp['fixture']), amb)
            z = float(self.cam.floor_z(*c)) + self.m.fixture_origin_height
            res.fixture = Detection('fixture', c, yaw, np.array([*c, z]), cov, out, n=len(idx))

        # The brick last: a brick seated in a holder (fixture or platform) takes the holder's yaw,
        # measured over 100 mm instead of the brick's 50 mm (at ~3 mm per pixel).
        if len(good) == 1:
            idx, c, a, cov, out = good[0]
            c, a = refine_moments(xy_b[idx], w_b[idx], c, a, self.m.brick_size)
            yaw, unsure = choose_yaw(a, math.radians(yp['brick']), amb)
            holder = self._holder(c, yaw, res)
            if holder is not None:
                yaw = choose_yaw(holder.yaw, yaw, amb)[0]
                res.debug['brick_yaw_from'] = holder.kind
            elif unsure:
                res.flag('brick yaw is near prior +-90 deg; cannot tell its +x end')
            det = self._brick(xy_b[idx], w_b[idx], hh[idx], c, yaw, cov, out)
            lo, hi = (v*1e-3 for v in acc['brick_height_tol_mm'])
            dz = det.z_rel - self.m.brick_origin_height
            if not lo <= dz <= hi:
                res.flag(f'brick sits {dz*1e3:+.1f} mm from a seated height')
            res.brick = det
        return res

    def _holder(self, c, yaw, res):
        """The fixture/platform the brick is seated in: centred on it (platform: on its seat) and aligned."""
        tol = self.rig['holder']
        for h in (res.fixture, res.platform):
            if h is None:
                continue
            seat = h.centre if h.kind == 'fixture' else h.pos[:2] + rot2(h.yaw) @ np.array([self.m.platform_centre_x, 0.])
            d = (c - seat) @ rot2(h.yaw)
            if abs(d[0]) <= tol['along_m'] and abs(d[1]) <= tol['across_m'] and \
                    abs(wrap(2*(yaw - h.yaw)))/2 <= math.radians(tol['yaw_deg']):
                return h
        return None

    def _brick(self, xy, w, h, c, yaw, cov, out):
        """Brick pose: footprint centre and yaw, height (and optionally tilt) from the CAD top surface."""
        inner = inside_rect(xy, c, yaw, self.m.brick_size, grow=-0.002)
        loc = (xy[inner] - c) @ rot2(yaw)
        top = self.m.brick_top(loc[:, 0], loc[:, 1])
        r = h[inner] - top
        ok = np.isfinite(r)
        loc, r = loc[ok], r[ok]
        if len(r) < 5:
            z, mad = float('nan'), float('nan')
        else:
            z = float(np.median(r))
            mad = float(1.4826*np.median(np.abs(r - z)))
        roll = pitch = 0.
        if self.rig['estimate_tilt'] and len(r) >= 20:
            keep = np.abs(r - z) <= 3*max(mad, 1e-3)
            A = np.column_stack([np.ones(keep.sum()), loc[keep]])
            z, sx, sy = np.linalg.lstsq(A, r[keep], rcond=None)[0]
            pitch, roll = -math.atan(sx), math.atan(sy)
        pos = np.array([c[0], c[1], float(self.cam.floor_z(*c)) + z])
        return Detection('brick', c, yaw, pos, cov, out, z_rel=z, z_mad=mad, roll=roll, pitch=pitch, n=len(xy))


# ---- debug overlay -----------------------------------------------------------------------------
def overlay(frame, res: SceneResult, cam, model: Model):
    import cv2
    img = frame.color.copy()
    tint = {'brick': (0, 255, 0), 'platform': (255, 0, 255), 'fixture': (0, 255, 255)}
    for k, m in res.debug.get('masks', {}).items():
        img[m] = (0.5*img[m] + 0.5*np.array(tint[k])).astype(np.uint8)
    sizes = {'brick': model.brick_size, 'platform': model.platform_size, 'fixture': model.fixture_size}
    y = 30
    for det in (res.brick, res.platform, res.fixture):
        if det is None:
            continue
        L, W = sizes[det.kind]
        corners = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]]) @ rot2(det.yaw).T + det.centre
        z = cam.floor_z(corners[:, 0], corners[:, 1]) + (0.03 if det.kind == 'brick' else 0.01)
        uv = cam.project(np.column_stack([corners, z]))
        cv2.polylines(img, [np.round(uv).astype(np.int32)], True, tint[det.kind], 2)
        tip = det.centre + rot2(det.yaw) @ np.array([L/2 + 0.02, 0.])        # +x arrow
        a, b = cam.project(np.array([[*det.centre, z[0]], [*tip, z[0]]]))
        cv2.arrowedLine(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), tint[det.kind], 2)
        txt = (f'{det.kind}: ({det.pos[0]*1e3:.1f}, {det.pos[1]*1e3:.1f}, {det.pos[2]*1e3:.1f}) mm '
               f'yaw {math.degrees(det.yaw):.1f} cov {det.coverage:.2f}')
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tint[det.kind], 2)
        y += 26
    for msg in res.issues:
        cv2.putText(img, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 26
    return img
