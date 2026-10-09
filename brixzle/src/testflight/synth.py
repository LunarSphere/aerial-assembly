"""Synthetic overhead RealSense frames for tests and ``testflight demo``.

Renders the floor (with AprilTags), the blue fixture, the pink platform and the pink brick as a
height field seen by a pinhole camera: each pixel's ray is intersected with the height field by
fixed-point iteration, colours are 2x supersampled, and depth gets Gaussian noise that grows with
range squared (about 3 mm at 2 m, like a D455) plus dropped pixels.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .calib import rays_world
from .camera import Frame
from .model import Model, load_model

PINK_BRICK = (113, 110, 236)     # BGR: HSV about (0, 135, 236), as measured on the lab photos
PINK_PLATFORM = (112, 105, 240)
BLUE = (125, 45, 52)


@dataclass
class Placed:
    kind: str                     # 'brick' | 'platform' | 'fixture'
    x: float                      # footprint centre (m)
    y: float
    yaw_deg: float = 0.
    z_rel: float = None           # brick: origin height above the floor (default: seated)


@dataclass
class SynthScene:
    objects: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)          # id -> (centre (x, y), yaw_deg)
    tag_size: float = 0.10
    dictionary: str = 'DICT_APRILTAG_25h9'
    floor: tuple = (0., 0., 0.)                        # z = a x + b y + c

    def survey(self):
        c = self.floor
        return {'dictionary': self.dictionary, 'size_m': self.tag_size,
                'tags': {str(k): {'center': [x, y, c[0]*x + c[1]*y + c[2]], 'yaw_deg': yaw}
                         for k, ((x, y), yaw) in self.tags.items()}}


def lab_scene(objects=()):
    """Six tags laid out like the lab photos (2 x 3 on the floor of a ~1.2 x 2.0 m cage)."""
    tags = {0: ((-0.35, 0.70), 0.), 1: ((0.35, 0.70), 0.), 2: ((-0.35, 0.0), 0.), 3: ((0.35, 0.0), 0.),
            4: ((-0.35, -0.70), 0.), 5: ((0.35, -0.70), 0.)}
    return SynthScene(list(objects), tags)


def look_at(C, target, ref=(1., 0., 0.)):
    """(R, t) with X_cam = R X + t; image x follows ``ref`` projected, z looks at ``target``."""
    C, target, ref = (np.asarray(v, float) for v in (C, target, ref))
    z = target - C
    z /= np.linalg.norm(z)
    x = ref - (ref @ z)*z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ C


def d455_K(w=1280, h=720, hfov_deg=90.):
    f = w/2/math.tan(math.radians(hfov_deg)/2)
    return np.array([[f, 0, w/2 - 0.5], [0, f, h/2 - 0.5], [0, 0, 1.]])


class _HeightField:
    def __init__(self, scene: SynthScene, model: Model, cell=0.0005):
        self.cell, self.floor = cell, np.asarray(scene.floor, float)
        objs = scene.objects
        if not objs:
            self.lo, self.h = np.zeros(2), np.zeros((1, 1))
            self.col = np.zeros((1, 1, 3))
            return
        pts = np.array([[o.x, o.y] for o in objs])
        self.lo = pts.min(axis=0) - 0.09
        hi = pts.max(axis=0) + 0.09
        n = np.ceil((hi - self.lo)/cell).astype(int)
        gx, gy = np.meshgrid(self.lo[0] + (np.arange(n[0]) + 0.5)*cell, self.lo[1] + (np.arange(n[1]) + 0.5)*cell,
                             indexing='ij')
        self.h = np.zeros(n)
        self.col = np.zeros((*n, 3))
        for o in sorted(objs, key=lambda o: {'fixture': 0, 'platform': 0, 'brick': 1}[o.kind]):
            a = math.radians(o.yaw_deg)
            xl = (gx - o.x)*math.cos(a) + (gy - o.y)*math.sin(a)
            yl = -(gx - o.x)*math.sin(a) + (gy - o.y)*math.cos(a)
            if o.kind == 'brick':
                L, W = model.brick_size
                z0 = model.brick_origin_height if o.z_rel is None else o.z_rel
                top = z0 + model.brick_top(xl, yl)
                color = PINK_BRICK
            else:
                L, W = model.fixture_size if o.kind == 'fixture' else model.platform_size
                px, pz = model.fixture_profile if o.kind == 'fixture' else model.platform_profile
                base = model.fixture_origin_height if o.kind == 'fixture' else model.platform_origin_height
                top = np.maximum(base + np.interp(xl, px, pz), 0.0015)
                color = BLUE if o.kind == 'fixture' else PINK_PLATFORM
            inside = (np.abs(xl) <= L/2) & (np.abs(yl) <= W/2) & np.isfinite(top)
            upd = inside & (top > self.h)
            self.h[upd] = top[upd]
            self.col[upd] = color

    def lookup(self, x, y):
        i = np.floor((x - self.lo[0])/self.cell).astype(int)
        j = np.floor((y - self.lo[1])/self.cell).astype(int)
        ok = (i >= 0) & (i < self.h.shape[0]) & (j >= 0) & (j < self.h.shape[1])
        h = np.zeros(x.shape)
        col = np.full((*x.shape, 3), np.nan)
        h[ok] = self.h[i[ok], j[ok]]
        c = self.col[i[ok], j[ok]]
        c[self.h[i[ok], j[ok]] <= 0] = np.nan
        col[ok] = c
        return h, col


def _tag_images(scene, px=140):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, scene.dictionary))
    return {k: cv2.aruco.generateImageMarker(d, k, px, borderBits=1) for k in scene.tags}


def render(scene: SynthScene, R, t, K, shape=(720, 1280), model=None, ss=2):
    """Noise-free (color BGR uint8, depth m) of the scene."""
    model = model or load_model()
    h, w = shape
    u, v = np.meshgrid((np.arange(w*ss) + 0.5)/ss - 0.5, (np.arange(h*ss) + 0.5)/ss - 0.5)
    C, d = rays_world(np.column_stack([u.ravel(), v.ravel()]), R, t, K, np.zeros(5))
    a, b, c = scene.floor
    hf = _HeightField(scene, model)

    def floor_at(x, y):
        return a*x + b*y + c
    X = C + ((c - C[2])/d[:, 2])[:, None]*d
    s = (floor_at(X[:, 0], X[:, 1]) - C[2])/d[:, 2]          # floor hit (planar floor: one refinement)
    # March the rays that pass over the objects from 6 cm down to the floor in 0.25 mm steps.
    top = 0.06
    s_top = (c + top - C[2])/d[:, 2]
    Xt = C + s_top[:, None]*d
    hi = hf.lo + np.array(hf.h.shape)*hf.cell
    near = np.nonzero((np.minimum(Xt[:, 0], X[:, 0]) <= hi[0]) & (np.maximum(Xt[:, 0], X[:, 0]) >= hf.lo[0]) &
                      (np.minimum(Xt[:, 1], X[:, 1]) <= hi[1]) & (np.maximum(Xt[:, 1], X[:, 1]) >= hf.lo[1]))[0]
    steps = np.linspace(0., 1., 241)
    for chunk in np.array_split(near, max(1, len(near)//20000)):
        ss_ = s_top[chunk, None] + steps[None, :]*(s[chunk] - s_top[chunk])[:, None]
        P = C + ss_[..., None]*d[chunk, None, :]
        hh, _ = hf.lookup(P[..., 0], P[..., 1])
        below = P[..., 2] <= floor_at(P[..., 0], P[..., 1]) + hh
        first = np.argmax(below, axis=1)
        s[chunk] = np.where(below.any(axis=1), ss_[np.arange(len(chunk)), first], s[chunk])
    X = C + s[:, None]*d
    hh, col = hf.lookup(X[:, 0], X[:, 1])
    # Floor: grey woven texture.
    g = 150 + 18*np.sin(X[:, 0]*900)*np.sin(X[:, 1]*700) + 10*np.sin((X[:, 0] + X[:, 1])*2300)
    img = np.repeat(g[:, None], 3, axis=1)
    tags = _tag_images(scene)
    for k, ((tx, ty), yaw) in scene.tags.items():
        ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ul = (X[:, 0] - tx)*ca + (X[:, 1] - ty)*sa
        vl = -(X[:, 0] - tx)*sa + (X[:, 1] - ty)*ca
        S = scene.tag_size
        paper = (np.abs(ul) <= 0.7*S) & (np.abs(vl) <= 0.7*S)
        img[paper] = 245
        m = (np.abs(ul) < S/2) & (np.abs(vl) < S/2)
        im = tags[k]
        n = im.shape[0]
        cols = np.clip(((ul[m]/S + 0.5)*n).astype(int), 0, n - 1)
        rows = np.clip(((0.5 - vl[m]/S)*n).astype(int), 0, n - 1)
        img[m] = np.where(im[rows, cols] > 127, 245, 20)[:, None]
    obj = np.isfinite(col[:, 0])
    shade = 0.85 + 0.15*np.clip(hh/0.05, 0, 1)                   # a little height shading
    img[obj] = col[obj]*shade[obj, None]
    img = img.reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
    depth = (s.reshape(h, ss, w, ss)[:, ss//2, :, ss//2]).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), depth


class SynthCamera:
    """Renders once; every ``capture`` adds fresh noise (as if a new burst were taken)."""

    def __init__(self, scene, R, t, K=None, shape=(720, 1280), seed=0, depth_sigma_2m=0.0015,
                 drop=0.02, color_sigma=3., model=None):
        self.K = d455_K(shape[1], shape[0]) if K is None else K
        self.R, self.t = R, t
        self.color, self.depth = render(scene, R, t, self.K, shape, model)
        self.rng = np.random.default_rng(seed)
        self.depth_sigma_2m, self.drop, self.color_sigma = depth_sigma_2m, drop, color_sigma

    def capture(self, n=None):
        r = self.rng
        sig = self.depth_sigma_2m*(self.depth/2.0)**2
        depth = self.depth + r.normal(size=self.depth.shape)*sig
        depth[r.random(self.depth.shape) < self.drop] = 0.
        color = np.clip(self.color + r.normal(size=self.color.shape)*self.color_sigma, 0, 255).astype(np.uint8)
        return Frame(color, depth.astype(np.float32), self.K, np.zeros(5))

    def close(self):
        pass
