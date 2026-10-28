"""Kinematic stacking animation: a B brick released at the aim point over an A brick, sliding into its seats.

Geometry only, no physics: at each horizontal offset the upper brick sits at the lowest height it can
reach from above without overlapping the lower brick (convex pieces). The path settles X first (the
sawtooth faces are steeper than the Y slope): the leaning tooth rides on its tip to the slot mouth and
enters along its axis from the stroke side (+x); then the brick slides down the Y slope into its nest. Rendered
with MuJoCo's offscreen renderer (brixzle.render). Not a simulation: it shows how the parts fit, not
whether a real drop seats.
"""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from brixzle.render import render
from brixzle.scene import MM, _mesh

from . import geometry as G
from .params import BBParams


class _Hulls:
    """Convex pieces as half-space equations with bounding boxes, for fast point-inside tests."""

    def __init__(self, pieces):
        self.eqs = [ConvexHull(v).equations for v in pieces]
        self.lo = np.array([v.min(0) for v in pieces])
        self.hi = np.array([v.max(0) for v in pieces])

    def hit(self, pts, shift, depth):
        pts = pts - shift  # into this set's frame
        for e, lo, hi in zip(self.eqs, self.lo, self.hi):
            near = np.all((pts >= lo - depth) & (pts <= hi + depth), axis=1)
            if near.any():
                q = pts[near]
                if np.any(np.all(np.c_[q, np.ones(len(q))] @ e.T <= -depth, axis=1)):
                    return True
        return False


def _samples(pieces, n, rng):
    return np.vstack([rng.dirichlet(np.ones(len(v)), n) @ v for v in pieces] + list(pieces))


def rest_height(p: BBParams, offsets, n=120, depth=0.05, seed=0, step=0.1, lower='A', upper='B'):
    """Height z (mm, relative to the seated course pitch H0) of the upper brick at each (dx, dy) offset.

    From the previous offset's height, the brick moves across and then down until it touches: the
    lowest height reachable from above, not the lowest clear one (a seated tooth is clear at z = 0
    but blocked straight above by its hook, so a plain bisection would leave it hanging).
    """
    rng = np.random.default_rng(seed)
    low = G.brick_pieces(p, lower)
    up = [q + [0., 0., p.H0] for q in G.brick_pieces(p, upper)]
    low_h, up_h = _Hulls(low), _Hulls(up)
    low_pts, up_pts = _samples(low, n, rng), _samples(up, n, rng)

    def clear(dx, dy, z):
        d = np.array([dx, dy, z])
        return not (low_h.hit(up_pts + d, 0., depth) or up_h.hit(low_pts, d, depth))

    z, out = p.H0 + 10., []
    for dx, dy in offsets:
        while not clear(dx, dy, z):  # moving across ran into something: rise over it
            z += step
        lo = z
        while lo - step > -3. and clear(dx, dy, lo - step):
            lo -= step
        hi, lo = lo, lo - step  # hi clear, lo blocked (or the floor of the search)
        for _ in range(6):
            mid = (lo + hi)/2
            lo, hi = (lo, mid) if clear(dx, dy, mid) else (mid, hi)
        z = hi
        out.append(z)
    return np.array(out)


def entry_x(p: BBParams):
    """Keel X offset at which the leaning tooth's tip is over the slot mouth: the stroke starts there.

    The tooth trails toward -x along its axis (brixzle: every tooth leans alpha from vertical), so it
    enters from +x; a brick landing left of that rides on its tooth tip down the ramp until then.
    """
    return p.tooth_L*math.sin(math.radians(p.alpha)) + 0.5


def path(p: BBParams, below='A', step=0.4):
    """Horizontal offsets from the aim point: in X to the stroke start, down the stroke onto the seat,
    then down the Y slope into the nest."""
    ax, ay = G.aim(p, below)
    ex = max(ax, entry_x(p))
    legs = [((ax, ay), (ex, ay)), ((ex, ay), (0., ay)), ((0., ay), (0., 0.))]
    pts = []
    for (x0, y0), (x1, y1) in legs:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0)/step))
        pts += [(x0 + (x1 - x0)*t, y0 + (y1 - y0)*t) for t in np.linspace(0, 1, n, endpoint=False)]
    return pts + [(0., 0.)]


def scene(p: BBParams, lower='A', upper='B', lower_alpha=0.45):
    """MJCF: the lower brick as static geoms, the upper brick on a free joint; no contacts."""
    root = ET.Element('mujoco', {'model': 'biasbrick-stack'})
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', {'offwidth': '1280', 'offheight': '960'})
    ET.SubElement(visual, 'headlight', {'ambient': '0.4 0.4 0.4', 'diffuse': '0.6 0.6 0.6'})
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', {'type': 'skybox', 'builtin': 'gradient', 'rgb1': '1 1 1', 'rgb2': '0.75 0.8 0.88',
                                     'width': '32', 'height': '32'})
    low, up = G.brick_pieces(p, lower), G.brick_pieces(p, upper)
    for k, v in enumerate(low):
        _mesh(asset, f'low{k}', v)
    for k, v in enumerate(up):
        _mesh(asset, f'up{k}', v)
    world = ET.SubElement(root, 'worldbody')
    floor = min(v[:, 2].min() for v in low)
    ET.SubElement(world, 'geom', {'type': 'plane', 'size': '0.3 0.3 0.01', 'pos': f'0 0 {floor*MM - 0.002}',
                                  'rgba': '0.85 0.85 0.85 1', 'contype': '0', 'conaffinity': '0'})
    for k in range(len(low)):
        ET.SubElement(world, 'geom', {'type': 'mesh', 'mesh': f'low{k}', 'rgba': f'0.2 0.45 0.8 {lower_alpha}',
                                      'contype': '0', 'conaffinity': '0'})
    body = ET.SubElement(world, 'body', {'name': 'upper', 'pos': f'0 0 {p.H0*MM}'})
    ET.SubElement(body, 'freejoint')
    for k in range(len(up)):
        ET.SubElement(body, 'geom', {'type': 'mesh', 'mesh': f'up{k}', 'rgba': '0.95 0.55 0.15 1',
                                     'contype': '0', 'conaffinity': '0', 'mass': '0.001'})
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def frames(p: BBParams, drop=25., fall_frames=40, hold_frames=45, step=0.4):
    """qpos frames (m) and the offsets and heights (mm) they come from."""
    offs = path(p, 'A', step)
    z = rest_height(p, offs)
    q = []
    for t in np.linspace(0, 1, fall_frames):
        q.append((offs[0], z[0] + drop*(1 - t*t)))
    q += list(zip(offs, z))
    q += [q[-1]]*hold_frames
    out = [np.array([dx*MM, dy*MM, (p.H0 + dz)*MM, 1., 0., 0., 0.]) for (dx, dy), dz in q]
    return out, offs, z


VIEWS = {  # name: (lookat in mm relative to socket 0's seat, distance m, azimuth, elevation)
    'stack': (None, 0.36, -135, -25),
    'stack-socket': ((0., 15., 20.), 0.16, -60, -15),
    'stack-y-slide': ((0., 15., 20.), 0.17, 180, -5),
}


def _look(p: BBParams, rel):
    if rel is None:
        return [0., 10.*MM, (p.H0 + 20.)*MM]
    x0 = p.socket_x(0) + G.seat_x(p, 'A')
    return [(x0 + rel[0])*MM, rel[1]*MM, (p.H0 + rel[2])*MM]


def animate(p: BBParams, out_dir, step=0.4):
    """Write one video per view and a contact sheet PNG."""
    model = scene(p)
    qs, offs, z = frames(p, step=step)
    for name, (rel, dist, az, el) in VIEWS.items():
        render(model, qs, out_dir/f'{name}.mp4', azimuth=az, elevation=el, lookat=_look(p, rel), distance=dist, fps=30)
    _contact_sheet(p, model, qs, out_dir/'stack-frames.png')
    return {'offsets_mm': [list(o) for o in offs], 'rest_height_mm': z.tolist(), 'frames': len(qs),
            'aim_mm': list(G.aim(p, 'A'))}


def _contact_sheet(p, model, qs, out, n=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = mujoco.MjData(model)
    idx = np.linspace(0, len(qs) - 1, n).astype(int)
    fig, axes = plt.subplots(len(VIEWS), n, figsize=(4*n, 3.5*len(VIEWS)))
    with mujoco.Renderer(model, 360, 480) as r:
        for row, (name, (rel, dist, az, el)) in enumerate(VIEWS.items()):
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = _look(p, rel), dist, az, el
            for col, i in enumerate(idx):
                data.qpos[:] = qs[i]
                mujoco.mj_forward(model, data)
                r.update_scene(data, cam)
                axes[row, col].imshow(r.render())
                axes[row, col].set_title(f'{name}, frame {i}')
                axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
