"""Kinematic stacking animation: one brick released at the aim point over another, sliding into its seats.

Geometry only, no physics: at each horizontal offset the upper brick sits at the lowest height where
none of its feet overlap the lower brick's hoppers (bisection on the convex pieces). The path follows
the slopes: down the ramp the foot landed on to the crease, then down the crease into the slanted slot.
Rendered with MuJoCo's offscreen renderer (brixzle.render). Not a simulation: it shows how the parts
fit, not whether a real drop seats.
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


def _eqs(pieces):
    return [ConvexHull(v).equations for v in pieces]


def _hit(eqs, pts, depth):
    h = np.c_[pts, np.ones(len(pts))]
    return any(np.any(np.all(h @ e.T <= -depth, axis=1)) for e in eqs)


def _samples(pieces, n, rng):
    return np.vstack([rng.dirichlet(np.ones(len(v)), n) @ v for v in pieces] + list(pieces))


def rest_height(p: BBParams, offsets, n=400, depth=0.05, seed=0, step=0.1):
    """Height z (mm, relative to the seated course pitch H0) of the upper brick at each (dx, dy) offset.

    From the previous offset's height, the brick moves across and then down until it touches: the
    lowest height reachable from above, not the lowest clear one (a seated tooth is clear at z = 0
    but blocked straight above by the hook, so a plain bisection would leave it hanging).
    """
    rng = np.random.default_rng(seed)
    lower = [q for i in range(p.n_vox) for q in G.hopper_pieces(p, p.seat_xy(i))]
    feet = [q + [0., 0., p.H0] for s in G.feet_seats(p) for q in G.foot_pieces(p, s)]
    low_eqs, low_pts = _eqs(lower), _samples(lower, n, rng)
    feet_pts = _samples(feet, n, rng)

    def clear(dx, dy, z):
        d = np.array([dx, dy, z])
        return not (_hit(low_eqs, feet_pts + d, depth) or _hit(_eqs([f + d for f in feet]), low_pts, depth))

    z, out = p.guide_h + 5., []
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


def path(p: BBParams, step=0.4):
    """Horizontal offsets from the aim point down the ramp to the crease, then down the crease to the seat."""
    ax, ay = p.aim
    c = min(ax, ay)
    legs = [((ax, ay), (c, c)), ((c, c), (0., 0.))]
    pts = []
    for (x0, y0), (x1, y1) in legs:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0)/step))
        pts += [(x0 + (x1 - x0)*t, y0 + (y1 - y0)*t) for t in np.linspace(0, 1, n, endpoint=False)]
    return pts + [(0., 0.)]


def scene(p: BBParams, lower_alpha=0.45):
    """MJCF: the lower brick as static geoms, the upper brick on a free joint; no contacts."""
    pieces = G.brick_pieces(p)
    root = ET.Element('mujoco', {'model': 'biasbrick-stack'})
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', {'offwidth': '1280', 'offheight': '960'})
    ET.SubElement(visual, 'headlight', {'ambient': '0.4 0.4 0.4', 'diffuse': '0.6 0.6 0.6'})
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', {'type': 'skybox', 'builtin': 'gradient', 'rgb1': '1 1 1', 'rgb2': '0.75 0.8 0.88',
                                     'width': '32', 'height': '32'})
    for k, v in enumerate(pieces):
        _mesh(asset, f'piece{k}', v)
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', {'type': 'plane', 'size': '0.3 0.3 0.01', 'pos': f'0 0 {-p.foot_L*MM - 0.002}',
                                  'rgba': '0.85 0.85 0.85 1', 'contype': '0', 'conaffinity': '0'})
    for k in range(len(pieces)):
        ET.SubElement(world, 'geom', {'type': 'mesh', 'mesh': f'piece{k}', 'rgba': f'0.2 0.45 0.8 {lower_alpha}',
                                      'contype': '0', 'conaffinity': '0'})
    body = ET.SubElement(world, 'body', {'name': 'upper', 'pos': f'0 0 {p.H0*MM}'})
    ET.SubElement(body, 'freejoint')
    for k in range(len(pieces)):
        ET.SubElement(body, 'geom', {'type': 'mesh', 'mesh': f'piece{k}', 'rgba': '0.95 0.55 0.15 1',
                                     'contype': '0', 'conaffinity': '0', 'mass': '0.001'})
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def frames(p: BBParams, drop=25., fall_frames=40, hold_frames=45, step=0.4):
    """qpos frames (m) and the offsets and heights (mm) they come from."""
    offs = path(p, step)
    z = rest_height(p, offs)
    q = []
    for t in np.linspace(0, 1, fall_frames):
        q.append((offs[0], z[0] + drop*(1 - t*t)))
    q += list(zip(offs, z))
    q += [q[-1]]*hold_frames
    out = [np.array([dx*MM, dy*MM, (p.H0 + dz)*MM, 1., 0., 0., 0.]) for (dx, dy), dz in q]
    return out, offs, z


def animate(p: BBParams, out_dir, step=0.4):
    """Write stack.mp4 (overview), stack-corner.mp4 (close on one seat) and a contact sheet PNG."""
    model = scene(p)
    qs, offs, z = frames(p, step=step)
    sx, sy = p.seat_xy(0)
    mid = [0., 0., (p.H0 - p.foot_L/2)*MM]
    corner = [sx*MM, sy*MM, (p.bar_h + p.slot_h)*MM]
    render(model, qs, out_dir/'stack.mp4', azimuth=-135, elevation=-25, lookat=mid, distance=0.34, fps=30)
    render(model, qs, out_dir/'stack-corner.mp4', azimuth=-120, elevation=-20, lookat=corner, distance=0.13,
           fps=30)
    _contact_sheet(model, qs, out_dir/'stack-frames.png', mid, corner)
    return {'offsets_mm': [list(o) for o in offs], 'rest_height_mm': z.tolist(), 'frames': len(qs)}


def _contact_sheet(model, qs, out, mid, corner, n=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = mujoco.MjData(model)
    idx = np.linspace(0, len(qs) - 1, n).astype(int)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 7))
    with mujoco.Renderer(model, 360, 480) as r:
        for row, (look, dist, az, el) in enumerate(((mid, 0.34, -135, -25), (corner, 0.13, -120, -20))):
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, dist, az, el
            for col, i in enumerate(idx):
                data.qpos[:] = qs[i]
                mujoco.mj_forward(model, data)
                r.update_scene(data, cam)
                axes[row, col].imshow(r.render())
                axes[row, col].set_title(f'frame {i}')
                axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
