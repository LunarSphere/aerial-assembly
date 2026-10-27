"""DF-brick geometry as convex pieces (mm), the source of truth for collision and the CAD solid.

Brick frame: x along the brick, centred on its middle voxel; y across; z = 0 is the rail bottom.
The feet hang below z = 0, the receivers stand on the rail top (z = bar_h).

A receiver wall is a 2D section in (r, z), where r is the distance from the receiver centre
along the wall's outward normal: a vertical throat plate, then a flared plate whose inner face
runs from the top of the throat to the ridge at r = extent. A pocket has four walls, mitred at
the corners (lateral |s| <= r): each piece is the section's prism cut by two half-spaces, so it
is convex. A slot has only the two walls normal to Y, as straight prisms over the voxel.
"""
import math

import numpy as np

from .params import DFParams


def _box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def wall_sections(p: DFParams, z0):
    """(r, z) polygons of one wall: the throat plate and the flared plate."""
    t, E, w = p.throat, p.extent, p.wall_t
    f = math.radians(p.flare)
    zp = z0 + p.parallel_len
    ztop = zp + p.flare_h
    drop = w/math.sin(f)  # vertical thickness of the flared plate
    throat = [(t, z0), (t + w, z0), (t + w, zp), (t, zp)]
    flared = [(t, zp), (E, ztop), (E, ztop - drop), (t, max(z0, zp - drop))]
    return throat, flared


def inner_face_z(p: DFParams, r, z0):
    """Height of a receiver's inner face at distance r (throat <= r <= extent)."""
    return z0 + p.parallel_len + (r - p.throat)/math.tan(math.radians(p.flare))


def _wall_piece(section, centre, normal, mitre, half_len):
    n = np.array([normal[0], normal[1], 0.])
    lat = np.array([-normal[1], normal[0], 0.])
    c = np.array([centre[0], centre[1], 0.])
    pts = []
    for r, z in section:
        s = r if mitre else half_len
        for sign in (-1, 1):
            pts.append(c + r*n + sign*s*lat + np.array([0, 0, z]))
    return np.array(pts)


def receiver_pieces(p: DFParams, kind, centre=(0., 0.), z0=None, yaw=0.):
    """Convex pieces of one receiver (pocket or slot); slot walls are normal to the yawed Y axis."""
    z0 = p.bar_h if z0 is None else z0
    a = math.radians(yaw)
    ex, ey = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    if kind == 'pocket':
        normals, mitre = [ex, (-ex[0], -ex[1]), ey, (-ey[0], -ey[1])], True
    elif kind == 'slot':
        normals, mitre = [ey, (-ey[0], -ey[1])], False
    else:
        raise ValueError(f'unknown receiver kind {kind!r}')
    return [_wall_piece(sec, centre, nrm, mitre, p.extent) for nrm in normals for sec in wall_sections(p, z0)]


def foot_piece(p: DFParams, x):
    a, c, L = p.half_foot, p.tip_chamfer, p.foot_L
    pts = []
    for h, z in ((a, 0.), (a, -L + c), (a - c, -L)):
        pts += [[x + sx*h, sy*h, z] for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(pts, float)


def cup_sections(p: DFParams):
    """(r, z) polygons of one side of a cup foot's shell (brick frame, rail bottom z = 0).

    Its outer face is the receiver's inner face moved in by the clearance: straight past the
    throat, then flared to just inside the ridge, then a short vertical skirt up to the rail.
    """
    f = math.radians(p.flare)
    w, c, a, E = p.wall_t, p.clearance, p.half_foot, p.extent
    z_tip = -p.foot_L
    zp = z_tip + p.parallel_len
    zg = z_tip + p.guide_h
    rise = w/math.sin(f)  # vertical thickness of the flared plate
    flared = [(a, zp), (E - c, zg), (E - c, min(zg + rise, 0.)), (a, zp + rise)]
    skirt = [(E - c - w, zg), (E - c, zg), (E - c, 0.), (E - c - w, 0.)]
    return [flared, skirt]


def foot_pieces(p: DFParams, x):
    if p.foot_kind == 'peg':
        return [foot_piece(p, x)]
    a, ch = p.half_foot, p.tip_chamfer
    z_tip = -p.foot_L
    top = z_tip + p.parallel_len + p.wall_t/math.sin(math.radians(p.flare))
    core = []
    for h, z in ((a, top), (a, z_tip + ch), (a - ch, z_tip)):
        core += [[x + sx*h, sy*h, z] for sx in (-1, 1) for sy in (-1, 1)]
    out = [np.array(core, float)]
    for nrm in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        out += [_wall_piece(sec, (x, 0.), nrm, True, 0.) for sec in cup_sections(p)]
    return out


def feet_x(p: DFParams):
    return [p.voxel_x(0), p.voxel_x(p.n_vox - 1)]


def rail_width(p: DFParams):
    """The rail carries every throat's floor and wall feet."""
    return 2*(p.throat + p.wall_t)


def rail_piece(p: DFParams):
    half, wy = p.length/2 - p.end_gap/2, rail_width(p)/2
    return _box(-half, half, -wy, wy, 0., p.bar_h)


def rib_top(p: DFParams):
    """Rib top next to an inner ridge: 0.5 mm into the flared walls' underside, so they fuse."""
    f = math.radians(p.flare)
    return inner_face_z(p, p.U/2 - p.rib_t/2, p.bar_h) - p.wall_t/math.sin(f) + 0.5


def bore_radius(p: DFParams):
    return p.fork_d/2 + p.fork_clearance


def channels(p: DFParams):
    """(x, z) of the two straight Y tine bores, in the ribs under the ridges either side of centre."""
    r = bore_radius(p)
    z = rib_top(p) - 1.2 - r
    xs = [p.voxel_x(i) + p.U/2 for i in range(p.n_vox - 1)]
    if len(xs) > 2:  # use the two ribs nearest the centre
        xs = sorted(xs, key=abs)[:2]
    if len(xs) == 1:  # n = 2: one rib; the bores straddle it inside the rail-wide rib
        xs = [xs[0] - p.rib_t/4, xs[0] + p.rib_t/4]
    return [(float(x), float(z)) for x in sorted(xs)]


def rib_pieces(p: DFParams, bored=False):
    """Under each inner ridge: a band across the full depth around the bore, on a post over the rail."""
    top, E, half = rib_top(p), p.extent, p.rib_t/2
    r, wy = bore_radius(p), rail_width(p)/2
    zb = channels(p)[0][1]
    band = zb - r - 1.2
    xs = [p.voxel_x(i) + p.U/2 for i in range(p.n_vox - 1)]
    bored_x = {round(x, 6) for x, _ in channels(p)} if bored else set()
    out = []
    for xb in xs:
        out.append(_box(xb - half, xb + half, -wy, wy, 0., band))
        if round(xb, 6) not in bored_x:
            out.append(_box(xb - half, xb + half, -E, E, band, top))
            continue
        # Square hole around the bore, as brixzle.cad.bored_pieces.
        out += [_box(xb - half, xb + half, -E, E, band, zb - r),
                _box(xb - half, xb + half, -E, E, zb + r, top),
                _box(xb - half, xb - r, -E, E, zb - r, zb + r),
                _box(xb + r, xb + half, -E, E, zb - r, zb + r)]
    return out


def brick_pieces(p: DFParams, bored=False):
    pieces = [rail_piece(p), *[q for x in feet_x(p) for q in foot_pieces(p, x)], *rib_pieces(p, bored)]
    for i in range(p.n_vox):
        pieces += receiver_pieces(p, p.receiver_kind(i), (p.voxel_x(i), 0.))
    return pieces


def base_pieces(p: DFParams, receivers, thickness=4.):
    """Anchored base: receivers [(x, y, kind, yaw)] (world mm) with floors at z = 0 on a slab."""
    pieces = []
    for x, y, kind, yaw in receivers:
        pieces.append(_box(x - p.U/2, x + p.U/2, y - p.U/2, y + p.U/2, -thickness, 0.))
        pieces += receiver_pieces(p, kind, (x, y), 0., yaw)
    return pieces
