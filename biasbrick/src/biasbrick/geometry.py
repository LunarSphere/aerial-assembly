"""biasbrick geometry: brixzle's A/B profile cut into sockets, swept one-sided in Y, joined by a rod.

Everything is convex pieces (mm), the source of truth for collision and the CAD solid, as in
brixzle: a socket is one voxel of brixzle's A (or B) outer profile (``profile.brick_parts``,
clipped to the voxel), decomposed into convex polygons (``convex.convex_parts``) and swept through
Y in two affine bands either side of the Y valley (``rise``): the main slope toward +y and the
nest face toward -y, so every piece stays convex. The flange is a band under the socket's top line,
swept along the main slope past the +y face.
The rod and the fork sleeves are boxes and prisms in the brick frame; nothing of them is sheared.
"""
import math

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from brixzle import profile as pr
from brixzle.convex import convex_parts

from .params import BBParams


def _geoms(g):
    if g.is_empty:
        return []
    gs = [g] if g.geom_type == 'Polygon' else [h for h in getattr(g, 'geoms', []) if h.geom_type == 'Polygon']
    return [h for h in gs if h.area > 1e-3]


def _box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def rise(p: BBParams, y):
    """Z shift of the swept socket at y: zero along the Y valley, up the main slope toward +y and up
    the nest face toward -y (an asymmetric sawtooth in Y, which the keel copies)."""
    d = np.asarray(y, float) - p.valley_y
    return np.where(d >= 0, p.shear_k*d, -p.nest_k*d)


def bands(p: BBParams, y0, y1):
    """[y0, y1] split at the valley, so each part is one affine band."""
    v = p.valley_y
    return [(a, b) for a, b in ((y0, min(y1, v)), (max(y0, v), y1)) if b - a > 1e-9]


def sweep(p: BBParams, polys, y0, y1, dx=0.):
    """Convex pieces of (x, z) polygons swept from y0 to y1 through the Y sawtooth, moved by dx."""
    pieces = []
    for poly in polys:
        for g in _geoms(poly):
            for q in convex_parts(np.asarray(g.simplify(1e-4).exterior.coords)[:-1]):
                for a, b in bands(p, y0, y1):
                    near = np.column_stack([q[:, 0] + dx, np.full(len(q), a), q[:, 1] + rise(p, a)])
                    far = np.column_stack([q[:, 0] + dx, np.full(len(q), b), q[:, 1] + rise(p, b)])
                    pieces.append(np.vstack([near, far]))
    return pieces


def _flags(p: BBParams, part=None):
    """(top mirrored, bottom mirrored, top slots reversed, bottom teeth reversed) of a part."""
    return pr.PART_FLAGS[part or p.part]


def top_line(p: BBParams, x, part=None):
    """Socket top surface along the Y valley (socket frame, x centred)."""
    bp = p.profile()
    return pr.sawtooth(bp, np.asarray(x, float) + bp.U/2, _flags(p, part)[0]) + bp.H0


def bottom_line(p: BBParams, x, part=None):
    bp = p.profile()
    return pr.sawtooth(bp, np.asarray(x, float) + bp.U/2, _flags(p, part)[1])


def _centre(g, U):
    return affinity.translate(g, -U/2, 0.)


def socket_profile(p: BBParams, part=None):
    """One voxel of brixzle's part profile, x centred: outer, lightening holes, slots and teeth."""
    part = part or p.part
    bp = p.profile()
    parts = pr.brick_parts(bp, part)
    e = bp.end_gap/2
    lo, hi = parts['outer'].bounds[1] - 1, parts['outer'].bounds[3] + 1
    pieces = _geoms(parts['outer'].intersection(box(e, lo, bp.U - e, hi)))
    outer = max(pieces, key=lambda g: g.area)
    holes = []
    inner = box(e + bp.wall, lo, bp.U - e - bp.wall, hi)
    for h in pr.lightening_holes(bp, parts):
        holes += [g for g in _geoms(h.intersection(inner)) if g.area > 6.]
    mine = lambda pts: [i for i, v in enumerate(pts) if 0 < v[0] < bp.U]  # noqa: E731
    slots = [parts['slots'][i] for i in mine(parts['valleys'])]
    teeth = [parts['teeth'][i] for i in mine(parts['keels'])]
    valley = parts['valleys'][mine(parts['valleys'])[0]]
    keel = parts['keels'][mine(parts['keels'])[0]]
    # Under the nest face the socket has no tooth: the slot channel bends at the Y valley, so a rigid
    # tooth there would be lifted against its hook while the brick slides down the main slope.
    xs = pr.breakpoints(bp, -1., bp.U + 1., _flags(p, part)[1])
    above = Polygon(np.vstack([np.column_stack([xs, pr.sawtooth(bp, xs, _flags(p, part)[1]) - 0.05]),
                               [[xs[-1], hi], [xs[0], hi]]]))
    toothless = max(_geoms(outer.intersection(above)), key=lambda g: g.area)
    return {'outer': _centre(outer, bp.U), 'toothless': _centre(toothless, bp.U),
            'holes': [_centre(h, bp.U) for h in holes],
            'slots': [_centre(s, bp.U) for s in slots], 'teeth': [_centre(t, bp.U) for t in teeth],
            'valley': (float(valley[0] - bp.U/2), float(valley[1])),
            'keel': (float(keel[0] - bp.U/2), float(keel[1])), 'fragments': len(pieces)}


def seat_x(p: BBParams, below):
    """X of the seat (the valley of a ``below`` socket) from the socket centre."""
    bp = p.profile()
    v = bp.U - bp.valley_u if pr.PART_FLAGS[below][0] else bp.valley_u
    return v - bp.U/2


def x_band(p: BBParams, below='A'):
    """X offsets of the upper keel from the seat that land on the socket below, a half tooth in from its edges."""
    v = seat_x(p, below)
    return -p.U/2 - v + p.tooth_w/2, p.U/2 - v - p.tooth_w/2


def y_band(p: BBParams):
    """Y offsets from the seat that land on the slopes: the nest face below, the main slope and flange above."""
    return -p.nest_w, p.D/2 + p.flange - p.valley_y


def aim(p: BBParams, below='A'):
    """(x, y) aim offset of the upper brick from its seat (calibrated by capture maps, ``trials.best_aim``)."""
    return p.aim_x, p.aim_y


def _band(p: BBParams, above, below, part=None):
    """(x, z) region from ``below`` under to ``above`` over the socket's top line."""
    bp = p.profile()
    e = bp.end_gap/2
    xs = pr.breakpoints(bp, e, bp.U - e, _flags(p, part)[0]) - bp.U/2
    top = top_line(p, xs, part)
    return Polygon(np.vstack([np.column_stack([xs, top + above]), np.column_stack([xs, top - below])[::-1]]))


def flange_profile(p: BBParams, slots, part=None):
    """The top surface's skin with the slot channels left open (a tooth slides down its channel)."""
    return _band(p, 0., p.flange_t, part).difference(unary_union(slots))


def body_bands(p: BBParams, prof):
    """(profile, y0, y1) of a socket body: toothless under the nest face and ``tooth_setback`` up the main
    slope, the full profile (with the tooth) beyond."""
    start = p.valley_y + p.tooth_setback
    return [(prof['toothless'], -p.D/2, start), (prof['outer'], start, p.D/2)]


def socket_pieces(p: BBParams, dx=0., part=None):
    prof = socket_profile(p, part)
    pieces = [q for poly, y0, y1 in body_bands(p, prof) for q in sweep(p, [poly], y0, y1, dx)]
    if p.flange > 0:
        pieces += sweep(p, [flange_profile(p, prof['slots'], part)], p.D/2, p.D/2 + p.flange, dx)
    return pieces


ROD_INSET = 1.  # the rod reaches this far into each socket's end face, so they fuse


def end_faces(p: BBParams, part=None):
    """(z low, z high) the socket holds along the Y valley over the rod's inset from each end face.

    The ends run through the sawtooth peaks, but the top and bottom fall away from a peak, so the
    window is taken over the whole inset (else the rod pokes out of the top into the next course).
    """
    e = p.end_gap/2
    xs = np.array([-p.U/2 + e, -p.U/2 + e + ROD_INSET, p.U/2 - e - ROD_INSET, p.U/2 - e])
    return float(np.max(bottom_line(p, xs, part))), float(np.min(top_line(p, xs, part)))


def rod_band(p: BBParams, part=None):
    """(z0, z1, floor, ceiling): the rod, centred in the part of the end faces every y of the rod sees."""
    lo, hi = end_faces(p, part)
    y0, y1 = p.rod_y - p.rod_w/2, p.rod_y + p.rod_w/2
    ys = [y0, y1] + ([p.valley_y] if y0 < p.valley_y < y1 else [])
    floor, ceil = lo + float(np.max(rise(p, ys))), hi + float(np.min(rise(p, ys)))
    mid = (floor + ceil)/2
    return mid - p.rod_h/2, mid + p.rod_h/2, floor, ceil


def cavity_half(p: BBParams):
    return p.fork_d/2 + p.cav_play


def sleeve_band(p: BBParams, part=None):
    """(z0, z1) of the sleeves: H0 - sleeve_gap tall, centred on the rod."""
    z0, z1, _, _ = rod_band(p, part)
    h = (p.H0 - p.sleeve_gap)/2
    return (z0 + z1)/2 - h, (z0 + z1)/2 + h


def sleeve_x(p: BBParams):
    return [-p.tine_spacing/2, p.tine_spacing/2]


def sleeve_y(p: BBParams):
    """Y span of the sleeves; the fork enters at the -y end."""
    return p.cav_y - p.cav_depth/2, p.cav_y + p.cav_depth/2


def sleeve_height(p: BBParams):
    """Height a sleeve needs: floor, the square inside, the V roof and a roof wall."""
    h = cavity_half(p)
    return 2*p.cav_wall + 2*h + h*math.tan(math.radians(p.roof_angle))


def _sleeve(p: BBParams, part=None):
    """Heights of a sleeve's floor top, side tops (where the V roof starts) and apex."""
    z0, z1 = sleeve_band(p, part)
    h = cavity_half(p)
    floor = z0 + p.cav_wall
    side = floor + 2*h
    return {'h': h, 'z0': z0, 'z1': z1, 'floor': floor, 'side': side,
            'apex': side + h*math.tan(math.radians(p.roof_angle))}


def channels(p: BBParams, part=None):
    """(x, z) of each tine's centre when the brick hangs on it: wedged into the V roof's apex."""
    apex = _sleeve(p, part)['apex']
    return [(float(xs), float(apex - p.fork_d/2/math.cos(math.radians(p.roof_angle)))) for xs in sleeve_x(p)]


def rod_and_sleeve_pieces(p: BBParams, part=None):
    z0, z1, _, _ = rod_band(p, part)
    ya, yb = p.rod_y - p.rod_w/2, p.rod_y + p.rod_w/2
    y0, y1 = sleeve_y(p)
    end = p.pitch/2 - p.U/2 + p.end_gap/2 + ROD_INSET
    s, cw = _sleeve(p, part), p.cav_wall
    h, top = s['h'], s['z1']
    pieces, edges = [], [-end]
    for xs in sleeve_x(p):
        pieces.append(_box(xs - h - cw, xs + h + cw, y0, y1, s['z0'], s['floor']))
        for xa, xb in ((xs - h - cw, xs - h), (xs + h, xs + h + cw)):
            pieces.append(_box(xa, xb, y0, y1, s['z0'], top))
        for sign in (-1, 1):
            sec = [(xs + sign*h, s['side']), (xs, s['apex']), (xs, top), (xs + sign*h, top)]
            pieces.append(np.array([[x, y, z] for x, z in sec for y in (y0, y1)], float))
        edges += [xs - h, xs + h]
    edges.append(end)
    # The rod runs between the sleeves' insides, so the tines pass through.
    for xa, xb in zip(edges[::2], edges[1::2]):
        pieces.append(_box(xa, xb, ya, yb, z0, z1))
    return pieces


def brick_pieces(p: BBParams, part=None):
    pieces = []
    for i in (0, 1):
        pieces += socket_pieces(p, p.socket_x(i), part)
    return pieces + rod_and_sleeve_pieces(p, part)


def base_pieces(p: BBParams, sites, part='A', thickness=None):
    """Anchored base receiving a course-0 ``part`` at each site x (world mm): base top and flange."""
    bp = p.profile()
    thickness = thickness or bp.tooth_L + bp.slot_extra + 4
    _, m, _, rev = pr.PART_FLAGS[part]
    outer, valleys = pr.base_outer(bp, 1, thickness, 1, part)
    slots = [_centre(pr.slot(bp, v, mirrored=m, reversed=rev), bp.U) for v in valleys]
    outer = _centre(outer, bp.U)
    xs = pr.breakpoints(bp, 0., bp.U, m)
    top = pr.sawtooth(bp, xs, m)
    xs = xs - bp.U/2
    flange = Polygon(np.vstack([np.column_stack([xs, top]), np.column_stack([xs, top - p.flange_t])[::-1]]))
    pieces = []
    for x in sites:
        pieces += sweep(p, [outer], -p.D/2, p.D/2, x)
        if p.flange > 0:
            pieces += sweep(p, [flange.difference(unary_union(slots))], p.D/2, p.D/2 + p.flange, x)
    return pieces
