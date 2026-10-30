"""biasbrick3d geometry as convex pieces (mm), the source of truth for collision and the CAD solid.

Brick frame: x along the brick, centred on its middle voxel; y across; z = 0 is the rail bottom.
The hoppers stand on the rail top (z = bar_h); the cup feet hang below z = 0.

Each piece is built in seat-local coordinates (du, dv) from a seat's centre at its slot floor,
then moved to the seat. The slot plates and the tooth lean: a point h above the slot floor shifts
by h lean_k along +du and +dv (alpha from vertical along the diagonal), so going down, the seat
moves into the -x-y corner. Above the slot nothing leans. A hopper is symmetric about its diagonal
du = dv, so the Y part (the X part mirrored in y, placed at yaw 90) has the same hoppers in the
world frame.
"""
import math

import numpy as np

from .params import BBParams


def _box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def _lean(pts, k, z_ref):
    """Shear points by k per mm above z_ref along +x and +y (seat-local, so about the seat)."""
    pts = np.array(pts, float)
    h = pts[:, 2] - z_ref
    pts[:, 0] += k*h
    pts[:, 1] += k*h
    return pts


def _plate(poly, z_of, below, above):
    """Prism over an (du, dv) polygon between z_of(du, dv) - below and z_of(du, dv) + above (z_of linear)."""
    return np.array([[u, v, z_of(u, v) + dz] for u, v in poly for dz in (-below, above)], float)


def _extrude_v(poly, v0, v1):
    """(du, z) polygon extruded from dv = v0 to v1."""
    return np.array([[u, v, z] for u, z in poly for v in (v0, v1)], float)


def _extrude_u(poly, u0, u1):
    """(dv, z) polygon extruded from du = u0 to u1."""
    return np.array([[u, v, z] for v, z in poly for u in (u0, u1)], float)


def _drop(p: BBParams):
    """Vertical thickness of a plate of normal thickness wall_t on a ramp."""
    return p.wall_t/math.cos(math.radians(p.ramp))


def ramp_z(p: BBParams, d, z0):
    """Height of a hopper's ramp surface at distance d (along +x or +y) from its seat centre."""
    return z0 + p.slot_h + (d - p.throat - p.lean_shift)*math.tan(math.radians(p.ramp))


def ridge_z(p: BBParams, z0):
    return z0 + p.guide_h


def _wall_top(p: BBParams, z0):
    """(d, z) corners of a side wall's top: stop_h above the ramp, capped at the ridge."""
    t, s, zp, top = p.throat, p.lean_shift, z0 + p.slot_h, ridge_z(p, z0)
    T = math.tan(math.radians(p.ramp))
    d_cap = t + s + max(p.ramp_h - p.stop_h, 0.)/T
    return [(t + s, min(zp + p.stop_h, top)), (d_cap, top), (p.reach, top)]


def hopper_pieces(p: BBParams, seat=(0., 0.), z0=None):
    """Convex pieces of one corner hopper whose seat centre is at ``seat`` (x, y) and slot floor at z0."""
    z0 = p.bar_h if z0 is None else z0
    t, w, s, R = p.throat, p.wall_t, p.lean_shift, p.reach
    zp = z0 + p.slot_h
    drop = _drop(p)
    pieces = []
    # Slot: four leaning plates round the seat; the -x and -y ones overhang the seated tooth (the hook).
    for u0, u1, v0, v1 in ((t, t + w, -t - w, t + w), (-t - w, -t, -t - w, t + w),
                           (-t, t, t, t + w), (-t, t, -t - w, -t)):
        pieces.append(_lean(_box(u0, u1, v0, v1, z0, zp), p.lean_k, z0))
    # Ramps: +x below the diagonal crease, +y above it.
    rz = lambda d: ramp_z(p, d, z0)  # noqa: E731
    pieces.append(_plate([(t + s, -t + s), (R, -t + s), (R, R), (t + s, t + s)], lambda u, v: rz(u), drop, 0.))
    pieces.append(_plate([(-t + s, t + s), (t + s, t + s), (R, R), (-t + s, R)], lambda u, v: rz(v), drop, 0.))
    # Side walls on -x and -y: stop_h above the slot mouth, then along each ramp's edge up to the ridge.
    flat = min(zp + p.stop_h, ridge_z(p, z0))
    pieces.append(_box(-t - w + s, t + s, -t - w + s, -t + s, zp - 1., flat))
    pieces.append(_box(-t - w + s, -t + s, -t - w + s, t + s, zp - 1., flat))
    top = _wall_top(p, z0)
    band = [(t + s, zp - drop), (R, rz(R) - drop)] + top[::-1]
    pieces.append(_extrude_v(band, -t - w + s, -t + s))
    pieces.append(_extrude_u(band, -t - w + s, -t + s))
    return [q + np.array([seat[0], seat[1], 0.]) for q in pieces]


def foot_pieces(p: BBParams, seat=(0., 0.)):
    """A nesting cup under the seat: its outer faces copy the hopper, moved toward the seat by the clearance.

    The tooth is the leaning, tapered, chamfered square that enters the slot, coaxial with it and
    ``slot_extra`` short of its floor; the +x and +y plates copy the ramps; the -x and -y plates copy
    the side walls and close the shell up to the rail.
    """
    a, c, ch, w, s, R = p.half_foot, p.clearance, p.tip_chamfer, p.wall_t, p.lean_shift, p.reach
    T = math.tan(math.radians(p.ramp))
    floor = -p.foot_L  # the slot floor of the course below, when seated
    zp = floor + p.slot_h
    tip = floor + p.slot_extra
    drop = _drop(p)
    cup = lambda d: zp + (d - a - s)*T  # noqa: E731  outer face: the ramp moved toward the seat by c
    a_tip = a - p.tooth_taper/2
    tooth = []
    for h, z in ((a, zp + drop), (a_tip, tip + ch), (a_tip - ch, tip)):
        tooth += [[sx*h, sy*h, z] for sx in (-1, 1) for sy in (-1, 1)]
    pieces = [_lean(tooth, p.lean_k, floor)]
    e = R - c
    pieces.append(_plate([(a + s, -a + s), (e, -a + s), (e, e), (a + s, a + s)], lambda u, v: cup(u), 0., drop))
    pieces.append(_plate([(-a + s, a + s), (a + s, a + s), (e, e), (-a + s, e)], lambda u, v: cup(v), 0., drop))
    pieces.append(_box(e - w, e, -a + s, e, -p.top_gap, 0.))
    pieces.append(_box(-a + s, e, e - w, e, -p.top_gap, 0.))
    side = [(-a + s, zp), (a + s, zp), (e, cup(e)), (e, 0.), (-a + s, 0.)]
    pieces.append(_extrude_v(side, -a + s, -a + s + w))
    pieces.append(_extrude_u(side, -a + s, -a + s + w))
    return [q + np.array([seat[0], seat[1], 0.]) for q in pieces]


def feet_seats(p: BBParams):
    return [p.seat_xy(0), p.seat_xy(p.n_vox - 1)]


def rail_width(p: BBParams):
    """The rail carries every throat and the side walls' feet."""
    return 2*(p.throat + p.wall_t)


def rail_piece(p: BBParams):
    x0 = p.voxel_x(0) - p.U/2 + p.end_gap/2
    x1 = p.voxel_x(p.n_vox - 1) + p.U/2 - p.end_gap/2
    y0 = p.seat - p.throat - p.wall_t
    return _box(x0, x1, y0, y0 + rail_width(p), 0., p.bar_h)


def cavity_half(p: BBParams):
    """Half-width of a fork cavity's square inside."""
    return p.fork_d/2 + p.cav_play


def _cavity(p: BBParams, i):
    """Section numbers of the fork cavity under the +x ridge of voxel i (brick frame, mm)."""
    h, cw = cavity_half(p), p.cav_wall
    xs = p.seat_xy(i)[0]
    xr = p.voxel_x(i) + p.U/2 - p.end_gap/2  # the ramp's far edge
    xc = xr - cw - h
    under = lambda x: ramp_z(p, x - xs, p.bar_h) - _drop(p) + 0.5  # noqa: E731  fused 0.5 mm into the ramp
    apex = under(xc - h - cw) - cw
    side = apex - h*math.tan(math.radians(p.roof_angle))
    return {'xc': xc, 'xl': xc - h - cw, 'xr': xr, 'h': h, 'under': under, 'apex': apex, 'side': side,
            'floor': side - 2*h}


def _cavity_sections(p: BBParams, i, grow=0.):
    """(x, z) polygons of one cavity's shell; ``grow`` opens the walls and floor outward (the mouth)."""
    c = _cavity(p, i)
    xc, h, cw, under = c['xc'], c['h'], p.cav_wall, c['under']
    fl = c['floor'] - grow
    xl, xi, xo, xr = xc - h - cw - grow, xc - h - grow, xc + h + grow, xc + h + cw + grow
    left = [(xl, fl - cw), (xi, fl - cw), (xi, under(xi)), (xl, under(xl))]
    right = [(xo, fl - cw), (xr, fl - cw), (xr, under(xr)), (xo, under(xo))]
    floor = [(xi, fl - cw), (xo, fl - cw), (xo, fl), (xi, fl)]
    roof_l = [(xc - h, c['side']), (xc, c['apex']), (xc, under(xc)), (xc - h, under(xc - h))]
    roof_r = [(xc, c['apex']), (xc + h, c['side']), (xc + h, under(xc + h)), (xc, under(xc))]
    return [left, right, floor, roof_l, roof_r]


def cavity_y(p: BBParams):
    """Y span of the fork cavities: from the brick's -y face (where the fork enters) for cav_depth."""
    y0 = p.seat - p.throat - p.wall_t
    return y0, y0 + p.cav_depth


def cavity_voxels(p: BBParams):
    """Voxels whose +x ridge carries a cavity: the two inner ridges nearest the centre."""
    ridges = list(range(p.n_vox - 1))
    if len(ridges) > 2:
        ridges = sorted(ridges, key=lambda i: abs(p.voxel_x(i) + p.U/2))[:2]
    return sorted(ridges)


def channels(p: BBParams):
    """(x, z) of each tine's centre when the brick hangs on it: wedged into the V roof's apex."""
    out = []
    for i in cavity_voxels(p):
        c = _cavity(p, i)
        out.append((float(c['xc']), float(c['apex'] - p.fork_d/2/math.cos(math.radians(p.roof_angle)))))
    return out


def cavity_pieces(p: BBParams):
    """Shelled square tubes with a V roof, hung from the ramp underside and posted to the rail."""
    y0, y1 = cavity_y(p)
    ym = y0 + min(p.cav_chamfer, p.cav_depth)
    pieces = []
    for i in cavity_voxels(p):
        c = _cavity(p, i)
        for open_, sec in zip(_cavity_sections(p, i, p.cav_chamfer), _cavity_sections(p, i)):
            pieces.append(np.array([[x, y1, z] for x, z in sec] + [[x, ym, z] for x, z in sec], float))
            pieces.append(np.array([[x, ym, z] for x, z in sec] + [[x, y0, z] for x, z in open_], float))
        post_top = c['floor'] - p.cav_wall
        if post_top > p.bar_h:
            pieces.append(_box(c['xl'], c['xr'], y0, y0 + rail_width(p), p.bar_h - 0.5, post_top + 0.5))
    return pieces


def _mirror(pieces):
    return [q*np.array([1., -1., 1.]) for q in pieces]


def brick_pieces(p: BBParams):
    pieces = [rail_piece(p)]
    for i in range(p.n_vox):
        pieces += hopper_pieces(p, p.seat_xy(i))
    for seat in feet_seats(p):
        pieces += foot_pieces(p, seat)
    pieces += cavity_pieces(p)
    return _mirror(pieces) if p.part == 'Y' else pieces


def base_pieces(p: BBParams, voxels, thickness=4.):
    """Anchored base: a hopper over each world voxel centre (x, y), throat bottoms at z = 0, on a slab."""
    pieces = []
    for x, y in voxels:
        pieces.append(_box(x - p.U/2, x + p.U/2, y - p.U/2, y + p.U/2, -thickness, 0.))
        pieces += hopper_pieces(p, (x + p.seat, y + p.seat), 0.)
    return pieces
