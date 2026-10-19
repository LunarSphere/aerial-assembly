"""2D XZ profile of the brick and the anchored base, built with shapely.

The A-orientation brick spans local x in [-U, U] (two voxels, a voxel
boundary at x=0). Its top surface is the sawtooth ``T`` raised by ``H0`` and
its bottom surface is the mirrored sawtooth ``T'``. Rotating A by 180 degrees
about Z gives B, whose bottom is ``T`` and whose top is ``T'``; courses
alternate A/B, so every interface nests. Each top valley carries a slanted
slot whose far wall continues the steep face, and each bottom keel point
carries a matching tooth. Teeth and slots lean in opposite directions in one
brick, which forces the alternating-angle (ABA) layering.

Profile z is the world z at y = 0; the Y gutter later shears each Y half by
``tan(gutter) * |y|``.
"""
import math

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from .params import BrickParams


def _unit(deg):
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


def sawtooth(p: BrickParams, x, mirrored=False):
    """Periodic hopper surface (peaks at voxel boundaries, height p.amplitude)."""
    h, U, uv = p.amplitude, p.U, p.valley_u
    u = np.mod(np.asarray(x, dtype=float), U)
    if mirrored:
        u = U - u
    tan_t, tan_f = math.tan(math.radians(p.theta)), math.tan(math.radians(p.phi))
    return np.where(u <= uv, h - u*tan_t, (u - uv)*tan_f)


def breakpoints(p: BrickParams, x0, x1, mirrored=False):
    uv = p.U - p.valley_u if mirrored else p.valley_u
    xs = {x0, x1}
    k0, k1 = math.floor(x0/p.U) - 1, math.ceil(x1/p.U) + 1
    for k in range(k0, k1 + 1):
        for x in (k*p.U, k*p.U + uv):
            if x0 < x < x1:
                xs.add(x)
    return np.array(sorted(xs))


def interface(p: BrickParams, mirrored, reversed=False):
    """Slot/tooth geometry for a sawtooth surface.

    Returns (valley u within voxel, axis d pointing into the lower body, normal n
    toward the slot's open side, mouth edge direction r pointing up from the
    valley along the face the mouth chamfer cuts).

    Normally the slot follows the steep face (its far wall continues it) and
    the mouth chamfer cuts the long ramp. ``reversed`` mirrors the axis about
    vertical: the slot then leans under the steep face with its back wall just
    above the long ramp (inside the nesting cone while alpha <= 90 - theta),
    and the chamfer cuts the steep face instead. Used by the 'ab' brick pair so
    every tooth leans against the build direction.
    """
    if not mirrored:   # T: long ramp left of valley, steep face to the right
        u, d, n, r, steep = p.valley_u, -_unit(p.phi), _unit(90 + p.phi), _unit(180 - p.theta), _unit(p.phi)
    else:
        u, d, n, r, steep = p.U - p.valley_u, _unit(-p.phi), _unit(90 - p.phi), _unit(p.theta), _unit(180 - p.phi)
    if reversed:
        flip = np.array([-1., 1.])
        return u, d*flip, n*flip, steep
    return u, d, n, r


def _quad(origin, a, b, ta, tb):
    o = np.asarray(origin, dtype=float)
    return Polygon([o + ta[0]*a + tb[0]*b, o + ta[1]*a + tb[0]*b,
                    o + ta[1]*a + tb[1]*b, o + ta[0]*a + tb[1]*b])


def tooth_width(p: BrickParams, t):
    """Tooth width at distance t along its axis (root t=0 to tip t=L)."""
    return p.tooth_w - p.tooth_taper*np.clip(t, 0., None)/p.tooth_L


def _tip_chamfer(p: BrickParams):
    return min(p.tip_chamfer, 0.45*tooth_width(p, p.tooth_L), 0.45*p.tooth_L)


def _barb_span(p: BrickParams):
    """(t_catch, length) of the barb along the tooth axis; it ends where the tip chamfer starts."""
    bl = max(0.3, min(p.barb_len, p.tooth_L - _tip_chamfer(p) - 0.5))
    return p.tooth_L - _tip_chamfer(p) - bl, bl


def slot(p: BrickParams, valley, mirrored, reversed=False):
    """Tapered slot below a valley (profile coordinates of the valley point).

    The far wall continues the steep face; the near wall follows the tooth
    taper offset by the seat gap 2*clearance, so a seated tooth is held with
    small play while the entry gap is taper + 2*clearance.
    """
    _, d, n, r = interface(p, mirrored, reversed)
    gap = 2*p.clearance
    depth = p.tooth_L + p.slot_extra
    reach = (p.amplitude + p.H0)/math.sin(math.radians(p.phi)) + 1
    v = np.asarray(valley, dtype=float)
    mouth_w = p.tooth_w + gap
    cut = Polygon([v - reach*d, v + depth*d, v + depth*d + (tooth_width(p, depth) + gap)*n,
                   v + mouth_w*n, v - reach*d + mouth_w*n])
    # Mouth: near wall (s = mouth_w) meets the long ramp at m.
    A = np.column_stack([d, -r])
    t, _ = np.linalg.solve(A, -mouth_w*n)
    m = v + mouth_w*n + t*d
    ch = p.mouth_chamfer
    mouth = Polygon([m, m + ch*r, m + ch*d])
    pieces = [cut, mouth]
    if p.barb_h > 0:
        # Pocket in the near wall where the seated barb rests (clear of contact when seated).
        t0, bl = _barb_span(p)
        c = max(p.clearance, 0.1)
        pieces.append(_quad(v, d, n, (t0 - 0.2 - c, t0 + bl + c), (0., tooth_width(p, t0) + gap + p.barb_h + c)))
    return unary_union(pieces)


def barb(p: BrickParams, keel, mirrored, reversed=False):
    """Snap barb on the tooth's near flank: ramp toward the tip, flat catch face toward the root."""
    _, d, n, _ = interface(p, mirrored, reversed)
    t0, bl = _barb_span(p)
    o = np.asarray(keel, dtype=float)
    at = lambda t, s: o + t*d + s*n
    return Polygon([at(t0, tooth_width(p, t0) - 0.3), at(t0, tooth_width(p, t0) + p.barb_h),
                    at(t0 + bl, tooth_width(p, t0 + bl)), at(t0 + bl, tooth_width(p, t0 + bl) - 0.3)])


def lip(p: BrickParams, apex, mirrored, reversed=False):
    """Capture lip: a centred tapered tooth hanging from the bottom apex at the brick's middle."""
    _, d, n, _ = interface(p, mirrored, reversed)
    o = np.asarray(apex, dtype=float)
    w0, w1 = p.lip_w/2, 0.35*p.lip_w
    root = 0.4*p.H0  # anchors the lip; stays below the top surface for any bounded theta/phi
    return Polygon([o - root*d - w0*n, o - w0*n, o + p.lip_L*d - w1*n, o + p.lip_L*d + w1*n,
                    o + w0*n, o - root*d + w0*n])


def lip_slot(p: BrickParams, apex, mirrored, reversed=False):
    """Receiver for a lip, centred on a top apex; the entry is wider than the seat by the lip taper."""
    _, d, n, _ = interface(p, mirrored, reversed)
    o = np.asarray(apex, dtype=float)
    c = p.clearance
    depth = p.lip_L + p.slot_extra
    reach = (p.amplitude + p.H0)/math.sin(math.radians(p.phi)) + 1
    w0, w1 = p.lip_w/2 + c + 0.5, 0.35*p.lip_w + c
    return Polygon([o - reach*d - w0*n, o - w0*n, o + depth*d - w1*n, o + depth*d + w1*n,
                    o + w0*n, o - reach*d + w0*n])


def end_region(p: BrickParams, d):
    """Region between the brick's two end curves (scarfed and stepped ends tile by translation 2U).

    Each end is a line through the end's mid-height along e (vertical blended toward
    the insertion axis d by end_scarf); the lower half is offset by end_step toward +x.
    The curve is a graph along e, so a brick can still slide along e past its neighbour.
    """
    k = p.end_scarf
    e = (1 - k)*np.array([0., -1.]) + k*np.asarray(d)
    e = e/np.linalg.norm(e)
    ne = np.array([-e[1], e[0]])
    if ne[0] < 0:
        ne = -ne
    S = 3*(p.H0 + p.amplitude) + p.tooth_L
    zc = p.amplitude + p.H0/2

    # Neighbours are separated by end_gap across both the scarf faces and the ledge face.
    off = p.end_gap/2*(ne - np.sign(p.end_step)*e)

    def curve(P):
        return [P - S*e, P, P + p.end_step*ne, P + p.end_step*ne + S*e]
    left = curve(np.array([-p.U, zc]) + off)
    right = curve(np.array([p.U, zc]) - off)
    return Polygon(left[::-1] + right).buffer(0)


def tooth(p: BrickParams, keel, mirrored, reversed=False):
    """Tapered tooth hanging from a keel point; mirrored is the sawtooth it must enter."""
    _, d, n, _ = interface(p, mirrored, reversed)
    w, L = p.tooth_w, p.tooth_L
    w_tip = tooth_width(p, L)
    ch = min(p.tip_chamfer, 0.45*w_tip, 0.45*L)
    o = np.asarray(keel, dtype=float)
    root = p.tooth_w
    near = lambda t: o + t*d + tooth_width(p, t)*n
    pts = [o - root*d, o + L*d, o + L*d + (w_tip - ch)*n, near(L - ch), near(0.), o - root*d + w*n]
    return Polygon(pts)


def _strip(p: BrickParams, x0, x1, top_mirrored, bottom_mirrored, bottom_offset):
    xt = breakpoints(p, x0, x1, top_mirrored)
    xb = breakpoints(p, x0, x1, bottom_mirrored)
    top = np.column_stack([xt, sawtooth(p, xt, top_mirrored) + p.H0])
    bottom = np.column_stack([xb, sawtooth(p, xb, bottom_mirrored) + bottom_offset])
    return Polygon(np.vstack([top, bottom[::-1]]))


def brick_parts(p: BrickParams, part='A'):
    """Outer profile (no internal holes) and feature list of a brick part.

    'alternate': the A part; the bottom is the mirrored sawtooth (teeth lean
    opposite to the top slots, forcing ABA courses; B = A rotated 180 deg).
    'uniform': the bottom repeats the top (constant-thickness chevron; every
    tooth leans the same way).
    'ab': two distinct parts on the alternating body, both world-aligned. A has
    bottom T' with reversed teeth and top T with normal slots; B has bottom T
    with normal teeth and top T' with reversed slots. Every tooth tip trails
    toward -x, so every joint hooks against a +x overhang.
    """
    U, H0 = p.U, p.H0
    if p.lean_mode == 'ab' and part == 'B':
        top_m, bottom_m, top_rev, bottom_rev = True, False, True, False
    elif p.lean_mode == 'ab':
        top_m, bottom_m, top_rev, bottom_rev = False, True, False, True
    else:
        top_m, bottom_m, top_rev, bottom_rev = False, not p.uniform, False, False
    # Vertical ends through the coincident top/bottom peaks: the thinnest
    # section (H0). Course neighbours tile by translation; the order planner
    # keeps the slanted final approach clear of already placed neighbours.
    shaped_ends = p.end_scarf > 0 or p.end_step != 0
    if shaped_ends:
        M = p.H0 + 2*p.amplitude + abs(p.end_step) + 5
        body = _strip(p, -U - M, U + M, top_mirrored=top_m, bottom_mirrored=bottom_m, bottom_offset=0.)
        body = body.intersection(end_region(p, interface(p, bottom_m, bottom_rev)[1]))
    else:
        body = _strip(p, -U + p.end_gap/2, U - p.end_gap/2, top_mirrored=top_m,
                      bottom_mirrored=bottom_m, bottom_offset=0.)
    keel_u = U - p.valley_u if bottom_m else p.valley_u
    valley_u = U - p.valley_u if top_m else p.valley_u
    keels = [np.array([xl + keel_u, 0.]) for xl in (-U, 0.)]
    valleys = [np.array([xl + valley_u, H0]) for xl in (-U, 0.)]
    teeth = [tooth(p, k, mirrored=bottom_m, reversed=bottom_rev) for k in keels]
    slots = [slot(p, v, mirrored=top_m, reversed=top_rev) for v in valleys]
    lips, lip_slots = [], []
    if p.lip_L > 0:
        h = p.amplitude
        lips = [lip(p, (0., h), mirrored=bottom_m, reversed=bottom_rev)]
        lip_slots = [lip_slot(p, (x, H0 + h), mirrored=top_m, reversed=top_rev) for x in (-U, 0., U)]
    barbs = [barb(p, k, mirrored=bottom_m, reversed=bottom_rev) for k in keels] if p.barb_h > 0 else []
    cuts = slots + lip_slots
    if shaped_ends:
        # Slanted ends reach into the neighbours' voxels: give way to their teeth and carry their slots.
        shift = lambda gs: [affinity.translate(g, dx, 0.) for g in gs for dx in (-2*U, 2*U)]
        body = body.difference(unary_union(shift(teeth + lips)).buffer(max(p.clearance, p.end_gap/2), join_style=2))
        cuts = cuts + shift(cuts)
    outer = unary_union([body, *teeth, *lips]).difference(unary_union(cuts))
    # Opening removes zero-width spikes left where cuts graze the steep faces.
    outer = outer.buffer(-0.01, join_style=2).buffer(0.01, join_style=2)
    fragments = 1 if outer.geom_type == 'Polygon' else len(outer.geoms)
    outer = _largest(outer)
    # Barbs are soft-contact catches: kept out of the rigid outer, added to the printed solid.
    return {'outer': outer, 'teeth': teeth + lips, 'slots': slots + lip_slots, 'barbs': barbs,
            'keels': keels, 'valleys': valleys, 'fragments': fragments}


def _largest(geom):
    if geom.geom_type == 'Polygon':
        return geom
    return max(geom.geoms, key=lambda g: g.area)


def shear_k(p: BrickParams):
    return math.tan(math.radians(p.gutter))


def shear_drop(p: BrickParams):
    """Rise of the sheared surfaces at the Y faces relative to the centre band."""
    return shear_k(p)*max(0., p.D/2 - p.gutter_flat)


def y_bands(p: BrickParams):
    """Piecewise-affine Y sweep: [(y0, y1, z_shift_at_y0, z_shift_at_y1)].

    The centre band |y| <= gutter_flat is a plain extrusion; the side bands
    rise at the gutter angle toward the faces, forming a flat-bottomed V.
    """
    f = min(p.gutter_flat, p.D/2) if p.gutter > 0 else p.D/2
    drop = shear_drop(p) if p.gutter > 0 else 0.
    bands = [(-f, f, 0., 0.)] if f > 0 else []
    if f < p.D/2:
        bands += [(f, p.D/2, 0., drop), (-f, -p.D/2, 0., drop)]
    return bands


def channel_footprint(p: BrickParams, center):
    """Profile-space region a straight Y tine bore sweeps through the gutter shear."""
    x, z = center
    r = p.fork_d/2 + p.fork_clearance
    return LineString([(x, z - shear_drop(p)), (x, z)]).buffer(r, quad_segs=8)


def fork_channels(p: BrickParams, parts=None, step=0.25, min_spacing=8.0):
    """Two straight through-Y tine bores, as high as possible, >= min_spacing apart.

    Centres are world (x, z) at y = 0. Each bore keeps ``wall`` of material
    around its sheared footprint so a straight rod passes the full depth.
    Returns [] when no pair fits.
    """
    parts = parts or brick_parts(p)
    solid = parts['outer']
    zmin, zmax = solid.bounds[1], solid.bounds[3]
    tops = []
    for x in np.arange(-p.U + step, p.U, step):
        for z in np.arange(zmax, zmin, -step):
            if solid.contains(channel_footprint(p, (x, z)).buffer(p.wall)):
                tops.append((float(x), float(z)))
                break
    # The pair must straddle the centre of mass so the brick hangs level.
    cx = solid.centroid.x
    best, best_z = [], -np.inf
    for i, a in enumerate(tops):
        for b in tops[i+1:]:
            straddles = a[0] + 2 <= cx <= b[0] - 2
            if straddles and b[0] - a[0] >= min_spacing and min(a[1], b[1]) > best_z:
                best, best_z = [a, b], min(a[1], b[1])
    return best


def lightening_holes(p: BrickParams, parts=None, channels=None):
    """Through-Y holes leaving skins, ribs, and material around slots/teeth/channels."""
    parts = parts or brick_parts(p)
    if not p.lighten:
        return []
    outer = parts['outer']
    keep = [s.buffer(p.wall) for s in parts['slots']]
    keep += [t.buffer(p.wall) for t in parts['teeth'] + parts.get('barbs', [])]
    for c in channels or []:
        if c is not None:
            keep.append(channel_footprint(p, c).buffer(p.wall))
    zlo, zhi = outer.bounds[1] - 1, outer.bounds[3] + 1
    for x in (-p.U, 0., p.U, -p.U + p.valley_u, p.valley_u, -p.valley_u, p.U - p.valley_u):
        keep.append(box(x - p.wall/2, zlo, x + p.wall/2, zhi))
    region = outer.buffer(-p.wall, join_style=2).difference(unary_union(keep))
    region = region.buffer(-0.4, join_style=2).buffer(0.4, join_style=2)
    geoms = [region] if region.geom_type == 'Polygon' else list(getattr(region, 'geoms', []))
    holes = []
    for g in geoms:
        if g.area > 6.0 and g.minimum_rotated_rectangle.length > 0:
            g = g.simplify(0.05)
            if g.is_valid and not g.is_empty:
                holes.append(g)
    return holes


def base_outer(p: BrickParams, voxels, thickness=6.0, direction=1):
    """Anchored base: top matches a course-0 brick's bottom, with slots; flat bottom.

    In 'ab' mode the slots lean against the build ``direction`` (reversed for +x,
    normal for -x, which receives a rotated B part).
    """
    m = not p.uniform
    x0, x1 = 0., voxels*p.U
    xs = breakpoints(p, x0, x1, mirrored=m)
    top = np.column_stack([xs, sawtooth(p, xs, mirrored=m)])
    body = Polygon(np.vstack([top, [[x1, -thickness], [x0, -thickness]]]))
    vu = p.U - p.valley_u if m else p.valley_u
    valleys = [np.array([i*p.U + vu, 0.]) for i in range(voxels)]
    rev = p.lean_mode == 'ab' and direction > 0
    cuts = [slot(p, v, mirrored=m, reversed=rev) for v in valleys]
    if p.lip_L > 0:
        cuts += [lip_slot(p, (i*p.U, p.amplitude), mirrored=m, reversed=rev) for i in range(voxels + 1)]
    cuts = unary_union(cuts)
    return _largest(body.difference(cuts)), valleys


def mirror_x(geom):
    return affinity.scale(geom, xfact=-1, yfact=1, origin=(0, 0))
