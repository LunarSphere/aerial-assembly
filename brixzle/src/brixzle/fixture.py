"""Pick fixture: a row of anchored base strips, each holding one brick in a known pose.

Each slot is the anchored base of a course-0 placement (``cad.build_base`` over voxels
[-1, 3), the tower's base), so the waiting brick is seated and hooked exactly as on a real
first course. A slot strip is only D deep in Y, so both Y faces are open for the fork's
tines. Slots sit ``pitch`` apart along X (at least the prop-guard span). A slot may be turned
180 degrees about Z (``orientation='B'``), pre-orienting bricks for odd courses in the
'alternate' lean mode; otherwise the drone yaws 180 degrees in transit. Also builds the
launch stand: two rails under the prop-guard legs so the hanging gripper clears the floor.
Units: mm and degrees; poses are (pos_mm, quat_wxyz).
"""
from dataclasses import asdict, dataclass, field
import math

import numpy as np

from . import cad, lattice as L
from .params import BrickParams

SLOT_BASE = (-1, 3)   # voxel range under each slot's course-0 placement Placement(0, 0)


@dataclass(frozen=True)
class FixtureParams:
    origin: tuple = (0., -300., 0.)      # mm, slot 0 seat frame origin (base top at z = origin z)
    pitch: float = 160.                  # mm between slot centres along X
    prerotate_B: bool = True             # B slots turned 180 degrees vs. yawing in transit
    guard_span: float = 139.6            # mm, cf21B prop-guard outer width
    margin: float = 10.                  # mm extra between neighbouring slot and drone
    stand_xy: tuple = (-250., -300.)     # mm, launch stand centre
    stand_z: float = 60.                 # mm, rail top
    floor_z: float = -40.                # mm, below everything
    funnel: bool = True                  # tapered tine guides in front of each slot's bores
    funnel_mouth: float = 5.0            # mm, half-width of the guide's entry
    funnel_len: float = 6.0              # mm along y
    funnel_gap: float = 1.0              # mm between the guide's exit and the brick's face
    funnel_t: float = 1.6                # mm, guide wall thickness (printable)
    hooked: bool = False                 # True: real course-0 slots (pick must back out along the tooth
                                         # axis); False: vertical pockets, the brick lifts straight out
    pocket_clearance: float = 0.4        # mm around the tooth's vertical sweep

    def to_dict(self):
        return asdict(self)


def _rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def slot_pose(p: BrickParams, fp: FixtureParams, index, orientation):
    """Seated brick pose (mm, wxyz) in slot ``index``; the slot frame turns with the orientation."""
    centre = np.asarray(fp.origin, float) + np.array([index*fp.pitch, 0., 0.])
    yaw = 0. if orientation == 'A' else 180.
    quat = np.array([math.cos(math.radians(yaw)/2), 0, 0, math.sin(math.radians(yaw)/2)])
    return centre, quat, yaw


def funnel_pieces(p: BrickParams, fp: FixtureParams, channels):
    """Convex walls of one pyramidal tine guide per bore, in the slot (brick) frame, mm.

    The guide sits on the approach side (-y) of the brick: its mouth (half-width
    ``funnel_mouth``) faces the drone, its exit matches the bore. A tine tip anywhere in the
    mouth slides along a wall and the soft-hovering drone is pushed onto the bore axis. The
    top is open (the planner aims the tines slightly low), so lifting is never obstructed.
    """
    r_exit = p.fork_d/2 + p.fork_clearance
    y_exit = -p.D/2 - fp.funnel_gap
    y_mouth = y_exit - fp.funnel_len
    W, t = fp.funnel_mouth, fp.funnel_t
    pieces = []
    for x, z in channels:
        # Side walls (normal to x) and a floor ramp (normal to z); open on top so the tines,
        # and the brick on them, can lift straight out after engaging.
        for axis, sign in ((0, -1), (0, 1), (2, -1)):
            inner = []
            for y, h in ((y_mouth, W), (y_exit, r_exit)):
                for s2 in (-1, 1):
                    pt = np.array([x, y, z], float)
                    pt[axis] += sign*h
                    pt[2 - axis] += s2*(h + t)
                    inner.append(pt)
            inner = np.array(inner)
            outer = inner.copy()
            outer[:, axis] += sign*t
            pieces.append(np.vstack([inner, outer]))
    return pieces


def build_fixture(p: BrickParams, fp: FixtureParams, orientations, channels=None):
    """Bundle: slots [{index, orientation, pos, quat}], static collision pieces (world mm), stand.

    ``orientations`` lists 'A'/'B' per slot. The brick origin of slot i is at its pos;
    Placement(0, 0) puts the brick centre at x = U over the base, so the base is shifted -U.
    """
    if p.lean_mode == 'ab':
        raise NotImplementedError("Fixture slots for 'ab' lean mode parts are not built yet")
    base = cad.build_base(p, SLOT_BASE[1] - SLOT_BASE[0])
    seat, _ = L.pose(p, L.Placement(0, 0))
    outer = slot_outer(p, fp)
    local = cad._pieces(outer, p)
    guides = funnel_pieces(p, fp, channels) if fp.funnel and channels else []
    slots, pieces = [], []
    for i, o in enumerate(orientations):
        pos, quat, yaw = slot_pose(p, fp, i, o)
        R = _rotz(yaw)
        slots.append({'index': i, 'orientation': o, 'pos': pos.tolist(), 'quat': quat.tolist()})
        pieces += [piece @ R.T + pos for piece in local + guides]
    stand = []
    sx, sy = fp.stand_xy
    for side in (-1, 1):
        x = sx + side*35.36
        lo = np.array([x - 6, sy - 60, fp.floor_z])
        hi = np.array([x + 6, sy + 60, fp.stand_z])
        stand.append(cad_box(lo, hi))
    return {'params': fp.to_dict(), 'slots': slots, 'collision': pieces, 'stand': stand,
            'base_outer': outer, 'local': local, 'guides': guides}


def slot_outer(p: BrickParams, fp: FixtureParams):
    """Slot base profile in the slot frame (seated brick origin at 0), mm.

    Hooked: the anchored course-0 base. Unhooked: the same sawtooth, with each tooth's
    vertical sweep pocketed out so the brick is located (sawtooth in x, gutter in y) but lifts
    straight up.
    """
    from shapely import affinity
    from shapely.ops import unary_union
    from . import profile as pr
    thickness = p.tooth_L + p.slot_extra + 4
    outer, _ = pr.base_outer(p, SLOT_BASE[1] - SLOT_BASE[0], thickness)
    seat, _ = L.pose(p, L.Placement(0, 0))
    outer = affinity.translate(outer, SLOT_BASE[0]*p.U - seat[0], 0)
    if fp.hooked:
        return outer
    teeth = pr.brick_parts(p, 'A')['teeth']
    rise = p.tooth_L + p.slot_extra + 3
    sweep = unary_union([affinity.translate(t, 0, dz) for t in teeth for dz in np.linspace(0, rise, 16)])
    pocket = sweep.convex_hull if len(teeth) == 1 else unary_union(
        [unary_union([affinity.translate(t, 0, dz) for dz in np.linspace(0, rise, 16)]).convex_hull for t in teeth])
    return pr._largest(outer.difference(pocket.buffer(fp.pocket_clearance, join_style=2)))


def cad_box(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def stand_home(fp: FixtureParams, guard_bottom=-15.3):
    """Drone origin (mm) resting on the stand's rails."""
    return np.array([fp.stand_xy[0], fp.stand_xy[1], fp.stand_z - guard_bottom])


def fixture_rules(p: BrickParams, fp: FixtureParams, fixture, gripper):
    """Signed margins (>= 0 satisfied): slot spacing, open Y faces, approach corridor, stand."""
    out = {}
    out['slot_pitch'] = fp.pitch - (fp.guard_span + fp.margin)
    # The base strip spans |y| <= D/2; the tines enter from beyond D/2 + protrude.
    ys = np.vstack(fixture['local'])[:, 1]
    out['y_faces_open'] = (p.D/2 + 1e-6) - float(np.max(np.abs(ys)))
    # Approach corridor: while the drone stands off on the -Y side of a slot, nothing of the
    # neighbouring slots may lie within the prop-guard square.
    out['corridor_clear'] = fp.pitch - fp.guard_span/2 - 2*p.U - fp.margin
    if fp.funnel:
        # Engaged, the gripper plate must stay behind the guide's mouth.
        out['plate_behind_funnel'] = gripper['params']['gap'] - (fp.funnel_gap + fp.funnel_len) - 0.5
        out['funnel_printable'] = fp.funnel_t - 1.2
    plate_bottom = min(float(np.min(np.asarray(c)[:, 2])) for c in gripper['collision'])
    out['stand_clears_gripper'] = (fp.stand_z - (-15.3)) + plate_bottom - fp.floor_z - 5.
    return out


def export(p: BrickParams, fixture, out):
    """Printable fixture: the slot strips swept like ``baseplate``, joined by a flat bar."""
    from pathlib import Path
    import cadquery as cq
    from . import profile as pr
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    thickness = p.tooth_L + p.slot_extra + 4
    drop = pr.shear_drop(p) if p.gutter > 0 else 0.
    outer, _ = pr.base_outer(p, SLOT_BASE[1] - SLOT_BASE[0], thickness + drop + 1.)
    strip = cad._swept(outer, [], p)
    seat, _ = L.pose(p, L.Placement(0, 0))
    strip = strip.translate(cq.Vector(SLOT_BASE[0]*p.U - seat[0], 0, 0))
    shape = None
    for s in fixture['slots']:
        piece = strip.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 0 if s['orientation'] == 'A' else 180)
        piece = piece.translate(cq.Vector(*s['pos']))
        shape = piece if shape is None else shape.fuse(piece)
    xs = [s['pos'][0] for s in fixture['slots']]
    y0, z0 = fixture['slots'][0]['pos'][1], fixture['slots'][0]['pos'][2]
    bar = cq.Solid.makeBox(max(xs) - min(xs) + 4*p.U, 8., thickness, cq.Vector(min(xs) - 2*p.U, y0 - 4., z0 - thickness))
    shape = shape.fuse(bar)
    box = cq.Solid.makeBox(max(xs) - min(xs) + 6*p.U, p.D + 20, thickness + p.amplitude + p.tooth_L + 10,
                           cq.Vector(min(xs) - 3*p.U, y0 - p.D/2 - 10, z0 - thickness))
    shape = shape.intersect(box).clean()
    mesh = cad._mesh(shape)
    mesh.export(out/'fixture.stl')
    cq.exporters.export(shape, str(out/'fixture.step'))
    return out/'fixture.stl'
