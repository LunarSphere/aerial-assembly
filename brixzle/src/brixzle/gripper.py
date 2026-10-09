"""Passive grippers (no actuators): parameters, CadQuery solids, mass properties, rules.

Every gripper is built in the drone body frame (mm, g; origin at the motor-plane centre,
+z up, +y the engagement direction) around the brick it carries: ``build_gripper`` returns
the printable solid, its mesh and mass properties, convex collision pieces (same scheme as
``cad``), and ``carry``, the carried brick's pose in the drone frame. The carried brick's
centre of mass sits on the drone's z axis by construction.

Families: ``fork`` (two straight Y tines through the brick's tine bores; pick by +y
translation into a fixtured brick, release by seating and withdrawing along -y), and the
others registered in ``FAMILIES``.
"""
from dataclasses import asdict, dataclass, fields
import math

import cadquery as cq
import numpy as np
import trimesh

from .drone_params import DroneConfig, G
from .params import PLA_DENSITY, BrickParams

# g/mm^3. Pultruded carbon rod and steel rod are bought, not printed.
DENSITY = {'pla': PLA_DENSITY, 'cf': 1.55e-3, 'steel': 7.85e-3}
# Flexural yield / strength used for the tine bending rule (MPa), and Young's modulus (GPa).
STRENGTH = {'pla': 50., 'cf': 1200., 'steel': 350.}
MODULUS = {'pla': 3.0, 'cf': 120., 'steel': 200.}
# Underside of the PCB and bottom header pins (cf21B meshes): the mount clips on here.
PCB_BOTTOM = -1.2
HEADER_BOTTOM = -2.4
HEADER_HALF = (9.4, 11.4)
HEADER_PIN_Y = 11.0      # mm, the two 1x10 bottom header rows (2.0 mm pitch, x = -9 ... 9)
PIN_HOLE_D = 1.0         # mm, clearance for the header pins in the printed bracket
GUARD_BOTTOM = -15.3
GUARD_RADIUS = 34.0
MOTOR_XY = 35.355


@dataclass(frozen=True)
class ForkParams:
    family: str = 'fork'
    tine_d: float = 2.0          # rod diameter (brick bore is fork_d + 2 fork_clearance)
    tine_material: str = 'pla'   # printed in one piece with the plate; 'cf'/'steel' are bought rods
                                 # (see pin_length below)
    protrude: float = 4.0        # tine length beyond the brick's far face
    lead_in: float = 2.0         # conical tip length
    gap: float = 8.0             # plate to the brick's near face (room for the fixture's tine guide)
    clear_guard: float = 8.0     # brick top below the prop-guard bottom (hover error + aim-low + margin)
    plate_t: float = 1.6         # printed plate thickness (y)
    plate_margin: float = 3.0    # plate beyond the outer tines (x)
    bracket_t: float = 2.0       # mount bracket thickness (z)
    detent_h: float = 0.0        # optional bump near the tip (soft contact), mm
    # Bought dowel pins pressed into the plate (pin_length > 0, e.g. MISUMI MSH2-30): each pin
    # sits flush with the back of a boss behind the plate, so whatever of the pin is not exposed
    # (gap + D + protrude) is the press depth. protrude < 0 means the tips stop inside the bores.
    pin_length: float = 0.       # mm, 0 = tines printed with the plate
    press_d: float = 1.9         # printed hole for the pin (interference fit; tune with the coupon)
    boss_d: float = 5.0          # boss around each pressed pin

    def to_dict(self):
        return asdict(self)

    def exposed(self, p: BrickParams):
        """Tine length in front of the plate (mm)."""
        return self.gap + p.D + self.protrude

    def press_depth(self, p: BrickParams):
        """Pin length held by the plate and its boss (mm); 0 for printed tines."""
        return self.pin_length - self.exposed(p) if self.pin_length else 0.


FAMILIES = {'fork': ForkParams}
BOUNDS = {'fork': {'tine_d': (1.2, 2.4), 'protrude': (1.0, 10.0), 'lead_in': (0.5, 4.0), 'gap': (2.0, 14.0),
                   'clear_guard': (1.0, 12.0), 'plate_t': (1.2, 3.0), 'detent_h': (0.0, 0.3)}}


def params_from_dict(d):
    cls = FAMILIES[d.get('family', 'fork')]
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names - {'_comment'}
    if unknown:
        raise ValueError(f'Unknown gripper parameters: {sorted(unknown)}')
    return cls(**{k: v for k, v in d.items() if k in names})


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _rod_piece(x, z, y0, y1, r, lead, n=12):
    """Convex vertex set of a Y rod from y0 to y1 with a conical tip of length ``lead``."""
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring = np.column_stack([x + r*np.cos(a), np.zeros(n), z + r*np.sin(a)])
    tip = np.column_stack([x + .25*r*np.cos(a), np.zeros(n), z + .25*r*np.sin(a)])
    out = []
    for y, pts in ((y0, ring), (y1 - lead, ring), (y1, tip)):
        q = pts.copy()
        q[:, 1] = y
        out.append(q)
    return np.vstack(out)


def _part_mass(mesh, density):
    m = mesh.copy()
    m.density = density
    return m.mass, m.center_mass, m.moment_inertia


def _combine(parts):
    """Mass, COM and inertia about the COM of [(mass, com, inertia_about_own_com)]."""
    M = sum(m for m, _, _ in parts)
    com = sum(m*np.asarray(c) for m, c, _ in parts)/M
    I = np.zeros((3, 3))
    for m, c, Ic in parts:
        d = np.asarray(c) - com
        I += Ic + m*(np.dot(d, d)*np.eye(3) - np.outer(d, d))
    return M, com, I


def carried_pose(gp, brick):
    """Brick origin in the drone frame (mm) when hanging on the tines: COM on the z axis."""
    (x1, z1), (x2, z2) = brick['channels']
    cx = brick['com'][0]
    top = brick['mesh'].bounds[1][2] if 'mesh' in brick else brick['outer'].bounds[3]
    zc = max(z1, z2)
    # Highest tine z such that the brick top clears the guard bottom by clear_guard.
    tine_z = GUARD_BOTTOM - gp.clear_guard - (top - zc)
    return np.array([-cx, 0., tine_z - zc])


def build_gripper(gp, brick, p: BrickParams, solid=True):
    """Fork: a mount bracket fixed under the bottom header pins, a vertical plate, two tines.

    The plate, tines and carried brick are slid together in x and y until the combined COM
    (bracket included) is on the drone axis; the bracket stays on the pins and is stretched
    to reach the plate.
    """
    if gp.family != 'fork':
        raise NotImplementedError(gp.family)
    if len(brick['channels']) != 2:
        raise ValueError('Fork gripper needs a brick with two tine bores')
    origin0 = carried_pose(gp, brick)
    tines0 = [(x + origin0[0], z + origin0[2]) for x, z in brick['channels']]
    r = gp.tine_d/2
    y_face0 = -(p.D/2 + gp.gap)
    y_tip0 = p.D/2 + gp.protrude
    # Pressed pins run back through the plate into a boss and end flush with its back face.
    seat = gp.press_depth(p)
    y_back0 = y_face0 - max(seat, gp.plate_t)
    xs = [t[0] for t in tines0]
    x_lo0, x_hi0 = min(xs) - r - gp.plate_margin, max(xs) + r + gp.plate_margin
    z_bracket = HEADER_BOTTOM - gp.bracket_t
    z_plate_lo = min(t[1] for t in tines0) - max(r + p.wall, gp.boss_d/2 if seat else 0.)
    m_b = brick['mass_g']

    def pieces(move):
        plate = _box((x_lo0, y_face0 - gp.plate_t, z_plate_lo), (x_hi0, y_face0, HEADER_BOTTOM)) + move
        rods = [_rod_piece(x + move[0], z, (y_back0 if seat else y_face0) + move[1], y_tip0 + move[1], r, gp.lead_in)
                for x, z in tines0]
        bosses = [_rod_piece(x + move[0], z, y_back0 + move[1], y_face0 - gp.plate_t + move[1], gp.boss_d/2, 0.)
                  for x, z in tines0] if seat > gp.plate_t else []
        b_lo = (min(-HEADER_HALF[0] - 2, x_lo0 + move[0]), min(-HEADER_PIN_Y - 2, y_face0 + move[1] - gp.plate_t),
                z_bracket)
        b_hi = (max(HEADER_HALF[0] + 2, x_hi0 + move[0]), HEADER_PIN_Y + 2, HEADER_BOTTOM)
        printed = [plate, *bosses]
        moving = [_part_mass(trimesh.convex.convex_hull(c), PLA_DENSITY) for c in printed]
        moving += [_part_mass(trimesh.convex.convex_hull(c), DENSITY[gp.tine_material]) for c in rods]
        return printed, rods, _box(b_lo, b_hi), b_lo, b_hi, moving

    move = np.zeros(3)
    for _ in range(50):   # bracket extent depends on the shift, the shift on the bracket's mass
        _, _, bracket, _, _, moving = pieces(move)
        mass, com, _ = _combine(moving + [_part_mass(trimesh.convex.convex_hull(bracket), PLA_DENSITY)])
        # Zero the combined (gripper + brick) COM in x, y by moving everything but the bracket.
        resid = mass*com[:2] + m_b*(origin0[:2] + move[:2] + np.asarray(brick['com'][:2]))
        m_moving = sum(m for m, _, _ in moving) + m_b
        move[:2] -= resid/m_moving
        if np.max(np.abs(resid/m_moving)) < 1e-12:
            break
    printed, rods, bracket, b_lo, b_hi, moving = pieces(move)
    collision = [*printed, bracket]
    mass, com, inertia = _combine(moving + [_part_mass(trimesh.convex.convex_hull(bracket), PLA_DENSITY)])
    origin = origin0 + move
    tines = [(x + move[0], z) for x, z in tines0]
    y_face, y_tip, y_back = y_face0 + move[1], y_tip0 + move[1], y_back0 + move[1]
    # Tines collide as capsules: smooth, cheap, and the hemispherical tip is a natural lead-in.
    capsules = [{'type': 'capsule', 'size': r, 'fromto': [x, y_face, z, x, y_tip - r, z]} for x, z in tines]
    mesh = trimesh.util.concatenate([trimesh.convex.convex_hull(c) for c in [*printed, bracket, *rods]])
    bundle = {'params': gp.to_dict(), 'family': gp.family, 'collision': collision, 'primitives': capsules,
              'mass_g': mass, 'com': com.tolist(), 'inertia': inertia.tolist(), 'tines': tines,
              'carry': origin.tolist(), 'axis': [0., 1., 0.], 'y_face': y_face, 'y_tip': y_tip,
              'engage_depth': y_tip - y_face, 'press_depth': seat, 'shift': move[:2].tolist(), 'mesh': mesh}
    if solid:
        shape = cq.Solid.makeBox(x_hi0 - x_lo0, gp.plate_t, HEADER_BOTTOM - z_plate_lo,
                                 cq.Vector(x_lo0 + move[0], y_face - gp.plate_t, z_plate_lo))
        shape = shape.fuse(cq.Solid.makeBox(b_hi[0] - b_lo[0], b_hi[1] - b_lo[1], gp.bracket_t, cq.Vector(*b_lo)))
        # Two rows of 10 holes at 2.0 mm pitch for the bottom header pins (cf21B connector-pins mesh).
        for row in (-HEADER_PIN_Y, HEADER_PIN_Y):
            for k in range(10):
                x = -9.0 + 2.0*k
                shape = shape.cut(cq.Solid.makeCylinder(PIN_HOLE_D/2, gp.bracket_t + 2,
                                                        cq.Vector(x, row, z_bracket - 1), cq.Vector(0, 0, 1)))
        rods_s = None
        y_root = y_back if seat else y_face
        for x, z in tines:
            rod = cq.Solid.makeCylinder(r, y_tip - gp.lead_in - y_root, cq.Vector(x, y_root, z), cq.Vector(0, 1, 0))
            rod = rod.fuse(cq.Solid.makeCone(r, .25*r, gp.lead_in, cq.Vector(x, y_tip - gp.lead_in, z),
                                             cq.Vector(0, 1, 0)))
            rods_s = rod if rods_s is None else rods_s.fuse(rod)
            if seat:
                if seat > gp.plate_t:
                    shape = shape.fuse(cq.Solid.makeCylinder(gp.boss_d/2, seat - gp.plate_t + .01,
                                                             cq.Vector(x, y_back, z), cq.Vector(0, 1, 0)))
                # Through hole: the pin presses in from the front and is set flush with the back.
                shape = shape.cut(cq.Solid.makeCylinder(gp.press_d/2, seat + 2, cq.Vector(x, y_back - 1, z),
                                                        cq.Vector(0, 1, 0)))
        if gp.tine_material == 'pla' and not seat:
            shape = shape.fuse(rods_s)
        bundle.update(shape=shape.clean(), rods=rods_s.clean())
    return bundle


@dataclass(frozen=True)
class GripperRuleSettings:
    wall: float = 1.2              # printable wall (BrickParams.wall)
    com_xy_max: float = 3.0        # combined COM off the drone axis (mm)
    load_factor: float = 3.0       # tine bending: carried weight x this (lift-off jerk, swing)
    deflection_max: float = 0.3    # tine tip deflection under the brick (mm); bore clearance is 0.3
    tilt_max_deg: float = 10.      # release motion may not need more tilt than this
    release_accel: float = 0.5     # m/s^2 used by the release primitive
    mu: float = 0.35               # PLA on rod / PLA on PLA
    residual_load: float = 0.3     # fraction of brick weight still on the tines at withdrawal
    release_sf: float = 2.0        # seat retention must exceed tine drag by this factor
    fit_margin: float = 0.1        # radial clearance kept in the bore (mm)
    past_com: float = 3.0          # tine tips at least this far beyond the brick's mid-depth (mm)
    press_min: float = 6.0         # pressed pin length held by the PLA, about 3 pin diameters (mm)


def gripper_rules(gp, g, brick, p: BrickParams, drone: DroneConfig, s: GripperRuleSettings = GripperRuleSettings()):
    """Signed margins (>= 0 satisfied), in the style of ``rules.rules``."""
    out = {}
    m_brick = brick['mass_g']
    payload = g['mass_g'] + m_brick
    dm = drone.margins(payload)
    out['payload_limit'] = dm['payload_limit_g']
    out['thrust_to_weight'] = dm['tw']
    carry = np.asarray(g['carry'])
    top = brick['mesh'].bounds[1][2] if 'mesh' in brick else brick['outer'].bounds[3]
    out['clears_guards'] = GUARD_BOTTOM - (carry[2] + top) - 1.0
    gr_hi = max(np.max(np.asarray(c)[:, 2]) for c in g['collision'])
    # Everything the gripper adds sits below the PCB; the Lighthouse deck on top sees the sky.
    out['clears_deck'] = PCB_BOTTOM - gr_hi       # nothing above the PCB: the Lighthouse deck sees the sky
    # Gripper + brick footprint must stay inside the circle that clears the props' disks below.
    pts = np.vstack([np.asarray(c)[:, :2] for c in g['collision']])
    reach = np.max(np.abs(pts), axis=0)
    out['inside_frame'] = MOTOR_XY + GUARD_RADIUS - max(reach)
    brick_com = carry + np.asarray(brick['com'])
    com = (g['mass_g']*np.asarray(g['com']) + m_brick*brick_com)/payload
    out['com_below_drone'] = -com[2]
    out['com_on_axis'] = s.com_xy_max - float(np.hypot(*com[:2]))
    out['printable_wall'] = min(gp.plate_t, gp.bracket_t) - s.wall
    bore_r = p.fork_d/2 + p.fork_clearance
    out['tine_fits_bore'] = bore_r - gp.tine_d/2 - s.fit_margin
    # Tines may stop short of the far face (short bought pins), but must reach past mid-depth.
    engaged = min(p.D, p.D + gp.protrude)
    out['tine_past_com'] = engaged - p.D/2 - s.past_com
    if gp.pin_length:
        out['press_depth'] = gp.press_depth(p) - s.press_min
        out['press_interference'] = gp.tine_d - gp.press_d
    # Each tine is a cantilever from the plate carrying half the brick at mid-span of its engagement.
    span = gp.gap + engaged/2
    F = s.load_factor*m_brick*1e-3*G/2
    I = math.pi*gp.tine_d**4/64
    stress = F*span*(gp.tine_d/2)/I
    out['tine_strength'] = STRENGTH[gp.tine_material]/stress - 1
    defl = (m_brick*1e-3*G/2)*span**3/(3*MODULUS[gp.tine_material]*1e3*I)
    out['tine_stiffness'] = s.deflection_max - defl
    out['release_tilt'] = s.tilt_max_deg - math.degrees(math.atan(s.release_accel/G))
    # Withdrawal: the seated brick's Y gutter must resist the tines' friction drag.
    retain = m_brick*1e-3*G*math.tan(math.radians(p.gutter) + math.atan(s.mu)) if p.gutter > 0 else 0.
    drag = s.mu*s.residual_load*m_brick*1e-3*G
    out['release_force'] = retain/(s.release_sf*drag) - 1
    out['fork_fits'] = 1. if len(brick['channels']) == 2 else -1.
    return out


def feasible(margins):
    return all(v >= 0 for v in margins.values())


COUPON_HOLES = (1.8, 1.85, 1.9, 1.95, 2.0)


def press_coupon(gp, depth, holes=COUPON_HOLES):
    """Test block for the pin press fit: one hole per diameter (smallest marked by a notch).

    Exported standing on z = 0 with the holes horizontal (along y), as they are in the
    gripper printed bracket-down; ``depth`` is the gripper's press depth.
    """
    pitch, h = gp.boss_d + 2., gp.boss_d + 2.
    w = pitch*len(holes)
    shape = cq.Solid.makeBox(w, depth, h)
    for k, d in enumerate(holes):
        shape = shape.cut(cq.Solid.makeCylinder(d/2, depth + 2, cq.Vector(pitch*(k + .5), -1, h/2), cq.Vector(0, 1, 0)))
    shape = shape.cut(cq.Solid.makeBox(1., depth + 2, 1., cq.Vector(pitch/2 - .5, -1, h - 1.)))
    return shape.clean()


def export(bundle, out):
    from pathlib import Path
    from .cad import _mesh
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if 'shape' in bundle:
        # Print the fused solid (pin holes included); the collision hull mesh overlaps itself.
        _mesh(bundle['shape']).export(out/'gripper.stl')
        cq.exporters.export(bundle['shape'], str(out/'gripper_printed.step'))
    else:
        bundle['mesh'].export(out/'gripper_collision.stl')
    if 'shape' in bundle:
        if bundle['params']['tine_material'] != 'pla' or bundle['press_depth']:
            cq.exporters.export(bundle['rods'], str(out/'gripper_rods.step'))
        if bundle['press_depth']:
            gp = params_from_dict(bundle['params'])
            _mesh(press_coupon(gp, bundle['press_depth'])).export(out/'press_coupon.stl')
    return out
