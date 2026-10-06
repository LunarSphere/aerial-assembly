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
GUARD_BOTTOM = -15.3
GUARD_RADIUS = 34.0
MOTOR_XY = 35.355


@dataclass(frozen=True)
class ForkParams:
    family: str = 'fork'
    tine_d: float = 2.0          # rod diameter (brick bore is fork_d + 2 fork_clearance)
    tine_material: str = 'pla'   # printed in one piece with the plate; 'cf'/'steel' are bought rods
    protrude: float = 4.0        # tine length beyond the brick's far face
    lead_in: float = 2.0         # conical tip length
    gap: float = 8.0             # plate to the brick's near face (room for the fixture's tine guide)
    clear_guard: float = 8.0     # brick top below the prop-guard bottom (hover error + aim-low + margin)
    plate_t: float = 1.6         # printed plate thickness (y)
    plate_margin: float = 3.0    # plate beyond the outer tines (x)
    bracket_t: float = 2.0       # mount bracket thickness (z)
    detent_h: float = 0.0        # optional bump near the tip (soft contact), mm

    def to_dict(self):
        return asdict(self)


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
    if gp.family != 'fork':
        raise NotImplementedError(gp.family)
    if len(brick['channels']) != 2:
        raise ValueError('Fork gripper needs a brick with two tine bores')
    origin = carried_pose(gp, brick)
    tines = [(x + origin[0], z + origin[2]) for x, z in brick['channels']]
    r = gp.tine_d/2
    y_face = -(p.D/2 + gp.gap)
    y_tip = p.D/2 + gp.protrude
    xs = [t[0] for t in tines]
    x_lo, x_hi = min(xs) - r - gp.plate_margin, max(xs) + r + gp.plate_margin
    z_bracket = HEADER_BOTTOM - gp.bracket_t
    z_plate_lo = min(t[1] for t in tines) - r - p.wall
    plate_lo = (x_lo, y_face - gp.plate_t, z_plate_lo)
    plate_hi = (x_hi, y_face, HEADER_BOTTOM)
    bracket_lo = (-HEADER_HALF[0] - 2, y_face - gp.plate_t, z_bracket)
    bracket_hi = (HEADER_HALF[0] + 2, HEADER_HALF[1], HEADER_BOTTOM)
    collision = [_box(plate_lo, plate_hi), _box(bracket_lo, bracket_hi)]
    rods = [_rod_piece(x, z, y_face, y_tip, r, gp.lead_in) for x, z in tines]
    meshes = {'printed': trimesh.util.concatenate([trimesh.convex.convex_hull(collision[0]),
                                                   trimesh.convex.convex_hull(collision[1])]),
              'tines': [trimesh.convex.convex_hull(c) for c in rods]}
    parts = [_part_mass(trimesh.convex.convex_hull(c), PLA_DENSITY) for c in collision[:2]]
    parts += [_part_mass(m, DENSITY[gp.tine_material]) for m in meshes['tines']]
    mass, com, inertia = _combine(parts)
    # Slide gripper and brick together in x and y so the combined COM sits on the drone axis.
    m_b = brick['mass_g']
    sx, shift = -(mass*com[:2] + m_b*(origin[:2] + np.asarray(brick['com'][:2])))/(mass + m_b)
    move = np.array([sx, shift, 0.])
    collision = [c + move for c in collision]
    moved = [(x + sx, z) for x, z in tines]
    # Tines collide as capsules: smooth, cheap, and the hemispherical tip is a natural lead-in.
    capsules = [{'type': 'capsule', 'size': r, 'fromto': [x, y_face + shift, z, x, y_tip + shift - r, z]}
                for x, z in moved]
    origin = origin + move
    com = com + move
    for m in [meshes['printed'], *meshes['tines']]:
        m.apply_translation(move)
    bundle = {'params': gp.to_dict(), 'family': gp.family, 'collision': collision, 'primitives': capsules,
              'mass_g': mass,
              'com': com.tolist(), 'inertia': inertia.tolist(), 'tines': moved, 'carry': origin.tolist(),
              'axis': [0., 1., 0.], 'y_face': y_face + shift, 'y_tip': y_tip + shift, 'engage_depth': y_tip - y_face,
              'y_shift': shift,
              'mesh': trimesh.util.concatenate([meshes['printed'], *meshes['tines']])}
    if solid:
        shape = cq.Solid.makeBox(x_hi - x_lo, gp.plate_t, HEADER_BOTTOM - z_plate_lo,
                                 cq.Vector(x_lo, y_face - gp.plate_t, z_plate_lo))
        shape = shape.fuse(cq.Solid.makeBox(bracket_hi[0] - bracket_lo[0], bracket_hi[1] - bracket_lo[1],
                                            gp.bracket_t, cq.Vector(*bracket_lo)))
        rods = None
        for x, z in tines:
            rod = cq.Solid.makeCylinder(r, y_tip - gp.lead_in - y_face, cq.Vector(x, y_face, z), cq.Vector(0, 1, 0))
            rod = rod.fuse(cq.Solid.makeCone(r, .25*r, gp.lead_in, cq.Vector(x, y_tip - gp.lead_in, z),
                                             cq.Vector(0, 1, 0)))
            rods = rod if rods is None else rods.fuse(rod)
        if gp.tine_material == 'pla':
            shape = shape.fuse(rods)
        bundle.update(shape=shape.clean().translate(cq.Vector(*move)), rods=rods.clean().translate(cq.Vector(*move)))
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
    # Each tine is a cantilever from the plate carrying half the brick at mid-span of the bore.
    span = gp.gap + p.D/2
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


def export(bundle, out):
    from pathlib import Path
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle['mesh'].export(out/'gripper.stl')
    if 'shape' in bundle:
        cq.exporters.export(bundle['shape'], str(out/'gripper_printed.step'))
        if bundle['params']['tine_material'] != 'pla':
            cq.exporters.export(bundle['rods'], str(out/'gripper_rods.step'))
    return out
