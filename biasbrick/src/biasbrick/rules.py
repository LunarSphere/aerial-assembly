"""Analytic design rules for the biasbrick, checked before simulation. Margin >= 0 means satisfied.

The joint rules are brixzle's (``brixzle.rules``) on the socket profile: every face slides, the
tooth enters inside the nesting cone, the taper absorbs tilt, and the leaning tooth hooks. New here:
the Y sawtooth's main slope must slide although the keel rides in the X sawtooth's V (which
multiplies friction), every landing must fall on the main slope, the socket and flange must catch
the measured error plus the yaw the rod length adds, the rod and the fork sleeves must fit in one
course, and the brick must hang level from the fork within the payload.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from . import geometry as G
from .params import BBParams


@dataclass(frozen=True)
class RuleSettings:
    mu_max: float = 0.4            # PLA-on-PLA upper estimate (as brixzle)
    slide_margin_deg: float = 8.
    placement_error: float = 14.8  # X-Y approach error the sockets must catch (mm); set from Exp 3
    mass_max: float = 24.73        # payload_max 30 g minus brixzle's 5.27 g V0 fork (same tine spacing and
                                   # depth; not re-derived for the sleeves)
    min_feature: float = 1.2       # printable wall (3 perimeters at 0.4 mm)
    residual_tilt_deg: float = 6.  # tilt left when the tooth meets the slot (as brixzle)
    max_play_deg: float = 6.       # seated rotational play of a single-tooth joint (as brixzle)
    fork_insert_error: float = 2.5  # tine-to-sleeve error at pickup the sleeve must absorb (mm); a guess,
                                    # not measured (the fixture's tine guides capture +/-5 mm)
    hang_margin: float = 1.
    support_margin: float = 2.     # landing COM this far inside the flange's far edge
    course_gap: float = 0.5        # sleeves below the next course's parts


def settings_for_tolerance(tolerance, base: RuleSettings = RuleSettings()):
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError('tolerance must be finite and >= 0')
    return replace(base, placement_error=float(tolerance))


def landing(p: BBParams, error, below='A'):
    """Nearest and farthest landing offsets (x, y) of the upper keel from its seat."""
    ax, ay = G.aim(p, below)
    yx, yy = p.yaw_shift
    return (ax - error - yx, ay - error - yy), (ax + error + yx, ay + error + yy)


def rules(p: BBParams, bundle, s: RuleSettings = RuleSettings()):
    friction = math.degrees(math.atan(s.mu_max)) + s.slide_margin_deg
    bp = p.profile()
    out = {}
    # Every face slides: brixzle's ramp and steep face, and the Y slope (the valley lines run at it).
    out['long_ramp_slides'] = p.theta - friction
    out['steep_face_slides'] = bp.phi - friction
    # The keel slides down the Y slope inside the X sawtooth's V: the V faces' normal forces sum to about
    # W / cos(face), so the friction resisting the slide is mu / cos(face) of the weight (steepest face,
    # conservative). Without this factor a 33 deg slope stalled at mu 0.5 (runs/biasbrick/evidence/bb2-mean-*-mu0.5).
    groove = max(p.theta, bp.phi)
    out['y_slope_slides'] = p.slope - (math.degrees(math.atan(s.mu_max/math.cos(math.radians(groove))))
                                       + s.slide_margin_deg)
    out['nest_face_slides'] = p.nest_angle - friction
    # Catchment, around the actual aim: X on the socket below (A or B), Y on the nest face or the main
    # slope, with the COM of a far landing still over the flange. Full coverage of a ring of the error
    # (plus yaw) is the strict test; the capture maps and drops measure the real rate.
    worst = {}
    for below in ('A', 'B'):
        (nx, ny), (fx, fy) = landing(p, s.placement_error, below)
        lo, hi = G.x_band(p, below)
        worst.setdefault('capture_x', []).append(min(nx - lo, hi - fx))
        worst.setdefault('y_on_slopes', []).append(ny - G.y_band(p)[0])
        worst.setdefault('capture_y', []).append(p.D/2 + p.flange - (fy + bundle['com'][1]) - s.support_margin)
    out.update({k: min(v) for k, v in worst.items()})
    # brixzle's joint rules on the socket profile.
    width = p.tooth_w + 2*p.clearance
    out['nest_cone'] = (90 - p.theta) - p.alpha
    out['clearance_absorbs_tilt'] = (p.tooth_taper + 2*p.clearance
                                     - p.tooth_L*math.sin(math.radians(s.residual_tilt_deg)))
    out['seated_play'] = s.max_play_deg - math.degrees(math.atan(2*p.clearance/p.tooth_L))
    out['hook_depth'] = p.tooth_L*math.sin(math.radians(p.alpha)) - width/2
    out['tooth_printable'] = p.tooth_w - p.tooth_taper - p.tip_chamfer - s.min_feature
    # One course: the rod fits the end faces and the sleeves stay below the next course's rod and sleeves.
    z0, z1, floor, ceil = G.rod_band(p, bundle.get('part'))
    out['rod_in_end_faces'] = min(z0 - floor, ceil - z1)
    out['sleeve_fits_course'] = (p.H0 - p.sleeve_gap) - G.sleeve_height(p)
    y0, y1 = G.sleeve_y(p)
    out['sleeves_on_rod'] = min(p.rod_y - p.rod_w/2 - y0, y1 - p.rod_y - p.rod_w/2)
    out['sleeves_between_sockets'] = (p.pitch/2 - p.U/2) - (p.tine_spacing/2 + G.cavity_half(p) + p.cav_wall) - 1.
    # Printability and payload.
    out['wall_printable'] = min(p.wall, p.flange_t, p.cav_wall) - s.min_feature
    out['single_piece'] = 1. - bundle.get('fragments', 1)
    out['mass_max'] = s.mass_max - bundle['mass_g']
    # Passive fork in the sleeves: a coarse tine enters, the V roof centres the hanging brick on it, the
    # tines sit above the COM and straddle it, and the sleeves span its Y.
    com = bundle['com']
    out['cavity_capture'] = p.cav_play - s.fork_insert_error
    out['roof_centres'] = p.roof_angle - friction
    out['hangs_stable'] = bundle['channels'][0][1] - com[2] - s.hang_margin
    xs = sorted(c[0] for c in bundle['channels'])
    out['tines_straddle_com'] = min(com[0] - xs[0], xs[-1] - com[0]) - 1.
    y0, y1 = bundle['sleeve_y']
    out['com_in_sleeve_span'] = min(com[1] - y0, y1 - com[1]) - 2.
    return out


def feasible(margins, skip=()):
    return all(v >= 0 for k, v in margins.items() if k not in skip)


def violations(margins):
    return np.array([max(0., -v) for v in margins.values()])


def size_for_tolerance(tolerance, base: BBParams = BBParams(), step=0.5):
    """Smallest socket width U and flange (rounded up to ``step``) whose catchment covers the X-Y tolerance.

    X: the keel lands on the socket's own top (``geometry.x_band``, U - tooth_w wide), so U >= tooth_w +
    2 (tolerance + yaw x + the aim's offset from the band's centre): capture_x exactly. (At the old aim-blind
    size, 36.5 mm, far -x landings jammed; 40 and 43 mm seat 99/96/94 and 100/96/96% against 97/94/91:
    runs/biasbrick/current/bb7-*.)
    Y: the farthest landing, 2 (tolerance + yaw y) + aim_margin up the slope, keeps its COM
    support_margin inside the flange (conservative: the aim sits lower, and the nest face catches too). The flange moves the COM, so the flange and the sleeves (centred on the COM) are iterated on
    the built brick. The yaw shifts grow with the pitch U + rod_L, so the rod length costs catchment.
    """
    from .cad import build_brick
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError('tolerance must be finite and >= 0')
    a = math.radians(base.yaw_allow)
    U = base.U
    for _ in range(50):  # the yaw shift and the seat's offset depend on U
        yaw_x = (U + base.rod_L)/2*(1 - math.cos(a))
        lo, hi = G.x_band(replace(base, U=U), 'A')
        off = abs(base.aim_x - (lo + hi)/2)  # the aim sits off the catchment's centre
        U_new = math.ceil((base.tooth_w + 2*(tolerance + yaw_x + off))/step - 1e-9)*step
        if U_new == U:
            break
        U = U_new
    p = replace(base, U=float(U), aim_r=float(tolerance))
    far_y = 2*(tolerance + p.yaw_shift[1]) + p.aim_margin
    com_y = base.cav_y
    for _ in range(10):
        flange = max(0., math.ceil((far_y + com_y + 0.5 + RuleSettings.support_margin - p.D/2)/step - 1e-9)*step)
        reach = p.D/2 - p.rod_w/2 - 1.  # the rod must meet the sockets' end faces
        cav_y, rod_y = round(com_y, 1), round(min(max(com_y, -reach), reach), 1)
        depth = max(base.cav_depth, math.ceil(2*(abs(cav_y - rod_y) + p.rod_w/2 + 1.)))  # sleeves cover the rod
        p = replace(p, flange=float(flange), cav_y=float(cav_y), rod_y=float(rod_y), cav_depth=float(depth))
        new = build_brick(p)['com'][1]
        if abs(new - com_y) < 0.1:
            break
        com_y = new
    return p
