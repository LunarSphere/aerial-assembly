"""Analytic design rules for the biasbrick3d, checked before simulation. Margin >= 0 means satisfied.

Same principles as brixzle and dfbrick: every guiding face slides under the worst friction, the
catchment covers the drone's approach error, clearance absorbs tilt without wedging, the part
prints in one piece, and it hangs level from the passive fork within the payload budget. New
here: the aim bias must keep every landing on the ramp side of the seat (one-sided), and the
ramps' diagonal crease, shallower than either ramp, must also slide.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from .params import BBParams


@dataclass(frozen=True)
class RuleSettings:
    mu_max: float = 0.4            # PLA-on-PLA upper estimate (as brixzle)
    slide_margin_deg: float = 8.
    placement_error: float = 14.8  # X-Y approach error the hoppers must capture (mm); set from Exp 3
    mass_max: float = 21.6         # payload_max 30 g minus an 8.4 g fork, as dfbrick (fork not yet resized)
    min_feature: float = 1.2       # printable wall (3 perimeters at 0.4 mm)
    residual_tilt_deg: float = 6.  # tilt left when a foot meets its throat (as brixzle)
    max_yaw_play_deg: float = 1.   # seated yaw play allowed by the two feet
    fork_insert_error: float = 2.5  # tine-to-cavity error at pickup the cavity must absorb (mm); a guess,
                                    # not measured (the fixture's tine guides capture +/-5 mm)
    hang_margin: float = 1.
    com_band_margin: float = 2.    # COM this far inside the cavities' Y span, so the brick hangs level
    min_top_gap: float = 0.5       # rail above the ridges of the course below


def settings_for_tolerance(tolerance, base: RuleSettings = RuleSettings()):
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError('tolerance must be finite and >= 0')
    return replace(base, placement_error=float(tolerance))


def crease_angle(p: BBParams):
    """Slope of the diagonal crease where the +x and +y ramps meet (degrees from horizontal)."""
    return math.degrees(math.atan(math.tan(math.radians(p.ramp))/math.sqrt(2)))


def capture(p: BBParams):
    """Farthest X or Y offset (from the seat) at which the foot tip still lands on its own hopper's ramp."""
    return p.reach - (p.half_foot - p.tip_chamfer)


def landing(p: BBParams, error):
    """Nearest and farthest per-axis landing offsets from the seat, (x, y) each, for an X-Y error radius."""
    near = tuple(a - error - s for a, s in zip(p.aim, p.yaw_shift))
    far = tuple(a + error + s for a, s in zip(p.aim, p.yaw_shift))
    return near, far


def rules(p: BBParams, bundle, s: RuleSettings = RuleSettings()):
    friction = math.degrees(math.atan(s.mu_max)) + s.slide_margin_deg
    out = {}
    # Every face a falling foot can touch slides it toward the seat.
    out['ramp_slides'] = p.ramp - friction
    out['crease_slides'] = crease_angle(p) - friction
    # The slot's lower (+x, +y) walls: steepest along their corner, which leans alpha from vertical.
    out['slot_slides'] = (90 - p.alpha) - friction
    out['chamfer_slides'] = 45. - friction
    # Aim bias: every landing is on the ramp side of the seat (never on a side wall or the ridge
    # behind it) and on this hopper's ramps (never past the far ridge).
    near, far = landing(p, s.placement_error)
    out['one_sided_x'], out['one_sided_y'] = near
    out['capture_x'], out['capture_y'] = (capture(p) - f for f in far)
    # The final stroke along the tooth axis lies inside the cone the ramps and crease leave open
    # (brixzle's nest_cone): along x (or y) the tooth leans atan(lean_k), along the diagonal alpha.
    out['nest_cone'] = min((90 - p.ramp) - math.degrees(math.atan(p.lean_k)), (90 - crease_angle(p)) - p.alpha)
    # Entry: the taper plus clearance must absorb the residual tilt over the tooth (as brixzle).
    out['clearance_absorbs_tilt'] = (p.tooth_taper + 2*p.clearance
                                     - p.tooth_L*math.sin(math.radians(s.residual_tilt_deg)))
    # Wedging screen (Whitney peg-in-hole, small-angle; as dfbrick). Approximate, not a proof.
    c_ratio = p.clearance/p.half_foot
    out['anti_wedge'] = math.degrees(c_ratio/s.mu_max) - math.degrees(math.atan(2*p.clearance/p.tooth_L))
    out['seated_yaw_play'] = s.max_yaw_play_deg - math.degrees(math.atan(2*p.clearance/p.baseline))
    # Hook (brixzle's hook_depth, per axis): the tooth tip sits past half the slot width under the
    # overhanging -x and -y walls, so a seated foot cannot be lifted straight out.
    out['hook_depth'] = p.tooth_h*p.lean_k - (p.half_foot + p.clearance)
    out['tooth_printable'] = p.foot_w - p.tooth_taper - 2*p.tip_chamfer - s.min_feature
    out['rail_clears_ridges'] = p.top_gap - s.min_top_gap
    # Printability.
    out['wall_printable'] = p.wall_t - s.min_feature
    out['cavity_wall_printable'] = p.cav_wall - s.min_feature
    out['single_piece'] = 1. - bundle.get('fragments', 1)
    out['mass_max'] = s.mass_max - bundle['mass_g']
    # Passive fork in the shelled cavities: a coarse tine still enters, the V roof centres the hanging
    # brick on it, the tines sit above the COM and straddle it, and the cavities span its Y.
    out['cavity_capture'] = p.cav_play - s.fork_insert_error
    out['roof_centres'] = p.roof_angle - friction
    com = bundle['com']
    y0, y1 = bundle['cavity_y']
    out['hangs_stable'] = bundle['channels'][0][1] - com[2] - s.hang_margin
    xs = sorted(c[0] for c in bundle['channels'])
    out['tines_straddle_com'] = min(com[0] - xs[0], xs[-1] - com[0]) - 1.
    out['com_in_cavity_span'] = min(com[1] - y0, y1 - com[1]) - s.com_band_margin
    return out


def feasible(margins, skip=()):
    return all(v >= 0 for k, v in margins.items() if k not in skip)


def violations(margins):
    return np.array([max(0., -v) for v in margins.values()])


def size_for_tolerance(tolerance, base: BBParams = BBParams(), step=0.5):
    """Smallest voxel pitch (rounded up to ``step``) whose aim bias and catchment cover the X-Y tolerance.

    The aim is tolerance + yaw shift + aim_margin per axis, so the farthest landing is
    2 (tolerance + yaw shift) + aim_margin from the seat; capture must reach it. Capture is
    U - end_gap - wall_t - throat - (half_foot - tip_chamfer), and the Y yaw shift is
    (n - 1) U / 2 sin(yaw_allow), which sets U.
    """
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError('tolerance must be finite and >= 0')
    lean = (base.n_vox - 1)/2*math.sin(math.radians(base.yaw_allow))
    fixed = base.end_gap + base.wall_t + base.throat + base.half_foot - base.tip_chamfer
    U = (2*tolerance + base.aim_margin + fixed)/(1 - 2*lean)
    U = math.ceil(U/step - 1e-9)*step
    return replace(base, U=float(U), aim_r=float(tolerance), cav_depth=min(base.cav_depth, U - base.end_gap))
