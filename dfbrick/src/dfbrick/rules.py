"""Analytic design rules for the DF brick, checked before simulation. Margin >= 0 means satisfied.

Same principles as brixzle.rules: every guiding face slides under the worst friction, capture
covers the drone's approach error, clearance absorbs tilt without wedging, the part prints in
one piece, and it hangs level from the passive fork within the payload budget.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from . import geometry as G
from .params import DFParams


@dataclass(frozen=True)
class RuleSettings:
    mu_max: float = 0.4            # PLA-on-PLA upper estimate (as brixzle)
    slide_margin_deg: float = 8.
    placement_error: float = 10.   # X-Y approach error the receivers must capture (mm); set from Exp 3
    mass_max: float = 24.7         # payload_max 30 g (brixzle drone_params) minus the 5.27 g fork
                                   # (brixzle docs/exp2/gripper-fork-v0/gripper.json)
    min_feature: float = 1.2       # printable wall (3 perimeters at 0.4 mm)
    residual_tilt_deg: float = 6.  # tilt left when a foot meets its throat (as brixzle)
    max_yaw_play_deg: float = 1.   # seated yaw play allowed by the two feet
    hang_margin: float = 1.
    min_top_gap: float = 0.5       # rail above the ridges of the course below


def settings_for_tolerance(tolerance, base: RuleSettings = RuleSettings()):
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError('tolerance must be finite and >= 0')
    return replace(base, placement_error=float(tolerance))


def capture(p: DFParams):
    """Largest X or Y offset at which the foot tip still lands inside its receiver's mouth (mm)."""
    return p.extent - (p.half_foot - p.tip_chamfer)


def rules(p: DFParams, bundle, s: RuleSettings = RuleSettings()):
    friction = math.degrees(math.atan(s.mu_max)) + s.slide_margin_deg
    out = {}
    # Every face a falling foot can touch slides it home: the flare and the 45 degree tip chamfer.
    out['flare_slides'] = (90 - p.flare) - friction
    out['chamfer_slides'] = 45. - friction
    # Coarse capture: the mouth reaches the ridge, so a foot tip anywhere inside it seats.
    # Square receivers capture the same in X and Y.
    out['capture_xy'] = capture(p) - s.placement_error
    # Entry: the chamfer plus clearance must absorb the residual tilt over the guided stroke.
    out['clearance_absorbs_tilt'] = (p.tip_chamfer + p.clearance
                                     - p.parallel_len*math.sin(math.radians(s.residual_tilt_deg)))
    # Wedging screen (Whitney peg-in-hole, small-angle): two-point wedging needs a tilt above about
    # c/mu, with c the clearance ratio (radial clearance / foot half-width). The parallel throat
    # holds an engaged foot within atan(2 clearance / parallel_len). Approximate, not a proof.
    c_ratio = p.clearance/p.half_foot
    out['anti_wedge'] = math.degrees(c_ratio/s.mu_max) - math.degrees(math.atan(2*p.clearance/p.parallel_len))
    # Distributed funnel payoff: the wide baseline leaves little seated yaw play.
    out['seated_yaw_play'] = s.max_yaw_play_deg - math.degrees(math.atan(2*p.clearance/p.baseline))
    out['rail_clears_ridges'] = p.top_gap - s.min_top_gap
    # Printability.
    out['wall_printable'] = p.wall_t - s.min_feature
    out['bore_web'] = p.rib_t/2 - G.bore_radius(p) - s.min_feature
    out['single_piece'] = 1. - bundle.get('fragments', 1)
    out['mass_max'] = s.mass_max - bundle['mass_g']
    # Passive fork: the brick hangs below the tines, which straddle the COM.
    z_tine = bundle['channels'][0][1]
    out['hangs_stable'] = z_tine + p.fork_d/2 - bundle['com'][2] - s.hang_margin
    xs = sorted(c[0] for c in bundle['channels'])
    out['tines_straddle_com'] = min(bundle['com'][0] - xs[0], xs[-1] - bundle['com'][0]) - 1.
    return out


def feasible(margins, skip=()):
    return all(v >= 0 for k, v in margins.items() if k not in skip)


def violations(margins):
    return np.array([max(0., -v) for v in margins.values()])


def size_for_tolerance(tolerance, base: DFParams = DFParams(), margin=1.0, step=0.5):
    """Smallest voxel pitch (rounded up to ``step``) whose capture covers ``tolerance + margin``."""
    need = tolerance + margin + (base.half_foot - base.tip_chamfer) + base.end_gap/2
    U = math.ceil(2*need/step)*step
    return replace(base, U=float(U))
