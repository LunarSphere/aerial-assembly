"""Analytic design rules: the reasons behind the geometry, checked before simulation.

Each rule returns a signed margin; >= 0 means satisfied. NSGA-III uses the
negated margins as inequality constraints (g <= 0), and the CLI prints them as
the design's justification.
"""
from dataclasses import dataclass
import math

import numpy as np

from .params import BrickParams


@dataclass(frozen=True)
class RuleSettings:
    mu_max: float = 0.4          # PLA-on-PLA upper estimate the design must still slide on
    slide_margin_deg: float = 8  # "steeper than friction suggests"
    placement_error: float = 10.  # worst-case X placement error (mm), coarse drone
    placement_error_y: float = 6.  # Y capture target (1.5 sigma); traded against straight tine bores
    mass_min: float = 1.
    mass_max: float = 25.
    min_feature: float = 1.2     # printable wall (3 perimeters @ 0.4 mm)
    load_factor: float = 20.     # tooth must carry this many brick weights in shear
    pla_shear_mpa: float = 20.   # conservative printed-PLA interlayer shear strength
    residual_tilt_deg: float = 6.  # tilt left when the tooth meets the slot (3 sigma of release tilt)
    hang_margin: float = 1.0     # tine contact must sit this far above the COM (mm)
    max_play_deg: float = 6.     # seated rotational play allowed in a single-tooth joint


def rules(p: BrickParams, bundle, s: RuleSettings = RuleSettings()):
    friction = math.degrees(math.atan(s.mu_max)) + s.slide_margin_deg
    width = p.tooth_w + 2*p.clearance
    entry_gap = p.tooth_taper + 2*p.clearance
    out = {}
    # Every surface a falling brick can land on must slide it home.
    out['long_ramp_slides'] = p.theta - friction
    out['steep_face_slides'] = p.phi - friction
    out['gutter_slides'] = (p.gutter - friction) if p.gutter > 0 else -1.
    # Coarse capture: the valley catchment is the full voxel in X and the full
    # gutter in Y; the slot must still accept a tooth arriving off-centre.
    out['capture_x'] = p.U/2 - p.tooth_w/2 - s.placement_error
    out['capture_y'] = (p.D/2 - p.gutter_flat - s.placement_error_y) if p.gutter > 0 else -s.placement_error_y
    # Final approach along the tooth axis must lie inside the V-nesting cone:
    # extraction direction may not be steeper into the long ramp than the
    # ramp normal allows (alpha <= 90 - theta).
    out['nest_cone'] = (90 - p.theta) - p.alpha
    # Wedging: a tooth tilted by the residual release error makes two-point
    # contact in the slot; the diametral clearance must absorb that tilt over
    # the engaged length (sim sensitivity: clearance dominated jamming).
    out['clearance_absorbs_tilt'] = entry_gap - p.tooth_L*math.sin(math.radians(s.residual_tilt_deg))
    # Seated play lets a single-tooth cantilever rotate by ~atan(2c / L) before
    # the hook bears; overhangs need that small.
    out['seated_play'] = s.max_play_deg - math.degrees(math.atan(2*p.clearance/p.tooth_L))
    # Hook: the tooth tip must sit laterally past half the slot width so a
    # straight vertical lift is blocked (anti-tip under overhang moments).
    out['hook_depth'] = p.tooth_L*math.sin(math.radians(p.alpha)) - width/2
    out['single_piece'] = 1. - bundle.get('fragments', 1)
    # Mass window (favour light; optimizer minimises separately).
    out['mass_min'] = bundle['mass_g'] - s.mass_min
    out['mass_max'] = s.mass_max - bundle['mass_g']
    # Printability and strength.
    out['tooth_printable'] = min(p.tooth_w - p.tooth_taper - p.tip_chamfer,
                                 p.H0 - p.tooth_L*math.cos(math.radians(p.alpha))) - s.min_feature
    shear_area = p.tooth_w*p.D*0.6  # mm^2, gutter-reduced effective depth
    load = s.load_factor*bundle['mass_g']*1e-3*9.81  # N
    out['tooth_strength'] = shear_area*s.pla_shear_mpa - load
    # Passive fork: two bores exist and the brick hangs below them.
    if bundle['channels']:
        z_tine = bundle['channels'][0][1]
        out['fork_fits'] = 1.
        out['hangs_stable'] = z_tine + p.fork_d/2 - bundle['com'][2] - s.hang_margin
        xs = sorted(c[0] for c in bundle['channels'])
        out['tines_straddle_com'] = min(bundle['com'][0] - xs[0], xs[-1] - bundle['com'][0]) - 1.
    else:
        out['fork_fits'] = -1.
        out['hangs_stable'] = -1.
        out['tines_straddle_com'] = -1.
    return out


def feasible(margins):
    return all(v >= 0 for v in margins.values())


def violations(margins):
    return np.array([max(0., -v) for v in margins.values()])
