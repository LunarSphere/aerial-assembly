"""Analytic design rules for polycube bricks; margins >= 0 mean satisfied."""
from dataclasses import dataclass
import math

import numpy as np

from .params import BrickParams


@dataclass(frozen=True)
class RuleSettings:
    mu_max: float = 0.4            # PLA-on-PLA upper estimate that must still slide
    slide_margin_deg: float = 8.   # steeper than friction suggests
    placement_error: float = 8.    # mm per axis the funnel must capture
    mass_min: float = 1.
    mass_max: float = 25.
    payload_g: float = 40.         # drone payload
    min_feature: float = 1.2       # printable wall (3 perimeters at 0.4 mm)
    residual_tilt_deg: float = 6.  # tilt left when the peg meets the socket
    hang_margin: float = 1.0       # tine contact above the COM (mm)


def rules(p: BrickParams, bundle, s: RuleSettings = RuleSettings()):
    friction = math.degrees(math.atan(s.mu_max)) + s.slide_margin_deg
    out = {}
    # Every landing face is a funnel face: must slide home.
    out['funnel_slides'] = p.theta - friction
    # Capture: funnel catchment is the cell; the peg tip must still clear the far rim.
    out['capture'] = p.a - (p.peg_w - p.peg_taper)/2 - s.placement_error
    # Final approach along the peg axis must stay inside the pyramid nesting cone.
    out['nest_cone'] = (90 - p.theta) - abs(p.peg_lean)
    # The slanted final stroke must itself slide (axis steeper than friction).
    out['stroke_slides'] = (90 - abs(p.peg_lean)) - friction
    # Wedging: entry gap absorbs residual tilt over the engaged length.
    out['entry_gap'] = p.peg_taper + 2*p.clearance - p.peg_L*math.sin(math.radians(s.residual_tilt_deg))
    # Hook: the lobe's far wall leans with the peg, so a vertical lift is blocked
    # once the tip offset exceeds the seated gap (sim: 15 deg stops pillar creep).
    out['hook'] = p.peg_L*math.tan(math.radians(abs(p.peg_lean))) - 2*p.clearance
    # Socket floor stays inside the cell.
    out['socket_floor'] = (p.H - p.peg_L - p.socket_extra) - s.min_feature
    out['peg_printable'] = (p.peg_w - p.peg_taper) - 2*p.tip_chamfer - s.min_feature + 1.2
    out['mass_min'] = bundle['mass_g'] - s.mass_min if 'mass_g' in bundle else 1.
    out['mass_max'] = s.mass_max - bundle['mass_g'] if 'mass_g' in bundle else 1.
    # Passive fork.
    if bundle['bores']:
        out['fork_fits'] = 1.
        if 'com' in bundle:
            z_tine = bundle['bores'][0][1]
            ys = sorted(b[0] for b in bundle['bores'])
            out['hangs_stable'] = z_tine + p.fork_d/2 - bundle['com'][2] - s.hang_margin
            out['tines_straddle_com'] = (min(bundle['com'][1] - ys[0], ys[-1] - bundle['com'][1]) - 1.) \
                if len(ys) > 1 else -1.
    else:
        out['fork_fits'] = -1.
    return out


def feasible(margins, ignore=()):
    return all(v >= 0 for k, v in margins.items() if k not in ignore)
