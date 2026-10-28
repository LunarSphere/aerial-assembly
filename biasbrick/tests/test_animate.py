import numpy as np
import pytest

from biasbrick import animate as A, geometry as G
from biasbrick.params import DEFAULT


def test_path_goes_through_the_stroke_start_then_down_the_slope():
    p = DEFAULT
    offs = A.path(p, step=2.)
    assert offs[0] == pytest.approx(G.aim(p, 'A'))
    assert max(o[0] for o in offs) >= A.entry_x(p)  # the tooth enters its slot from the stroke side (+x)
    assert offs[-1] == (0., 0.)


def test_brick_rides_the_slope_and_seats():
    """Kinematic stacking: on the slope the brick sits at shear_k * dy, and it ends seated."""
    p = DEFAULT
    offs = A.path(p, step=0.5)  # small steps: the tooth moves across inside its slot between descents
    z = A.rest_height(p, offs, n=60)
    slide = [(o[1], h) for o, h in zip(offs, z) if o[0] == 0.]
    assert all(h == pytest.approx(p.shear_k*dy, abs=0.3) for dy, h in slide)
    assert abs(z[-1]) < 0.2  # seated, to the sampled-overlap tolerance (0.05 mm depth, sparse samples)
