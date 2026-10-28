import numpy as np
import pytest

from biasbrick import animate as A
from biasbrick.params import DEFAULT


def test_path_runs_from_the_aim_down_the_crease_to_the_seat():
    p = DEFAULT
    offs = A.path(p, step=2.)
    assert offs[0] == pytest.approx(p.aim)
    assert offs[-1] == (0., 0.)
    c = min(p.aim)
    assert any(np.allclose(o, (c, c)) for o in offs)  # reaches the crease, then follows it


def test_brick_slides_down_and_seats_on_its_cups():
    """Kinematic stacking: the height falls along the slide and ends at the seat, the cups on the ramps."""
    p = DEFAULT
    z = A.rest_height(p, A.path(p, step=2.), n=150)
    assert np.all(np.diff(z) <= 0.2)
    assert -0.6 < z[-1] <= 0.
