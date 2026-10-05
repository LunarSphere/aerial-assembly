import pytest

from brixzle import baseplate as B
from brixzle.params import V0


@pytest.fixture(scope='module')
def plates():
    return {n: B.build_baseplate(V0, n) for n in (2, 3)}


def test_watertight_and_footprint(plates):
    for n, b in plates.items():
        m = b['mesh']
        assert m.is_watertight
        ext = m.bounds[1] - m.bounds[0]
        assert abs(ext[0] - n*V0.U) < 0.1 and abs(ext[1] - n*V0.D) < 0.1


def test_flat_bottom_and_volume_scales(plates):
    assert abs(plates[2]['mesh'].bounds[0][2] + plates[2]['thickness']) < 1e-3
    ratio = plates[3]['mesh'].volume/plates[2]['mesh'].volume
    assert 2.0 < ratio < 2.5  # ~ (3/2)^2 = 2.25
