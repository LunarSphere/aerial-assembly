import math

import numpy as np
import pytest
import trimesh

from dfbrick import cad, geometry as G, rules as R
from dfbrick.params import DEFAULT, DFParams


@pytest.fixture(scope='module')
def brick():
    return cad.build_brick(DEFAULT)


def test_params_validate():
    with pytest.raises(ValueError):
        DFParams.from_dict({'U': 38., 'tooth_L': 3.})
    with pytest.raises(ValueError):
        DFParams(receiver_mode='cone')
    with pytest.raises(ValueError):
        DFParams(U=8.)
    with pytest.raises(ValueError):
        DFParams(flare=float('nan'))
    assert DFParams.from_dict({**DEFAULT.to_dict(), 'derived': {}}) == DEFAULT


def test_derived_sizes_close_the_funnel():
    p = DEFAULT
    assert p.extent == pytest.approx(p.U/2 - p.end_gap/2)
    # The flare reaches the ridge exactly; the rail above clears it by top_gap.
    assert G.inner_face_z(p, p.extent, 0.) == pytest.approx(p.guide_h)
    assert p.foot_L - p.guide_h == pytest.approx(p.top_gap)
    assert p.baseline == pytest.approx(2*p.U)


def test_pieces_are_convex_and_solid(brick):
    for v in brick['collision'] + brick['collision_bored']:
        hull = trimesh.convex.convex_hull(v)
        assert hull.volume > 0
        # Every vertex is on the hull (pieces are convex by construction, nothing dropped).
        assert len(np.unique(np.round(v, 6), axis=0)) == len(np.unique(np.round(hull.vertices, 6), axis=0))


def test_solid_is_one_watertight_piece(brick):
    assert brick['fragments'] == 1
    assert brick['mesh'].is_watertight
    # Bores and overlaps aside, the solid matches the collision volume within 10%.
    assert brick['volume'] == pytest.approx(cad.collision_volume(brick), rel=0.1)


def test_mass_and_fork(brick):
    assert 10. < brick['mass_g'] < 24.7
    margins = R.rules(DEFAULT, brick)
    assert R.feasible(margins), {k: v for k, v in margins.items() if v < 0}
    (x0, z0), (x1, z1) = brick['channels']
    assert z0 == z1 and x0 < brick['com'][0] < x1


def test_slot_mode_drops_x_walls():
    p = DFParams(receiver_mode='pocket-slot')
    assert [p.receiver_kind(i) for i in range(3)] == ['pocket', 'slot', 'slot']
    assert len(G.receiver_pieces(p, 'pocket')) == 8 and len(G.receiver_pieces(p, 'slot')) == 4
    slot = np.vstack(G.receiver_pieces(p, 'slot'))
    assert np.abs(slot[:, 1]).min() == pytest.approx(p.throat)  # walls only across Y


def test_rules_flag_bad_designs(brick):
    margins = R.rules(DEFAULT, brick, R.settings_for_tolerance(30.))
    assert margins['capture_xy'] < 0
    steep = DFParams(flare=65.)
    assert R.rules(steep, cad.build_brick(steep, solid=False))['flare_slides'] < 0
    # Both wedge terms scale with clearance; what matters is a guided stroke long against the foot.
    short = DFParams(parallel_len=2.)
    assert R.rules(short, cad.build_brick(short, solid=False))['anti_wedge'] < 0


def test_wide_baseline_limits_yaw_play():
    assert math.degrees(math.atan(2*DEFAULT.clearance/DEFAULT.baseline)) < 0.5


@pytest.mark.parametrize('tol', [14.8, 38.5])
def test_sizing_covers_tolerance(tol):
    p = R.size_for_tolerance(tol)
    assert R.capture(p) >= tol + 1.0
    assert R.capture(DFParams(**{**p.to_dict(), 'U': p.U - 0.5})) < tol + 1.0
