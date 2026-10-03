import math

import numpy as np
import pytest
from dataclasses import replace

from brixzle import cad, lattice as L, profile as pr, rules as R, structures as S
from brixzle.convex import convex_parts
from brixzle.params import V0


def test_sawtooth_spans_voxel_and_peaks_at_boundaries():
    x = np.array([0., V0.U, 2*V0.U])
    assert np.allclose(pr.sawtooth(V0, x), V0.amplitude)
    assert np.isclose(pr.sawtooth(V0, V0.valley_u), 0.)
    assert np.isclose(pr.sawtooth(V0, V0.U - V0.valley_u, mirrored=True), 0.)


def test_convex_parts_cover_polygon():
    L_shape = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
    parts = convex_parts(L_shape)
    area = sum(0.5*abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1))) for q in parts)
    assert math.isclose(area, 6.0)
    assert 2 <= len(parts) <= 3


@pytest.fixture(scope='module')
def brick():
    return cad.build_brick(V0)


def test_v0_mass_within_window_and_watertight(brick):
    assert 1 <= brick['mass_g'] <= 25
    assert brick['mesh'].is_watertight


def test_collision_volume_matches_outer_profile(brick):
    assert math.isclose(cad.collision_volume(brick), brick['profile_area']*V0.D, rel_tol=0.01)


def test_v0_satisfies_design_rules(brick):
    margins = R.rules(V0, brick)
    assert R.feasible(margins), {k: v for k, v in margins.items() if v < 0}


def test_fork_bores_straddle_com_above_it(brick):
    xs = sorted(c[0] for c in brick['channels'])
    assert len(xs) == 2 and xs[0] < brick['com'][0] < xs[1]
    assert all(c[1] > brick['com'][2] for c in brick['channels'])


@pytest.mark.parametrize('mode', ['alternate', 'uniform'])
def test_courses_nest_without_overlap(mode):
    p = replace(V0, lean_mode=mode)
    outer = pr.brick_parts(p)['outer']
    lower = L.footprint(p, outer, L.Placement(0, 0))
    upper = L.footprint(p, outer, L.Placement(1, 1))
    base, _ = pr.base_outer(p, 4, 12)
    assert lower.intersection(upper).area < 0.5
    assert lower.intersection(base).area < 0.5
    # They touch: nesting is a contact, not a gap.
    assert lower.distance(upper) < 0.05 and lower.distance(base) < 0.05


def test_rules_flag_shallow_ramp():
    p = replace(V0, theta=20.)
    margins = R.rules(p, cad.build_brick(p, solid=False))
    assert margins['long_ramp_slides'] < 0
