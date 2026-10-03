"""'ab' lean mode: distinct A/B parts whose teeth all trail against the build direction."""
from dataclasses import replace

import numpy as np
import pytest

from brixzle import cad, lattice as L, profile as pr, rules as R, structures as S, trials as T
from brixzle.params import V0

AB = replace(V0, lean_mode='ab')


@pytest.fixture(scope='module')
def bricks():
    return cad.build_bricks(AB)


def test_parts_are_single_watertight_and_rule_feasible(bricks):
    assert set(bricks) == {'A', 'B'}
    for name, b in bricks.items():
        assert b['fragments'] == 1 and b['mesh'].is_watertight, name
        margins = R.rules(AB, b)
        assert R.feasible(margins), (name, {k: v for k, v in margins.items() if v < 0})


@pytest.mark.parametrize('direction', [1, -1])
def test_staircase_footprints_nest_without_overlap(direction):
    outer = {k: pr.brick_parts(AB, k)['outer'] for k in 'AB'}
    st = S.overhang(5, direction)
    base, _ = pr.base_outer(AB, st.bases[0][1] - st.bases[0][0], 12, direction)
    from shapely import affinity
    base = affinity.translate(base, st.bases[0][0]*AB.U, 0)
    shapes = [L.footprint(AB, outer, b) for b in st.bricks]
    for i, s in enumerate(shapes):
        assert s.intersection(base).area < 0.5
        for t in shapes[:i]:
            assert s.intersection(t).area < 0.5


def test_every_tooth_trails_against_build_direction():
    for direction in (1, -1):
        for b in S.overhang(4, direction).bricks:
            assert np.sign(L.insertion_axis(AB, b)[0]) == -direction


@pytest.mark.parametrize('part', ['A', 'B'])
def test_aligned_drop_seats(bricks, part):
    rows, _ = T.drop_trials(AB, bricks, T.TrialConfig(error=T.ErrorModel(ideal=True)), offsets=[0.], part=part)
    assert rows[0]['outcome'] == 'seated'


def test_tight_ab_overhang_beats_alternate():
    # Measured: tight 'ab' reaches 10 tracked/ideal, alternate 4-6 (README table).
    cfg = T.TrialConfig(error=T.ErrorModel(ideal=True, track_supporter=True))
    tight = replace(AB, clearance=0.05)
    r, _ = T.assemble(tight, cad.build_bricks(tight), S.overhang(12, 1), cfg)
    assert r['stable_bricks'] >= 8
