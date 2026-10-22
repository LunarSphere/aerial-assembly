"""A/B is the only brick design: V0 is the A/B pair, one fork carries both parts, the fixture holds both."""
import json
from pathlib import Path

import numpy as np
import pytest
from shapely import affinity

from brixzle import cad, fixture as Fx, gripper as Gr, lattice as L, planner as P, profile as pr, structures as S
from brixzle.drone_params import CF21B
from brixzle.params import V0, BrickParams

CONFIGS = Path(__file__).parents[1]/'configs'


def test_v0_is_the_ab_config():
    ab = json.loads((CONFIGS/'ab.json').read_text())
    assert BrickParams.from_dict(ab) == V0
    assert 'lean_mode' not in V0.to_dict()


def test_from_dict_accepts_legacy_ab_and_rejects_other_lean_modes():
    assert BrickParams.from_dict({**V0.to_dict(), 'lean_mode': 'ab'}) == V0
    for mode in ('alternate', 'uniform'):
        with pytest.raises(ValueError, match='A/B'):
            BrickParams.from_dict({**V0.to_dict(), 'lean_mode': mode})


@pytest.fixture(scope='module')
def bricks():
    return cad.build_bricks(V0)


def test_one_fork_fits_both_parts(bricks):
    assert set(bricks) == {'A', 'B'}
    assert bricks['A']['channels'] == bricks['B']['channels']
    xs = sorted(c[0] for c in bricks['A']['channels'])
    for b in bricks.values():
        assert xs[0] < b['com'][0] < xs[1]
        assert all(c[1] > b['com'][2] for c in b['channels'])


def test_fixture_slots_seat_their_part(bricks):
    fp = Fx.FixtureParams(hooked=True)
    for part in 'AB':
        slot = Fx.slot_outer(V0, fp, part)
        brick = pr.brick_parts(V0, part)['outer']
        assert brick.intersection(slot).area < 0.5, part
        assert brick.distance(slot) < 0.05, part
    # The other part's teeth would not fit: the slots really are part-specific.
    wrong = pr.brick_parts(V0, 'B')['outer'].intersection(Fx.slot_outer(V0, fp, 'A')).area
    assert wrong > 5.


def test_fixture_holds_both_parts_and_keeps_its_rules(bricks):
    g = Gr.build_gripper(Gr.ForkParams(), bricks['A'], V0)
    fp = Fx.FixtureParams()
    fx = Fx.build_fixture(V0, fp, ['A', 'B'], bricks['A']['channels'])
    assert [s['kind'] for s in fx['slots']] == ['A', 'B']
    margins = Fx.fixture_rules(V0, fp, fx, g)
    assert all(v >= 0 for v in margins.values()), margins


def test_tower_plan_alternates_parts_and_their_slots_and_masses(bricks):
    g = Gr.build_gripper(Gr.ForkParams(), bricks['A'], V0)
    plan, layout = P.compile_plan(V0, S.tower(4), bricks, g, Fx.FixtureParams(), CF21B)
    parts = [b['part'] for b in plan['bricks']]
    assert parts == ['A', 'B', 'A', 'B']
    for b in plan['bricks']:
        assert layout['slot_kinds'][b['slot']] == b['slot_kind'] == b['part']
        loaded = next(s['value'] for s in b['pick']['segments'] if s.get('name') == 'ctrlMel.mass')
        empty = plan['settings']['ctrl_mass_kg']['empty']
        assert loaded == pytest.approx(empty + bricks[b['part']]['mass_g']*1e-3)


def test_overhang_minus_uses_rotated_slots(bricks):
    g = Gr.build_gripper(Gr.ForkParams(), bricks['A'], V0)
    plan, _ = P.compile_plan(V0, S.overhang(2, -1), bricks, g, Fx.FixtureParams(), CF21B)
    for b in plan['bricks']:
        assert b['orientation'] == 'B' and b['slot_kind'] == b['part'] + 'r'


def test_courses_nest_without_overlap():
    outer = {k: pr.brick_parts(V0, k)['outer'] for k in 'AB'}
    lower = L.footprint(V0, outer, L.Placement(0, 0))
    upper = L.footprint(V0, outer, L.Placement(1, 1))
    base, _ = pr.base_outer(V0, 4, 12)
    assert lower.intersection(upper).area < 0.5 and lower.intersection(base).area < 0.5
    assert lower.distance(upper) < 0.05 and lower.distance(base) < 0.05
