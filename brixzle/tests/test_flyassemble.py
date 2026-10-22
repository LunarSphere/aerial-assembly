import numpy as np
import pytest

from brixzle import cad, fixture as Fx, flyassemble as FA, gripper as Gr, lattice as L, planner as P, structures as S
from brixzle.drone_params import CF21B
from brixzle.estimator import LighthouseModel
from brixzle.params import V0

QUIET = LighthouseModel(bias_sigma_mm=0, bias_gradient_mm_per_m=0, jitter_mm=0, vel_noise_mps=0, att_noise_deg=0,
                        latency_s=0)


@pytest.fixture(scope='module')
def parts():
    bricks = cad.build_bricks(V0)
    return bricks, Gr.build_gripper(Gr.ForkParams(), bricks['A'], V0)


def test_fixture_rules_and_open_faces(parts):
    bricks, g = parts
    fp = Fx.FixtureParams()
    fx = Fx.build_fixture(V0, fp, ['A', 'Br'], bricks['A']['channels'])
    margins = Fx.fixture_rules(V0, fp, fx, g)
    assert all(v >= 0 for v in margins.values()), margins
    # A turned slot is yawed 180 degrees.
    assert len(fx['slots']) == 2 and fx['slots'][1]['quat'][3] == pytest.approx(1.)


@pytest.mark.parametrize('name', ['tower', 'wall', 'bridge', 'overhang+', 'overhang-'])
def test_plan_follows_assembly_order_and_supporters(parts, name):
    bricks, g = parts
    st = S.CATALOG[name]()
    plan, layout = P.compile_plan(V0, st, bricks, g, Fx.FixtureParams(), CF21B)
    order = L.assembly_order(V0, st)
    assert [b['name'] for b in plan['bricks']] == [b.name for b in order]
    placed = set()
    for b in plan['bricks']:
        sup = b['place']['frame']['supporter']
        assert sup.startswith('base') or sup in placed
        placed.add(b['name'])
        assert b['pick']['segments'][0]['op'] == 'go_to' and b['place']['segments'][-1]['op'] == 'go_to'
    assert plan['settings']['transit_z'] > max(b.course for b in order)*V0.H0*1e-3
    assert plan['settings']['battery']['per_brick_s'] < plan['settings']['battery']['flight_time_s']


def test_a_then_b_pick_and_place_without_sensor_noise():
    cfg = FA.FlyConfig()
    cfg.lighthouse = QUIET
    res, plan, h = FA.fly_assemble(V0, Gr.ForkParams(), 'tower', cfg, seed=0, bricks=2)
    assert [b['part'] for b in plan['bricks']] == ['A', 'B']
    assert res['mission_status'] == 'complete' and res['successes'] == 2
    for b in res['bricks']:
        assert b['true_outcome'] == 'seated' and np.abs(b['verify'][-1]['true_err_mm']).max() < 0.5, b['brick']
    assert not [w for w in res['watchdog'] if 'watchdog' in w[1]]
