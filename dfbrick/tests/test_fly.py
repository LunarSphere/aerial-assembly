import numpy as np
import pytest

from brixzle import gripper as Gr
from brixzle.estimator import LighthouseModel
from brixzle.flyassemble import FlyConfig
from brixzle.trials import ErrorModel

from dfbrick import fly as F
from dfbrick.params import DEFAULT

QUIET = LighthouseModel(bias_sigma_mm=0, bias_gradient_mm_per_m=0, jitter_mm=0, vel_noise_mps=0, att_noise_deg=0,
                        latency_s=0)


def test_plan_places_supporters_first_with_vertical_descent():
    plan, layout, brick, gripper, structure = F.make(DEFAULT, Gr.ForkParams(), 'tower', FlyConfig(), bricks=3)
    names = [b['name'] for b in plan['bricks']]
    assert names == ['b0_x0_0', 'b1_x0_0', 'b2_x0_0']
    assert plan['bricks'][1]['place']['frame']['supporter'] == 'b0_x0_0'
    for b in plan['bricks']:
        moves = [np.asarray(s['to']) for s in b['place']['segments'] if s['phase'] == 'DESCEND' and 'to' in s]
        assert all(np.allclose(m[:2], moves[0][:2]) for m in moves)  # straight down


def test_fork_carries_a_df_brick():
    p = DEFAULT
    plan, layout, brick, gripper, _ = F.make(p, Gr.ForkParams(), 'tower', FlyConfig(), bricks=1)
    margins = F.gripper_margins(p, Gr.ForkParams(), brick, gripper, FlyConfig().drone)
    assert margins['tine_fits_bore'] >= 0 and margins['com_on_axis'] >= 0, margins
    assert layout['payload_g'] == pytest.approx(brick['mass_g'] + gripper['mass_g'])


def test_two_brick_tower_without_sensor_noise():
    cfg = FlyConfig()
    cfg.lighthouse = QUIET
    res, plan, h = F.fly_assemble(DEFAULT, Gr.ForkParams(), 'tower', cfg, seed=0, bricks=2)
    assert res['mission_status'] == 'complete' and res['successes'] == 2, [b['true_outcome'] for b in res['bricks']]
    for b in res['bricks']:
        assert np.abs(b['verify'][-1]['true_err_mm']).max() < 1.0, b['brick']


def test_measured_bias_is_injected_per_placement():
    cfg = FlyConfig()
    cfg.lighthouse = QUIET
    err = ErrorModel(radial_mode='ring', radius_mm=6., aim_bias=0.)
    res, plan, h = F.fly_assemble(DEFAULT, Gr.ForkParams(), 'tower', cfg, seed=0, bricks=1, error=err)
    b = res['bricks'][0]
    assert np.hypot(*b['placement_bias_mm'][:2]) == pytest.approx(6.)
    assert np.allclose(h.lighthouse.b0, 0.)  # cleared after the place was verified
    assert b['true_outcome'] == 'seated', b['verify']
