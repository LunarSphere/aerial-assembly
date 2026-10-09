from dataclasses import replace

import numpy as np
import pytest

from brixzle import cad, gripper as Gr
from brixzle.drone_params import CF21B
from brixzle.params import V0


@pytest.fixture(scope='module')
def brick():
    return cad.build_brick(V0)


def test_fork_v0_is_feasible_and_hangs_brick_on_axis(brick):
    gp = Gr.ForkParams()
    g = Gr.build_gripper(gp, brick, V0)
    m = Gr.gripper_rules(gp, g, brick, V0, CF21B)
    assert Gr.feasible(m), {k: v for k, v in m.items() if v < 0}
    brick_com = np.asarray(g['carry']) + brick['com']
    combined = (g['mass_g']*np.asarray(g['com']) + brick['mass_g']*brick_com)/(g['mass_g'] + brick['mass_g'])
    assert np.allclose(combined[:2], 0, atol=1e-9) and np.hypot(*brick_com[:2]) < 5.
    # Tines sit where the brick's bores are, once the brick is at its carried pose.
    for (x, z), (bx, bz) in zip(g['tines'], brick['channels']):
        assert np.isclose(x, bx + g['carry'][0]) and np.isclose(z, bz + g['carry'][2])
    assert 2. < g['mass_g'] < 8.


@pytest.mark.parametrize('change, rule', [
    (dict(tine_d=2.7), 'tine_fits_bore'),
    (dict(plate_t=0.8), 'printable_wall'),
    (dict(clear_guard=-5.), 'clears_guards'),
])
def test_bad_fork_designs_violate_the_matching_rule(brick, change, rule):
    gp = replace(Gr.ForkParams(), **change)
    g = Gr.build_gripper(gp, brick, V0, solid=False)
    m = Gr.gripper_rules(gp, g, brick, V0, CF21B)
    assert m[rule] < 0


def test_payload_and_thrust_limits_bind(brick):
    from brixzle.drone_params import DroneConfig
    gp = Gr.ForkParams(tine_material='steel')
    g = Gr.build_gripper(gp, brick, V0, solid=False)
    assert Gr.gripper_rules(gp, g, brick, V0, DroneConfig(payload_max_g=15.))['payload_limit'] < 0
    assert Gr.gripper_rules(gp, g, brick, V0, DroneConfig(thrust_source='crazyflow'))['thrust_to_weight'] < 0


def test_gripper_rods_are_convex_and_export(brick, tmp_path):
    g = Gr.build_gripper(Gr.ForkParams(), brick, V0)
    Gr.export(g, tmp_path)
    assert (tmp_path/'gripper.stl').exists() and (tmp_path/'gripper_printed.step').exists()
    assert g['shape'].isValid() and g['rods'].isValid()


def test_pressed_msh2_30_pins_fit_their_length_and_export_a_coupon(brick, tmp_path):
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).parents[1]/'configs'/'gripper-fork-msh2-30.json').read_text())
    gp = Gr.params_from_dict(d)
    g = Gr.build_gripper(gp, brick, V0)
    m = Gr.gripper_rules(gp, g, brick, V0, CF21B)
    assert Gr.feasible(m), {k: v for k, v in m.items() if v < 0}
    # Exposed tine + press depth is the bought pin; the pin ends flush with the boss's back face.
    assert np.isclose(gp.exposed(V0) + g['press_depth'], 30.)
    assert np.isclose(g['shape'].BoundingBox().ymin, g['y_face'] - g['press_depth'], atol=1e-6)
    Gr.export(g, tmp_path)
    assert (tmp_path/'press_coupon.stl').exists() and (tmp_path/'gripper_rods.step').exists()


def test_tines_stopping_before_the_brick_com_violate_tine_past_com(brick):
    gp = replace(Gr.ForkParams(), protrude=-12.)
    g = Gr.build_gripper(gp, brick, V0, solid=False)
    assert Gr.gripper_rules(gp, g, brick, V0, CF21B)['tine_past_com'] < 0
