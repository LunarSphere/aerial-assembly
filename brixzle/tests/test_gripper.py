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
    com_world = np.asarray(g['carry']) + brick['com']
    assert abs(com_world[0]) < 1e-9 and abs(com_world[1]) < 1e-9
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
