import mujoco
import pytest

from aerial_assembly.chain import _model
from aerial_assembly.config import Physics


def _body_bundle():
    geom = {'type': 'box', 'size': [.02, .02, .02]}
    return {
        'collision': [geom],
        'visual': [geom],
        'inertial': {
            'mass': .48,
            'com': [0., 0., 0.],
            'matrix': [[1e-4, 0., 0.], [0., 1e-4, 0.], [0., 0., 1e-4]],
        },
    }


def test_chain_model_locks_complement_only_when_configured():
    bundle = _body_bundle()
    physics = Physics(timestep=.002, contact_timeconst=.005)

    free_model, _ = _model(bundle, bundle, 1, physics)
    bolted_model, xml = _model(bundle, bundle, 1, physics,
                               complement_fixed=True, base_floor_z=.02)

    assert free_model.neq == 0
    assert bolted_model.neq == 1
    assert bolted_model.eq_active0[0]
    assert '<weld name="base_world_lock" body1="base" active="true"' in xml
    assert bolted_model.body('base').pos[2] == .02

    data = mujoco.MjData(bolted_model)
    mujoco.mj_resetData(bolted_model, data)
    base_joint = bolted_model.joint('base_free').id
    qpos_address = bolted_model.jnt_qposadr[base_joint]
    initial = data.qpos[qpos_address:qpos_address+7].copy()
    for _ in range(20):
        mujoco.mj_step(bolted_model, data)
    assert data.qpos[qpos_address:qpos_address+7] == pytest.approx(initial, abs=1e-5)
