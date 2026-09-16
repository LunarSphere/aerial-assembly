from dataclasses import replace

import mujoco
import numpy as np
import pytest

from aerial_assembly.config import Envelope, Physics, Release, TrialSettings
from aerial_assembly.experiments import summarize, wilson
from aerial_assembly.model import build_model, initial_state, set_state
from aerial_assembly.simulation import classify, pose_metrics, run_drop


def test_free_fall_matches_gravity(bundle, model):
    state = initial_state(bundle, Release(height=1))
    data = set_state(model, state)
    t = .01
    mujoco.mj_step(model, data, nstep=round(t/model.opt.timestep))
    assert data.qvel[2] == pytest.approx(-9.81*t, abs=1e-9)
    assert data.qpos[2] == pytest.approx(state['qpos'][2]-.5*9.81*t*t, abs=3e-6)
    assert data.ncon == 0


def test_release_rotation_preserves_com_and_velocity(bundle):
    from aerial_assembly.config import rotation
    a = initial_state(bundle, Release())
    b = initial_state(bundle, Release(rpy_deg=(4,5,6), velocity=(.1,.2,.3), angular_velocity=(.3,.1,.2)))
    R = rotation(b['qpos'][3:])
    com = R.apply(bundle['inertial']['com'])
    a_com = rotation(a['qpos'][3:]).apply(bundle['inertial']['com']) + a['qpos'][:3]
    b_com = com + b['qpos'][:3]
    assert np.allclose(a_com[:2], b_com[:2])
    assert np.allclose(np.asarray(b['qvel'][:3])+np.cross([.3,.1,.2],com), [.1,.2,.3])


def test_aligned_repeatable_and_half_timestep(bundle, model):
    settings = TrialSettings(duration=.3, dwell=.1)
    state = initial_state(bundle)
    a, trace_a = run_drop(model, bundle, state, settings)
    b, trace_b = run_drop(model, bundle, state, settings)
    assert a['status'] == 'success'
    assert a == b
    assert np.array_equal(trace_a, trace_b)
    assert not np.array_equal(trace_a[0], trace_a[-1])
    refined = build_model(bundle, replace(Physics(), timestep=Physics().timestep/2))[0]
    c, _ = run_drop(refined, bundle, state, settings, record=False)
    assert c['status'] == a['status']
    assert c['final']['max_seating_gap'] < settings.gap_tolerance
    assert a['max_penetration'] < bundle['max_penetration']


def test_target_stays_seated(bundle, model):
    state = {'qpos': [*bundle['target']['pos'], *bundle['target']['quat']], 'qvel': [0]*6}
    r, _ = run_drop(model, bundle, state, TrialSettings(duration=.15, dwell=.1), record=False)
    assert r['status'] == 'success'


def test_miss_and_intersecting_initial_state(bundle, model):
    result, _ = run_drop(model, bundle, initial_state(bundle, Release(offset=(.3,0))),
                         TrialSettings(duration=.3, dwell=.1), record=False)
    assert result['status'] == 'missed_receiver'
    state = {'qpos': [0,0,.039,1,0,0,0], 'qvel': [0]*6}
    r, _ = run_drop(model, bundle, state, TrialSettings(duration=.1,dwell=.05))
    assert r['status'] == 'invalid'
    assert r['invalid_reason'] == 'intersecting_initial_state'


def test_final_dwell_required_and_invalid_overrides_success(bundle):
    settings = TrialSettings()
    final = {'linear_speed':0,'angular_speed':0,'translation_error':0,'angle_error':0}
    assert classify(final, .49, bundle, settings)[0] != 'success'
    assert classify(final, .5, bundle, settings)[0] == 'success'
    assert classify(final, 1, bundle, settings, 'excessive_penetration')[0] == 'invalid'
    assert classify(final, 1, bundle, settings, touched_floor=True)[0] == 'missed_receiver'


def test_quaternion_sign_does_not_change_score(bundle):
    a = pose_metrics(bundle, bundle['target']['pos'], [1,0,0,0])
    b = pose_metrics(bundle, bundle['target']['pos'], [-1,0,0,0])
    assert a == b


def test_randomization_and_statistics():
    assert Envelope().sample(10,42) == Envelope().sample(10,42)
    assert Envelope().sample(10,42) != Envelope().sample(10,43)
    low, high = wilson(0,100)
    assert low == pytest.approx(0,abs=1e-15) and .03 < high < .04
    s = summarize([{'status':'invalid','settling_time':None}, {'status':'success','settling_time':.2}])
    assert s['success_fraction_among_valid'] == 1
    assert s['invalid_fraction'] == .5
    assert not s['eligible_for_ranking']


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        Physics(timestep=.01)
    with pytest.raises(ValueError):
        TrialSettings(duration=.1,dwell=.5)
    with pytest.raises(ValueError):
        Release(height=-1)
    with pytest.raises(ValueError):
        Envelope(offset=(-1,0)).sample(1,42)
