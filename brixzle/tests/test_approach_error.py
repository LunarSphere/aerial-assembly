"""Exp 3 measured approach error -> trial error model and minimum design tolerance (#25)."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from brixzle import approach_error as AE, cad, cli, fixture as Fx, rules as R, trials as T
from brixzle.params import V0

RADII_M = [0.004, 0.010, 0.012, 0.020, 0.030]


def _metrics(tmp_path, radii=RADII_M, kind='approach-accuracy-metrics'):
    approaches = [{'trial': 'trial-01', 'tag_id': i, 'positioning_error_m': r,
                   'positioning_error_xy_m': [r, 0.], 'control_error_m': r/4} for i, r in enumerate(radii)]
    approaches.append({'trial': 'trial-01', 'tag_id': 9, 'positioning_error_m': None})  # missing camera
    m = {'version': 1, 'kind': kind, 'positioning_source': 'drone_marker',
         'trials_included': ['trial-01'], 'trials_excluded': [],
         'summary': {'positioning': {'overall': {
             'n': len(radii), 'mean_error_m': float(np.mean(radii)),
             'rmse_m': float(np.sqrt(np.mean(np.square(radii)))), 'std_error_m': float(np.std(radii)),
             'max_error_m': max(radii), 'bias_xy_m': [-0.008, 0.004]}}},
         'approaches': approaches}
    path = tmp_path/'metrics.json'
    path.write_text(json.dumps(m))
    return path


def test_summary_from_metrics(tmp_path):
    s = AE.load(_metrics(tmp_path))
    assert s['kind'] == 'approach-error'
    assert s['n'] == 5
    assert s['radii_mm'] == pytest.approx([4., 10., 12., 20., 30.])
    assert s['mean_mm'] == pytest.approx(15.2)
    assert s['max_mm'] == pytest.approx(30.)
    assert s['bias_xy_mm'] == pytest.approx([-8., 4.])  # recorded, not applied
    assert s['source']['metrics'].endswith('metrics.json')
    assert s['source']['positioning_source'] == 'drone_marker'


def test_summary_round_trips_as_config(tmp_path):
    s = AE.load(_metrics(tmp_path))
    cfg = tmp_path/'error.json'
    cfg.write_text(json.dumps(s))
    assert AE.load(cfg) == s


def test_rejects_wrong_kind_and_non_finite(tmp_path):
    with pytest.raises(ValueError):
        AE.load(_metrics(tmp_path, kind='approach-accuracy'))
    with pytest.raises(ValueError):
        AE.load(_metrics(tmp_path, radii=[0.01, float('nan')]))
    with pytest.raises(ValueError):
        AE.error_model(AE.load(_metrics(tmp_path)), 'median')


@pytest.mark.parametrize('stat,radius', [('mean', 15.2), ('max', 30.), ('p95', 28.)])
def test_ring_models_sample_fixed_radius_any_direction(tmp_path, stat, radius):
    err = AE.error_model(AE.load(_metrics(tmp_path)), stat)
    assert err.radial_mode == 'ring' and err.aim_bias == 0.
    assert err.radius_mm == pytest.approx(radius)
    assert err.source['statistic'] == stat
    rng = np.random.default_rng(0)
    xy = np.array([[s['dx'], s['dy']] for s in (err.sample(rng) for _ in range(200))])
    assert np.hypot(*xy.T) == pytest.approx(radius)
    assert abs(xy.mean(0)).max() < 0.25*radius  # zero-mean: the measured bias is not applied
    assert (xy[:, 0] < 0).any() and (xy[:, 1] < 0).any()


def test_empirical_model_bootstraps_measured_radii(tmp_path):
    err = AE.error_model(AE.load(_metrics(tmp_path)), 'empirical')
    assert err.radial_mode == 'bootstrap'
    rng = np.random.default_rng(1)
    r = {round(float(np.hypot(s['dx'], s['dy'])), 6) for s in (err.sample(rng) for _ in range(100))}
    assert r <= {4., 10., 12., 20., 30.}
    assert len(r) > 1


def test_unmeasured_terms_keep_defaults_and_are_labelled(tmp_path):
    err = AE.error_model(AE.load(_metrics(tmp_path)), 'mean')
    d = T.ErrorModel()
    assert (err.sigma_yaw, err.sigma_tilt, err.clearance, err.sigma_vel) == \
        (d.sigma_yaw, d.sigma_tilt, d.clearance, d.sigma_vel)
    assert set(err.source['unmeasured']) == {'sigma_yaw', 'sigma_tilt', 'clearance', 'sigma_vel'}


def test_gauss_mode_unchanged():
    """The default model draws exactly what it drew before #25 (seeded runs reproduce)."""
    err, rng, ref = T.ErrorModel(), np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(5):
        s = err.sample(rng)
        dx, dy = np.clip(ref.normal(0, 4., 2), -12., 12.)
        expect = dict(dx=float(dx + 3.), dy=float(dy), yaw=float(ref.normal(0, 5.)),
                      roll=float(ref.normal(0, 3.)), pitch=float(ref.normal(0, 3.)),
                      clear=float(ref.uniform(5., 30.)), vel=ref.normal(0, 0.03, 3).tolist())
        assert {k: s[k] for k in expect} == expect


@pytest.mark.parametrize('stat,tol', [('mean', 15.2), ('max', 30.), ('p95', 28.), ('empirical', None)])
def test_required_tolerance(tmp_path, stat, tol):
    s = AE.load(_metrics(tmp_path))
    expect = tol if tol is not None else float(np.percentile(s['radii_mm'], 95))
    assert AE.required_tolerance(s, stat) == pytest.approx(expect)


def test_placement_capture_rule_uses_required_tolerance():
    bundle = cad.build_brick(V0, solid=False)
    # V0 capture_x headroom is U/2 - tooth_w/2 = 10 mm, capture_y is D/2 - gutter_flat = 9.5 mm.
    ok = R.rules(V0, bundle, R.settings_for_tolerance(5.))
    assert ok['capture_x'] >= 0 and ok['capture_y'] >= 0
    bad = R.rules(V0, bundle, R.settings_for_tolerance(14.8))
    assert bad['capture_x'] < 0 and bad['capture_y'] < 0


def test_pickup_funnel_rule_uses_required_tolerance():
    fp = Fx.FixtureParams()
    assert Fx.funnel_capture_margin(fp, 4.) >= 0
    assert Fx.funnel_capture_margin(fp, 14.8) < 0
    assert Fx.funnel_capture_margin(replace(fp, funnel=False), 1.) < 0


def test_capture_margin_from_simulated_sweep():
    bricks = cad.build_bricks(V0)
    # V0 seats over roughly -7.5..+16.5 mm (README); a 5 mm band holds, 14.8 mm does not.
    assert T.capture_margin(V0, bricks, 5.) >= 0
    assert T.capture_margin(V0, bricks, 14.8) < 0


def test_capture_radius_is_measured_from_the_aim():
    rows = [{'dx': 3. + d, 'aim': 3., 'outcome': 'seated' if abs(d) <= 4 else 'jammed'} for d in range(-8, 9)]
    assert T.capture_radius(rows) == 4.


def test_cli_records_error_source(tmp_path):
    cfg_path = tmp_path/'error.json'
    cfg_path.write_text(json.dumps(AE.load(_metrics(tmp_path))))
    args = cli.parser().parse_args(['drop', '--out', str(tmp_path/'o'), '--approach-error', str(cfg_path),
                                    '--error-stat', 'max'])
    cfg = cli._trial_config(args)
    rec = T.as_dict(cfg)['error']
    assert rec['radial_mode'] == 'ring' and rec['radius_mm'] == pytest.approx(30.)
    assert rec['source']['statistic'] == 'max'
    assert rec['source']['config'] == str(cfg_path)
    json.dumps(rec, allow_nan=False)


def test_committed_exp3_r4_config():
    """configs/error-exp3-r4.json holds the runs/exp3-r4 positioning error (git-ignored source)."""
    s = AE.load(AE.CONFIG_DIR/'error-exp3-r4.json')
    assert s['source']['run'] == 'runs/exp3-r4'
    assert s['n'] == len(s['radii_mm']) == 24
    assert s['mean_mm'] == pytest.approx(14.82, abs=0.01)
    assert s['max_mm'] == pytest.approx(38.50, abs=0.01)
    assert math.isclose(max(s['radii_mm']), s['max_mm'])


def _eval_config(tmp_path, radii):
    from brixzle import optimize as O
    cfg = tmp_path/'error.json'
    cfg.write_text(json.dumps(AE.load(_metrics(tmp_path, radii=radii))))
    return O, O.EvalConfig(drop_samples=1, frictions=(0.35,), overhang_n=2, structures=(),
                           approach_error={'config': str(cfg), 'statistic': 'max'})


def test_optimizer_rejects_design_below_measured_tolerance(tmp_path):
    O, ec = _eval_config(tmp_path, [0.005, 0.0148])
    obj, margins, details = O.evaluate(V0, ec)
    assert margins['capture_x'] < 0
    assert details['skipped'] == 'rules'
    assert details['approach_error']['statistic'] == 'max'
    assert 'capture_tolerance' in O.constraint_names()


def test_optimizer_scores_design_above_measured_tolerance(tmp_path):
    O, ec = _eval_config(tmp_path, [0.001, 0.004])
    obj, margins, details = O.evaluate(V0, ec)
    assert margins['capture_x'] >= 0 and margins['capture_tolerance'] >= 0
    assert 'scores' in details, details
    assert details['approach_error']['required_tolerance_mm'] == pytest.approx(4.)
