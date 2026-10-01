from dataclasses import replace
import pytest

from brixzle.evaluation import reliability_lower_bound, trial_seeds
from brixzle.models import Evaluation, RunConfig
from brixzle.optimization import pareto_indices, weighted_order
from brixzle.simulation import sample_disturbance


def test_exact_reliability_lower_bound():
    assert reliability_lower_bound(0, 200) == 0
    assert reliability_lower_bound(200, 200) == pytest.approx(0.05**(1/200))
    assert reliability_lower_bound(58, 58) < 0.95
    assert reliability_lower_bound(59, 59) >= 0.95
    assert reliability_lower_bound(195, 200) < 0.95
    with pytest.raises(ValueError):
        reliability_lower_bound(2, 1)


def test_held_out_seeds_and_disturbances():
    c = RunConfig()
    search, held = trial_seeds(12, 100), trial_seeds(12, 100, "validate")
    assert not set(search) & set(held)
    assert sample_disturbance(c.uncertainty, search[0], 3) == sample_disturbance(c.uncertainty, search[0], 3)
    assert sample_disturbance(c.uncertainty, search[0], 0)["clearance_error_mm"] == sample_disturbance(c.uncertainty, search[0], 3)["clearance_error_mm"]


def test_pareto_uses_only_feasible_candidates_and_preferences_are_optional():
    def e(f, feasible=True):
        return Evaluation("h", {}, f, [0, 0, 0] if feasible else [1, 0, 0], {}, feasible)
    values = [e([0, 2, 3, 10, 0]), e([0.2, 1, 2, 5, 0]), e([0.5, 3, 4, 12, 1]), e([0, 0, 0, 0, 0], False)]
    assert set(pareto_indices(values)) == {0, 1}
    assert weighted_order(values[:3], [1, 0, 0, 0, 0])[0] == 0
    assert weighted_order(values[:3], [0, 0, 0, 1, 0])[0] == 1


def test_config_rejects_bad_error_bounds_and_solver_settings():
    c = RunConfig()
    with pytest.raises(ValueError):
        replace(c, uncertainty=replace(c.uncertainty, lateral_mm=-1)).validate()
    with pytest.raises(ValueError):
        replace(c, simulation=replace(c.simulation, timestep_s=0.1)).validate()
    with pytest.raises(ValueError):
        replace(c, search=replace(c.search, population=5.5)).validate()


def test_configuration_typos_have_actionable_errors(tmp_path):
    from brixzle.models import load_config
    p = tmp_path / "config.json"
    p.write_text('{"simulation": {"timestep": 0.001}}')
    with pytest.raises(ValueError, match="Unknown fields"):
        load_config(p)
    p.write_text('{"simulation": {"hold_s": "five"}}')
    with pytest.raises(ValueError, match="hold_s"):
        load_config(p)


def test_failures_are_included_capped_and_cached(monkeypatch, tmp_path):
    from brixzle.assembly import benchmark_suite
    from brixzle.evaluation import evaluate_design
    from brixzle.geometry import generate_design
    from brixzle.models import DesignParameters, TrialResult
    design = generate_design(DesignParameters())
    calls = []
    def trial(d, plan, seed=0, **kwargs):
        calls.append(seed)
        return TrialResult(plan.target.name, seed, seed == 1, None if seed == 1 else "jam",
                           2, 43 if seed == 1 else 0.1, 0, d.mass_g)
    monkeypatch.setattr("brixzle.evaluation.generate_design", lambda *a, **k: design)
    monkeypatch.setattr("brixzle.evaluation.run_trial", trial)
    target = benchmark_suite(False)[1]
    evaluation = evaluate_design(DesignParameters(), [target], seeds=[1, 2], output_dir=tmp_path)
    summary = evaluation.benchmarks["tower"]
    assert summary["success_rate"] == 0.5
    assert summary["mean_capped_time_s"] == (43+7*35)/2
    assert summary["failures"] == {"jam": 1}
    assert evaluation.objectives[0] == 0.5
    assert not summary["meets_reliability_target"]
    cached = evaluate_design(DesignParameters(), [target], seeds=[1, 2], output_dir=tmp_path)
    assert cached == evaluation
    assert calls == [1, 2]
