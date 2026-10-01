from dataclasses import asdict, replace
import json
import pickle

import numpy as np
from brixzle.models import Evaluation, RunConfig
from brixzle.optimization import run_search


def test_nsga_generations_checkpoint_and_completed_resume(monkeypatch, tmp_path):
    """Exercise the real NSGA-III loop with cheap deterministic objective data."""
    def evaluate(parameters, benchmark_targets=None, config=None, seeds=None, output_dir=None):
        p = asdict(parameters)
        f = [parameters.clearance_mm, parameters.pitch_mm, parameters.guide_height_mm,
             parameters.wall_mm, abs(parameters.approach_bias_mm)]
        return Evaluation(str(p), p, f, [0, 0, 0], {}, True)
    monkeypatch.setattr("brixzle.optimization.evaluate_design", evaluate)
    c = RunConfig()
    c = replace(c, search=replace(c.search, population=5, generations=2, workers=1, trials=1))
    result = run_search(c, tmp_path, ("combined",), ("ibeam",), validate=False)
    checkpoint = tmp_path / "combined-ibeam" / "checkpoint.pkl"
    with checkpoint.open("rb") as stream:
        saved = pickle.load(stream)
    assert not saved["algorithm"].has_next()
    assert saved["algorithm"].evaluator.n_eval == 10
    assert "numpy_random_state" in saved and "python_random_state" in saved
    resumed = run_search(c, tmp_path, ("combined",), ("ibeam",), resume=True, validate=False)
    assert resumed == result
    assert json.loads((tmp_path / "combined-ibeam" / "progress.json").read_text())["generation"] == 2
