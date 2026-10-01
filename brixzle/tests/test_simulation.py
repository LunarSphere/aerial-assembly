from dataclasses import replace
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from brixzle.assembly import benchmark_suite, plan_assembly
from brixzle.geometry import generate_design
from brixzle.models import DesignParameters, RunConfig, Target, UncertaintyProfile, Voxel
from brixzle.simulation import build_scene, run_trial, sample_disturbance


@pytest.fixture(scope="module")
def nominal():
    c = RunConfig(uncertainty=UncertaintyProfile(lateral_mm=0, vertical_mm=0, yaw_deg=0, tilt_deg=0,
        clearance_error_mm=0, mass_fraction=0, tracking_mm=0))
    c = replace(c, simulation=replace(c.simulation, manipulation_mode="fork"))
    return generate_design(DesignParameters(), c), c


@pytest.mark.integration
def test_scene_has_no_artificial_connections_or_overhang_floor(nominal):
    d, c = nominal
    plan = plan_assembly(d, benchmark_suite(False)[2], c)
    xml, mapping = build_scene(d, plan, c, sample_disturbance(c.uncertainty, 1234, len(plan.order)))
    root = ET.fromstring(xml)
    assert root.find("equality") is None
    assert len(root.findall(".//freejoint")) == 4  # three bricks and the fork
    assert len(root.findall(".//geom[@name='fixture0']")) == 1
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Explicit mass and inertia use the CAD result, never the sum of hull masses.
    assert model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "brick1")] == pytest.approx(d.mass_g/1000)


@pytest.mark.integration
def test_nominal_passive_pickup_and_release(nominal, tmp_path):
    d, c = nominal
    result = run_trial(d, plan_assembly(d, benchmark_suite(False)[0], c), seed=1234, config=c, output_dir=tmp_path)
    assert result.success, result
    assert result.peak_force_n > d.mass_g/1000*9.81
    assert result.placements[0]["errors"]["b0"]["position_mm"] <= c.simulation.position_tolerance_mm
    assert (tmp_path / "scene.xml").exists()
    assert (tmp_path / "final_state.json").exists()


@pytest.mark.integration
def test_no_lift_attachment_hides_lost_payload(nominal):
    d, c = nominal
    # Move the pickup channels beyond the fork without changing geometry. A
    # fake attachment would still lift this brick; physical contact cannot.
    impossible = replace(d, pickup_surface_mm=100)
    result = run_trial(impossible, plan_assembly(impossible, benchmark_suite(False)[0], c), seed=1234, config=c)
    assert not result.success
    assert result.reason == "missed_pickup"


@pytest.mark.integration
def test_adjacent_rail_and_peg_seats_do_not_interpenetrate(nominal):
    d, c = nominal
    for offset in ((1, 0, 0), (0, 0, 1)):
        target = Target("mate", [Voxel("a", (0, 0, 0), anchored=True), Voxel("b", offset)])
        plan = plan_assembly(d, target, c)
        scene, mapping = build_scene(d, plan, c, sample_disturbance(c.uncertainty, 0, 1))
        model = mujoco.MjModel.from_xml_string(scene)
        data = mujoco.MjData(model)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "brick1_joint")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr+3] = mapping["targets"]["b"]
        mujoco.mj_forward(model, data)
        assert all(data.contact[i].dist >= -1e-6 for i in range(data.ncon))


@pytest.mark.integration
def test_floating_seating_removes_virtual_carrier_and_omits_pickup(nominal, tmp_path):
    d, c = nominal
    c = replace(c, simulation=replace(c.simulation, manipulation_mode="floating"))
    assert "pickup" not in [t.name for t in benchmark_suite(False, "floating")]
    target = Target("two_high", [Voxel("base", (0, 0, 0), anchored=True), Voxel("next", (0, 0, 1))])
    result = run_trial(d, plan_assembly(d, target, c), seed=1234, config=c, output_dir=tmp_path)
    assert result.success, result
    assert result.manipulation_mode == "floating"
    assert not result.transport_simulated
    state = json.loads((tmp_path / "final_state.json").read_text())
    assert np.count_nonzero(state["xfrc_applied"]) == 0
    assert np.count_nonzero(state["body_gravcomp"]) == 0
    root = ET.fromstring((tmp_path / "scene.xml").read_text())
    assert len(root.findall(".//freejoint")) == 1
    assert root.find("equality") is None


@pytest.mark.integration
def test_future_float_placeholders_are_excluded_and_prior_bricks_remain_free(nominal, tmp_path):
    d, c = nominal
    c = replace(c, simulation=replace(c.simulation, manipulation_mode="floating"))
    target = Target("three_high", [Voxel("base", (0, 0, 0), anchored=True),
        Voxel("next", (0, 0, 1)), Voxel("last", (0, 0, 2))])
    result = run_trial(d, plan_assembly(d, target, c), seed=1234, config=c, output_dir=tmp_path)
    assert result.success, result
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    state = json.loads((tmp_path / "final_state.json").read_text())
    for name in ("brick1", "brick2"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert state["body_gravcomp"][bid] == 0
        assert state["body_contype"][bid] == 1
    assert len(result.placements) == 2
    assert "next" in result.placements[-1]["errors"]


def test_engine_warning_is_captured_rejected_and_callback_restored(nominal, monkeypatch, tmp_path):
    from brixzle.models import TrialResult
    d, c = nominal
    c = replace(c, simulation=replace(c.simulation, manipulation_mode="floating"))
    old = mujoco.get_mju_user_warning()
    def fake(*args):
        mujoco.get_mju_user_warning()("synthetic Newton linesearch warning")
        return TrialResult("tower", 1234, True, None, 0, 1, 0, d.mass_g)
    monkeypatch.setattr("brixzle.simulation._floating_trial", fake)
    result = run_trial(d, plan_assembly(d, benchmark_suite(False)[1], c), seed=1234, config=c, output_dir=tmp_path)
    assert not result.success and result.numerical_warning
    assert result.disturbance["solver_warnings"] == ["synthetic Newton linesearch warning"]
    assert mujoco.get_mju_user_warning() == old
    assert json.loads((tmp_path / "trial.json").read_text())["numerical_warning"]


def test_sampled_actual_mass_is_a_hard_limit(nominal, monkeypatch):
    from brixzle.simulation import sample_disturbance
    d, c = nominal
    c = replace(c, simulation=replace(c.simulation, manipulation_mode="floating"),
                printing=replace(c.printing, max_mass_g=12),
                uncertainty=replace(c.uncertainty, mass_fraction=0.05))
    disturbance = sample_disturbance(c.uncertainty, 1234, 1)
    disturbance["mass_scale"] = 1.05
    monkeypatch.setattr("brixzle.simulation.sample_disturbance", lambda *a: disturbance)
    target = Target("two_high", [Voxel("base", (0, 0, 0), anchored=True), Voxel("next", (0, 0, 1))])
    result = run_trial(d, plan_assembly(d, target, c), config=c, manufactured_design=d)
    assert not result.success
    assert result.reason == "printed_mass_limit"
    assert result.mass_g > c.printing.max_mass_g
