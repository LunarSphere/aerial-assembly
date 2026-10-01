from dataclasses import replace
import pytest

from brixzle.assembly import benchmark_suite, connectivity_graph, plan_assembly
from brixzle.geometry import generate_design
from brixzle.models import DesignParameters, RunConfig, Target, Voxel


@pytest.fixture(scope="module")
def design():
    return generate_design(DesignParameters())


def test_mandatory_plans_have_exact_lattice_connections(design):
    for target in benchmark_suite(False):
        plan = plan_assembly(design, target)
        assert plan.feasible, plan.reason
        assert set(plan.order) == {v.id for v in target.voxels if not v.anchored}
        by_id = {v.id: v for v in target.voxels}
        for a, b, _, _ in plan.connections:
            assert sum(abs(x-y) for x, y in zip(by_id[a].cell, by_id[b].cell)) == 1
        done = {v.id for v in target.voxels if v.anchored} | {"table"}
        for brick in plan.order:
            assert set(plan.dependencies[brick]) <= done
            done.add(brick)


def test_topology_limit_is_explicit(design):
    pegs = replace(design, connectors=[c for c in design.connectors if c.kind == "peg"])
    assert not plan_assembly(pegs, benchmark_suite(False)[2]).feasible
    rails = replace(design, connectors=[c for c in design.connectors if c.kind == "rail"])
    assert not plan_assembly(rails, benchmark_suite(False)[1]).feasible


def test_rear_fork_withdrawal_cannot_cross_neighbor(design):
    target = Target("blocked", [Voxel("base", (0, 0, 0), anchored=True), Voxel("next", (0, 1, 0))])
    assert connectivity_graph(design, target).has_edge("base", "next")
    c = RunConfig()
    c = replace(c, simulation=replace(c.simulation, manipulation_mode="fork"))
    assert not plan_assembly(design, target, c).feasible
    assert plan_assembly(design, target).feasible


def test_rotated_geometry_preserves_compatibility(design):
    target = Target("rotated", [Voxel("base", (0, 0, 0), 1, True), Voxel("next", (0, 1, 0), 1)])
    assert plan_assembly(design, target).feasible


def test_disconnected_and_duplicate_voxels(design):
    assert not plan_assembly(design, Target("gap", [Voxel("a", (0, 0, 0), anchored=True), Voxel("b", (2, 0, 0))])).feasible
    with pytest.raises(ValueError):
        connectivity_graph(design, Target("duplicates", [Voxel("a", (0, 0, 0)), Voxel("b", (0, 0, 0))]))


def test_planner_backtracks_to_preserve_future_fork_access(design):
    target = Target("ordering", [Voxel("anchor0", (1, 0, 0), anchored=True),
        Voxel("anchor1", (1, 1, 0), anchored=True), Voxel("rear", (0, 0, 0)), Voxel("front", (0, 1, 0))])
    c = RunConfig()
    c = replace(c, simulation=replace(c.simulation, manipulation_mode="fork"))
    plan = plan_assembly(design, target, c)
    assert plan.feasible
    assert plan.order == ["front", "rear"]
