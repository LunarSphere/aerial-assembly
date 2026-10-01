"""Lattice connectivity, insertion access, and benchmark construction."""
from __future__ import annotations

import math
import networkx as nx
import numpy as np

from .models import AssemblyPlan, Connector, Design, RunConfig, Target, Voxel


def rotate_face(face, q):
    x, y, z = face
    for _ in range(q % 4):
        x, y = -y, x
    return (x, y, z)


def connectors_at(design: Design, voxel: Voxel) -> dict[tuple[int, int, int], Connector]:
    return {rotate_face(c.face, voxel.yaw_quarters): c for c in design.connectors}


def connectivity_graph(design: Design, target: Target) -> nx.Graph:
    graph = nx.Graph()
    cells = {v.cell: v for v in target.voxels}
    if len(cells) != len(target.voxels) or len({v.id for v in target.voxels}) != len(target.voxels):
        raise ValueError("Voxel cells and IDs must be unique")
    for v in target.voxels:
        if len(v.cell) != 3 or any(not isinstance(x, int) for x in v.cell) or v.cell[2] < 0:
            raise ValueError("Voxel cells must be integer triples at nonnegative height")
        graph.add_node(v.id, cell=v.cell, anchored=v.anchored, yaw_quarters=v.yaw_quarters)
    for v in target.voxels:
        for direction, connector in connectors_at(design, v).items():
            other = cells.get(tuple(a+b for a, b in zip(v.cell, direction)))
            if other is None:
                continue
            mate = connectors_at(design, other).get(tuple(-x for x in direction))
            if mate and mate.gender != connector.gender and mate.kind == connector.kind:
                graph.add_edge(v.id, other.id, connectors={v.id: connector.name, other.id: mate.name},
                               kind=connector.kind)
    return graph


def _accessible(design: Design, candidate: Voxel, placed: list[Voxel], config: RunConfig) -> bool:
    """Conservative voxel-level access filter, followed by physical collision trials."""
    x, y, z = candidate.cell
    # Vertical insertion cannot cross a previously placed voxel above the seat.
    if any(v.cell[:2] == (x, y) and v.cell[2] > z for v in placed):
        return False
    if config.simulation.manipulation_mode == "floating":
        return True
    # The fork bar and withdrawal sweep occupy the rear ray at bearing height.
    rear = rotate_face((0, -1, 0), candidate.yaw_quarters)
    for v in placed:
        dx, dy, dz = (v.cell[i]-candidate.cell[i] for i in range(3))
        if dz == 0 and dx*rear[1]-dy*rear[0] == 0 and dx*rear[0]+dy*rear[1] > 0:
            return False
    return True


def plan_assembly(design: Design, voxel_target: Target, config: RunConfig | None = None) -> AssemblyPlan:
    config = config or RunConfig()
    graph = connectivity_graph(design, voxel_target)
    anchors = [v for v in voxel_target.voxels if v.anchored]
    if any(v.cell[2] != 0 for v in anchors):
        return AssemblyPlan(voxel_target, [], [], {}, False, "Fixtures must sit on the stable base")
    if voxel_target.name != "pickup" and not anchors:
        return AssemblyPlan(voxel_target, [], [], {}, False, "Target has no anchored starting component")
    remaining = [v for v in voxel_target.voxels if not v.anchored]
    visited = set()
    exhausted = False
    def search(placed, pending, order, dependencies):
        nonlocal exhausted
        if not pending:
            return order, dependencies
        state = frozenset(v.id for v in placed)
        if state in visited:
            return None
        if len(visited) >= 20_000:
            exhausted = True
            return None
        visited.add(state)
        candidates = []
        for v in pending:
            supports = [p.id for p in placed if graph.has_edge(v.id, p.id) and p.cell[2] <= v.cell[2]]
            if voxel_target.name == "pickup":
                supports = ["table"]
            if supports and _accessible(design, v, placed, config):
                candidates.append((v.cell[2], -len(supports), v.cell, v.id, v, supports))
        for candidate in sorted(candidates, key=lambda a: a[:4]):
            *_, selected, supports = candidate
            result = search(placed+[selected], [v for v in pending if v.id != selected.id],
                            order+[selected.id], {**dependencies, selected.id: supports})
            if result is not None:
                return result
        return None
    selected_plan = search(anchors, remaining, [], {})
    if selected_plan is None:
        reason = ("Planning search budget exhausted (20000 states)" if exhausted else
                  "No compatible supported placement with clear insertion and withdrawal")
        return AssemblyPlan(voxel_target, [], [], {}, False, reason)
    order, dependencies = selected_plan
    connections = [(a, b, data["connectors"][a], data["connectors"][b]) for a, b, data in graph.edges(data=True)]
    return AssemblyPlan(voxel_target, order, connections, dependencies)


def benchmark_suite(include_diagnostics: bool = True, manipulation_mode: str = "fork") -> list[Target]:
    def make(name, cells, anchor_cells, mandatory=True):
        return Target(name, [Voxel(f"b{i}", tuple(c), 0, tuple(c) in anchor_cells)
                             for i, c in enumerate(cells)], mandatory)
    result = [
        make("pickup", [(0, 0, 0)], set()),
        make("tower", [(0, 0, z) for z in range(8)], {(0, 0, 0)}),
        make("cantilever", [(x, 0, 0) for x in range(4)], {(0, 0, 0)}),
        make("bridge", [(x, 0, 0) for x in range(5)], {(0, 0, 0), (4, 0, 0)}),
    ]
    if manipulation_mode == "floating":
        result = result[1:]  # Pickup cannot be validated by an external transport proxy.
    if include_diagnostics:
        result.extend([
            make("arch", [(0, 0, 0), (4, 0, 0), (0, 0, 1), (4, 0, 1),
                          (1, 0, 1), (3, 0, 1), (1, 0, 2), (3, 0, 2), (2, 0, 2)],
                 {(0, 0, 0), (4, 0, 0)}, False),
            make("dome", [(x, y, 0) for x in range(3) for y in range(3) if x != 1 or y != 1]
                 + [(1, 0, 1), (1, 2, 1), (0, 1, 1), (2, 1, 1), (1, 1, 1)],
                 {(x, y, 0) for x in range(3) for y in range(3) if x != 1 or y != 1}, False),
        ])
    return result


def target_from_dict(raw: dict) -> Target:
    return Target(raw["name"], [Voxel(v["id"], tuple(v["cell"]), v.get("yaw_quarters", 0),
                                      v.get("anchored", False)) for v in raw["voxels"]],
                  raw.get("mandatory", True))


def voxel_position(voxel: Voxel, pitch_mm: float, platform_height_m: float = 0.0) -> np.ndarray:
    return np.array([voxel.cell[0], voxel.cell[1], voxel.cell[2]+0.5]) * pitch_mm/1000 + np.array([0, 0, platform_height_m])
