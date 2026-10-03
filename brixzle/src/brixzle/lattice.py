"""Voxel lattice, brick placements, connectivity graph, and assembly order.

Lattice: voxel i spans x in [iU, (i+1)U]; course k sits at z = k H0. A brick
at (course k, start voxel i0) covers voxels i0 and i0+1 with its centre at
x = (i0+1) U. In 'alternate' lean mode even courses are orientation A and odd
courses B (A rotated 180 degrees about Z), so the alternating tooth lean is
enforced by the lattice; in 'uniform' mode every course is A. In 'ab' mode two
distinct parts alternate (A on even, B on odd courses), both unrotated, so every
tooth trails toward -x; a placement with direction=-1 (building toward -x) uses
the other part rotated 180 degrees, which flips the lean to +x.
Bases are anchored plates covering a voxel range and receive course 0.
"""
from dataclasses import dataclass, field
import math

import networkx as nx
import numpy as np
from shapely import affinity
from shapely.ops import unary_union

from . import profile as pr
from .params import BrickParams


@dataclass(frozen=True)
class Placement:
    course: int
    i0: int
    direction: int = 1  # 'ab' mode: build direction the teeth hook against

    @property
    def orientation(self):
        return 'A' if self.course % 2 == 0 else 'B'

    @property
    def voxels(self):
        return (self.i0, self.i0 + 1)

    @property
    def name(self):
        return f'b{self.course}_{self.i0}' + ('m' if self.direction < 0 else '')


@dataclass
class Structure:
    name: str
    bricks: list
    bases: list = field(default_factory=list)  # [(v0, v1[, direction])] voxel ranges, half-open

    def course(self, k):
        return [b for b in self.bricks if b.course == k]


def orientation(p: BrickParams, b: Placement):
    """'A' (unrotated) or 'B' (rotated 180 degrees about Z) frame for the placement."""
    if p.lean_mode == 'ab':
        return 'B' if b.direction < 0 else 'A'
    return 'A' if p.uniform else b.orientation


def part(p: BrickParams, b: Placement):
    """Which physical part sits at the placement ('A' unless in 'ab' mode)."""
    if p.lean_mode != 'ab':
        return 'A'
    even = b.course % 2 == 0
    return ('A' if even else 'B') if b.direction > 0 else ('B' if even else 'A')


def pose(p: BrickParams, b: Placement):
    """Seated pose (mm, quaternion wxyz) of a placement."""
    pos = np.array([(b.i0 + 1)*p.U, 0., b.course*p.H0])
    quat = np.array([1., 0, 0, 0]) if orientation(p, b) == 'A' else np.array([0., 0, 0, 1])
    return pos, quat


def insertion_axis(p: BrickParams, b: Placement):
    """Unit (x, z) direction the brick moves along during final seating."""
    if p.lean_mode == 'ab':
        # Both parts' teeth trail toward -x; the 180-degree rotation flips it.
        _, d, _, _ = pr.interface(p, mirrored=False)
        return d if orientation(p, b) == 'A' else d*np.array([-1., 1.])
    a_mirrored = not p.uniform
    mirrored = a_mirrored if orientation(p, b) == 'A' else not a_mirrored
    _, d, _, _ = pr.interface(p, mirrored=mirrored)
    return d


def footprint(p: BrickParams, outer, b: Placement, offset=(0., 0.)):
    """``outer`` is one profile polygon or {part: polygon}."""
    g = outer[part(p, b)] if isinstance(outer, dict) else outer
    g = g if orientation(p, b) == 'A' else pr.mirror_x(g)
    pos, _ = pose(p, b)
    return affinity.translate(g, pos[0] + offset[0], pos[2] + offset[1])


def connectivity(structure: Structure):
    """Directed support graph: supporter -> supported, one edge per engaged voxel."""
    g = nx.DiGraph()
    for i, spec in enumerate(structure.bases):
        g.add_node(f'base{i}', kind='base', voxels=tuple(spec[:2]))
    for b in structure.bricks:
        g.add_node(b.name, kind='brick', placement=b)
    occupied = {}
    for b in structure.bricks:
        for v in b.voxels:
            key = (b.course, v)
            if key in occupied:
                raise ValueError(f'Voxel {key} used by {occupied[key]} and {b.name}')
            occupied[key] = b.name
    for b in structure.bricks:
        for v in b.voxels:
            if b.course == 0:
                for i, spec in enumerate(structure.bases):
                    if spec[0] <= v < spec[1]:
                        g.add_edge(f'base{i}', b.name, voxel=v)
            elif (b.course - 1, v) in occupied:
                g.add_edge(occupied[(b.course - 1, v)], b.name, voxel=v)
    return g


def unsupported(graph):
    return [n for n, d in graph.nodes(data=True) if d['kind'] == 'brick' and graph.in_degree(n) == 0]


def _path_clear(p, outer, b, placed_shapes, samples=10):
    """Nominal approach: vertical drop, then the final tooth stroke along the axis."""
    d = insertion_axis(p, b)
    stroke = p.tooth_L + p.slot_extra
    start = -stroke*d
    offsets = [tuple(start*s) for s in np.linspace(0, 1, samples)]
    offsets += [(start[0], start[1] + dz) for dz in np.linspace(0, 3*(p.amplitude + p.H0), samples)]
    swept = unary_union([footprint(p, outer, b, o) for o in offsets]).buffer(-0.05)
    return not any(swept.intersection(s).area > 0.5 for s in placed_shapes)


def assembly_order(p: BrickParams, structure: Structure, outer):
    """Greedy topological order: supporters first, approach path clear of placed bricks.

    Prefers the lowest course, then the most-supported brick. Raises if the
    structure cannot be completed (e.g. a brick that must enter between two
    already placed neighbours against its tooth lean).
    """
    graph = connectivity(structure)
    if unsupported(graph):
        raise ValueError(f'Unsupported bricks: {unsupported(graph)}')
    placed, shapes, order = set(), [], []
    remaining = list(structure.bricks)
    bases = {n for n, d in graph.nodes(data=True) if d['kind'] == 'base'}
    while remaining:
        ready = [b for b in remaining
                 if all(u in placed or u in bases for u in graph.predecessors(b.name))]
        # Fill each course from the side the tooth stroke points to, so the
        # approach side stays open; fall back to any clear brick.
        ready.sort(key=lambda b: (b.course, np.sign(insertion_axis(p, b)[0])*-b.i0))
        for b in ready:
            if _path_clear(p, outer, b, shapes):
                break
        else:
            raise ValueError(f'No clear insertion path for {[b.name for b in ready]}')
        order.append(b)
        placed.add(b.name)
        shapes.append(footprint(p, outer, b))
        remaining.remove(b)
    return order


def tile(occupancy):
    """Running-bond tiling of a boolean grid [course, voxel] into 2-voxel bricks.

    Each run of occupied voxels is covered left to right, starting on the
    course's preferred parity when possible. Returns (placements, leftover voxels).
    """
    occ = np.asarray(occupancy, dtype=bool)
    bricks, leftover = [], []
    for k, row in enumerate(occ):
        v = 0
        n = len(row)
        while v < n:
            if not row[v]:
                v += 1
                continue
            end = v
            while end < n and row[end]:
                end += 1
            run_end, start = end, v
            if (end - v) % 2 == 1:
                # odd run: leave the single voxel on the side that keeps the bond
                if (v + k) % 2 == 1:
                    leftover.append((k, v))
                    start = v + 1
                else:
                    leftover.append((k, end - 1))
                    end -= 1
            for i in range(start, end - 1, 2):
                bricks.append(Placement(k, i))
            v = run_end
    return bricks, leftover


def mass_center_x(p: BrickParams, bricks, com_local_x):
    xs = [(b.i0 + 1)*p.U + (com_local_x if orientation(p, b) == 'A' else -com_local_x) for b in bricks]
    return float(np.mean(xs)) if xs else math.nan
