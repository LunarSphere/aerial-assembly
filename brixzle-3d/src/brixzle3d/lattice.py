"""3D voxel lattice: placements, support graph, assembly order, voxel tiling.

A placement is (course k, origin cell i, j, rotation r in quarter turns). The
brick frame origin sits at lattice corner (iU, jU, kH), and its cells rotate
about that corner. Every cell interface is identical and 4-fold symmetric, so
any rotation nests on any occupied cell below.
"""
from dataclasses import dataclass, field
import math

import networkx as nx
import numpy as np

from .geometry import rotate_cells
from .params import BrickParams


@dataclass(frozen=True)
class Placement:
    course: int
    i: int
    j: int
    r: int = 0

    @property
    def name(self):
        return f'b{self.course}_{self.i}_{self.j}_r{self.r}'


@dataclass
class Structure:
    name: str
    bricks: list
    bases: list = field(default_factory=list)  # list of cell sets [(i, j), ...]


def cells(p: BrickParams, b: Placement):
    return [(b.i + ci, b.j + cj) for ci, cj in rotate_cells(p.cells, b.r)]


def pose(p: BrickParams, b: Placement):
    """Seated pose: position (mm) and quaternion (w, x, y, z)."""
    pos = np.array([b.i*p.U, b.j*p.U, (b.course + 1)*p.H])  # anchored plates top out at z = H
    half = math.pi/4*(b.r % 4)
    return pos, np.array([math.cos(half), 0., 0., math.sin(half)])


def connectivity(p: BrickParams, structure: Structure):
    """Directed support graph supporter -> supported; one edge per seated cell."""
    g = nx.DiGraph()
    base_cells = {}
    for n, cs in enumerate(structure.bases):
        g.add_node(f'base{n}', kind='base')
        for c in cs:
            base_cells[c] = f'base{n}'
    occupied = {}
    for b in structure.bricks:
        g.add_node(b.name, kind='brick', placement=b)
        for c in cells(p, b):
            key = (b.course, *c)
            if key in occupied:
                raise ValueError(f'Cell {key} used by {occupied[key]} and {b.name}')
            occupied[key] = b.name
    for b in structure.bricks:
        for c in cells(p, b):
            below = base_cells.get(c) if b.course == 0 else occupied.get((b.course - 1, *c))
            if below is not None:
                if g.has_edge(below, b.name):
                    g[below][b.name]['cells'] += 1
                else:
                    g.add_edge(below, b.name, cells=1)
    return g


def unsupported(graph):
    return [n for n, d in graph.nodes(data=True) if d['kind'] == 'brick' and graph.in_degree(n) == 0]


def support_fraction(p: BrickParams, graph, b: Placement):
    return sum(d['cells'] for _, _, d in graph.in_edges(b.name, data=True))/len(p.cells)


def assembly_order(p: BrickParams, structure: Structure):
    """Supporters first; within a course, most-supported first (vertical drops never collide)."""
    g = connectivity(p, structure)
    if unsupported(g):
        raise ValueError(f'Unsupported bricks: {unsupported(g)}')
    placed, order = set(), []
    remaining = list(structure.bricks)
    while remaining:
        ready = [b for b in remaining if all(u in placed or g.nodes[u]['kind'] == 'base'
                                             for u in g.predecessors(b.name))]
        if not ready:
            raise ValueError('Cyclic support')
        ready.sort(key=lambda b: (b.course, -support_fraction(p, g, b), b.i, b.j))
        b = ready[0]
        order.append(b)
        placed.add(b.name)
        remaining.remove(b)
    return order


def tipping_bricks(p: BrickParams, structure: Structure, com_local):
    """Bricks whose own COM projects outside the hull of their seated cells.

    With vertical pegs a joint carries no tension, so such a brick tips on
    placement unless something above pins it first.
    """
    from scipy.spatial import Delaunay
    g = connectivity(p, structure)
    occupied = {(b.course, *c) for b in structure.bricks for c in cells(p, b)}
    base = {c for cs in structure.bases for c in cs}
    bad = []
    for b in structure.bricks:
        pos, q = pose(p, b)
        ang = 2*math.atan2(q[3], q[0])
        c, s = math.cos(ang), math.sin(ang)
        com = pos[:2] + np.array([c*com_local[0] - s*com_local[1], s*com_local[0] + c*com_local[1]])
        seated = [cl for cl in cells(p, b) if (cl in base if b.course == 0 else (b.course - 1, *cl) in occupied)]
        corners = np.array([((x + dx)*p.U, (y + dy)*p.U) for x, y in seated for dx in (0, 1) for dy in (0, 1)])
        if len(seated) == 0 or Delaunay(corners).find_simplex(com) < 0:
            bad.append(b.name)
    return bad


def _com_world(p, b, com_local):
    pos, q = pose(p, b)
    ang = 2*math.atan2(q[3], q[0])
    c, s_ = math.cos(ang), math.sin(ang)
    return pos[:2] + np.array([c*com_local[0] - s_*com_local[1], s_*com_local[0] + c*com_local[1]])


def _balanced(p, b, seated_cells, com_local, margin=0.1):
    """COM inside the union of seated cell squares (shrunk by margin*U)."""
    x, y = _com_world(p, b, com_local)/p.U
    return any(cx + margin <= x <= cx + 1 - margin and cy + margin <= y <= cy + 1 - margin
               for cx, cy in seated_cells)


def tile_course(p: BrickParams, k, free, below, com_local, limit=20000):
    """Exact cover of ``free`` cells by rotated bricks, each balanced on cells in ``below``.

    Depth-first over the lowest uncovered cell; returns placements or None.
    """
    free = set(free)
    shapes = [(r, rotate_cells(p.cells, r)) for r in range(4)]
    budget = [limit]

    def dfs(remaining, chosen):
        if not remaining:
            return chosen
        budget[0] -= 1
        if budget[0] < 0:
            return None
        c = min(remaining)
        for r, shape in shapes:
            for anchor in shape:
                oi, oj = c[0] - anchor[0], c[1] - anchor[1]
                cs = [(oi + a, oj + b) for a, b in shape]
                if not all(x in remaining for x in cs):
                    continue
                b = Placement(k, oi, oj, r)
                seated = [x for x in cs if x in below]
                if not seated or not _balanced(p, b, seated, com_local):
                    continue
                out = dfs(remaining - set(cs), chosen + [b])
                if out is not None:
                    return out
        return None
    return dfs(frozenset(free), [])


def tile(p: BrickParams, occupancy, base_cells, com_local, name='tiled'):
    """Tile a voxel occupancy {course: set of (i, j)} course by course into a Structure.

    Each brick must balance on its own seated cells (no joint tension assumed).
    Raises if a course cannot be covered.
    """
    bricks, below = [], set(base_cells)
    for k in sorted(occupancy):
        course = tile_course(p, k, occupancy[k], below, com_local)
        if course is None:
            raise ValueError(f'Course {k} cannot be tiled with balanced bricks')
        bricks += course
        below = set(occupancy[k])
    return Structure(name, bricks, [sorted(base_cells)])
