"""DF lattice: placements on a square voxel grid, poses, support graph and assembly order.

A placement covers n voxels along X or Y from cell (i0, j0) in course k. Its feet sit in the
first and last cell. The base floor is z = 0, so a course-k brick's frame (rail bottom) is at
foot_L + k H0. A brick is supported by whatever covers its foot cells one course down (the base
for course 0). Every insertion is a vertical drop into the receivers below, so there are no
hooks to order around: supporters first, then course by course.
"""
from dataclasses import dataclass
import math

import networkx as nx
import numpy as np

from .params import DFParams


@dataclass(frozen=True)
class Placement:
    course: int
    i0: int
    j0: int = 0
    axis: str = 'x'

    def __post_init__(self):
        if self.axis not in ('x', 'y'):
            raise ValueError("axis must be 'x' or 'y'")

    @property
    def name(self):
        return f'b{self.course}_{self.axis}{self.i0}_{self.j0}'


def cells(p: DFParams, b: Placement):
    if b.axis == 'x':
        return [(b.i0 + k, b.j0) for k in range(p.n_vox)]
    return [(b.i0, b.j0 + k) for k in range(p.n_vox)]


def feet_cells(p: DFParams, b: Placement):
    c = cells(p, b)
    return [c[0], c[-1]]


def yaw(b: Placement):
    return 0. if b.axis == 'x' else 90.


def pose(p: DFParams, b: Placement):
    """Brick frame position (mm) and wxyz quaternion."""
    xy = np.mean(cells(p, b), axis=0)*p.U
    a = math.radians(yaw(b))/2
    return np.array([xy[0], xy[1], p.foot_L + b.course*p.H0]), np.array([math.cos(a), 0., 0., math.sin(a)])


def receivers(p: DFParams, b: Placement):
    """[(cell, kind)] on top of brick b; slot walls follow the brick's own Y."""
    return [(c, p.receiver_kind(k)) for k, c in enumerate(cells(p, b))]


def owners(p: DFParams, structure):
    out = {}
    for b in structure.bricks:
        for c in cells(p, b):
            key = (b.course, c)
            if key in out:
                raise ValueError(f'{b.name} overlaps {out[key].name} at {c}')
            out[key] = b
    return out


def validate(p: DFParams, structure):
    own = owners(p, structure)
    for b in structure.bricks:
        if b.course and not any((b.course - 1, c) in own for c in feet_cells(p, b)):
            raise ValueError(f'{b.name} has no supported foot')
    return own


def connectivity(p: DFParams, structure):
    """Directed support graph: supporter -> brick, one edge per supported foot."""
    own = validate(p, structure)
    g = nx.MultiDiGraph()
    for b in structure.bricks:
        g.add_node(b.name, placement=b)
        if b.course == 0:
            continue
        for c in feet_cells(p, b):
            if (b.course - 1, c) in own:
                g.add_edge(own[(b.course - 1, c)].name, b.name)
    return g


def supporter(p: DFParams, structure, b: Placement):
    """The brick under b's first supported foot (None on the base)."""
    own = owners(p, structure)
    for c in feet_cells(p, b):
        if (b.course - 1, c) in own:
            return own[(b.course - 1, c)]
    return None


def assembly_order(p: DFParams, structure):
    """Course by course; a structure's key brick (if any) goes last in its course."""
    validate(p, structure)
    key = getattr(structure, 'key', None)
    return sorted(structure.bricks, key=lambda b: (b.course, b.name == key, b.axis, b.j0, b.i0))


def base_receivers(p: DFParams, structure, extra=()):
    """[(x, y, kind, yaw)] base receivers under course 0, matching each brick's top pattern.

    ``extra`` adds pocket receivers at further cells (e.g. neighbours a missed drop can land in).
    """
    out = {}
    for b in structure.bricks:
        if b.course == 0:
            for c, kind in receivers(p, b):
                out[c] = (kind, yaw(b))
    for c in extra:
        out.setdefault(c, ('pocket', 0.))
    return [(float(c[0]*p.U), float(c[1]*p.U), kind, yw) for c, (kind, yw) in sorted(out.items())]
