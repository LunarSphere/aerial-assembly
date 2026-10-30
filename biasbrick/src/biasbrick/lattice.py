"""biasbrick lattice: placements on a row of sites, poses, support and assembly order.

Sites sit ``pitch`` apart along x. A placement covers sites i0 and i0 + 1 in course k (its two
sockets); its frame is at x = (i0 + 1/2) pitch, y = 0, z = k H0, and its part alternates A (even
courses) / B (odd), as brixzle's pair. A socket is supported by the socket one course down at the
same site; a brick needs at least one supported socket (a staircase overhang hangs on one hooked
joint). Bricks run along x only: a brick along y would turn its sawtooth and Y nest by 90 degrees,
and no socket receives a crossing brick, so there are no corners (no ring).
"""
from dataclasses import dataclass

import numpy as np

from .params import BBParams


@dataclass(frozen=True)
class Placement:
    course: int
    i0: int

    @property
    def name(self):
        return f'b{self.course}_{self.i0}'

    @property
    def part(self):
        return 'A' if self.course % 2 == 0 else 'B'

    @property
    def sites(self):
        return (self.i0, self.i0 + 1)


IDENTITY = np.array([1., 0., 0., 0.])


def pose(p: BBParams, b: Placement):
    """Brick frame position (mm) and wxyz quaternion at its seat."""
    return np.array([(b.i0 + 0.5)*p.pitch, 0., b.course*p.H0]), IDENTITY


def owners(structure):
    out = {}
    for b in structure.bricks:
        for s in b.sites:
            key = (b.course, s)
            if key in out:
                raise ValueError(f'{b.name} overlaps {out[key].name} at site {s}')
            out[key] = b
    return out


def validate(structure):
    own = owners(structure)
    for b in structure.bricks:
        if b.course and not any((b.course - 1, s) in own for s in b.sites):
            raise ValueError(f'{b.name} has no supported socket')
    return own


def supported_sockets(structure, b: Placement):
    own = owners(structure)
    return [s for s in b.sites if (b.course - 1, s) in own]


def supporter(structure, b: Placement):
    """The brick under b's first supported socket (None on the base)."""
    own = owners(structure)
    for s in b.sites:
        if (b.course - 1, s) in own:
            return own[(b.course - 1, s)]
    return None


def assembly_order(structure):
    """Course by course, left to right; a structure's key brick (if any) goes last in its course."""
    validate(structure)
    key = getattr(structure, 'key', None)
    return sorted(structure.bricks, key=lambda b: (b.course, b.name == key, b.i0))


def base_sites(p: BBParams, structure):
    """World x of every site under course 0 (the base receives part A there)."""
    return sorted({s*p.pitch for b in structure.bricks if b.course == 0 for s in b.sites})
