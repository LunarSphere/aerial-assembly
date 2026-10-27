"""Trial structures, the same set as brixzle: tower, wall, bridge, overhang +/-, and a 3D ring."""
from dataclasses import dataclass, field

from .lattice import Placement
from .params import DFParams


@dataclass
class Structure:
    name: str
    bricks: list = field(default_factory=list)
    key: str = None  # brick that must go last (bridge span)


def tower(p: DFParams, h=6):
    return Structure('tower', [Placement(k, 0) for k in range(h)])


def wall(p: DFParams, width=3, courses=4):
    """Running bond: odd courses shift one voxel, so every inner joint is straddled by two feet.

    Odd courses step out one voxel at the +x end; that brick keeps its COM over the one below.
    """
    n = p.n_vox
    bricks = []
    for k in range(courses):
        bricks += [Placement(k, m*n + (k % 2)) for m in range(width)]
    return Structure('wall', bricks)


def bridge(p: DFParams, pier_h=2):
    """Two piers one voxel apart; the span's feet rest in the facing end receivers, placed last."""
    n = p.n_vox
    right = 2*n - 2  # the span (n voxels) reaches from the left pier's last cell to it
    bricks = [Placement(k, 0) for k in range(pier_h)] + [Placement(k, right) for k in range(pier_h)]
    span = Placement(pier_h, n - 1)
    if span.i0 + n - 1 != right:
        raise ValueError('span feet must land on both piers')
    return Structure('bridge', bricks + [span], key=span.name)


def overhang(p: DFParams, count=12, direction=+1):
    """Staircase: each brick shifts one voxel; its trailing foot rests on the middle of the one below."""
    return Structure(f'overhang{"+" if direction > 0 else "-"}',
                     [Placement(k, direction*k) for k in range(count)])


def ring(p: DFParams, side=7, courses=4):
    """Square ring of side ``side`` cells, pinwheel corners alternating by course (masonry corner bond).

    Each wall of a course is side - 1 cells long, so side - 1 must be a multiple of n_vox.
    """
    n, L = p.n_vox, side - 1
    if L % n:
        raise ValueError('side - 1 must be a multiple of n_vox')
    bricks = []
    for k in range(courses):
        lo, hi = (0, 1) if k % 2 == 0 else (1, 0)
        for m in range(0, L, n):
            bricks += [Placement(k, lo + m, 0, 'x'),       # south
                       Placement(k, L, lo + m, 'y'),       # east
                       Placement(k, hi + m, L, 'x'),       # north
                       Placement(k, 0, hi + m, 'y')]       # west
    return Structure('ring', bricks)


def catalog(p: DFParams):
    return {'tower': lambda: tower(p, 6), 'wall': lambda: wall(p, 3, 4), 'bridge': lambda: bridge(p, 2),
            'overhang+': lambda: overhang(p, 12, +1), 'overhang-': lambda: overhang(p, 12, -1),
            'ring': lambda: ring(p, 7, 4)}
