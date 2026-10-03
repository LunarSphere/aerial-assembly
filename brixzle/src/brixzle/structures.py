"""Trial structures on the lattice.

Anchored bases extend one voxel past the first course on every side that is
not deliberately cantilevered, so edge bricks get the same hopper catchment
as interior ones.
"""
from .lattice import Placement, Structure


def tower(height):
    """Stack bond column: every brick fully seated on two teeth."""
    return Structure(f'tower{height}', [Placement(k, 0) for k in range(height)], bases=[(-1, 3)])


def overhang(n, direction=1):
    """Single-brick staircase stepping one voxel per course past a 2-voxel base.

    Each brick after the first engages its supporter with one tooth only; its
    centre of mass sits over the supporter's edge, so only the interlock holds it.
    """
    if direction > 0:
        bricks, base = [Placement(k, k) for k in range(n)], (-1, 2)
    else:
        bricks, base = [Placement(k, -k, -1) for k in range(n)], (0, 3, -1)
    return Structure(f'overhang{"+" if direction > 0 else "-"}{n}', bricks, bases=[base])


def wall(width_bricks, height):
    """Running-bond wall; odd courses shift by one voxel."""
    bricks = []
    for k in range(height):
        shift = k % 2
        count = width_bricks - shift
        bricks += [Placement(k, shift + 2*j) for j in range(count)]
    return Structure(f'wall{width_bricks}x{height}', bricks, bases=[(-1, 2*width_bricks + 1)])


def bridge(arm_courses):
    """Two corbelled arms on separate anchored bases closed by one key brick."""
    m = 2*arm_courses
    bricks = []
    for k in range(arm_courses):
        bricks.append(Placement(k, k))
        bricks.append(Placement(k, m - k))
    bricks.append(Placement(arm_courses, arm_courses))
    return Structure(f'bridge{arm_courses}', bricks, bases=[(-1, 2), (m, m + 3)])


CATALOG = {
    'tower': lambda: tower(6),
    'overhang+': lambda: overhang(12, +1),
    'overhang-': lambda: overhang(12, -1),
    'overhang+20': lambda: overhang(20, +1),
    'overhang-20': lambda: overhang(20, -1),
    'wall': lambda: wall(3, 4),
    'bridge': lambda: bridge(3),
}
