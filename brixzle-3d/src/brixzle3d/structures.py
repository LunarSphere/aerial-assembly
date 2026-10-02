"""3D trial structures built from L trominoes (r = quarter turns about the origin corner).

Rotated L cells relative to its origin (i, j):
  r0: (0,0) (1,0) (0,1)      r1: (-1,0) (-1,1) (-2,0)
  r2: (-1,-1) (-2,-1) (-1,-2)  r3: (0,-1) (0,-2) (1,-1)

Two Ls tile a 3x2 block in two ways ("pair" partitions), which lets courses
cross their seams like masonry bond.
"""
from .lattice import Placement, Structure


def _rect(i0, i1, j0, j1):
    return [(i, j) for i in range(i0, i1) for j in range(j0, j1)]


def pair(k, x, y=0):
    """Two Ls covering [x, x+3) x [y, y+2) in course k."""
    return [Placement(k, x, y, 0), Placement(k, x + 3, y + 2, 2)]


def pillar(height):
    """2x2 spiral pillar: one L per course, rotating a quarter turn each course,
    so every L seats two cells on the one below."""
    origins = {0: (0, 0), 1: (2, 0), 2: (2, 2), 3: (0, 2)}
    bricks = [Placement(k, *origins[k % 4], k % 4) for k in range(height)]
    return Structure(f'pillar{height}', bricks, [_rect(-1, 3, -1, 3)])


def column(height):
    """Solid 2x3 column: two Ls per course, partitions alternate so seams cross."""
    bricks = []
    for k in range(height):
        if k % 2 == 0:
            bricks += [Placement(k, 0, 0, 0), Placement(k, 2, 3, 2)]   # (0,0)(1,0)(0,1) + (1,2)(0,2)(1,1)
        else:
            bricks += [Placement(k, 2, 0, 1), Placement(k, 0, 3, 3)]   # (1,0)(1,1)(0,0) + (0,2)(0,1)(1,2)
    return Structure(f'column{height}', bricks, [_rect(-1, 3, -1, 4)])


def overhang(n):
    """Single-L staircase stepping +x one cell per course: a pure cantilever
    (each L seats one cell on the L below). Needs joint tension to go far."""
    bricks = [Placement(k, k, 0, 0) for k in range(n)]
    return Structure(f'overhang{n}', bricks, [_rect(-1, 2, -1, 3)])


def wall(blocks, height):
    """Two-cell-thick wall of 3x2 pair blocks; partitions alternate per course so
    the internal L seams cross (every cell fully seated)."""
    bricks = []
    for k in range(height):
        for n in range(blocks):
            x = 3*n
            if k % 2 == 0:
                bricks += [Placement(k, x, 0, 0), Placement(k, x + 3, 2, 2)]
            else:
                bricks += [Placement(k, x, 2, 3), Placement(k, x + 3, 0, 1)]
    return Structure(f'wall{blocks}x{height}', bricks, [_rect(-1, 3*blocks + 1, -1, 3)])


CATALOG = {
    'pillar': lambda: pillar(8),
    'column': lambda: column(6),
    'overhang': lambda: overhang(8),
    'wall': lambda: wall(3, 4),
}
