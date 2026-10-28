"""Trial structures, as brixzle and dfbrick: tower, wall, bridge, overhang +/- (no ring: see lattice)."""
from dataclasses import dataclass, field

from .lattice import Placement


@dataclass
class Structure:
    name: str
    bricks: list = field(default_factory=list)
    key: str = None  # brick that must go last (bridge span)


def tower(h=6):
    return Structure('tower', [Placement(k, 0) for k in range(h)])


def wall(width=4, courses=4):
    """Running bond: odd courses shift one site. Like DF, a course keeps only bricks with both sockets
    over the course below, so with no half bricks the wall steps in at the ends (4-3-2-1)."""
    bricks, below = [], None
    for k in range(courses):
        row = [Placement(k, 2*m + (k % 2)) for m in range(width)]
        if below is not None:
            row = [b for b in row if all(s in below for s in b.sites)]
        if not row:
            break
        bricks += row
        below = {s for b in row for s in b.sites}
    return Structure('wall', bricks)


def bridge(pier_h=2):
    """Two piers side by side (sites 0-1 and 2-3); the span (sites 1-2) joins them, placed last."""
    bricks = [Placement(k, 0) for k in range(pier_h)] + [Placement(k, 2) for k in range(pier_h)]
    span = Placement(pier_h, 1)
    return Structure('bridge', bricks + [span], key=span.name)


def overhang(count=12, direction=+1):
    """Staircase: each brick shifts one site, so its trailing socket sits on the leading socket of the
    one below and its leading socket hangs: every joint carries the overhang through one hooked tooth."""
    return Structure(f'overhang{"+" if direction > 0 else "-"}', [Placement(k, direction*k) for k in range(count)])


def catalog():
    return {'tower': lambda: tower(6), 'wall': lambda: wall(4, 4), 'bridge': lambda: bridge(2),
            'overhang+': lambda: overhang(12, +1), 'overhang-': lambda: overhang(12, -1)}
