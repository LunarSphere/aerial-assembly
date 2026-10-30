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


def corbel(courses=6, width=2):
    """Corbelled overhang: every course is ``width`` bricks in running bond, shifted one site toward +x.

    Each course's outer brick reaches one site past the course below (one supported socket); the next
    course's brick across that joint pins its tail, so the overhang is a hooked cantilever tied back to the
    base rather than a chain of single hooks. Reach after k courses: k - 1 sites past the base.
    ``corbel`` (4 courses, 3 wide) is the overhang that stands: 3 sites, 363 mm, past the base; at the
    5th course the tip sags past H0/2 (runs/biasbrick/current/bb9-*).
    """
    return Structure(f'corbel{width}', [Placement(k, k + 2*m) for k in range(courses) for m in range(width)])


def catalog():
    return {'tower': lambda: tower(6), 'wall': lambda: wall(4, 4), 'bridge': lambda: bridge(2),
            'overhang+': lambda: overhang(12, +1), 'overhang-': lambda: overhang(12, -1),
            'corbel': lambda: corbel(4, 3), 'corbel2': lambda: corbel(6, 2), 'corbel3': lambda: corbel(6, 3)}
