"""Polycube brick parameters, the V0 guess, and optimizer bounds.

Lengths in millimetres, angles in degrees. A brick is a set of lattice cells
(default: the L tromino). Every cell carries the same interface, a square
funnel on top and a matching pyramid keel with a tapered peg below, so any
of the four 90-degree rotations nests on any occupied cell below.
"""
from dataclasses import asdict, dataclass, fields, replace
import math

PLA_DENSITY = 1.24e-3  # g/mm^3

SHAPES = {
    'L': ((0, 0), (1, 0), (0, 1)),
    'I2': ((0, 0), (1, 0)),
    'I3': ((0, 0), (1, 0), (2, 0)),
    'O': ((0, 0), (1, 0), (0, 1), (1, 1)),
    'T': ((0, 0), (1, 0), (2, 0), (1, 1)),
    'S': ((0, 0), (1, 0), (1, 1), (2, 1)),
    'L4': ((0, 0), (1, 0), (2, 0), (0, 1)),
}


@dataclass(frozen=True)
class BrickParams:
    shape: str = 'L'
    U: float = 24.0             # cell pitch in X and Y
    H: float = 13.0             # course pitch (constant vertical thickness of each cell)
    theta: float = 35.0         # funnel/keel face angle from horizontal
    peg_w: float = 5.0          # peg width at the root (square)
    peg_L: float = 5.0          # peg length below the keel tip
    peg_taper: float = 2.0      # width lost root -> tip (entry lead)
    peg_lean: float = 15.0      # hook angle from vertical; pegs lean toward body -x, sockets get 4 lobes
    tip_chamfer: float = 0.8
    clearance: float = 0.25     # half the seated peg/socket gap
    socket_extra: float = 1.0   # socket depth beyond the peg tip
    mouth_chamfer: float = 1.0  # 45-degree chamfer at the socket mouth
    gap: float = 0.3            # gap between neighbouring bricks
    wall: float = 1.2           # minimum material around tine bores
    fork_d: float = 2.0         # tine rod diameter
    fork_clearance: float = 0.3
    shell: float = 0.8          # printed perimeter thickness
    infill: float = 0.2         # printed infill fraction inside the shell

    @property
    def a(self):
        return self.U/2

    @property
    def h(self):
        """Funnel depth / keel height from the face angle."""
        return self.a*math.tan(math.radians(self.theta))

    @property
    def cells(self):
        return SHAPES[self.shape]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f'Unknown brick parameters: {sorted(unknown)}')
        return cls(**values)


V0 = BrickParams()

BOUNDS = {
    'U': (16.0, 32.0),
    'H': (8.0, 20.0),
    'theta': (28.0, 55.0),
    'peg_w': (2.5, 8.0),
    'peg_L': (2.0, 9.0),
    'peg_taper': (0.0, 3.0),
    'peg_lean': (0.0, 40.0),
    'clearance': (0.08, 0.6),
    'mouth_chamfer': (0.2, 2.5),
    'infill': (0.1, 0.5),
}


def with_vector(base, names, values):
    return replace(base, **{n: float(v) for n, v in zip(names, values)})
