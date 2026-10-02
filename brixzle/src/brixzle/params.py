"""Brick design parameters, the V0 educated guess, and optimizer bounds.

Lengths are millimetres and angles degrees; conversion to SI happens at the
MuJoCo boundary. The brick is a 2.5D profile in XZ spanning two voxels in X,
extruded through the depth D in Y and sheared into a V gutter across Y.
"""
from dataclasses import asdict, dataclass, fields, replace
import math


PLA_DENSITY = 1.24e-3  # g/mm^3


@dataclass(frozen=True)
class BrickParams:
    U: float = 25.0             # voxel pitch in X (brick length is 2U)
    D: float = 25.0             # depth in Y
    H0: float = 15.0            # course pitch (vertical thickness of the corrugated body)
    theta: float = 50.0         # long hopper ramp angle from horizontal
    alpha: float = 38.0         # tooth/slot lean from vertical; steep face is 90 - alpha
    gutter: float = 30.0        # Y gutter angle from horizontal (0 disables)
    gutter_flat: float = 3.0    # half-width of the unsheared centre band (keeps tine bores straight)
    tooth_w: float = 5.0        # tooth thickness perpendicular to its axis
    tooth_L: float = 6.5        # tooth length along its axis
    tip_chamfer: float = 1.5    # leading-edge chamfer on the tooth tip
    tooth_taper: float = 2.0    # tooth width lost from root to tip (entry lead)
    clearance: float = 0.3      # half the seated tooth/slot gap
    slot_extra: float = 1.0     # slot depth beyond tooth length
    mouth_chamfer: float = 1.5  # chamfer where the slot meets the long ramp
    end_gap: float = 0.3        # gap between neighbours in a course
    wall: float = 1.2           # minimum wall/rib thickness for lightening holes
    lighten: bool = True        # cut through-Y lightening holes
    lean_mode: str = 'alternate'  # 'alternate': B = A rotated; 'uniform': every course A; 'ab': distinct A/B parts, all teeth trail -x
    fork_d: float = 2.0         # fork tine (rod) diameter
    fork_clearance: float = 0.3 # radial clearance around each tine
    infill: float = 1.0         # printed mass fraction relative to solid

    @property
    def uniform(self):
        return self.lean_mode == 'uniform'

    @property
    def phi(self):
        """Steep face angle from horizontal; parallel to the tooth axis."""
        return 90.0 - self.alpha

    @property
    def amplitude(self):
        """Sawtooth height h so the long ramp plus steep face span one voxel."""
        return self.U/(1/math.tan(math.radians(self.theta)) + 1/math.tan(math.radians(self.phi)))

    @property
    def valley_u(self):
        """Valley position of the top sawtooth, measured from the voxel's left edge."""
        return self.amplitude/math.tan(math.radians(self.theta))

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

# Continuous NSGA-III search space (name -> (low, high)).
BOUNDS = {
    'U': (18.0, 32.0),
    'H0': (9.0, 20.0),
    'theta': (30.0, 65.0),
    'alpha': (10.0, 45.0),
    'gutter': (28.0, 45.0),
    'gutter_flat': (0.0, 10.0),
    'tooth_w': (2.5, 8.0),
    'tooth_L': (3.0, 9.0),
    'tip_chamfer': (0.3, 2.5),
    'tooth_taper': (0.0, 3.0),
    'clearance': (0.1, 0.6),
    'mouth_chamfer': (0.3, 3.0),
}


def with_vector(base, names, values):
    """Return base with the named fields replaced; D tracks U to keep cubic-ish voxels."""
    updates = {n: float(v) for n, v in zip(names, values)}
    if 'U' in updates:
        updates.setdefault('D', updates['U'])
    return replace(base, **updates)
