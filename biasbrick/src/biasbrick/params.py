"""biasbrick parameters. Units: mm and degrees (converted to SI only at the MuJoCo boundary).

A biasbrick is brixzle's A/B brick cut in half: two **sockets**, each one voxel of the A (or B)
profile with its sawtooth top, slot and leaning tooth, joined by a **rod** of free length. There is
no voxel grid: sockets sit on sites ``pitch`` apart, a brick spans two neighbouring sites, and the
next course shifts by one site (or none, for a tower).

The drone aims off the seat at (``aim_x``, ``aim_y``), picked from 2D capture maps (``trials.best_aim``)
so a ring of the measured error around it seats as often as possible:
- **X**: slightly toward the side the tooth stroke comes from (+x for both parts, brixzle's
  ``aim_bias``): the leaning tooth enters its slot along its axis from that side.
- **Y**: up the main slope, so landings fall on it (or, the nearest ones, on the nest face). The socket is swept
  through Y as an asymmetric sawtooth (brixzle's gutter, made one-sided): a long ``slope`` rising
  toward +y, continued past the socket's +y face by a thin corrugated **flange** that catches far
  landings, and a short steep **nest face** rising toward -y from the valley. The keel copies it, so
  a brick nests V-in-V in Y as in X; an overshoot climbs the nest face and falls back (a wall stop
  threw sliding bricks over it: runs/biasbrick/evidence/bb2x-*). The slot runs through Y, so a tooth that lands on the
  flange slides down its own channel into the seat.

Two shelled square **sleeves** with a V roof cross the rod and take brixzle's passive fork: the
rod is not sheared, so straight tines fit (they would not through a sheared socket).

Socket frame: x across the socket (centred), y along the shear (body from -D/2 to D/2), z = 0 at
the profile origin. Brick frame: sockets centred at x = -pitch/2 and +pitch/2.
"""
from dataclasses import asdict, dataclass, fields, replace
import math

from brixzle.params import BrickParams

PLA_DENSITY = 1.24e-3  # g/mm^3
PARTS = ('A', 'B')
# The brixzle profile fields a socket takes over unchanged.
PROFILE = ('U', 'H0', 'theta', 'alpha', 'tooth_w', 'tooth_L', 'tip_chamfer', 'tooth_taper', 'clearance',
           'slot_extra', 'mouth_chamfer', 'end_gap', 'wall', 'lighten')


@dataclass(frozen=True)
class BBParams:
    # Socket profile: brixzle V0 values except U (sized for the measured error; rules.size_for_tolerance).
    U: float = 41.             # socket width in X (one brixzle voxel)
    H0: float = 15.            # course pitch
    theta: float = 50.         # long ramp angle from horizontal
    alpha: float = 38.         # tooth and slot lean from vertical
    tooth_w: float = 5.
    tooth_L: float = 6.5
    tip_chamfer: float = 1.5
    tooth_taper: float = 2.
    clearance: float = 0.3
    slot_extra: float = 1.
    mouth_chamfer: float = 1.5
    end_gap: float = 0.3
    wall: float = 1.2          # skin and rib thickness of the lightened socket
    lighten: bool = True
    # One-sided Y.
    D: float = 20.             # socket body depth in Y
    slope: float = 42.         # main Y slope from horizontal, rising toward +y; the keel slides down it inside
                               # the X sawtooth's V, which multiplies friction by ~1/cos 52 (rules.y_slope_slides)
    flange: float = 43.5       # flange length past the socket's +y face (sized with U)
    flange_t: float = 1.2      # flange plate thickness (vertical)
    nest_angle: float = 60.    # short Y face from horizontal, rising toward -y from the valley
    nest_w: float = 12.        # its width in Y (the valley sits nest_w in from the socket's -y face). Near-side
                               # landings (aim - error - the rod's yaw shift) parked on a 5 mm face's ridge; 9 and
                               # 12 mm seat 96/92/89 and 97/94/90% against 78/77/84 (runs/biasbrick/current/bb5-nest*)
    tooth_setback: float = 0.  # teeth start this far up the main slope from the valley (3 and 5 mm did not cut
                               # the jams: runs/biasbrick/current/bb5-setback*)
    # Rod: a web between the sockets' end faces (fused into them), centred on y = rod_y.
    rod_L: float = 80.         # clear length between the sockets' end faces: 80 mm seats 78/77/84% against 57/62/61%
                               # at 40 mm (B onto A, Exp 3 mean, mu 0.2/0.35/0.5; runs/biasbrick/current/bb3-rod*)
    rod_w: float = 3.          # Y thickness
    rod_h: float = 11.         # Z height (a web: stiff in bending, light)
    rod_y: float = 7.5         # toward the COM (the flange pulls it +y), within the end faces
    # Fork sleeves: shelled square tubes along Y crossing the rod, V roof inside. In a tower the next
    # course's rod is H0 above, so a sleeve fits in a band H0 - sleeve_gap tall round the rod.
    fork_d: float = 2.
    cav_play: float = 2.8      # sleeve play per side around a tine
    cav_wall: float = 1.2
    roof_angle: float = 45.    # V roof faces from horizontal
    cav_depth: float = 24.     # sleeve length in Y
    cav_y: float = 8.7         # sleeve centre in Y: on the COM, so the brick hangs level
    tine_spacing: float = 24.  # between the two sleeves (brixzle's fork: 24.75 mm)
    sleeve_gap: float = 1.     # to the next course's rod and sleeves
    # Aim bias.
    aim_r: float = 14.82       # X-Y error the catchment is sized for (Exp 3 mean)
    aim_x: float = 2.          # aim from the seat (mm). X: the best worst-case ring score over the capture maps
    aim_y: float = 20.         # (runs/biasbrick/current/bb3-mean-map-*, bb4-rod80-map-*). Y: with the 12 mm nest,
                               # 20 seats best of 17 and 20 (bb5-nest12-aimy*); the maps ignore yaw
    yaw_allow: float = 5.      # yaw the catchment also covers: 1 sigma of the (unmeasured) ErrorModel yaw
    aim_margin: float = 2.     # sizing: the far landing's margin
    part: str = 'A'
    infill: float = 1.0

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name not in ('part', 'lighten') and not math.isfinite(v):
                raise ValueError(f'{f.name} must be finite')
        if self.part not in PARTS:
            raise ValueError(f'part must be one of {PARTS}')
        if not 0 < self.slope < 60:
            raise ValueError('slope must be in (0, 60) degrees from horizontal')
        if not 0 < self.nest_angle < 85:
            raise ValueError('nest_angle must be in (0, 85) degrees from horizontal')
        if self.nest_w >= self.D:
            raise ValueError('nest_w must be smaller than the socket depth D')
        if self.nest_w + self.tooth_setback >= self.D:
            raise ValueError('the teeth need some of the main slope: nest_w + tooth_setback < D')
        if not 0 < self.roof_angle < 90:
            raise ValueError('roof_angle must be in (0, 90) degrees from horizontal')
        positive = ('U', 'H0', 'D', 'flange_t', 'nest_w', 'rod_w', 'rod_h', 'fork_d', 'cav_wall', 'cav_depth',
                    'tine_spacing', 'infill')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError(f'{positive} must be > 0')
        nonneg = ('flange', 'tooth_setback', 'rod_L', 'cav_play', 'sleeve_gap', 'aim_r', 'yaw_allow', 'aim_margin')
        if any(getattr(self, k) < 0 for k in nonneg):
            raise ValueError(f'{nonneg} must be >= 0')

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d.pop('derived', None)
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f'unknown biasbrick parameters: {sorted(unknown)}')
        return cls(**d)

    def to_dict(self):
        return asdict(self)

    def profile(self):
        """The brixzle profile a socket is cut from (no gutter: the Y shear is biasbrick's own)."""
        return BrickParams(**{k: getattr(self, k) for k in PROFILE}, D=self.D, gutter=0.)

    def derived(self):
        keys = ('pitch', 'length', 'shear_k', 'nest_k', 'valley_y', 'yaw_shift')
        return {k: getattr(self, k) for k in keys}

    @property
    def pitch(self):
        """Site spacing: socket centre to socket centre."""
        return self.U + self.rod_L

    @property
    def length(self):
        return self.pitch + self.U

    @property
    def shear_k(self):
        """Rise per mm of the main Y slope (toward +y)."""
        return math.tan(math.radians(self.slope))

    @property
    def nest_k(self):
        """Rise per mm of the nest face (toward -y)."""
        return math.tan(math.radians(self.nest_angle))

    @property
    def valley_y(self):
        """Y of the Y valley (the seat line), nest_w in from the socket's -y face."""
        return -self.D/2 + self.nest_w

    @property
    def yaw_shift(self):
        """(x, y) a socket moves when the brick yaws by ``yaw_allow`` about its centre."""
        a = math.radians(self.yaw_allow)
        return (self.pitch/2*(1 - math.cos(a)), self.pitch/2*math.sin(a))

    def socket_x(self, i):
        """X of socket i (0 or 1) in the brick frame."""
        return (i - 0.5)*self.pitch


DEFAULT = BBParams()


def with_values(p: BBParams, **kw):
    return replace(p, **kw)
