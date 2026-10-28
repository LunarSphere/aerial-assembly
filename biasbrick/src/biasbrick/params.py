"""biasbrick parameters. Units: mm and degrees (converted to SI only at the MuJoCo boundary).

A biasbrick is n square voxels long (pitch U, depth U) on a rail, like a DF brick, but every voxel
carries a **corner hopper** instead of a symmetric funnel. Its seat (a slanted square slot) sits in the
voxel's -x-y corner. The +x and +y faces ramp up at ``ramp`` from the throat to the voxel's far
edges; the two ramps meet along a diagonal crease. The -x and -y sides are short side walls that
stand ``stop_h`` above the ramp (capped at the ridge). The drone aims at ``aim`` mm off the seat
along +x and +y, which covers the error ``aim_r`` plus the yaw allowance, so every landing falls
on the ramps and slides one way into the corner.

The seat is a slanted slot and each foot ends in a tapered tooth, both leaning ``alpha`` from
vertical toward the -x-y corner (brixzle's tooth and slot, along the diagonal): the ramps deliver
a foot moving toward -x-y and the slot continues that stroke down into the corner. A seated tooth
sits under the slot's overhanging -x and -y walls, so it cannot be lifted straight out: every
joint hooks against an overhang toward +x or +y. The tooth stops ``slot_extra`` short of the slot
floor, so a brick bears on its cup faces on the ramps (brixzle's keel on the V). Two nesting-cup
feet under the outer voxels copy the hopper 0.3 mm inside it (DF's finding). Two shelled square
tubes with a V roof under the inner ridges take brixzle's passive fork.

Brick frame: x along the brick, centred on its middle voxel; y across; z = 0 is the rail bottom.
Seat-local coordinates (du, dv) are measured from a seat's centre at the slot floor.
"""
from dataclasses import asdict, dataclass, fields, replace
import math

PLA_DENSITY = 1.24e-3  # g/mm^3
PARTS = ('X', 'Y')


@dataclass(frozen=True)
class BBParams:
    U: float = 47.5            # voxel pitch (X and Y); sized for the Exp 3 mean error (rules.size_for_tolerance)
    n_vox: int = 3             # voxels per brick; feet under the two outer ones
    bar_h: float = 3.          # rail height
    foot_w: float = 6.         # square tooth width at the root (horizontal section)
    tooth_L: float = 9.        # tooth length along its axis
    alpha: float = 38.         # tooth and slot lean from vertical toward -x-y (along the diagonal), as brixzle
    tooth_taper: float = 2.    # tooth width lost from root to tip (entry lead)
    tip_chamfer: float = 1.    # 45 degree chamfer round the tooth tip
    clearance: float = 0.3     # per side between the tooth root and the slot
    slot_extra: float = 1.     # slot deeper than the seated tooth, so the cups bear on the ramps
    ramp: float = 41.          # +x and +y faces from horizontal; the crease is shallower (rules.crease_angle)
    stop_h: float = 6.         # side walls stand this far above the slot top and the ramp (capped at the ridge)
    wall_t: float = 1.2        # plate thickness (printable minimum: 3 perimeters at 0.4 mm)
    top_gap: float = 1.0       # rail clearance above the ridges of the course below
    end_gap: float = 2.0       # between neighbouring hoppers (and bricks)
    aim_r: float = 14.8        # X-Y error the aim bias and catchment are sized for (Exp 3 mean)
    yaw_allow: float = 5.      # yaw the aim also covers: 1 sigma of the (unmeasured) ErrorModel yaw
    aim_margin: float = 1.     # nearest landing stays this far on the ramp side of the seat
    fork_d: float = 2.         # tine diameter
    cav_play: float = 3.       # fork cavity play per side around a tine
    cav_wall: float = 1.2      # fork cavity shell wall
    roof_angle: float = 45.    # V roof faces from horizontal: the hanging brick centres on the tine
    cav_chamfer: float = 2.    # 45 degree lead-in round the cavity mouth (-y face)
    cav_depth: float = 30.     # cavity length in Y from the -y face (must reach past the COM)
    part: str = 'X'            # 'X'; 'Y' is X mirrored in y, placed at yaw 90, so seats stay world -x-y
    infill: float = 1.0

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name != 'part' and not math.isfinite(v):
                raise ValueError(f'{f.name} must be finite')
        if self.part not in PARTS:
            raise ValueError(f'part must be one of {PARTS}')
        if int(self.n_vox) != self.n_vox or self.n_vox < 2:
            raise ValueError('n_vox must be an integer >= 2')
        if not 0 < self.ramp < 90:
            raise ValueError('ramp must be in (0, 90) degrees from horizontal')
        if not 0 <= self.alpha < 60:
            raise ValueError('alpha must be in [0, 60) degrees from vertical')
        if not 0 < self.roof_angle < 90:
            raise ValueError('roof_angle must be in (0, 90) degrees from horizontal')
        positive = ('U', 'bar_h', 'foot_w', 'tooth_L', 'wall_t', 'fork_d', 'cav_wall', 'cav_depth', 'infill')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError(f'{positive} must be > 0')
        nonneg = ('clearance', 'tip_chamfer', 'tooth_taper', 'slot_extra', 'top_gap', 'end_gap', 'stop_h', 'aim_r', 'yaw_allow', 'aim_margin',
                  'cav_play', 'cav_chamfer')
        if any(getattr(self, k) < 0 for k in nonneg):
            raise ValueError(f'{nonneg} must be >= 0')
        if self.tip_chamfer + self.tooth_taper/2 >= self.foot_w/2:
            raise ValueError('tip_chamfer plus half the taper must be smaller than half the tooth')
        if self.ramp_run <= 0:
            raise ValueError('U too small for the throat, side wall and lean')
        if self.cav_depth > self.U - self.end_gap:
            raise ValueError('cav_depth must fit in the voxel depth')

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

    def derived(self):
        keys = ('half_foot', 'throat', 'seat', 'reach', 'lean_k', 'tooth_h', 'slot_h', 'lean_shift', 'ramp_run', 'ramp_h', 'guide_h', 'foot_L',
                'H0', 'baseline', 'length', 'yaw_shift', 'aim')
        return {k: getattr(self, k) for k in keys}

    @property
    def half_foot(self):
        return self.foot_w/2

    @property
    def throat(self):
        """Half-width of the slot (horizontal section)."""
        return self.half_foot + self.clearance

    @property
    def seat(self):
        """Seat centre (slot floor) from the voxel centre, in both x and y (negative: the -x-y corner)."""
        return -self.U/2 + self.end_gap/2 + self.wall_t + self.throat

    @property
    def reach(self):
        """Seat centre to the far ridge (the +x and +y edges of the voxel)."""
        return self.U/2 - self.end_gap/2 - self.seat

    @property
    def lean_k(self):
        """Slot and tooth shift along +x (and +y) per mm of height: the diagonal lean alpha, per axis."""
        return math.tan(math.radians(self.alpha))/math.sqrt(2)

    @property
    def tooth_h(self):
        """Vertical height of the tooth."""
        return self.tooth_L*math.cos(math.radians(self.alpha))

    @property
    def slot_h(self):
        """Vertical height of the slot, floor to mouth."""
        return self.tooth_h + self.slot_extra

    @property
    def lean_shift(self):
        """How far the slot's mouth sits from its floor along +x and +y."""
        return self.slot_h*self.lean_k

    @property
    def ramp_run(self):
        """Horizontal run of a ramp, from the slot mouth to the ridge."""
        return self.reach - self.throat - self.lean_shift

    @property
    def ramp_h(self):
        return self.ramp_run*math.tan(math.radians(self.ramp))

    @property
    def guide_h(self):
        """Slot floor to ridge."""
        return self.slot_h + self.ramp_h

    @property
    def foot_L(self):
        return self.guide_h + self.top_gap

    @property
    def H0(self):
        """Course pitch."""
        return self.bar_h + self.foot_L

    @property
    def baseline(self):
        return (self.n_vox - 1)*self.U

    @property
    def length(self):
        return self.n_vox*self.U

    @property
    def yaw_shift(self):
        """(x, y) a foot moves when the brick yaws by ``yaw_allow`` about its centre."""
        a = math.radians(self.yaw_allow)
        return (self.baseline/2*(1 - math.cos(a)), self.baseline/2*math.sin(a))

    @property
    def aim(self):
        """Aim offset (x, y) from the seat: error plus yaw plus margin, so no landing is on the stop side."""
        return tuple(self.aim_r + s + self.aim_margin for s in self.yaw_shift)

    def voxel_x(self, i):
        """X of voxel i's centre in the brick frame (brick centred on x = 0)."""
        return (i - (self.n_vox - 1)/2)*self.U

    def seat_xy(self, i):
        """Seat centre of voxel i at the slot floor, in the brick frame (X part)."""
        return self.voxel_x(i) + self.seat, self.seat


DEFAULT = BBParams()


def with_values(p: BBParams, **kw):
    return replace(p, **kw)
