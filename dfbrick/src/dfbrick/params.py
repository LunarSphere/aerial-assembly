"""DF-brick parameters. Units: mm and degrees (converted to SI only at the MuJoCo boundary).

A DF brick is n square voxels long (pitch U, depth U). Every voxel carries a receiver on top: a
thin-walled square funnel whose walls run straight down for ``parallel_len`` (the parallel
sliding edges) and flare out at ``flare`` from vertical until they meet the neighbouring
voxel's funnel at a ridge, so the whole voxel is catchment and nothing is flat. Two chamfered
square feet hang under the outer voxels (baseline B = (n - 1) U) and seat in the receivers of the
course below. A rail joins the feet and the funnels; Y ribs under the ridges carry the two
straight tine bores of brixzle's passive fork.

Derived sizes follow from U, the foot, the clearance and the flare: the funnel height is what the
flare needs to reach the ridge, the foot is that long plus ``top_gap`` (so the rail above clears
the ridges), and the course pitch is rail plus foot.
"""
from dataclasses import asdict, dataclass, fields, replace
import math

PLA_DENSITY = 1.24e-3  # g/mm^3
RECEIVER_MODES = ('pocket', 'pocket-slot')
FOOT_KINDS = ('peg', 'cup')


@dataclass(frozen=True)
class DFParams:
    U: float = 44.5            # voxel pitch (X and Y); sized for the Exp 3 mean error (see rules)
    n_vox: int = 3             # voxels per brick; feet under the two outer ones
    bar_h: float = 3.          # rail height
    foot_w: float = 8.         # square foot width
    tip_chamfer: float = 1.5   # 45 degree chamfer round the foot tip
    clearance: float = 0.3     # per side between the foot and the parallel throat
    parallel_len: float = 4.   # straight throat length (guided stroke before seating)
    flare: float = 55.         # funnel wall angle from vertical
    wall_t: float = 1.2        # funnel wall thickness (printable minimum: 3 perimeters at 0.4 mm)
    top_gap: float = 1.0       # rail clearance above the ridges of the course below
    end_gap: float = 2.0       # between neighbouring funnels (and bricks) at a ridge; room for a
                               # brick dropped between two placed neighbours
    foot_kind: str = 'cup'     # 'peg': square prism; 'cup': hollow shell that nests in the receiver
                               # (prism tip through the throat, then the receiver's flare, offset by
                               # the clearance), so an off-centre foot slides on the ridge, not its tip
    rib_t: float = 5.4         # Y rib thickness (X) under each inner ridge, holds a tine bore
    rib_depth: float = 14.     # Y length of each bore: short, so a drone tilting as it withdraws the
                               # fork does not wedge the tines (0.3 mm clearance over 14 mm ~ 2.5 deg)
    fork_d: float = 2.         # tine diameter
    fork_clearance: float = 0.3
    receiver_mode: str = 'pocket'  # 'pocket': square funnels everywhere; 'pocket-slot': voxel 0 a
                                   # pocket, the others slots open along X (kinematic: one pocket
                                   # fixes X and Y, a slot fixes only Y)
    infill: float = 1.0

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name not in ('receiver_mode', 'foot_kind') and not math.isfinite(v):
                raise ValueError(f'{f.name} must be finite')
        if self.receiver_mode not in RECEIVER_MODES:
            raise ValueError(f'receiver_mode must be one of {RECEIVER_MODES}')
        if self.foot_kind not in FOOT_KINDS:
            raise ValueError(f'foot_kind must be one of {FOOT_KINDS}')
        if int(self.n_vox) != self.n_vox or self.n_vox < 2:
            raise ValueError('n_vox must be an integer >= 2')
        if not 0 < self.flare < 90:
            raise ValueError('flare must be in (0, 90) degrees from vertical')
        positive = ('U', 'bar_h', 'foot_w', 'parallel_len', 'wall_t', 'rib_t', 'rib_depth', 'fork_d', 'infill')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError(f'{positive} must be > 0')
        if min(self.clearance, self.tip_chamfer, self.top_gap, self.end_gap, self.fork_clearance) < 0:
            raise ValueError('clearances and chamfers must be >= 0')
        if self.tip_chamfer >= self.foot_w/2:
            raise ValueError('tip_chamfer must be smaller than half the foot')
        if self.extent <= self.throat + self.wall_t:
            raise ValueError('U too small for the throat and wall')

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d.pop('derived', None)
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f'unknown DF parameters: {sorted(unknown)}')
        return cls(**d)

    def to_dict(self):
        return asdict(self)

    def derived(self):
        keys = ('half_foot', 'throat', 'extent', 'flare_h', 'guide_h', 'foot_L', 'H0', 'baseline', 'length')
        return {k: getattr(self, k) for k in keys}

    @property
    def half_foot(self):
        return self.foot_w/2

    @property
    def throat(self):
        """Half-width of the parallel throat."""
        return self.half_foot + self.clearance

    @property
    def extent(self):
        """Half-width of a funnel's mouth (it reaches the ridge)."""
        return self.U/2 - self.end_gap/2

    @property
    def flare_h(self):
        return (self.extent - self.throat)/math.tan(math.radians(self.flare))

    @property
    def guide_h(self):
        return self.parallel_len + self.flare_h

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

    def voxel_x(self, i):
        """X of voxel i's centre in the brick frame (brick centred on x = 0)."""
        return (i - (self.n_vox - 1)/2)*self.U

    def receiver_kind(self, i):
        return 'pocket' if self.receiver_mode == 'pocket' or i == 0 else 'slot'


DEFAULT = DFParams()


def with_values(p: DFParams, **kw):
    return replace(p, **kw)
