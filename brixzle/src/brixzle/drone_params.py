"""Crazyflie 2.1 Brushless mass and thrust budget: payload limit and thrust-to-weight margins.

Masses in grams, forces in newtons. The two thrust sources disagree (crazyflow's identified
0.2 N/motor vs the datasheet's 30 gf); ``thrust_source`` picks the one the simulator and the
constraints use, and ``margins`` always reports both.
"""
from dataclasses import asdict, dataclass, field
import math

G = 9.81


@dataclass(frozen=True)
class DroneConfig:
    body_mass_g: float = 43.4          # crazyflow cf21B_500 (heavier than the datasheet's 34-37 g)
    deck_mass_g: float = 2.7           # Lighthouse deck
    thrust_source: str = 'datasheet'
    thrust_per_motor_N: dict = field(default_factory=lambda: {'datasheet': 0.2942, 'crazyflow': 0.2})
    tw_min: float = 1.5
    payload_max_g: float = 30.0
    flight_time_unloaded_s: float = 600.0

    def __post_init__(self):
        if self.thrust_source not in self.thrust_per_motor_N:
            raise ValueError(f'Unknown thrust_source {self.thrust_source!r}')
        values = [self.body_mass_g, self.deck_mass_g, self.tw_min, self.payload_max_g, self.flight_time_unloaded_s,
                  *self.thrust_per_motor_N.values()]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Drone config values must be finite and non-negative')

    @property
    def mass_g(self):
        return self.body_mass_g + self.deck_mass_g

    def thrust_max_N(self, source=None):
        return 4*self.thrust_per_motor_N[source or self.thrust_source]

    def t_w(self, payload_g=0., source=None):
        return self.thrust_max_N(source)/((self.mass_g + payload_g)*1e-3*G)

    def payload_at_tw(self, tw, source=None):
        """Largest payload (g) that keeps thrust-to-weight >= tw."""
        return self.thrust_max_N(source)/(tw*G)*1e3 - self.mass_g

    def margins(self, payload_g):
        """Signed margins (>= 0 satisfied) for the hard payload limit and T/W under both sources."""
        out = {'payload_limit_g': self.payload_max_g - payload_g,
               'tw': self.t_w(payload_g) - self.tw_min}
        for s in self.thrust_per_motor_N:
            out[f'tw_{s}'] = self.t_w(payload_g, s)
        out['binding'] = 'tw' if self.payload_at_tw(self.tw_min) < self.payload_max_g else 'payload'
        return out

    def flight_time_s(self, payload_g):
        """Flight time derated by momentum theory: hover power ~ (m g)^1.5 at a fixed battery energy."""
        return self.flight_time_unloaded_s*(self.mass_g/(self.mass_g + payload_g))**1.5

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in names})


CF21B = DroneConfig()
