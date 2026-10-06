"""Lighthouse V2 positioning as the state the controller sees: bias field, jitter, latency.

Numbers come from ``configs/lighthouse.json`` with citations. The true state is delayed by
``latency_s`` and corrupted by (a) a per-flight constant bias b0 and a linear bias gradient G
(systematic base-station geometry error, so nearby points share most of their error) and (b)
white jitter. With ``registered`` the build area is surveyed in the Lighthouse frame (e.g. by
landing the drone on the fixture and base), so b0 cancels and only G * distance remains.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np

from .cf.rot import mat_to_quat, quat_to_mat, rotvec_to_mat


@dataclass(frozen=True)
class LighthouseModel:
    bias_sigma_mm: float = 6.3            # per-axis sigma of the per-flight constant bias (Taffanel 2021)
    bias_gradient_mm_per_m: float = 10.0  # per-axis sigma of the linear bias field
    jitter_mm: float = 0.7                # white position noise (LH2 EKF static jitter)
    vel_noise_mps: float = 0.005          # white velocity-estimate noise (assumption)
    att_noise_deg: float = 0.1            # attitude-estimate noise (assumption; IMU-driven)
    latency_s: float = 0.01               # estimate delay (assumption)
    registered: bool = False              # build area surveyed in the Lighthouse frame
    scale: float = 1.0                    # multiplies every noise term (sensitivity study)

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for k, v in asdict(self).items() if k != 'registered'):
            raise ValueError('Lighthouse model values must be finite and non-negative')

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Lighthouse:
    """Noisy, delayed view of the true state (SI units; quaternions xyzw)."""

    def __init__(self, model: LighthouseModel, rng, dt=0.002, origin=(0., 0., 0.)):
        self.m = model
        self.rng = rng
        s = model.scale
        self.b0 = np.zeros(3) if model.registered else rng.normal(0, model.bias_sigma_mm*1e-3*s, 3)
        self.G = rng.normal(0, model.bias_gradient_mm_per_m*1e-3*s, (3, 3))
        self.origin = np.asarray(origin, float)
        self.set_rate(dt)

    def set_rate(self, dt):
        """Period (s) between ``observe`` calls; sizes the latency buffer."""
        self.delay = max(0, int(round(self.m.latency_s/dt)))
        self.buffer = deque(maxlen=self.delay + 1)

    def bias(self, pos):
        return self.b0 + self.G @ (np.asarray(pos) - self.origin)

    def observe(self, pos, quat, vel, ang_vel):
        """Push the current true state; return the estimate the controller sees."""
        self.buffer.append((np.array(pos), np.array(quat), np.array(vel), np.array(ang_vel)))
        p, _, v, _ = self.buffer[0]
        # Attitude and gyro rates are not delayed: the IMU drives them at 1 kHz on board.
        q, w = np.array(quat), np.array(ang_vel)
        s = self.m.scale
        p_est = p + self.bias(p) + self.rng.normal(0, self.m.jitter_mm*1e-3*s, 3)
        v_est = v + self.rng.normal(0, self.m.vel_noise_mps*s, 3)
        if self.m.att_noise_deg > 0:
            noise = rotvec_to_mat(np.radians(self.rng.normal(0, self.m.att_noise_deg*s, 3)))
            q = mat_to_quat(noise @ quat_to_mat(q))
        return p_est, q, v_est, w
