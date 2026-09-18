from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rotation(quat):
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-6:
        raise ValueError("Quaternions must be normalized [w, x, y, z]")
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def quaternion(rot):
    return rot.as_quat()[[3, 0, 1, 2]].tolist()


@dataclass(frozen=True)
class Physics:
    timestep: float = 0.0000125
    friction: float = 0.3
    contact_timeconst: float = 0.00005
    contact_dampratio: float = 1.0
    iterations: int = 100

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Physics settings must be finite")
        if self.timestep <= 0 or self.friction < 0 or self.contact_timeconst < 2*self.timestep:
            raise ValueError("Require dt > 0, friction >= 0, contact_timeconst >= 2*dt")
        if self.contact_dampratio <= 0 or self.iterations < 1:
            raise ValueError("Invalid damping or solver iteration count")


@dataclass(frozen=True)
class TrialSettings:
    duration: float = 3.0
    dwell: float = 0.5
    translation_tolerance: float = 0.0005
    angle_tolerance: float = math.pi / 180
    gap_tolerance: float = 0.0001
    linear_speed_tolerance: float = 0.001
    angular_speed_tolerance: float = math.pi / 180
    insertion_tolerance: float = 0.0005
    log_interval: float = 0.01

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Trial settings must be finite and positive")
        if self.dwell > self.duration:
            raise ValueError("Dwell must not exceed duration")


@dataclass(frozen=True)
class Release:
    # Height is additional global vertical clearance above separated block AABBs.
    height: float = 0.01
    offset: tuple = (0.0, 0.0)
    rpy_deg: tuple = (0.0, 0.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)  # COM velocity, world frame
    angular_velocity: tuple = (0.0, 0.0, 0.0)  # world frame, rad/s

    def __post_init__(self):
        values = [self.height, *self.offset, *self.rpy_deg, *self.velocity, *self.angular_velocity]
        if len(self.offset) != 2 or any(len(v) != 3 for v in (self.rpy_deg, self.velocity, self.angular_velocity)):
            raise ValueError("Release vectors have wrong lengths")
        if not np.isfinite(values).all() or self.height < 0:
            raise ValueError("Release values must be finite and height nonnegative")
