"""Poses and frames: positions in metres, quaternions [w, x, y, z] (the repo convention).

``wxyz_to_xyzw``/``xyzw_to_wxyz`` are the only conversions to scipy/crazyflow's scalar-last
order in the repo; ``brixzle.quat`` re-exports them.
"""
from dataclasses import dataclass
import math

import numpy as np


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return np.concatenate((q[..., 1:4], q[..., :1]), axis=-1)


def xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float)
    return np.concatenate((q[..., 3:4], q[..., :3]), axis=-1)


def _rot(q_wxyz):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(wxyz_to_xyzw(q_wxyz))


@dataclass(frozen=True)
class Pose:
    pos: tuple
    quat: tuple = (1., 0., 0., 0.)   # wxyz

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(float(v) for v in d['pos']), tuple(float(v) for v in d.get('quat', (1, 0, 0, 0))))

    def to_dict(self):
        return {'pos': list(self.pos), 'quat': list(self.quat)}

    @property
    def yaw(self):
        w, x, y, z = self.quat
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def apply(self, local):
        """World point of a point given in this frame."""
        return np.asarray(self.pos) + _rot(self.quat).apply(np.asarray(local, float))

    def rotate(self, v):
        return _rot(self.quat).apply(np.asarray(v, float))

    def compose(self, other):
        """self * other: ``other`` expressed in this frame, returned in the world frame."""
        q = (_rot(self.quat)*_rot(other.quat)).as_quat()
        return Pose(tuple(self.apply(other.pos)), tuple(xyzw_to_wxyz(q)))

    def inverse(self):
        r = _rot(self.quat).inv()
        return Pose(tuple(-r.apply(self.pos)), tuple(xyzw_to_wxyz(r.as_quat())))

    def relative_to(self, ref):
        """This pose in ``ref``'s frame."""
        return ref.inverse().compose(self)

    def angle_to(self, other):
        """Rotation angle (deg) between two orientations."""
        return math.degrees((_rot(self.quat).inv()*_rot(other.quat)).magnitude())
