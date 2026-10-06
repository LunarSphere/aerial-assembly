"""Small-rotation helpers for one drone per call (scipy ``Rotation`` costs ~10 us per op).

Quaternions are xyzw; euler angles follow scipy's extrinsic 'xyz' (R = Rz(yaw) Ry(pitch) Rx(roll)).
``tests/test_drone.py`` checks each helper against scipy.
"""
import math

import numpy as np


def quat_to_mat(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    s = 2.0/n if n > 0 else 0.
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([[1 - yy - zz, xy - wz, xz + wy],
                     [xy + wz, 1 - xx - zz, yz - wx],
                     [xz - wy, yz + wx, 1 - xx - yy]])


def mat_to_quat(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5/math.sqrt(t + 1.0)
        q = [(m[2, 1] - m[1, 2])*s, (m[0, 2] - m[2, 0])*s, (m[1, 0] - m[0, 1])*s, 0.25/s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0*math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25*s, (m[0, 1] + m[1, 0])/s, (m[0, 2] + m[2, 0])/s, (m[2, 1] - m[1, 2])/s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0*math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0])/s, 0.25*s, (m[1, 2] + m[2, 1])/s, (m[0, 2] - m[2, 0])/s]
    else:
        s = 2.0*math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0])/s, (m[1, 2] + m[2, 1])/s, 0.25*s, (m[1, 0] - m[0, 1])/s]
    q = np.array(q)
    return q/np.linalg.norm(q)


def mat_to_euler(m):
    return np.array([math.atan2(m[2, 1], m[2, 2]), -math.asin(max(-1., min(1., m[2, 0]))),
                     math.atan2(m[1, 0], m[0, 0])])


def euler_to_mat(rpy):
    a, b, c = rpy
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return np.array([[cc*cb, cc*sb*sa - sc*ca, cc*sb*ca + sc*sa],
                     [sc*cb, sc*sb*sa + cc*ca, sc*sb*ca - cc*sa],
                     [-sb, cb*sa, cb*ca]])


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def rotvec_to_mat(v):
    th = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v)/th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th)*K + (1 - math.cos(th))*(K @ K)
