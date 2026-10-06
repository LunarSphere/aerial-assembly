"""The single conversion point between brixzle/MuJoCo [w, x, y, z] and crazyflow/scipy [x, y, z, w]."""
import numpy as np


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return np.concatenate((q[..., 1:4], q[..., :1]), axis=-1)


def xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float)
    return np.concatenate((q[..., 3:4], q[..., :3]), axis=-1)
