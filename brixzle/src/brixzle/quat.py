"""Quaternion order conversion: [w, x, y, z] (brixzle/MuJoCo) <-> [x, y, z, w] (crazyflow/scipy).

Implemented once, in ``flight.frames``; re-exported here so brixzle never converts elsewhere.
"""
from flight.frames import wxyz_to_xyzw, xyzw_to_wxyz

__all__ = ['wxyz_to_xyzw', 'xyzw_to_wxyz']
