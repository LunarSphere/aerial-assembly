"""One flight program for the Brixzle pick-and-place mission.

Mission code talks only to the cflib API (``SyncCrazyflie``/``Crazyflie``, the high-level
and low-level commanders, params and logging) and to an injected ``Clock`` and
``BrickPoseSource``. The same code drives a real Crazyflie (``flight.hardware``) or the
MuJoCo simulator (``brixzle.simcf``, imported only for ``--sim``). This package must not
import CadQuery or MuJoCo.
"""
