"""Real test flight: overhead RealSense senses the pink brick and the pink platform.

Modules: ``model`` (object geometry from CAD), ``rig`` (lab config), ``camera`` (RealSense and
replayed captures), ``calib`` (AprilTag PnP into the Lighthouse frame), ``vision`` (colour + depth
detection), ``sense`` (the ``flight`` pose source), ``fieldplan`` (plan for the real cage),
``survey`` (Lighthouse readings with cflib, never arms), ``synth`` (synthetic frames for tests).
Units: metres and radians unless a name says otherwise (``_mm``, ``_deg``).
"""
