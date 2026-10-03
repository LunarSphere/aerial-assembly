"""Parametric drop-assembled bricks (brixzle)."""
import os
import sys

# MuJoCo picks its GL backend at import time; default to headless EGL on Linux.
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('MESA_SHADER_CACHE_DISABLE', 'true')
