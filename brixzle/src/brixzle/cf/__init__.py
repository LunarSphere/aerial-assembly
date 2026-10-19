"""Numpy port of crazyflow's Crazyflie Mellinger controller and first-principles dynamics.

Vendored from https://github.com/learnsyslab/crazyflow at d28ec70 (MIT License,
Copyright (c) 2025 Martin Schuck), files ``crazyflow/control/mellinger/{control.py,params.toml}``
and ``crazyflow/dynamics/first_principles/{dynamics.py,params.toml}``. The upstream functions
follow the array API but import JAX/flax at module level; this port keeps brixzle's runtime free of
JAX. Quaternions here are crazyflow's scalar-last ``xyzw``; convert only through ``brixzle.quat``.
``tests/test_drone.py`` cross-checks this port against upstream when crazyflow is importable.
"""
