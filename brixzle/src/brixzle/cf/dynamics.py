"""Rotor forces of crazyflow's first-principles model, split so MuJoCo integrates the rigid body.

Upstream ``dynamics()`` returns state derivatives. Here the rigid-body part (gravity, inertia,
gyroscopic ``w x Jw``, contacts) is MuJoCo's job; this module returns only the propulsion
wrench in the body frame plus the rotor-speed derivative, which the caller integrates.
"""
import numpy as np

RPM2RAD = 2*np.pi/60


def rotor_vel_dot(cmd, rotor_vel, rotor_dyn_coef):
    acc1, acc2, dec1, dec2 = rotor_dyn_coef
    up = acc1*(cmd - rotor_vel) + acc2*(cmd**2 - rotor_vel**2)
    down = dec1*(cmd - rotor_vel) + dec2*(cmd**2 - rotor_vel**2)
    return np.where(cmd > rotor_vel, up, down)


def motor_thrusts(rotor_vel, rpm2thrust):
    k0, k1, k2 = rpm2thrust
    return k0 + k1*rotor_vel + k2*rotor_vel**2


def body_wrench(rotor_vel, rotor_vel_dot_, ang_vel, vel_body, *, L, prop_inertia, rpm2thrust, rpm2torque,
                mixing_matrix, drag_matrix, thrust_scale=None, **_):
    """Propulsion and drag (force, torque) in the body frame, without gravity or w x Jw.

    ``thrust_scale`` (4,) multiplies each motor's thrust (ground effect, downwash).
    """
    f = motor_thrusts(rotor_vel, rpm2thrust)
    if thrust_scale is not None:
        f = f*thrust_scale
    c0, c1, c2 = rpm2torque
    tau_m = c0 + c1*rotor_vel + c2*rotor_vel**2
    torque = (mixing_matrix @ (f*L))*np.array([1., 1., 0.])
    torque = torque + (mixing_matrix @ tau_m)*np.array([0., 0., 1.])
    spin = mixing_matrix[-1]*prop_inertia
    h = float(spin @ (rotor_vel*RPM2RAD))
    h_dot = float(spin @ (rotor_vel_dot_*RPM2RAD))
    torque = torque + np.array([ang_vel[1]*h, -ang_vel[0]*h, h_dot])
    force = np.array([0., 0., f.sum()]) + drag_matrix @ vel_body
    return force, torque, f
