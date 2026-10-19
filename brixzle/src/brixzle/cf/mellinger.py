"""Mellinger controller (firmware reimplementation), numpy port of crazyflow's control.py.

Single drone, no batch axes. Quaternions are xyzw. The only change from upstream is
``pwm_headroom``: the motor PWM clip becomes ``pwm_max * pwm_headroom`` so a stronger
motor (datasheet thrust) saturates later without changing any controller gain, which upstream
couples to ``thrust_max`` through its linear PWM<->force maps.
"""
import numpy as np

from .rot import euler_to_mat, mat_to_euler, quat_to_mat, quat_yaw


def pwm2force(pwm, thrust_max, pwm_max):
    return pwm/pwm_max*thrust_max


def force2pwm(thrust, thrust_max, pwm_max):
    return thrust/thrust_max*pwm_max


def state2attitude(pos, quat, vel, cmd, pos_err_i=None, ctrl_freq=100, *, mass, kp, kd, ki, gravity_vec,
                   mass_thrust, int_err_max, thrust_max, pwm_max):
    """Full-state command [p(3), v(3), a(3), q_xyzw(4), w(3)] -> ([roll, pitch, yaw, thrust N], i_err)."""
    dt = 1/ctrl_freq
    pos_err = cmd[0:3] - pos
    vel_err = cmd[3:6] - vel
    i_err = np.zeros(3) if pos_err_i is None else pos_err_i
    i_err = np.clip(i_err + pos_err*dt, -int_err_max, int_err_max)
    target = mass*(cmd[6:9] - gravity_vec) + kp*pos_err + kd*vel_err + ki*i_err
    desired_yaw = quat_yaw(cmd[9:13])
    z_axis = quat_to_mat(quat)[:, 2]
    current_thrust = float(target @ z_axis)
    z_des = target/np.linalg.norm(target)
    x_c = np.array([np.cos(desired_yaw), np.sin(desired_yaw), 0.])
    y_des = np.cross(z_des, x_c)
    y_des /= np.linalg.norm(y_des)
    x_des = np.cross(y_des, z_des)
    rpy = mat_to_euler(np.stack((x_des, y_des, z_des), axis=-1))
    thrust = pwm2force(mass_thrust*current_thrust, thrust_max*4, pwm_max)
    return np.append(rpy, thrust), i_err


def attitude2force_torque(quat, ang_vel, cmd, prev_ang_vel=None, r_int_error=None, ctrl_freq=500,
                          ang_vel_des=None, prev_ang_vel_des=None, *, kR, kw, ki_m, kd_omega, int_err_max,
                          torque_pwm_max, thrust_max, pwm_min, pwm_max, L, thrust2torque, mixing_matrix,
                          pwm_headroom=1.0):
    """[roll, pitch, yaw, thrust] -> (force (1,), torque (3,), r_int_error), firmware legacy mixing."""
    dt = 1/ctrl_freq
    w_des = np.zeros(3) if ang_vel_des is None else ang_vel_des
    w_des_prev = w_des if prev_ang_vel_des is None else prev_ang_vel_des
    delta = euler_to_mat(cmd[:3]).T @ quat_to_mat(quat)
    eRM = delta - delta.T
    eR = np.array([eRM[2, 1], eRM[0, 2], eRM[1, 0]])
    ew = w_des - ang_vel
    prev = np.zeros(3) if prev_ang_vel is None else prev_ang_vel
    d_err = ((w_des - w_des_prev) - (ang_vel - prev))/dt
    d_err[2] = 0.
    r_int = np.zeros(3) if r_int_error is None else r_int_error
    r_int = np.clip(r_int - eR*dt, -int_err_max, int_err_max)
    torque_pwm = -kR*eR + kw*ew + ki_m*r_int + kd_omega*d_err
    torque_pwm = np.clip(torque_pwm, -torque_pwm_max, torque_pwm_max)
    force_des = cmd[3]
    if force_des <= 0:
        torque_pwm = np.zeros(3)
    force_pwm = force2pwm(force_des/4, thrust_max, pwm_max)
    torque_pwm = np.concatenate((torque_pwm[:2]/2, torque_pwm[2:]))
    pwms = force_pwm + torque_pwm @ mixing_matrix
    pwms = np.zeros(4) if np.all(pwms == 0) else np.clip(pwms, pwm_min, pwm_max*pwm_headroom)
    motor_forces = pwm2force(pwms, thrust_max, pwm_max)
    torque = (mixing_matrix @ motor_forces)*np.array([L, L, thrust2torque])
    return np.array([motor_forces.sum()]), torque, r_int


def force_torque2rotor_vel(force, torque, *, thrust_min, thrust_max, L, rpm2thrust, thrust2torque, mixing_matrix):
    """Collective force (1,) and body torque (3,) -> rotor speeds in RPM (4,)."""
    torque_forces = (torque*np.array([1/L, 1/L, 1/thrust2torque])) @ mixing_matrix
    motor_forces = (torque_forces + force)/4
    motor_forces = np.zeros(4) if np.all(force == 0) else np.clip(motor_forces, thrust_min, thrust_max)
    return motor_force2rotor_vel(motor_forces, rpm2thrust)


def motor_force2rotor_vel(motor_forces, rpm2thrust):
    c, b, a = rpm2thrust
    return (-b + np.sqrt(b**2 - 4*a*(c - motor_forces)))/(2*a)
