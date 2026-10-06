"""Simulated Crazyflie 2.1 Brushless inside the brixzle MuJoCo scene.

MuJoCo integrates the rigid body and resolves every contact (tine/bore, brick/fixture,
gripper/placed bricks). Each physics step this module applies the propulsion wrench of
crazyflow's first-principles rotor model (ported in ``brixzle.cf``), driven by crazyflow's
Mellinger controller at firmware rates (state 100 Hz, attitude 500 Hz) running on the
Lighthouse estimate rather than the true state. Units: SI; MuJoCo quaternions wxyz,
controller quaternions xyzw (converted only through ``brixzle.quat``).
"""
from dataclasses import dataclass
from importlib import resources
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .cf import dynamics as cfd, mellinger as mel, params as cfp
from .cf.rot import quat_to_mat, quat_yaw
from .drone_params import DroneConfig
from .estimator import Lighthouse
from .quat import wxyz_to_xyzw, xyzw_to_wxyz

NAME = 'cf'
GUARD_RADIUS = 0.034     # m, prop-guard ring around each motor (cf21B_prop-guards.stl)
GUARD_HALF_H = 0.0153    # m, guards double as legs: bottom at z = -15.3 mm
FRAME_HALF = (0.041, 0.041, 0.0087)
FRAME_Z = 0.0075
DECK_Z = 0.0185          # m, Lighthouse deck above the battery holder, on the top header pins
DECK_HALF = (0.0135, 0.0195, 0.00175)


def _numbers(values):
    return ' '.join(f'{float(v):.9g}' for v in np.ravel(values))


def mesh_dir():
    return resources.files('brixzle.cf') / 'meshes'


def add_drone(asset, world, cfg: DroneConfig, gripper=None, pos=(0., 0., 0.02), visual=True):
    """Append the drone body (with deck and an optional welded gripper bundle) to an MJCF tree.

    ``gripper`` is a ``gripper.build_gripper`` bundle in millimetres and grams, in the drone
    body frame (origin at the motor plane centre, +z up).
    """
    body = ET.SubElement(world, 'body', name=NAME, pos=_numbers(pos))
    ET.SubElement(body, 'freejoint', name=f'{NAME}_free')
    J = cfp.DYNAMICS['J']
    ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(cfg.body_mass_g*1e-3),
                  diaginertia=_numbers(np.diag(J)))
    col = dict(group='3', contype='1', conaffinity='1', rgba='.2 .2 .2 .3')
    ET.SubElement(body, 'geom', name=f'{NAME}_frame', type='box', size=_numbers(FRAME_HALF),
                  pos=f'0 0 {FRAME_Z}', **col)
    for i, (x, y, _) in enumerate(cfp.MOTOR_POS):
        ET.SubElement(body, 'geom', name=f'{NAME}_guard{i}', type='cylinder',
                      size=f'{GUARD_RADIUS} {GUARD_HALF_H}', pos=f'{x} {y} 0', **col)
    if visual:
        d = mesh_dir()
        for part, rgba in (('frame', '.3 .3 .3 1'), ('guards', '.1 .1 .1 1'), ('props', '.52 .9 .15 1')):
            ET.SubElement(asset, 'mesh', name=f'{NAME}_{part}', file=str(d / f'cf21B_{part}.stl'),
                          scale='1e-3 1e-3 1e-3')
            ET.SubElement(body, 'geom', name=f'{NAME}_vis_{part}', type='mesh', mesh=f'{NAME}_{part}',
                          group='2', contype='0', conaffinity='0', rgba=rgba, mass='0')
    deck = ET.SubElement(body, 'body', name=f'{NAME}_deck', pos=f'0 0 {DECK_Z}')
    m = cfg.deck_mass_g*1e-3
    a, b, c = (2*h for h in DECK_HALF)
    ET.SubElement(deck, 'inertial', pos='0 0 0', mass=str(m),
                  diaginertia=_numbers([m*(b*b + c*c)/12, m*(a*a + c*c)/12, m*(a*a + b*b)/12]))
    ET.SubElement(deck, 'geom', name=f'{NAME}_deck_vis', type='box', size=_numbers(DECK_HALF), group='2',
                  contype='0', conaffinity='0', rgba='.1 .4 .1 1', mass='0')
    if gripper is not None:
        _add_gripper(asset, body, gripper, visual)
    return body


def _add_gripper(asset, body, g, visual):
    MM = 1e-3
    grip = ET.SubElement(body, 'body', name=f'{NAME}_gripper')
    I = np.asarray(g['inertia'])*1e-9
    ET.SubElement(grip, 'inertial', mass=str(g['mass_g']*1e-3), pos=_numbers(np.asarray(g['com'])*MM),
                  fullinertia=_numbers([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
    for k, piece in enumerate(g['collision']):
        name = f'{NAME}_grip_c{k}'
        ET.SubElement(asset, 'mesh', name=name, vertex=_numbers(np.asarray(piece)*MM))
        ET.SubElement(grip, 'geom', name=name, type='mesh', mesh=name, group='3', contype='1',
                      conaffinity='1', rgba='.8 .2 .2 .4', **g.get('geom_attrs', {}))
    if visual and 'mesh' in g:
        mesh = g['mesh']
        ET.SubElement(asset, 'mesh', name=f'{NAME}_grip_vis', vertex=_numbers(mesh.vertices*MM),
                      face=' '.join(str(int(i)) for i in np.ravel(mesh.faces)))
        ET.SubElement(grip, 'geom', name=f'{NAME}_grip_vis', type='mesh', mesh=f'{NAME}_grip_vis', group='2',
                      contype='0', conaffinity='0', rgba='.85 .85 .85 1', mass='0')


@dataclass(frozen=True)
class ControlRates:
    state_hz: int = 100
    attitude_hz: int = 500


def firmware_mass(total_kg):
    """crazyflow controller mass that makes the hover feed-forward equal the true weight.

    crazyflow's linear PWM map scales the Mellinger thrust by k = mass_thrust * 4 * thrust_max / pwm_max
    (1.61 for cf21B_500), so with its default mass the drone hovers about 8 cm high (crazyflow itself
    does too). The sim reads the firmware's ``ctrlMel.mass`` (true kg) through this map.
    """
    c = cfp.CORE
    return total_kg/(c['mass_thrust']*4*c['thrust_max']/c['pwm_max'])


def full_state(pos, vel=(0, 0, 0), acc=(0, 0, 0), yaw=0., rates=(0, 0, 0)):
    """Mellinger full-state command [p, v, a, q_xyzw, w] (SI, yaw in rad)."""
    out = np.zeros(16)
    out[0:3], out[3:6], out[6:9], out[13:16] = pos, vel, acc, rates
    out[11], out[12] = np.sin(yaw/2), np.cos(yaw/2)
    return out


class Drone:
    """Controller + rotor model bound to the ``cf`` body; call ``before_step`` before each mj_step."""

    def __init__(self, model, data, cfg: DroneConfig, lighthouse: Lighthouse, rates=ControlRates(),
                 ctrl_mass=None, aero=None):
        self.model, self.data, self.cfg, self.lh, self.aero = model, data, cfg, lighthouse, aero
        self.body = model.body(NAME).id
        j = model.joint(f'{NAME}_free').id
        self.qpos, self.qvel = model.jnt_qposadr[j], model.jnt_dofadr[j]
        dt = model.opt.timestep
        self.att_every = max(1, round(1/(rates.attitude_hz*dt)))
        self.state_every = self.att_every*max(1, round(rates.attitude_hz/rates.state_hz))
        self.state_freq, self.att_freq = 1/(self.state_every*dt), 1/(self.att_every*dt)
        lighthouse.set_rate(self.att_every*dt)
        thrust = cfg.thrust_per_motor_N[cfg.thrust_source]
        self.p_state = cfp.state2attitude_params(mass=ctrl_mass)
        self.p_att = dict(cfp.attitude2force_torque_params(), pwm_headroom=thrust/cfp.CORE['thrust_max'])
        self.p_ft = cfp.force_torque2rotor_vel_params(thrust_max=thrust)
        self.p_dyn = dict(cfp.DYNAMICS)
        self.step = 0
        self.reset_controller()
        self.cmd = None          # full-state setpoint (16,), None => motors off
        self.level = False       # watchdog level-out: roll = pitch = 0, hold z
        self.thrusts = np.zeros(4)
        self.last_estimate = None

    def reset_controller(self):
        self.pos_err_i = np.zeros(3)
        self.r_int = np.zeros(3)
        self.prev_w = np.zeros(3)
        self.rpyt = np.zeros(4)
        self.rotor_cmd = np.zeros(4)
        self.rotor_vel = np.zeros(4)

    def set_ctrl_mass(self, kg):
        self.p_state = dict(self.p_state, mass=kg)

    def place(self, pos, yaw=0.):
        self.data.qpos[self.qpos:self.qpos+3] = pos
        self.data.qpos[self.qpos+3:self.qpos+7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
        self.data.qvel[self.qvel:self.qvel+6] = 0.

    def true_state(self):
        d = self.data
        pos = d.qpos[self.qpos:self.qpos+3].copy()
        quat = wxyz_to_xyzw(d.qpos[self.qpos+3:self.qpos+7])
        vel = d.qvel[self.qvel:self.qvel+3].copy()      # world frame
        ang_vel = d.qvel[self.qvel+3:self.qvel+6].copy()  # body frame
        return pos, quat, vel, ang_vel

    def before_step(self):
        """Update estimate, controller, rotors and the applied wrench at the attitude rate (500 Hz).

        Between updates the wrench is held, as in crazyflow's 500 Hz integration.
        """
        if self.step % self.att_every:
            self.step += 1
            return
        pos, quat, vel, w = self.true_state()
        est = self.lh.observe(pos, quat, vel, w)
        self.last_estimate = est
        if self.cmd is None:
            self.rotor_cmd = np.zeros(4)
            self.reset_integrators()
        else:
            if self.step % self.state_every == 0:
                cmd = self.cmd
                if self.level:
                    cmd = full_state([est[0][0], est[0][1], cmd[2]], yaw=quat_yaw(est[1]))
                self.rpyt, self.pos_err_i = mel.state2attitude(est[0], est[1], est[2], cmd, self.pos_err_i,
                                                               self.state_freq, **self.p_state)
                if self.level:
                    self.rpyt[:2] = 0.
            force, torque, self.r_int = mel.attitude2force_torque(
                est[1], est[3], self.rpyt, self.prev_w, self.r_int, self.att_freq, **self.p_att)
            self.prev_w = est[3]
            self.rotor_cmd = mel.force_torque2rotor_vel(force, torque, **self.p_ft)
        dt = self.model.opt.timestep*self.att_every
        rv_dot = cfd.rotor_vel_dot(self.rotor_cmd, self.rotor_vel, self.p_dyn['rotor_dyn_coef'])
        self.rotor_vel = np.maximum(self.rotor_vel + rv_dot*dt, 0.)
        rot = quat_to_mat(quat)
        scale = self.aero.thrust_scale(self, pos, rot) if self.aero is not None else None
        force_b, torque_b, self.thrusts = cfd.body_wrench(self.rotor_vel, rv_dot, w, rot.T @ vel,
                                                          thrust_scale=scale, **self.p_dyn)
        self.data.xfrc_applied[self.body, :3] = rot @ force_b
        self.data.xfrc_applied[self.body, 3:] = rot @ torque_b
        if self.aero is not None:
            self.aero.apply(self, pos, rot)
        self.step += 1

    def reset_integrators(self):
        self.pos_err_i = np.zeros(3)
        self.r_int = np.zeros(3)


def step(model, data, drone, n=1):
    for _ in range(n):
        drone.before_step()
        mujoco.mj_step(model, data)
