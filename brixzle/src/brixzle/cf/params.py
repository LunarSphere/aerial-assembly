"""cf21B_500 parameters copied from crazyflow d28ec70 (see package docstring).

``CONTROL`` mirrors ``control/mellinger/params.toml`` and ``DYNAMICS`` mirrors
``dynamics/first_principles/params.toml``. The controller mass (39.3 g) deliberately differs
from the physical mass (43.4 g) to mimic firmware behaviour; do not unify them (crazyflow SKILL.md).
"""
import numpy as np

MIXING = np.array([[-1., -1., 1., 1.],
                   [-1., 1., 1., -1.],
                   [-1., 1., -1., 1.]])
RPM2THRUST = np.array([0.0, -3.133427287299859e-7, 4.407354891648379e-10])
RPM2TORQUE = np.array([0.0, 1.65886356219615e-9, 2.4693477924534137e-12])
GRAVITY = np.array([0., 0., -9.81])

CORE = dict(
    mass=0.0393, L=0.035355, thrust2torque=0.00593893393599368, pwm_min=7000.0, pwm_max=65535.0,
    thrust_min=0.02136263065537499, thrust_max=0.2, mass_thrust=132000.,
    torque_pwm_max=np.array([32000., 32000., 32000.]),
)
STATE2ATTITUDE = dict(kp=np.array([0.4, 0.4, 1.25]), kd=np.array([0.2, 0.2, 0.5]),
                      ki=np.array([0.05, 0.05, 0.05]), int_err_max=np.array([2.0, 2.0, 0.4]))
ATTITUDE2FORCE_TORQUE = dict(kR=np.array([7e4, 7e4, 6e4]), kw=np.array([2e4, 2e4, 1.2e4]),
                             ki_m=np.array([0., 0., 500.]), kd_omega=np.array([200., 200., 0.]),
                             int_err_max=np.array([1., 1., 1500.]))

DYNAMICS = dict(
    mass=0.0434, J=np.diag([25e-6, 28e-6, 49e-6]), thrust_min=0.02136263065537499, thrust_max=0.2,
    L=0.035355, prop_inertia=38.93e-9, rpm2thrust=RPM2THRUST, rpm2torque=RPM2TORQUE,
    rotor_dyn_coef=np.array([13.996001897562685, 0.00011093207920685363, 5.933168530682111,
                             0.00031951312393561264]),
    mixing_matrix=MIXING,
    drag_matrix=np.diag([-0.021643637770852733, -0.021643637770852733, -0.02477138502714662]),
)

# Motor sites of cf21B_500.xml, metres in the body frame; order matches the mixing matrix columns.
MOTOR_POS = np.array([[0.03536, -0.03536, 0.], [-0.03536, -0.03536, 0.], [-0.03536, 0.03536, 0.],
                      [0.03536, 0.03536, 0.]])
PROP_RADIUS = 27.5e-3


def state2attitude_params(thrust_max=None, mass=None):
    out = dict(mass=CORE['mass'], gravity_vec=GRAVITY, mass_thrust=CORE['mass_thrust'],
               thrust_max=CORE['thrust_max'], pwm_max=CORE['pwm_max'], **STATE2ATTITUDE)
    if thrust_max is not None:
        out['thrust_max'] = thrust_max
    if mass is not None:
        out['mass'] = mass
    return out


def attitude2force_torque_params(thrust_max=None):
    keys = ('torque_pwm_max', 'thrust_max', 'pwm_min', 'pwm_max', 'L', 'thrust2torque')
    out = dict({k: CORE[k] for k in keys}, mixing_matrix=MIXING, **ATTITUDE2FORCE_TORQUE)
    if thrust_max is not None:
        out['thrust_max'] = thrust_max
    return out


def force_torque2rotor_vel_params(thrust_max=None):
    out = dict(thrust_min=CORE['thrust_min'], thrust_max=CORE['thrust_max'], L=CORE['L'],
               rpm2thrust=RPM2THRUST, thrust2torque=CORE['thrust2torque'], mixing_matrix=MIXING)
    if thrust_max is not None:
        out['thrust_max'] = thrust_max
    return out
