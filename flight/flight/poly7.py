"""Port of the firmware high-level planner's 7th-order, jerk-free segments.

crazyflie-firmware ``src/modules/src/pptraj.c`` (``poly7_nojerk``,
``piecewise_plan_7th_order_no_jerk``, ``piecewise_eval``) and ``planner.c``
(``plan_go_to_from``, ``plan_takeoff_or_landing``): one piece per command, x/y/z/yaw each a
7th-order polynomial with zero jerk at both ends; after the end the setpoint holds the final
position with zero velocity and acceleration.
"""
from dataclasses import dataclass
import math

import numpy as np


def poly7_nojerk(T, x0, dx0, ddx0, xf, dxf, ddxf):
    """Coefficients c0..c7 (ascending powers of t) of one axis."""
    if T <= 0:
        return np.array([xf, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    T2 = T*T
    c = np.zeros(8)
    c[0], c[1], c[2], c[3] = x0, dx0, ddx0/2, 0.
    c[4] = -(5*(14*x0 - 14*xf + 8*T*dx0 + 6*T*dxf + 2*T2*ddx0 - T2*ddxf))/(2*T**4)
    c[5] = (84*x0 - 84*xf + 45*T*dx0 + 39*T*dxf + 10*T2*ddx0 - 7*T2*ddxf)/T**5
    c[6] = -(140*x0 - 140*xf + 72*T*dx0 + 68*T*dxf + 15*T2*ddx0 - 13*T2*ddxf)/(2*T**6)
    c[7] = (2*(10*x0 - 10*xf + 5*T*dx0 + 5*T*dxf + T2*ddx0 - T2*ddxf))/T**7
    return c


def _eval(c, t):
    p = np.polynomial.polynomial
    return p.polyval(t, c), p.polyval(t, p.polyder(c)), p.polyval(t, p.polyder(c, 2))


def normalize_radians(a):
    return math.atan2(math.sin(a), math.cos(a))


def shortest_signed_angle(start, goal):
    return normalize_radians(goal - start)


@dataclass
class TrajEval:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: float = 0.
    yaw_rate: float = 0.


@dataclass
class Piece:
    t_begin: float
    duration: float
    coeffs: np.ndarray   # (4, 8): x, y, z, yaw

    def eval(self, t):
        tau = t - self.t_begin
        if tau >= self.duration:
            ev = self._at(self.duration)
            return TrajEval(ev.pos, np.zeros(3), np.zeros(3), ev.yaw, 0.)
        return self._at(max(0., tau))

    def _at(self, tau):
        vals = [_eval(c, tau) for c in self.coeffs]
        pos = np.array([v[0] for v in vals[:3]])
        vel = np.array([v[1] for v in vals[:3]])
        acc = np.array([v[2] for v in vals[:3]])
        return TrajEval(pos, vel, acc, float(vals[3][0]), float(vals[3][1]))

    def finished(self, t):
        return t - self.t_begin >= self.duration


def plan_7th_order_no_jerk(t, duration, p0, yaw0, v0, w0, a0, p1, yaw1, v1, w1, a1):
    c = [poly7_nojerk(duration, p0[i], v0[i], a0[i], p1[i], v1[i], a1[i]) for i in range(3)]
    c.append(poly7_nojerk(duration, yaw0, w0, 0., yaw1, w1, 0.))
    return Piece(t, duration, np.array(c))


def plan_go_to_from(t, curr: TrajEval, pos, yaw, duration, relative=False, linear=False):
    pos = np.asarray(pos, float)
    if relative:
        pos = pos + curr.pos
        yaw = yaw + curr.yaw
    cy = normalize_radians(curr.yaw)
    delta = shortest_signed_angle(cy, normalize_radians(yaw))
    goal_yaw = cy + delta
    if linear:
        vel = (pos - curr.pos)/duration
        omz = delta/duration
        return plan_7th_order_no_jerk(t, duration, curr.pos, cy, vel, omz, np.zeros(3), pos, goal_yaw, vel, omz,
                                      np.zeros(3))
    return plan_7th_order_no_jerk(t, duration, curr.pos, cy, curr.vel, curr.yaw_rate, curr.acc, pos, goal_yaw,
                                  np.zeros(3), 0., np.zeros(3))


def plan_takeoff_or_landing(t, curr_pos, curr_yaw, height, yaw, duration):
    goal = np.array(curr_pos, float)
    goal[2] = height
    cy = normalize_radians(curr_yaw)
    goal_yaw = cy + shortest_signed_angle(cy, normalize_radians(yaw))
    z = np.zeros(3)
    return plan_7th_order_no_jerk(t, duration, np.asarray(curr_pos, float), cy, z, 0., z, goal, goal_yaw, z, 0., z)


def limits(piece: Piece, samples=200):
    """Peak speed (m/s) and acceleration (m/s^2) along a piece (preflight checks)."""
    ts = piece.t_begin + np.linspace(0, piece.duration, samples)
    evs = [piece._at(t - piece.t_begin) for t in ts]
    return (max(float(np.linalg.norm(e.vel)) for e in evs), max(float(np.linalg.norm(e.acc)) for e in evs))
