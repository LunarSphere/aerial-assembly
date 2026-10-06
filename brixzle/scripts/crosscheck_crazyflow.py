"""Fly one open-air waypoint sequence, no payload, in pure crazyflow and in brixzle's MuJoCo port.

Reference (run in an environment with crazyflow installed, e.g. third_party/cfenv):
    python scripts/crosscheck_crazyflow.py reference tests/data/crazyflow_reference.npz
Compare (brixzle env):
    uv run python scripts/crosscheck_crazyflow.py compare tests/data/crazyflow_reference.npz
Both sides use crazyflow's cf21B_500 parameters unchanged (no deck, no thrust rescaling,
default controller mass), no estimator noise, state control at 100 Hz.
"""
import json
import sys

import numpy as np

# (t_start s, x, y, z, yaw rad); each setpoint is held until the next one starts.
WAYPOINTS = [(0., 0., 0., .5, 0.), (2., .3, 0., .5, 0.), (4., .3, .3, .7, 0.), (6., 0., .3, .7, .8),
             (8., 0., 0., .4, 0.), (10., 0., 0., .4, 0.)]
DURATION = 12.
RATE = 100


def setpoints():
    t = np.arange(int(DURATION*RATE))/RATE
    idx = np.searchsorted([w[0] for w in WAYPOINTS], t, side='right') - 1
    return t, np.array([WAYPOINTS[i][1:] for i in idx])


def reference(out):
    import os
    os.environ['SCIPY_ARRAY_API'] = '1'
    from scipy.spatial.transform import Rotation as R
    from crazyflow.sim import Sim
    sim = Sim(control='state', state_freq=RATE)
    sim.reset()
    start = np.array(WAYPOINTS[0][1:4])
    sim.data = sim.data.replace(states=sim.data.states.replace(
        pos=sim.data.states.pos.at[0, 0].set(start),
        rotor_vel=sim.data.states.rotor_vel.at[0, 0].set(np.ones(4)*20000.)))
    t, sp = setpoints()
    cmd = np.zeros((1, 1, 16))
    pos = []
    for row in sp:
        cmd[0, 0, :3] = row[:3]
        cmd[0, 0, 9:13] = R.from_euler('z', row[3]).as_quat()
        sim.state_control(cmd)
        sim.step(sim.freq//RATE)
        pos.append(np.asarray(sim.data.states.pos[0, 0]))
    np.savez(out, t=t, pos=np.array(pos), setpoints=sp)
    print(f'wrote {out}')


def port(rotor0=20000.):
    import mujoco
    from aerial_assembly.config import Physics
    from brixzle import cad, drone as D, scene
    from brixzle.drone_params import DroneConfig
    from brixzle.estimator import Lighthouse, LighthouseModel
    from brixzle.params import V0
    cfg = DroneConfig(deck_mass_g=0., thrust_source='crazyflow')
    phys = Physics(timestep=0.001, friction=0.35, contact_timeconst=0.004, iterations=50)
    start = np.array(WAYPOINTS[0][1:4])
    model, _, _ = scene.build_scene(V0, cad.build_brick(V0, solid=False), [(-20, -16)], 0, phys, visual=False,
                                    drone={'cfg': cfg, 'pos': start})
    data = mujoco.MjData(model)
    quiet = LighthouseModel(bias_sigma_mm=0, bias_gradient_mm_per_m=0, jitter_mm=0, vel_noise_mps=0,
                            att_noise_deg=0, latency_s=0)
    dr = D.Drone(model, data, cfg, Lighthouse(quiet, np.random.default_rng(0), model.opt.timestep))
    dr.place(start)
    dr.rotor_vel[:] = rotor0
    t, sp = setpoints()
    every = round(1/(RATE*model.opt.timestep))
    pos = []
    for row in sp:
        dr.cmd = D.full_state(row[:3], yaw=row[3])
        D.step(model, data, dr, every)
        pos.append(dr.true_state()[0])
    return t, np.array(pos)


def compare(path):
    ref = np.load(path)
    t, pos = port()
    err = np.linalg.norm(pos - ref['pos'], axis=1)
    out = {'rms_m': float(np.sqrt(np.mean(err**2))), 'max_m': float(err.max()),
           'rms_xyz_m': np.sqrt(np.mean((pos - ref['pos'])**2, axis=0)).tolist()}
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    {'reference': reference, 'compare': compare}[sys.argv[1]](sys.argv[2])
