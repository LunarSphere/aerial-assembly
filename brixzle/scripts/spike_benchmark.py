"""Phase-0 spike: wall-clock cost of the two drone-simulation architectures on this CPU.

(a) hybrid: the drone flies inside the brixzle MuJoCo scene for the whole placement;
(b) split: crazyflow (JAX) flies free-flight segments with the payload as added mass, and
    MuJoCo runs only the contact phases (engage, lift-off, descend, release).
Measures seconds of wall time per simulated second for each regime, then combines them with
a nominal per-brick phase budget. Run: uv run python scripts/spike_benchmark.py [cfenv-python]
"""
import json
import subprocess
import sys
import time

import mujoco
import numpy as np

from aerial_assembly.config import Physics
from brixzle import cad, drone as D, gripper as Gr, lattice as L, scene, structures as S
from brixzle.drone_params import CF21B
from brixzle.estimator import Lighthouse, LighthouseModel
from brixzle.params import V0

# Nominal per-brick phases (s): contact phases need MuJoCo either way.
FREE = {'takeoff/approach': 3.0, 'transit': 5.0, 'retreat/return': 4.0}
CONTACT = {'align+engage': 3.0, 'lift-off': 2.0, 'descend': 2.5, 'release': 2.0}


def mujoco_rate(n_placed, carry, seconds=3.0, sleep=True):
    brick = cad.build_brick(V0)
    g = Gr.build_gripper(Gr.ForkParams(), brick, V0)
    st = S.CATALOG['wall']()
    order = L.assembly_order(V0, st, {'A': brick['outer']})
    phys = Physics(timestep=0.001, friction=0.35, contact_timeconst=0.004, iterations=50)
    n = n_placed + 1
    model, _, info = scene.build_scene(V0, brick, st.bases, n, phys, visual=False, collision='collision_bored',
                                       drone={'cfg': CF21B, 'gripper': g, 'pos': (0, -0.2, 0.3)}, sleep=sleep)
    data = mujoco.MjData(model)
    for j, b in enumerate(order[:n_placed]):
        pos, quat = L.pose(V0, b)
        scene.activate(model, data, info, j, pos, quat)
    lh = Lighthouse(LighthouseModel(), np.random.default_rng(0), model.opt.timestep)
    payload = (g['mass_g'] + (brick['mass_g'] if carry else 0))*1e-3
    dr = D.Drone(model, data, CF21B, lh, ctrl_mass=(CF21B.mass_g*1e-3 + payload)/1.6113)
    start = np.array([0., -0.2, 0.3])
    dr.place(start)
    dr.rotor_vel[:] = 22000
    if carry:
        pos = start*1e3 + np.asarray(g['carry'])
        scene.activate(model, data, info, n - 1, pos, [1, 0, 0, 0])
    mujoco.mj_forward(model, data)
    steps = int(seconds/model.opt.timestep)
    t0 = time.perf_counter()
    for k in range(steps):
        a = k/steps
        dr.cmd = D.full_state(start + np.array([0.15*a, 0.1*a, -0.05*a]))
        D.step(model, data, dr)
    wall = time.perf_counter() - t0
    # The structure must still be standing (sleep must not freeze anything mid-motion).
    for j, b in enumerate(order[:n_placed]):
        pos, _ = L.pose(V0, b)
        assert np.linalg.norm(scene.body_pose(data, info, j)[0] - pos) < 1.0, f'brick {j} moved'
    held = None
    if carry:
        bp, _ = scene.body_pose(data, info, n - 1)
        held = float(np.linalg.norm(bp*1e-3 - (dr.true_state()[0] + np.asarray(g['carry'])*1e-3)))
    return wall/seconds, held


def crazyflow_rate(python):
    code = r'''
import os, time, json, numpy as np
os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.spatial.transform import Rotation as R
from crazyflow.sim import Sim
sim = Sim(control="state")
sim.reset()
c = np.zeros((1, 1, 16)); c[..., 9:13] = R.from_euler("z", 0).as_quat(); c[..., :3] = [0, 0, .5]
for _ in range(10):
    sim.state_control(c); sim.step(sim.freq//sim.control_freq)
t = time.perf_counter()
for k in range(500):
    c[..., 0] = k*1e-3
    sim.state_control(c); sim.step(sim.freq//sim.control_freq)
print(json.dumps({"wall_per_sim_s": (time.perf_counter() - t)/5.0}))
'''
    out = subprocess.run([python, '-c', code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])['wall_per_sim_s']


def main():
    cf_python = sys.argv[1] if len(sys.argv) > 1 else None
    res = {}
    for sleep in (False, True):
        for n_placed in (0, 6, 11):
            for carry in (False, True):
                r, held = mujoco_rate(n_placed, carry, sleep=sleep)
                res[f'mujoco_sleep{int(sleep)}_placed{n_placed}_carry{int(carry)}'] = {
                    'wall_per_sim_s': r, 'held_offset_m': held}
    if cf_python:
        res['crazyflow_free_flight'] = {'wall_per_sim_s': crazyflow_rate(cf_python)}
    free, contact = sum(FREE.values()), sum(CONTACT.values())
    for sleep in (0, 1):
        worst = max(v['wall_per_sim_s'] for k, v in res.items()
                    if k.startswith(f'mujoco_sleep{sleep}') and k.endswith('carry1'))
        res[f'per_brick_hybrid_sleep{sleep}_s'] = (free + contact)*worst
        if cf_python:
            res[f'per_brick_split_sleep{sleep}_s'] = free*res['crazyflow_free_flight']['wall_per_sim_s'] + contact*worst
    res['phases_s'] = {'free': FREE, 'contact': CONTACT}
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
