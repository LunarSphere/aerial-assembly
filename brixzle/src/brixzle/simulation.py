"""Force-limited passive-fork manipulation with physical contact only."""
from __future__ import annotations

from dataclasses import asdict, replace
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from pathlib import Path
import math
from threading import RLock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .assembly import voxel_position
from .geometry import generate_design
from .models import AssemblyPlan, Design, RunConfig, TrialResult, UncertaintyProfile, write_json

_WARNINGS = ContextVar("brixzle_solver_warnings", default=None)
_WARNING_LOCK = RLock()


def _capture_warnings(function):
    """Capture engine warnings, including Newton linesearch warnings.

    MuJoCo's Python callback is process-global. Trials in one process are
    serialized here; campaign parallelism uses separate worker processes.
    """
    sig = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _WARNING_LOCK:
            messages = []
            previous = mujoco.get_mju_user_warning()
            token = _WARNINGS.set(messages)
            def capture(message):
                message = str(message)
                if message not in messages and len(messages) < 32:
                    messages.append(message)
            mujoco.set_mju_user_warning(capture)
            try:
                result = function(*args, **kwargs)
                if messages:
                    result.success = False
                    result.numerical_warning = True
                    result.reason = result.reason or "numerical_instability"
                    result.disturbance["solver_warnings"] = messages
                    folder = sig.bind_partial(*args, **kwargs).arguments.get("output_dir")
                    if folder is not None:
                        write_json(folder / "trial.json", result)
                return result
            finally:
                mujoco.set_mju_user_warning(previous)
                _WARNINGS.reset(token)
    return wrapped


def quat(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z])


def rotation(q) -> Rotation:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _numbers(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def _element(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})


def sample_disturbance(profile: UncertaintyProfile, seed: int, n: int) -> dict:
    # A fixed draw layout gives every geometry the same disturbances.
    rng = np.random.default_rng(seed)
    return {
        "friction": float(rng.uniform(profile.friction_min, profile.friction_max)),
        "clearance_error_mm": float(rng.uniform(-profile.clearance_error_mm, profile.clearance_error_mm)),
        "mass_scale": float(rng.uniform(1-profile.mass_fraction, 1+profile.mass_fraction)),
        "placements": [{"position_mm": rng.uniform(-1, 1, 3).__mul__(
            [profile.lateral_mm, profile.lateral_mm, profile.vertical_mm]).tolist(),
            "angles_deg": rng.uniform(-1, 1, 3).__mul__(
                [profile.tilt_deg, profile.tilt_deg, profile.yaw_deg]).tolist()}
            for _ in range(n)],
        "tracking_seed": int(rng.integers(0, 2**31)),
    }


def build_scene(design: Design, plan: AssemblyPlan, config: RunConfig,
                disturbance: dict) -> tuple[str, dict]:
    p, f, s = design.parameters.pitch_mm/1000, config.fork, config.simulation
    root = ET.Element("mujoco", model="brixzle")
    _element(root, "compiler", angle="radian", inertiafromgeom="false")
    _element(root, "option", timestep=s.timestep_s, gravity="0 0 -9.81", solver="Newton",
             iterations=s.solver_iterations, tolerance="1e-10", cone="elliptic", integrator="implicitfast")
    _element(root.find("option"), "flag", multiccd="enable")
    defaults = _element(root, "default")
    _element(defaults, "geom", friction=f'{disturbance["friction"]} 0.002 0.0001',
             solref=f"{s.contact_timeconst_s} 1", solimp="0.95 0.99 0.0001", condim="4", margin="0")
    assets = _element(root, "asset")
    geometry_specs = []
    for i, piece in enumerate(design.collision_pieces):
        vertices = np.asarray(piece.vertices_mm)/1000
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        is_box = (len(vertices) == 8 and np.all(np.isclose(vertices, lo, atol=1e-11, rtol=0) |
                                               np.isclose(vertices, hi, atol=1e-11, rtol=0)))
        if is_box:
            geometry_specs.append({"type": "box", "pos": _numbers((lo+hi)/2), "size": _numbers((hi-lo)/2)})
        else:
            _element(assets, "mesh", name=f"piece{i}", vertex=_numbers(vertices.ravel()),
                     face=" ".join(str(v) for face in piece.faces for v in face))
            geometry_specs.append({"type": "mesh", "mesh": f"piece{i}"})
    world = _element(root, "worldbody")
    _element(world, "light", pos="0 -1 2", diffuse="0.8 0.8 0.8")
    _element(world, "camera", name="global", pos="0.45 -0.65 0.55", xyaxes="1 0.4 0 -0.2 0.5 1")
    _element(world, "geom", name="ground", type="plane", size="1 1 0.01", rgba="0.25 0.28 0.3 1")
    target_z = 0.04 if plan.target.name != "pickup" else 0.0
    mapping = {"names": {}, "targets": {}, "staging": {}, "target_z": target_z}
    order_index = {name: i for i, name in enumerate(plan.order)}
    for i, voxel in enumerate(plan.target.voxels):
        name = f"brick{i}"
        mapping["names"][voxel.id] = name
        target = voxel_position(voxel, design.parameters.pitch_mm, target_z)
        mapping["targets"][voxel.id] = target
        q = quat(Rotation.from_euler("z", voxel.yaw_quarters*math.pi/2))
        if voxel.anchored:
            _element(world, "geom", name=f"fixture{i}", type="box", pos=_numbers([target[0], target[1], target_z/2]),
                     size=_numbers([p/2, p/2, target_z/2]), rgba="0.4 0.4 0.45 1")
            pos = target
        else:
            j = order_index.get(voxel.id, i)
            pos = np.array([-(3+2*j)*p, -3*p, p/2])
            mapping["staging"][voxel.id] = pos
        body = _element(world, "body", name=name, pos=_numbers(pos), quat=_numbers(q))
        if not voxel.anchored:
            if s.manipulation_mode == "floating":
                body.set("gravcomp", "1")  # external staging placeholder until introduced
            _element(body, "freejoint", name=name+"_joint")
        scale = disturbance["mass_scale"]
        inertia = np.asarray(design.inertia_kg_m2)*scale
        _element(body, "inertial", pos=_numbers(np.asarray(design.center_of_mass_mm)/1000),
                 mass=design.mass_g/1000*scale,
                 fullinertia=_numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                      inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
        for j, attributes in enumerate(geometry_specs):
            _element(body, "geom", name=f"{name}_g{j}", **attributes, rgba="0.85 0.55 0.22 1")
    # Two tines and a rear bar, without any actuated grasp or attachment.
    if plan.order:
        first = plan.order[0]
        staging = mapping["staging"][first]
        first_voxel = next(v for v in plan.target.voxels if v.id == first)
        first_rotation = Rotation.from_euler("z", first_voxel.yaw_quarters*math.pi/2)
        start = staging+first_rotation.apply([0, -p/2-0.013-f.length_mm/1000, 0.05])
        start_quat = quat(first_rotation)
    else:
        start, start_quat = np.array([-0.5, -0.5, 0.25]), np.array([1, 0, 0, 0])
    fork = _element(world, "body", name="fork", pos=_numbers(start), quat=_numbers(start_quat))
    if s.manipulation_mode == "fork":
        _element(fork, "freejoint", name="fork_joint")
    _element(fork, "inertial", pos="0 0 0", mass=f.mass_g/1000,
             diaginertia=_numbers(np.array([f.length_mm**2, f.length_mm**2, f.length_mm**2+f.spacing_mm**2])*f.mass_g/1000*1e-6/12))
    for i, x in enumerate((-f.spacing_mm/2, f.spacing_mm/2)):
        _element(fork, "geom", name=f"tine{i}", type="box", pos=_numbers(np.array([x, f.length_mm/2, 0])/1000),
                 size=_numbers(np.array([f.width_mm, f.length_mm, f.thickness_mm])/2000), rgba="0.3 0.6 0.8 1")
        if f.lip_height_mm:
            _element(fork, "geom", name=f"lip{i}", type="box",
                     pos=_numbers(np.array([x, f.length_mm+0.4, f.thickness_mm/2+f.lip_height_mm/2])/1000),
                     size=_numbers(np.array([f.width_mm, 0.8, f.lip_height_mm])/2000), rgba="0.3 0.6 0.8 1")
    _element(fork, "geom", name="bar", type="box", pos=_numbers([0, -f.bar_depth_mm/2000, 0]),
             size=_numbers(np.array([f.spacing_mm+f.width_mm+2, f.bar_depth_mm, f.thickness_mm])/2000))
    return ET.tostring(root, encoding="unicode"), mapping


class PhysicsFailure(RuntimeError):
    pass


def _check_printed_mass(actual, config, disturbance):
    mass = actual.mass_g*disturbance["mass_scale"]
    disturbance["simulated_brick_mass_g"] = mass
    p = config.printing
    if not p.min_mass_g <= mass <= p.max_mass_g or mass+config.fork.mass_g > p.payload_g:
        raise PhysicsFailure("printed_mass_limit")
    return mass


class _Runner:
    def __init__(self, model, data, design, plan, config, mapping, disturbance):
        self.model, self.data, self.design, self.plan, self.config = model, data, design, plan, config
        self.mapping, self.disturbance = mapping, disturbance
        self.fork = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fork")
        self.fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fork_joint")
        self.dof = model.jnt_dofadr[self.fj]
        self.bricks = {key: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, val)
                       for key, val in mapping["names"].items()}
        self.reverse = {val: key for key, val in self.bricks.items()}
        self.allowed_pairs = {frozenset((self.bricks[a], self.bricks[b])) for a, b, _, _ in plan.connections}
        self.peak_force = 0.0
        self.collisions = 0
        self.previous_bad = set()
        self.last_bad_time = {}
        self.active = None
        self.rng = np.random.default_rng(disturbance["tracking_seed"])
        self.noise = np.zeros(3)
        self.noise_until = 0.0
        self.placement_start = 0.0
        self.phase = "approach"

    def step(self, goal, goal_q):
        m, d, s = self.model, self.data, self.config.simulation
        if d.time-self.placement_start > s.placement_timeout_s:
            raise PhysicsFailure("placement_timeout")
        if d.time >= self.noise_until:
            self.noise = self.rng.uniform(-1, 1, 3)*self.config.uncertainty.tracking_mm/1000
            self.noise_until = d.time+0.2
        velocity = d.qvel[self.dof:self.dof+3]
        force = s.position_gain_n_m*(goal+self.noise-d.xpos[self.fork])-s.damping_n_s_m*velocity
        force[2] += self.config.fork.mass_g/1000*9.81
        norm = np.linalg.norm(force)
        if norm > s.max_force_n:
            force *= s.max_force_n/norm
        current = rotation(d.xquat[self.fork])
        rot_error = (rotation(goal_q)*current.inv()).as_rotvec()
        angular_world = current.apply(d.qvel[self.dof+3:self.dof+6])
        torque = s.rotation_gain_nm_rad*rot_error-s.rotation_damping_nm_s*angular_world
        tnorm = np.linalg.norm(torque)
        if tnorm > s.max_torque_nm:
            torque *= s.max_torque_nm/tnorm
        d.xfrc_applied[:] = 0
        d.xfrc_applied[self.fork, :3] = force
        d.xfrc_applied[self.fork, 3:] = torque
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or any(w.number for w in d.warning) or _WARNINGS.get():
            raise PhysicsFailure("numerical_instability")
        if np.linalg.norm(d.xpos[self.fork]-goal) > 0.15:
            raise PhysicsFailure("controller_tracking_lost")
        bad = set()
        pair_forces = {}
        wrench = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            a, b = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if (a == 0 and b not in (self.active, self.fork)) or (b == 0 and a not in (self.active, self.fork)):
                continue  # Unrelated staged/previous table support has no metric contribution.
            mujoco.mj_contactForce(m, d, i, wrench)
            pair = frozenset((a, b))
            pair_forces[pair] = pair_forces.get(pair, 0.0)+float(np.linalg.norm(wrench[:3]))
            intended = pair in self.allowed_pairs
            if a == self.fork or b == self.fork:
                other = b if a == self.fork else a
                intended = False
                if other == self.active:
                    local = rotation(d.xquat[other]).inv().apply(con.pos-d.xpos[other])*1000
                    f = self.config.fork
                    tunnel_bottom = self.design.pickup_surface_mm-f.thickness_mm-f.lip_height_mm-2*f.clearance_mm
                    near_tine = min(abs(local[0]-f.spacing_mm/2), abs(local[0]+f.spacing_mm/2)) <= f.width_mm/2+f.clearance_mm+0.5
                    intended = (near_tine and abs(local[1]) <= self.design.parameters.pitch_mm/2+1 and
                                tunnel_bottom-0.5 <= local[2] <= self.design.pickup_surface_mm+f.lip_height_mm+1)
            elif a == 0 or b == 0:
                # The table supports staged bricks and pickup trials. Unsupported
                # target bricks touching the floor are a collapse, not success.
                intended = True
            if not intended and np.linalg.norm(wrench[:3]) > s.collision_force_threshold_n:
                bad.add(pair)
        # Count sustained events once, allowing brief contact-solver gaps.
        self.collisions += sum(d.time-self.last_bad_time.get(pair, -1e6) > 0.05 for pair in bad)
        for pair in bad:
            self.last_bad_time[pair] = d.time
        self.previous_bad = bad
        active_forces = [value for pair, value in pair_forces.items() if self.fork in pair or self.active in pair]
        self.peak_force = max(self.peak_force, np.linalg.norm(force), max(active_forces, default=0))

    def move(self, goal, goal_q, speed=None, dwell=0.15):
        s, d = self.config.simulation, self.data
        start, qstart = d.xpos[self.fork].copy(), d.xquat[self.fork].copy()
        goal = np.asarray(goal, dtype=float)
        duration = max(0.10, np.linalg.norm(goal-start)/(speed or s.move_speed_m_s))
        rstart, rend = rotation(qstart), rotation(goal_q)
        delta = (rend*rstart.inv()).as_rotvec()
        steps = math.ceil(duration/s.timestep_s)
        for i in range(steps):
            u = (i+1)/steps
            # Smooth endpoints reduce inertial slip in a passive grasp.
            fraction = u*u*(3-2*u)
            self.step(start+(goal-start)*fraction, quat(Rotation.from_rotvec(delta*fraction)*rstart))
        for _ in range(math.ceil(dwell/s.timestep_s)):
            self.step(goal, goal_q)

    def pose_errors(self, placed):
        results = {}
        for voxel in placed:
            bid = self.bricks[voxel.id]
            target = self.mapping["targets"][voxel.id]
            nominal = Rotation.from_euler("z", voxel.yaw_quarters*math.pi/2)
            angle = math.degrees((rotation(self.data.xquat[bid])*nominal.inv()).magnitude())
            results[voxel.id] = {"position_mm": float(np.linalg.norm(self.data.xpos[bid]-target)*1000),
                                 "angle_deg": angle}
        return results

    def hold_and_check(self, goal, q, placed):
        s = self.config.simulation
        # Allow the released brick to finish its gravity-driven seating motion.
        for _ in range(math.ceil(s.settle_s/s.timestep_s)):
            self.step(goal, q)
        maxima = {v.id: {"position_mm": 0.0, "angle_deg": 0.0} for v in placed}
        for _ in range(math.ceil(s.hold_s/s.timestep_s)):
            self.step(goal, q)
            errors = self.pose_errors(placed)
            for key, values in errors.items():
                for metric, value in values.items():
                    maxima[key][metric] = max(maxima[key][metric], value)
            if any(e["position_mm"] > s.position_tolerance_mm or e["angle_deg"] > s.angle_tolerance_deg
                   for e in errors.values()):
                return False, maxima
        return True, maxima


@_capture_warnings
def run_trial(design: Design, plan: AssemblyPlan, uncertainty: UncertaintyProfile | None = None,
              seed: int = 0, config: RunConfig | None = None, output_dir: Path | None = None,
              manufactured_design: Design | None = None) -> TrialResult:
    """Execute a complete benchmark; no contact is replaced by a grasp weld."""
    config = config or RunConfig()
    if uncertainty is not None:
        config = replace(config, uncertainty=uncertainty)
    config.validate()
    disturbed = sample_disturbance(config.uncertainty, seed, len(plan.order))
    cap = config.simulation.placement_timeout_s*max(1, len(plan.order))
    if not plan.feasible or not design.feasible:
        return TrialResult(plan.target.name, seed, False, plan.reason or "infeasible_design", 0, cap, 0,
                           design.mass_g, disturbance=disturbed,
                           manipulation_mode=config.simulation.manipulation_mode,
                           transport_simulated=config.simulation.manipulation_mode == "fork")
    if config.simulation.manipulation_mode == "floating":
        return _floating_trial(design, plan, seed, config, disturbed, output_dir, manufactured_design)
    records = []
    runner = None
    try:
        if manufactured_design is None and abs(disturbed["clearance_error_mm"]) > 1e-12:
            manufactured_design = generate_design(replace(design.parameters,
                clearance_mm=design.parameters.clearance_mm+disturbed["clearance_error_mm"]), config)
        actual = manufactured_design or design
        _check_printed_mass(actual, config, disturbed)
        scene, mapping = build_scene(actual, plan, config, disturbed)
        model = mujoco.MjModel.from_xml_string(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        runner = _Runner(model, data, actual, plan, config, mapping, disturbed)
        voxels = {v.id: v for v in plan.target.voxels}
        placed = [v for v in plan.target.voxels if v.anchored]
        f, p = config.fork, actual.parameters.pitch_mm/1000
        grip_offset = np.array([0, p/2+f.clearance_mm/1000-f.length_mm/1000,
                               (actual.pickup_surface_mm-f.thickness_mm/2-f.lip_height_mm-f.clearance_mm)/1000])
        for index, name in enumerate(plan.order):
            voxel = voxels[name]
            highest = max(mapping["targets"][v.id][2] for v in placed+[voxel])
            safe_z = highest+actual.parameters.pitch_mm/2000+0.025
            bid = runner.bricks[name]
            runner.active = bid
            runner.placement_start = data.time
            snapshots = []
            def snapshot(phase):
                snapshots.append({"phase": phase, "brick_position_m": data.xpos[bid].tolist(),
                                  "fork_position_m": data.xpos[runner.fork].tolist(),
                                  "brick_quaternion": data.xquat[bid].tolist()})
            nominal = Rotation.from_euler("z", voxel.yaw_quarters*math.pi/2)
            q = quat(nominal)
            staging = mapping["staging"][name]
            pickup = staging+nominal.apply(grip_offset)
            rear = nominal.apply([0, -(f.length_mm/1000+0.008), 0])
            approach = pickup+rear
            runner.phase = "approach"
            runner.move([*data.xpos[runner.fork][:2], safe_z], data.xquat[runner.fork])
            runner.move([approach[0], approach[1], safe_z], q)
            runner.move(approach, q)
            runner.phase = "pickup"
            runner.move(pickup, q, config.simulation.insert_speed_m_s)
            runner.phase = "lift"
            lifted = pickup.copy()
            lifted[2] = safe_z
            runner.move(lifted, q, dwell=0.3)
            snapshot("lift")
            if data.xpos[bid][2] < staging[2]+p/2:
                raise PhysicsFailure("missed_pickup")
            expected = data.xpos[runner.fork]-nominal.apply(grip_offset)
            if np.linalg.norm(data.xpos[bid]-expected) > p/2:
                raise PhysicsFailure("lost_payload")
            error = disturbed["placements"][index]
            pose_rotation = nominal*Rotation.from_euler("xyz", error["angles_deg"], degrees=True)
            pose_error = nominal.apply(np.array(error["position_mm"])/1000+
                                       [actual.parameters.approach_bias_mm/1000, 0, 0])
            destination = mapping["targets"][name]+pose_error
            goal = destination+pose_rotation.apply(grip_offset)
            error_q = quat(pose_rotation)
            runner.phase = "transport"
            runner.move([goal[0], goal[1], safe_z], error_q)
            snapshot("transport")
            expected = data.xpos[runner.fork]-pose_rotation.apply(grip_offset)
            if np.linalg.norm(data.xpos[bid]-expected) > p/2:
                raise PhysicsFailure("lost_payload")
            runner.phase = "insert"
            runner.move(goal, error_q, config.simulation.insert_speed_m_s, dwell=0.3)
            snapshot("insert")
            runner.phase = "withdraw"
            withdrawn = goal+pose_rotation.apply([0, -(f.length_mm/1000+0.008), 0])
            runner.move(withdrawn, error_q, config.simulation.insert_speed_m_s)
            snapshot("withdraw")
            runner.phase = "retreat"
            retreat = np.array([withdrawn[0], withdrawn[1], safe_z])
            runner.move(retreat, q)
            placed.append(voxel)
            runner.phase = "stability"
            stable, errors = runner.hold_and_check(retreat, q, placed)
            records.append({"brick": name, "success": stable, "errors": errors,
                            "time_s": float(data.time-runner.placement_start), "phase": runner.phase,
                            "snapshots": snapshots})
            if not stable:
                own = errors[name]
                reason = "collapse" if any(e["position_mm"] > actual.parameters.pitch_mm/2 for e in errors.values()) else "seating_error"
                raise PhysicsFailure(reason)
        result = TrialResult(plan.target.name, seed, True, None, float(runner.peak_force),
                             float(data.time), runner.collisions, disturbed["simulated_brick_mass_g"],
                             records, disturbed)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "scene.xml").write_text(scene)
            write_json(output_dir / "final_state.json", {"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist()})
    except (PhysicsFailure, ValueError, RuntimeError) as exc:
        reason = str(exc) if isinstance(exc, PhysicsFailure) else f"simulation_error: {type(exc).__name__}: {exc}"
        if runner is not None and runner.active is not None and (not records or records[-1]["success"]):
            records.append({"brick": runner.reverse.get(runner.active), "success": False,
                            "phase": runner.phase, "reason": reason,
                            "snapshots": snapshots if "snapshots" in locals() else []})
        result = TrialResult(plan.target.name, seed, False, reason,
                             float(runner.peak_force) if runner else 0.0,
                             float(runner.data.time) if runner else cap,
                             runner.collisions if runner else 0, disturbed.get("simulated_brick_mass_g", design.mass_g), records, disturbed,
                             "numerical" in reason)
        if output_dir is not None and runner is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "scene.xml").write_text(scene)
            write_json(output_dir / "final_state.json", {"qpos": runner.data.qpos.tolist(), "qvel": runner.data.qvel.tolist()})
    if output_dir is not None:
        write_json(output_dir / "trial.json", result)
        write_json(output_dir / "config.json", config)
    return result


def _floating_trial(design, plan, seed, config, disturbance, output_dir, manufactured_design):
    """External transport proxy followed by unassisted gravity/contact seating.

    Initializing each new brick above its connector represents the unmodeled
    transport step. Previously placed bricks are never reset or constrained.
    A compliant virtual carrier lowers the brick before release. No force,
    weld, or pose correction is applied after release.
    """
    s = config.simulation
    records, peak, collisions, last_bad = [], 0.0, 0, {}
    data, scene = None, None
    reason = None
    try:
        actual = manufactured_design
        if actual is None:
            delta = disturbance["clearance_error_mm"]
            actual = generate_design(replace(design.parameters, clearance_mm=design.parameters.clearance_mm+delta), config) if delta else design
        _check_printed_mass(actual, config, disturbance)
        scene, mapping = build_scene(actual, plan, config, disturbance)
        model = mujoco.MjModel.from_xml_string(scene)
        data = mujoco.MjData(model)
        voxels = {v.id: v for v in plan.target.voxels}
        bodies = {key: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, value) for key, value in mapping["names"].items()}
        # Compile with all collision shapes enabled so their bounding-volume
        # trees remain available. Exclude externally staged future bricks only
        # after compilation, and restore both geom and body masks on introduction.
        for v in plan.target.voxels:
            if not v.anchored:
                bid = bodies[v.id]
                start, count = model.body_geomadr[bid], model.body_geomnum[bid]
                model.geom_contype[start:start+count] = 0
                model.geom_conaffinity[start:start+count] = 0
                model.body_contype[bid] = 0
                model.body_conaffinity[bid] = 0
        mujoco.mj_forward(model, data)
        placed = [v for v in plan.target.voxels if v.anchored]
        expected = {frozenset((bodies[a], bodies[b])) for a, b, _, _ in plan.connections}
        for index, name in enumerate(plan.order):
            voxel, bid = voxels[name], bodies[name]
            start, count = model.body_geomadr[bid], model.body_geomnum[bid]
            model.geom_contype[start:start+count] = 1
            model.geom_conaffinity[start:start+count] = 1
            model.body_contype[bid] = 1
            model.body_conaffinity[bid] = 1
            model.body_gravcomp[bid] = 0
            j = model.body_jntadr[bid]
            qa, va = model.jnt_qposadr[j], model.jnt_dofadr[j]
            nominal = Rotation.from_euler("z", voxel.yaw_quarters*math.pi/2)
            error = disturbance["placements"][index]
            estimated = nominal*Rotation.from_euler("xyz", error["angles_deg"], degrees=True)
            offset = nominal.apply(np.array(error["position_mm"])/1000+
                                   [actual.parameters.approach_bias_mm/1000, 0, 0])
            tracking_rng = np.random.default_rng(np.random.SeedSequence([disturbance["tracking_seed"], index]))
            offset += tracking_rng.uniform(-1, 1, 3)*config.uncertainty.tracking_mm/1000
            connected = [v for v in placed if any({name, v.id} == {a, b} for a, b, _, _ in plan.connections)]
            side_mate = any(v.cell[2] == voxel.cell[2] for v in connected)
            height_mm = (actual.parameters.pitch_mm+actual.parameters.guide_height_mm
                         if side_mate else actual.parameters.peg_height_mm+1.0)
            # Sufficient clearance for the small provisional roll/pitch envelope.
            height_mm += actual.parameters.pitch_mm*math.sin(math.radians(config.uncertainty.tilt_deg))
            release = mapping["targets"][name]+offset+[0, 0, height_mm/1000]
            data.qpos[qa:qa+3] = release
            data.qpos[qa+3:qa+7] = quat(estimated)
            data.qvel[va:va+6] = 0
            data.qacc_warmstart[:] = 0
            mujoco.mj_forward(model, data)
            placed.append(voxel)
            placed_body_ids = {bodies[v.id] for v in placed}
            nominal_quats = {v.id: quat(Rotation.from_euler("z", v.yaw_quarters*math.pi/2)) for v in placed}
            maxima = {v.id: {"position_mm": 0.0, "angle_deg": 0.0} for v in placed}
            started = data.time
            final_carrier = mapping["targets"][name]+offset+[0, 0, s.floating_release_gap_mm/1000]
            approach_s = max(0.1, (height_mm-s.floating_release_gap_mm)/1000/s.insert_speed_m_s)
            duration = approach_s+s.settle_s+s.hold_s
            if duration > s.placement_timeout_s:
                raise PhysicsFailure("placement_timeout")
            stable = True
            for _ in range(math.ceil(duration/s.timestep_s)):
                elapsed = data.time-started
                data.xfrc_applied[:] = 0
                if elapsed < approach_s:
                    fraction = elapsed/approach_s
                    desired = release+(final_carrier-release)*fraction
                    desired_velocity = (final_carrier-release)/approach_s
                    force = (s.floating_position_gain_n_m*(desired-data.xpos[bid])+
                             s.floating_damping_n_s_m*(desired_velocity-data.qvel[va:va+3]))
                    force[2] += model.body_mass[bid]*9.81
                    norm = np.linalg.norm(force)
                    if norm > s.max_force_n:
                        force *= s.max_force_n/norm
                    current = rotation(data.xquat[bid])
                    torque = (s.floating_rotation_gain_nm_rad*(estimated*current.inv()).as_rotvec()-
                              s.floating_rotation_damping_nm_s*current.apply(data.qvel[va+3:va+6]))
                    norm = np.linalg.norm(torque)
                    if norm > s.max_torque_nm:
                        torque *= s.max_torque_nm/norm
                    data.xfrc_applied[bid, :3] = force
                    data.xfrc_applied[bid, 3:] = torque
                mujoco.mj_step(model, data)
                if not np.all(np.isfinite(data.qpos)) or any(w.number for w in data.warning) or _WARNINGS.get():
                    raise PhysicsFailure("numerical_instability")
                wrench, pair_forces, bad = np.zeros(6), {}, set()
                for i in range(data.ncon):
                    con = data.contact[i]
                    a, b = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
                    if a not in placed_body_ids and b not in placed_body_ids:
                        continue
                    pair = frozenset((a, b))
                    mujoco.mj_contactForce(model, data, i, wrench)
                    force = float(np.linalg.norm(wrench[:3]))
                    if a in placed_body_ids or b in placed_body_ids:
                        pair_forces[pair] = pair_forces.get(pair, 0.0)+force
                    if a != 0 and b != 0 and pair not in expected and force > s.collision_force_threshold_n:
                        bad.add(pair)
                peak = max(peak, max(pair_forces.values(), default=0))
                collisions += sum(data.time-last_bad.get(pair, -1e6) > 0.05 for pair in bad)
                for pair in bad:
                    last_bad[pair] = data.time
                if data.time-started >= approach_s+s.settle_s:
                    for v in placed:
                        pos_error = float(np.linalg.norm(data.xpos[bodies[v.id]]-mapping["targets"][v.id])*1000)
                        cos_half = abs(float(np.dot(data.xquat[bodies[v.id]], nominal_quats[v.id])))
                        angle_error = math.degrees(2*math.acos(min(1, cos_half)))
                        maxima[v.id]["position_mm"] = max(maxima[v.id]["position_mm"], pos_error)
                        maxima[v.id]["angle_deg"] = max(maxima[v.id]["angle_deg"], angle_error)
                        stable &= pos_error <= s.position_tolerance_mm and angle_error <= s.angle_tolerance_deg
                    if not stable:
                        break
            records.append({"brick": name, "success": bool(stable), "errors": maxima,
                            "time_s": float(data.time-started), "phase": "gravity_seating",
                            "approach_start_position_m": release.tolist(), "approach_height_mm": height_mm,
                            "virtual_carrier_removed_at_s": started+approach_s,
                            "release_gap_mm": s.floating_release_gap_mm})
            if not stable:
                reason = "collapse" if any(e["position_mm"] > actual.parameters.pitch_mm/2 for e in maxima.values()) else "seating_error"
                break
        result = TrialResult(plan.target.name, seed, reason is None, reason, peak, float(data.time), collisions,
                             disturbance["simulated_brick_mass_g"], records, disturbance, False, "floating", False)
    except (ValueError, RuntimeError) as exc:
        reason = str(exc) if isinstance(exc, PhysicsFailure) else f"simulation_error: {type(exc).__name__}: {exc}"
        result = TrialResult(plan.target.name, seed, False, reason, peak, float(data.time) if data else 0,
                             collisions, disturbance.get("simulated_brick_mass_g", design.mass_g), records,
                             disturbance, "numerical" in reason, "floating", False)
    if output_dir is not None:
        if scene is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            if data is not None:
                root = ET.fromstring(scene)
                for body in root.findall(".//body"):
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.attrib["name"])
                    body.set("gravcomp", str(float(model.body_gravcomp[bid])))
                scene = ET.tostring(root, encoding="unicode")
            (output_dir / "scene.xml").write_text(scene)
        if data is not None:
            write_json(output_dir / "final_state.json", {"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                "xfrc_applied": data.xfrc_applied.tolist(), "body_gravcomp": model.body_gravcomp.tolist(),
                "geom_contype": model.geom_contype.tolist(), "geom_conaffinity": model.geom_conaffinity.tolist(),
                "body_contype": model.body_contype.tolist(), "body_conaffinity": model.body_conaffinity.tolist()})
        write_json(output_dir / "trial.json", result)
        write_json(output_dir / "config.json", config)
    return result
