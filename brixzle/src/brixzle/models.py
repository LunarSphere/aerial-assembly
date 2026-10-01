"""Serializable public interfaces; CAD uses mm and simulation uses SI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import tempfile
from typing import Any, Literal

Family = Literal["pegs", "rails", "combined"]


def _validate_field_types(profile):
    for item in fields(profile):
        value, default = getattr(profile, item.name), item.default
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(default, float):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        elif isinstance(default, str):
            valid = isinstance(value, str)
        elif isinstance(default, tuple):
            valid = isinstance(value, (tuple, list)) and all(isinstance(v, (int, float)) and
                not isinstance(v, bool) and math.isfinite(v) for v in value)
        else:
            continue
        if not valid:
            raise ValueError(f"Invalid type or nonfinite value for {type(profile).__name__}.{item.name}")


@dataclass(frozen=True)
class DesignParameters:
    family: Family = "combined"
    pitch_mm: float = 32.0
    wall_mm: float = 1.2
    rib_mm: float = 1.2
    reinforcement: Literal["web", "ibeam"] = "ibeam"
    peg_width_mm: float = 5.0
    peg_height_mm: float = 2.5
    rail_depth_mm: float = 2.0
    rail_neck_mm: float = 5.0
    rail_head_mm: float = 8.0
    clearance_mm: float = 0.25
    guide_angle_deg: float = 65.0
    guide_height_mm: float = 2.0
    approach_bias_mm: float = 0.0

    def validate(self) -> None:
        _validate_field_types(self)
        if self.family not in ("pegs", "rails", "combined"):
            raise ValueError("Unknown connector family")
        if self.reinforcement not in ("web", "ibeam"):
            raise ValueError("Unknown reinforcement")
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 24 <= self.pitch_mm <= 48:
            raise ValueError("pitch_mm must be between 24 and 48")
        if min(self.wall_mm, self.rib_mm, self.clearance_mm, self.peg_height_mm,
               self.peg_width_mm, self.rail_depth_mm, self.rail_neck_mm,
               self.guide_height_mm) <= 0:
            raise ValueError("Thicknesses and connector dimensions must be positive")
        if not 45 <= self.guide_angle_deg <= 80:
            raise ValueError("Guide angle must be 45–80 degrees from horizontal")
        if self.rail_head_mm <= self.rail_neck_mm:
            raise ValueError("Retaining rail head must exceed neck width")
        if self.wall_mm * 4 >= self.pitch_mm:
            raise ValueError("Walls leave no interior space")
        if self.peg_width_mm <= 2 * self.peg_height_mm / math.tan(math.radians(self.guide_angle_deg)):
            raise ValueError("Peg taper leaves no tip")


@dataclass(frozen=True)
class ForkProfile:
    spacing_mm: float = 12.0  # distance between tine centers
    width_mm: float = 3.0
    thickness_mm: float = 2.0
    length_mm: float = 55.0
    lip_height_mm: float = 2.0  # rigid upturned tips; no grasp actuator
    bar_depth_mm: float = 4.0
    clearance_mm: float = 0.4
    mass_g: float = 10.0


@dataclass(frozen=True)
class PrintProfile:
    nozzle_mm: float = 0.4
    layer_mm: float = 0.2
    density_g_cm3: float = 1.24
    min_wall_lines: int = 3
    max_bridge_mm: float = 16.0
    min_overhang_angle_deg: float = 45.0
    min_mass_g: float = 1.0
    max_mass_g: float = 25.0
    payload_g: float = 40.0
    youngs_modulus_pa: float = 2.0e9
    allowable_stress_pa: float = 12.0e6
    calibrated: bool = False


@dataclass(frozen=True)
class UncertaintyProfile:
    lateral_mm: float = 2.0
    vertical_mm: float = 1.0
    yaw_deg: float = 5.0
    tilt_deg: float = 3.0
    friction_min: float = 0.20
    friction_max: float = 0.45
    clearance_error_mm: float = 0.08
    mass_fraction: float = 0.05
    tracking_mm: float = 0.2
    calibrated: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    manipulation_mode: Literal["floating", "fork"] = "floating"
    timestep_s: float = 0.001
    solver_iterations: int = 80
    contact_timeconst_s: float = 0.01
    position_tolerance_mm: float = 0.5
    angle_tolerance_deg: float = 1.0
    hold_s: float = 5.0
    settle_s: float = 1.0
    floating_release_gap_mm: float = 0.5
    floating_position_gain_n_m: float = 20.0
    floating_damping_n_s_m: float = 0.5
    floating_rotation_gain_nm_rad: float = 0.05
    floating_rotation_damping_nm_s: float = 0.001
    move_speed_m_s: float = 0.08
    insert_speed_m_s: float = 0.015
    position_gain_n_m: float = 400.0
    damping_n_s_m: float = 6.0
    rotation_gain_nm_rad: float = 2.0
    rotation_damping_nm_s: float = 0.02
    max_force_n: float = 1.0
    max_torque_nm: float = 0.02
    placement_timeout_s: float = 35.0
    collision_force_threshold_n: float = 0.02


@dataclass(frozen=True)
class SearchConfig:
    population: int = 32
    generations: int = 20
    trials: int = 8
    workers: int = 1
    seed: int = 1234
    shortlist: int = 10
    reevaluate_trials: int = 50
    finalists: int = 3
    validation_trials: int = 200
    reliability: float = 0.95
    confidence: float = 0.95
    weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class RunConfig:
    fork: ForkProfile = field(default_factory=ForkProfile)
    printing: PrintProfile = field(default_factory=PrintProfile)
    uncertainty: UncertaintyProfile = field(default_factory=UncertaintyProfile)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def validate(self) -> None:
        f, p, u, s, q = self.fork, self.printing, self.uncertainty, self.simulation, self.search
        for profile in (f, p, u, s, q):
            _validate_field_types(profile)
        for section in asdict(self).values():
            for key, val in section.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool) and not math.isfinite(val):
                    raise ValueError(f"{key} must be finite")
        if min(f.spacing_mm, f.width_mm, f.thickness_mm, f.length_mm, f.mass_g, f.clearance_mm) <= 0 or f.lip_height_mm < 0:
            raise ValueError("Fork dimensions, clearance, and mass must be positive")
        if f.spacing_mm <= f.width_mm:
            raise ValueError("Fork tines overlap")
        if min(p.nozzle_mm, p.layer_mm, p.density_g_cm3, p.min_wall_lines, p.max_bridge_mm,
               p.youngs_modulus_pa, p.allowable_stress_pa) <= 0:
            raise ValueError("Print dimensions and material properties must be positive")
        if not 0 < p.min_mass_g <= p.max_mass_g < p.payload_g:
            raise ValueError("Inconsistent mass limits")
        if not 0 <= u.friction_min <= u.friction_max or u.clearance_error_mm < 0 or not 0 <= u.mass_fraction < 1:
            raise ValueError("Invalid uncertainty range")
        if min(u.lateral_mm, u.vertical_mm, u.yaw_deg, u.tilt_deg, u.tracking_mm) < 0:
            raise ValueError("Error bounds must be nonnegative")
        if min(s.timestep_s, s.hold_s, s.move_speed_m_s, s.insert_speed_m_s, s.max_force_n,
               s.max_torque_nm, s.contact_timeconst_s, s.placement_timeout_s,
               s.position_tolerance_mm, s.angle_tolerance_deg) <= 0:
            raise ValueError("Simulation times, limits, and tolerances must be positive")
        if s.manipulation_mode not in ("floating", "fork") or s.settle_s < 0:
            raise ValueError("Invalid manipulation mode or settling time")
        if min(s.floating_position_gain_n_m, s.floating_damping_n_s_m, s.position_gain_n_m,
               s.damping_n_s_m, s.rotation_gain_nm_rad, s.rotation_damping_nm_s,
               s.floating_rotation_gain_nm_rad, s.floating_rotation_damping_nm_s) <= 0 or s.floating_release_gap_mm < 0:
            raise ValueError("Controller gains must be positive and release gap nonnegative")
        if s.solver_iterations < 1 or s.contact_timeconst_s < 2 * s.timestep_s:
            raise ValueError("Contact time constant must span at least two timesteps")
        if min(q.population, q.generations, q.trials, q.workers, q.shortlist,
               q.reevaluate_trials, q.finalists, q.validation_trials) < 1:
            raise ValueError("Search counts must be positive")
        if q.population < 5 or not 0 < q.reliability < 1 or not 0.5 < q.confidence < 1:
            raise ValueError("Invalid search population or reliability target")
        if len(q.weights) != 5 or min(q.weights) < 0 or sum(q.weights) <= 0:
            raise ValueError("Five nonnegative ranking weights are required")


@dataclass
class Connector:
    name: str
    face: tuple[int, int, int]
    gender: str
    kind: str
    frame_mm: tuple[float, float, float]
    insertion_axis: tuple[int, int, int] = (0, 0, -1)


@dataclass
class ConvexPiece:
    vertices_mm: list[list[float]]
    faces: list[list[int]]


@dataclass
class Design:
    parameters: DesignParameters
    mass_g: float
    center_of_mass_mm: list[float]
    inertia_kg_m2: list[list[float]]
    connectors: list[Connector]
    collision_pieces: list[ConvexPiece]
    pickup_surface_mm: float
    geometry_hash: str
    structural_screen: dict[str, float]
    diagnostics: list[str] = field(default_factory=list)
    feasible: bool = True


@dataclass(frozen=True)
class Voxel:
    id: str
    cell: tuple[int, int, int]
    yaw_quarters: int = 0
    anchored: bool = False

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Voxel IDs must be nonempty strings")
        if len(self.cell) != 3 or any(not isinstance(v, int) or isinstance(v, bool) for v in self.cell):
            raise ValueError("Voxel cells must be integer triples")
        if not isinstance(self.yaw_quarters, int) or isinstance(self.yaw_quarters, bool) or not isinstance(self.anchored, bool):
            raise ValueError("Voxel rotation must be an integer and anchored must be boolean")


@dataclass
class Target:
    name: str
    voxels: list[Voxel]
    mandatory: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError("Target names must be simple file-safe identifiers")
        if not self.voxels:
            raise ValueError("Targets must contain at least one voxel")


@dataclass
class AssemblyPlan:
    target: Target
    order: list[str]
    connections: list[tuple[str, str, str, str]]
    dependencies: dict[str, list[str]]
    feasible: bool = True
    reason: str | None = None


@dataclass
class TrialResult:
    benchmark: str
    seed: int
    success: bool
    reason: str | None
    peak_force_n: float
    assembly_time_s: float
    collisions: int
    mass_g: float
    placements: list[dict[str, Any]] = field(default_factory=list)
    disturbance: dict[str, Any] = field(default_factory=dict)
    numerical_warning: bool = False
    manipulation_mode: str = "fork"
    transport_simulated: bool = True


@dataclass
class Evaluation:
    design_hash: str
    parameters: dict[str, Any]
    objectives: list[float]
    constraints: list[float]
    benchmarks: dict[str, dict[str, Any]]
    feasible: bool
    diagnostics: list[str] = field(default_factory=list)


def canonical(value: Any) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical(value).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement keeps result/checkpoint files usable after interruption.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".",
                                     suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(json.dumps(asdict(value) if hasattr(value, "__dataclass_fields__") else value,
                                   indent=2, allow_nan=False)+"\n")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def load_config(path: Path | None = None) -> RunConfig:
    raw = {} if path is None else json.loads(path.read_text())
    classes = {"fork": ForkProfile, "printing": PrintProfile, "uncertainty": UncertaintyProfile,
               "simulation": SimulationConfig, "search": SearchConfig}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(raw) - set(classes)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for key, cls in classes.items():
        section = raw.get(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"Configuration section {key} must be an object")
        unexpected = set(section)-{f.name for f in fields(cls)}
        if unexpected:
            raise ValueError(f"Unknown fields in {key}: {sorted(unexpected)}")
    result = RunConfig(**{key: cls(**raw.get(key, {})) for key, cls in classes.items()})
    result.validate()
    return result


def load_parameters(path: Path | None = None, family: str | None = None) -> DesignParameters:
    raw = {} if path is None else json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw)-{f.name for f in fields(DesignParameters)}:
        raise ValueError("Parameters must be an object containing known design fields")
    if family is not None:
        raw["family"] = family
    result = DesignParameters(**raw)
    result.validate()
    return result
