"""Generate one CAD solid and matching convex contact pieces."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math
import numpy as np
from scipy.spatial import ConvexHull

from .models import Connector, Design, DesignParameters, RunConfig, digest, write_json
from .polyhedra import box, prism, csg, hull


def _square(cx: float, cy: float, width: float):
    return [(cx + x * width / 2, cy + y * width / 2) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _rotate(vertices: np.ndarray, quarters: int) -> np.ndarray:
    angle = quarters * math.pi / 2
    r = np.array([[math.cos(angle), -math.sin(angle), 0],
                  [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    return vertices @ r.T


def _solid(vertices: np.ndarray):
    import cadquery as cq
    h = hull(vertices)
    if h is None:
        raise ValueError("Degenerate CAD polyhedron")
    faces = []
    for indices, equation in zip(h.simplices, h.equations):
        a, b, c = h.points[indices]
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:
            indices = indices[::-1]
        wire = cq.Wire.makePolygon([cq.Vector(*h.points[i]) for i in indices], close=True)
        faces.append(cq.Face.makeFromWires(wire))
    return cq.Solid.makeSolid(cq.Shell.makeShell(faces))


def _recipe(p: DesignParameters, config: RunConfig):
    f = config.fork
    h, t, c = p.pitch_mm / 2, p.wall_mm, p.clearance_mm
    pickup = 2.0
    sources = [box((0, 0, 0), (p.pitch_mm,) * 3)]
    cutters = [box((0, 0, 0), (p.pitch_mm - 2*t,) * 3)]
    # Pickup bearings join both y walls. A web or I-beam carries the bearing
    # load into the top face instead of relying on a thin unsupported shell.
    for x in (-f.spacing_mm / 2, f.spacing_mm / 2):
        sources.append(box((x, 0, pickup + p.rib_mm / 2),
                           (f.width_mm + 2*f.clearance_mm + 2*t, p.pitch_mm, p.rib_mm)))
        height = h - t - pickup
        sources.append(box((x, 0, pickup + height/2), (p.rib_mm, p.pitch_mm, height)))
        if p.reinforcement == "ibeam":
            sources.append(box((x, 0, h - t - p.rib_mm/2),
                               (f.width_mm + 2*t, p.pitch_mm, p.rib_mm)))
        tunnel_height = f.thickness_mm+f.lip_height_mm+2*f.clearance_mm
        cutters.append(box((x, 0, pickup - tunnel_height/2),
                           (f.width_mm + 2*f.clearance_mm, p.pitch_mm + 2, tunnel_height)))

    connectors = []
    if p.family in ("pegs", "combined"):
        offset = h - 2*t - p.peg_width_mm / 2
        tip = p.peg_width_mm - 2*p.peg_height_mm / math.tan(math.radians(p.guide_angle_deg))
        for x in (-offset, offset):
            for y in (-offset, offset):
                sources.append(prism(_square(x, y, p.peg_width_mm), h, h+p.peg_height_mm,
                                     _square(x, y, tip)))
                cutters.append(prism(_square(x, y, p.peg_width_mm + 2*c), -h-0.01,
                                     -h+p.peg_height_mm+c, _square(x, y, tip+2*c)))
        connectors.extend([Connector("top", (0, 0, 1), "male", "peg", (0, 0, h)),
                           Connector("bottom", (0, 0, -1), "female", "peg", (0, 0, -h))])

    if p.family in ("rails", "combined"):
        d, neck, head = p.rail_depth_mm, p.rail_neck_mm, p.rail_head_mm
        tongue_xy = [(h, -neck/2), (h, neck/2), (h+d, head/2), (h+d, -head/2)]
        female_xy = [(-h-0.02, -neck/2-c), (-h-0.02, neck/2+c),
                     (-h+d+c, head/2+c), (-h+d+c, -head/2-c)]
        widen = p.guide_height_mm / math.tan(math.radians(p.guide_angle_deg))
        wide_xy = [(x, y + math.copysign(widen, y)) for x, y in female_xy]
        # New bricks extend toward +x/+y with downward male insertion into
        # the existing brick's upward-open female socket. Reversing these
        # genders lets an overhang slide downward out of an open socket.
        for q, face in ((2, (-1, 0, 0)), (3, (0, -1, 0))):
            sources.append(_rotate(box((h-t/2, 0, 0), (t, neck, p.pitch_mm-2*t)), q))
            sources.append(_rotate(prism(tongue_xy, -h+t, h-t), q))
            # Socket backing preserves a load path even though the shell is hollow.
            sources.append(_rotate(box((-h+(d+c+t)/2, 0, 0), (d+c+t, head+2*c+2*t, p.pitch_mm)), q))
            cutters.append(_rotate(prism(female_xy, -h+t, h-p.guide_height_mm), q))
            cutters.append(_rotate(prism(female_xy, h-p.guide_height_mm, h+0.02, wide_xy), q))
            name = "x" if q == 2 else "y"
            connectors.extend([Connector(name+"-", face, "male", "rail", tuple(h*i for i in face)),
                               Connector(name+"+", tuple(-i for i in face), "female", "rail",
                                         tuple(-h*i for i in face))])
    # Hollowing is applied BEFORE ribs/housings: the source shell is formed
    # separately, then all source additions receive connector/channel cuts.
    return sources, cutters, connectors, pickup


def generate_design(parameters: DesignParameters, config: RunConfig | None = None,
                    output_dir: Path | None = None) -> Design:
    """Generate a validated design; infeasible engineering limits are diagnostics.

    Invalid geometry raises ValueError. All consumers use this same API rather
    than independently inferring mass or closing connector cavities.
    """
    import cadquery as cq
    config = config or RunConfig()
    config.validate()
    parameters.validate()
    p, f, printing = parameters, config.fork, config.printing
    if f.length_mm < p.pitch_mm + 5:
        raise ValueError("Fork is too short to support the full brick")
    if f.spacing_mm/2 + f.width_mm/2 + f.clearance_mm + p.wall_mm >= p.pitch_mm/2:
        raise ValueError("Fork channels do not fit the brick")
    if p.rib_mm >= p.pitch_mm/2-p.wall_mm-2:
        raise ValueError("Rib exceeds available bearing height")
    sources, cutters, connectors, pickup = _recipe(p, config)
    shell = _solid(sources[0]).cut(_solid(cutters[0]))
    # Triangular faces are unified before Booleans to reduce CAD-kernel work.
    additions = [_solid(vertices) for vertices in sources[1:]]
    shape = shell.fuse(*additions).clean() if additions else shell.clean()
    for cutter in cutters[1:]:
        shape = shape.cut(_solid(cutter)).clean()
    solids = shape.Solids()
    if not shape.isValid() or len(solids) != 1 or shape.Volume() <= 0:
        raise ValueError("CAD must form one connected, valid solid")
    # Build exactly the same shell + additions in convex CSG.
    shell_pieces = csg([sources[0]], [cutters[0]])
    base = [np.asarray(piece.vertices_mm) for piece in shell_pieces]
    collision = csg(base + sources[1:], cutters[1:])
    volume = shape.Volume()
    collision_volume = sum(ConvexHull(piece.vertices_mm).volume for piece in collision)
    if abs(collision_volume-volume) > max(0.02, volume * 1e-5):
        raise ValueError(f"CAD/contact volume mismatch: {volume:.6f} vs {collision_volume:.6f} mm³")
    density_kg_mm3 = printing.density_g_cm3 * 1e-6
    mass = volume * printing.density_g_cm3 / 1000
    com = cq.Shape.centerOfMass(shape).toTuple()
    inertia = np.asarray(cq.Shape.matrixOfInertia(shape)) * density_kg_mm3 * 1e-6
    diagnostics = []
    min_wall = printing.nozzle_mm * printing.min_wall_lines
    if min(p.wall_mm, p.rib_mm) + 1e-8 < min_wall:
        diagnostics.append(f"Minimum wall/rib is below {min_wall:g} mm")
    low_mass = mass*(1-config.uncertainty.mass_fraction)
    high_mass = mass*(1+config.uncertainty.mass_fraction)
    if low_mass < printing.min_mass_g or high_mass > printing.max_mass_g:
        diagnostics.append(f"Mass range {low_mass:.3f}–{high_mass:.3f} g outside allowed range")
    if mass*(1+config.uncertainty.mass_fraction) + f.mass_g > printing.payload_g:
        diagnostics.append("Brick and fork exceed payload including mass uncertainty")
    if p.clearance_mm <= config.uncertainty.clearance_error_mm:
        diagnostics.append("Manufacturing error can close the connector clearance")
    if p.family in ("rails", "combined"):
        retaining_lip = (p.rail_head_mm-p.rail_neck_mm)/2-p.clearance_mm
        if retaining_lip < printing.nozzle_mm:
            diagnostics.append("Rail retention leaves less than one extrusion width")
    if p.guide_angle_deg < printing.min_overhang_angle_deg:
        diagnostics.append("Guide angle violates support-free profile")
    # This checks bridge spans in the chosen +z printing orientation. A slicer
    # and coupon check are still required; this is not a printability certificate.
    roof_span = max(f.spacing_mm-p.rib_mm,
                    (p.pitch_mm-f.spacing_mm)/2-p.wall_mm-p.rib_mm/2)
    tunnel_span = f.width_mm + 2*f.clearance_mm
    if max(roof_span, tunnel_span) > printing.max_bridge_mm:
        diagnostics.append("Roof or pickup tunnel exceeds configured bridge span")
    friction_margin = math.tan(math.radians(p.guide_angle_deg))-config.uncertainty.friction_max
    if friction_margin <= 0:
        diagnostics.append("Guide cannot slide at maximum configured friction")
    # Conservative shell beam estimate for three full-pitch extensions, using
    # only the two side webs. Connector strength is a separate screen.
    length = 3*p.pitch_mm*1e-3
    height = p.pitch_mm*1e-3
    thickness = p.wall_mm*1e-3
    second_moment = 2*thickness*height**3/12
    force = 3*mass/1000*9.81
    stress = force*length*(height/2)/second_moment
    deflection = force*length**3/(3*printing.youngs_modulus_pa*second_moment)
    rail_root_area = max(1e-12, p.rail_neck_mm*(p.pitch_mm-2*p.wall_mm)*1e-6)
    rail_shear = force/rail_root_area
    screen = {"friction_margin": friction_margin, "three_extension_stress_pa": stress,
              "three_extension_deflection_mm": deflection*1000, "rail_shear_pa": rail_shear,
              "collision_volume_mm3": collision_volume, "cad_volume_mm3": volume,
              "roof_bridge_span_mm": roof_span}
    if max(stress, rail_shear) > printing.allowable_stress_pa:
        diagnostics.append("Analytical structural screen exceeds allowable stress")
    design = Design(p, mass, list(com), inertia.tolist(), connectors, collision, pickup,
                    digest({"parameters": asdict(p), "fork": asdict(f), "printing": asdict(printing),
                            "collision_pieces": [asdict(piece) for piece in collision],
                            "geometry_version": 2}), screen, diagnostics, not diagnostics)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(shape, str(output_dir / "brick.step"))
        cq.exporters.export(shape, str(output_dir / "brick.stl"), tolerance=0.03, angularTolerance=0.1)
        cq.exporters.export(shape, str(output_dir / "brick.svg"), exportType="SVG",
                            opt={"width": 800, "height": 600, "showHidden": False})
        write_json(output_dir / "design.json", design)
        write_json(output_dir / "print_profile.json", printing)
    return design
