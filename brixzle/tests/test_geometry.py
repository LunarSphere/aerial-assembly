from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from brixzle.geometry import generate_design
from brixzle.models import DesignParameters, RunConfig
from brixzle.polyhedra import box, csg


@pytest.fixture(scope="module")
def default_design():
    return generate_design(DesignParameters())


def inside_any(design, point):
    return any(np.all(ConvexHull(piece.vertices_mm).equations @ np.r_[point, 1] < -1e-8)
               for piece in design.collision_pieces)


def test_collision_cavity_survives_boolean_subtraction():
    pieces = csg([box((0, 0, 0), (10, 10, 10))], [box((0, 0, 0), (6, 6, 12))])
    assert sum(ConvexHull(p.vertices_mm).volume for p in pieces) == pytest.approx(640)
    assert not any(np.all(ConvexHull(p.vertices_mm).equations[:, 3] < -1e-8) for p in pieces)


@pytest.mark.integration
def test_cad_mass_contact_volume_and_channels(default_design):
    d = default_design
    assert d.feasible
    assert 1 <= d.mass_g <= 25
    assert d.mass_g+RunConfig().fork.mass_g < 40
    assert d.structural_screen["collision_volume_mm3"] == pytest.approx(d.structural_screen["cad_volume_mm3"], rel=1e-5)
    assert np.linalg.eigvalsh(d.inertia_kg_m2).min() > 0
    # The hollow center and both tine passages remain open, while pickup
    # bearing surfaces and the retaining rail are real collision solids.
    assert not inside_any(d, [0, 0, 0])
    assert not inside_any(d, [6, -15.9, 1])
    assert not inside_any(d, [-6, 15.9, 1])
    assert inside_any(d, [6, 0, 2.5])
    assert inside_any(d, [-17, 0, 0])


@pytest.mark.integration
def test_export_is_one_solid_and_nonempty(default_design, tmp_path):
    d = generate_design(DesignParameters(), output_dir=tmp_path)
    import cadquery as cq
    imported = cq.importers.importStep(str(tmp_path / "brick.step")).val()
    assert len(imported.Solids()) == 1
    assert imported.Volume() == pytest.approx(d.structural_screen["cad_volume_mm3"], rel=1e-5)
    assert (tmp_path / "brick.stl").stat().st_size > 1000


@pytest.mark.integration
def test_small_wall_and_payload_limits_reject_candidates():
    d = generate_design(DesignParameters(wall_mm=0.8))
    assert not d.feasible
    assert any("wall" in diagnostic for diagnostic in d.diagnostics)


@pytest.mark.integration
def test_uncertain_print_mass_cannot_exceed_individual_limit():
    c = RunConfig()
    c = replace(c, printing=replace(c.printing, max_mass_g=12.0))
    d = generate_design(DesignParameters(), c)
    assert d.mass_g < 12.0
    assert not d.feasible
    assert any("Mass range" in diagnostic for diagnostic in d.diagnostics)


def test_invalid_and_nonfinite_geometry_is_rejected():
    with pytest.raises(ValueError):
        DesignParameters(pitch_mm=float("nan")).validate()
    with pytest.raises(ValueError):
        DesignParameters(rail_head_mm=2).validate()
