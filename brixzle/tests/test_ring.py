"""3D rings from the A/B bricks: walls along X and Y joined by turn pieces."""
import numpy as np
import pytest

from brixzle import lattice as L, profile as pr, ring as Rg, trials as T
from brixzle.params import V0


def test_ring_layout_keeps_column_yaw_and_alternates_parts():
    bricks = Rg.ring(6, 4)
    grid = Rg.check_columns(bricks)
    assert sum(1 for k, _ in grid if k == 0) == 20  # perimeter of a 6x6 ring
    assert {b.body for b in bricks} == {'L3A', 'S2A', 'K2A', 'I3B', 'S2B', 'K2B'}
    assert all(b.part == 'AB'[b.course % 2] for b in bricks)


def test_column_rule_rejects_a_turned_cell():
    bricks = Rg.ring(6, 2)
    i = next(i for i, b in enumerate(bricks) if b.course == 1 and b.kind == 'S2')
    b = bricks[i]
    bricks[i] = Rg.Brick(b.kind, b.course, b.x, b.y, (b.yaw + 2) % 4)
    with pytest.raises(ValueError, match='yaw'):
        Rg.check_columns(bricks)


def test_every_course_needs_its_closer_last():
    order = Rg.assembly_order(Rg.ring(6, 2))
    for k in (0, 1):
        course = [b for b in order if b.course == k]
        assert course[-1].kind == 'K2'
    # Without a closer the strokes chase each other round the loop.
    def straight(b):   # the S2 on the closer's two cells (K2's frame sits on its first cell)
        dx, dy = Rg.DIRS[b.yaw % 4]
        return Rg.Brick('S2', b.course, b.x + dx/2, b.y + dy/2, b.yaw)
    no_closer = [straight(b) if b.kind == 'K2' else b for b in Rg.ring(6, 1)]
    with pytest.raises(ValueError, match='insertion order'):
        Rg.assembly_order(no_closer)


def test_strokes_follow_the_ab_tooth_lean():
    for b in Rg.ring(6, 2):
        assert Rg.stroke(b) == tuple(-np.asarray(Rg.DIRS[b.yaw % 4]))


def test_cells_match_their_part_profile():
    for part in 'AB':
        outer, _ = Rg.cell_profile(V0, part, trim_hi=True, lighten=False)   # the brick's +x end keeps its end gap
        right = pr.brick_parts(V0, part)['outer']
        from shapely import affinity
        from shapely.geometry import box
        voxel = affinity.translate(right, -V0.U/2, 0).intersection(box(-V0.U/2, -50, V0.U/2, 50))
        assert outer.symmetric_difference(voxel).area < 1.0, part


def test_ring_rejects_odd_side():
    with pytest.raises(ValueError):
        Rg.ring(7, 2)


@pytest.fixture(scope='module')
def parts():
    return Rg.build_parts(V0)


def test_parts_watertight(parts):
    assert set(parts) >= {'L3A', 'S2A', 'K2A', 'I3B', 'S2B', 'K2B'}
    for name, b in parts.items():
        assert b['mesh'].is_watertight, name


def test_ideal_two_course_ring_seats(parts):
    r, _ = Rg.assemble(V0, parts, Rg.ring(6, 2), T.TrialConfig(error=T.ErrorModel(ideal=True)))
    assert r['status'] == 'complete' and r['seated'] == r['total'], [
        (s['brick'], s['outcome']) for s in r['stages'] if s['outcome'] != 'seated']
