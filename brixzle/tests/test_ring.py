"""3D rings from 2.5D bricks: walls along X and Y joined by turn pieces."""
import pytest

from brixzle import ring as Rg, trials as T
from brixzle.params import V0


def test_ring_layout_alternates_every_column():
    bricks = Rg.ring(6, 4)
    grid = Rg.check_columns(bricks)
    assert sum(1 for k, _ in grid if k == 0) == 20  # perimeter of a 6x6 ring
    kinds = {b.kind for b in bricks}
    assert kinds == {'S2', 'L3', 'I3', 'K2'}


def test_odd_courses_need_the_closer_last():
    order = Rg.assembly_order(Rg.ring(6, 2))
    assert order[-1].kind == 'K2'
    assert all(not Rg._blocked(b, set()) for b in order if b.course == 0)


def test_ring_rejects_odd_side():
    with pytest.raises(ValueError):
        Rg.ring(7, 2)


@pytest.fixture(scope='module')
def parts():
    return Rg.build_parts(V0)


def test_parts_watertight(parts):
    for name, b in parts.items():
        assert b['mesh'].is_watertight, name


def test_ideal_two_course_ring_seats(parts):
    r, _ = Rg.assemble(V0, parts, Rg.ring(6, 2), T.TrialConfig(error=T.ErrorModel(ideal=True)))
    assert r['status'] == 'complete' and r['seated'] == r['total']
