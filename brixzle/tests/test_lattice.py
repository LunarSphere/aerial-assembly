import pytest

from brixzle import lattice as L, profile as pr, structures as S
from brixzle.params import V0


@pytest.fixture(scope='module')
def outer():
    return pr.brick_parts(V0)['outer']


def test_courses_alternate_orientation():
    assert L.orientation(V0, L.Placement(0, 0)) == 'A'
    assert L.orientation(V0, L.Placement(1, 0)) == 'B'


def test_support_graph_counts_engaged_voxels():
    g = L.connectivity(S.wall(3, 2))
    assert g.in_degree('b1_1') == 2  # running bond: two supporters, one tooth each
    assert not L.unsupported(g)


def test_unsupported_brick_detected():
    st = L.Structure('floating', [L.Placement(0, 0), L.Placement(2, 0)], bases=[(0, 2)])
    assert L.unsupported(L.connectivity(st)) == ['b2_0']


def test_overlapping_voxels_rejected():
    st = L.Structure('clash', [L.Placement(0, 0), L.Placement(0, 1)], bases=[(0, 4)])
    with pytest.raises(ValueError):
        L.connectivity(st)


def test_bridge_key_brick_placed_last(outer):
    order = L.assembly_order(V0, S.bridge(3), outer)
    assert order[-1] == L.Placement(3, 3)


def test_course_filled_against_tooth_stroke(outer):
    order = [b for b in L.assembly_order(V0, S.wall(3, 1), outer)]
    # A course strokes toward +x, so it is filled right to left.
    assert [b.i0 for b in order] == [4, 2, 0]


def test_tile_running_bond():
    bricks, leftover = L.tile([[1, 1, 1, 1], [0, 1, 1, 1]])
    assert L.Placement(0, 0) in bricks and L.Placement(0, 2) in bricks
    assert L.Placement(1, 1) in bricks and leftover == [(1, 3)]
