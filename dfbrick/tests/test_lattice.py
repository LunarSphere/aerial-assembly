import pytest

from dfbrick import lattice as L, structures as S
from dfbrick.params import DEFAULT, DFParams


def test_pose_stacks_feet_on_rail_tops():
    p = DEFAULT
    pos0, _ = L.pose(p, L.Placement(0, 0))
    pos1, _ = L.pose(p, L.Placement(1, 0))
    assert pos0[2] == pytest.approx(p.foot_L)  # foot tips on the base floors (z = 0)
    assert pos1[2] - pos0[2] == pytest.approx(p.H0)
    assert pos0[0] == pytest.approx(p.U)  # centre of voxels 0..2


def test_y_bricks_turn_a_quarter():
    p = DEFAULT
    b = L.Placement(0, 6, 0, 'y')
    assert L.cells(p, b) == [(6, 0), (6, 1), (6, 2)]
    assert L.feet_cells(p, b) == [(6, 0), (6, 2)]
    _, q = L.pose(p, b)
    assert q == pytest.approx([2**-0.5, 0, 0, 2**-0.5])


def test_support_graph_and_order():
    p = DEFAULT
    s = S.wall(p, 3, 4)
    g = L.connectivity(p, s)
    order = L.assembly_order(p, s)
    seen = set()
    for b in order:
        assert all(u in seen for u in g.predecessors(b.name)), b.name
        seen.add(b.name)
    # Running bond: every brick above course 0 rests on two bricks, except at the stepped ends.
    for b in s.bricks:
        if b.course:
            row = [c.i0 for c in s.bricks if c.course == b.course]
            end = b.i0 == (max(row) if b.course % 2 else min(row))
            assert len(set(g.predecessors(b.name))) == (1 if end else 2), b.name


def test_bridge_key_goes_last():
    p = DEFAULT
    s = S.bridge(p, 2)
    order = L.assembly_order(p, s)
    assert order[-1].name == s.key
    assert len(set(L.connectivity(p, s).predecessors(s.key))) == 2


def test_overhang_keeps_one_foot_supported():
    p = DEFAULT
    s = S.overhang(p, 5, +1)
    g = L.connectivity(p, s)
    for b in s.bricks[1:]:
        assert len(list(g.predecessors(b.name))) == 1


def test_ring_is_a_closed_pinwheel():
    p = DEFAULT
    s = S.ring(p, 7, 2)
    for k in range(2):
        cells = [c for b in s.bricks if b.course == k for c in L.cells(p, b)]
        assert len(cells) == len(set(cells)) == 24  # the 7x7 perimeter
    g = L.connectivity(p, s)
    for b in s.bricks:
        if b.course:
            assert len(set(g.predecessors(b.name))) >= 1


def test_overlap_and_floating_rejected():
    p = DEFAULT
    with pytest.raises(ValueError):
        L.validate(p, S.Structure('bad', [L.Placement(0, 0), L.Placement(0, 2)]))
    with pytest.raises(ValueError):
        L.validate(p, S.Structure('float', [L.Placement(0, 0), L.Placement(1, 5)]))


def test_base_receivers_follow_mode():
    s = S.tower(DEFAULT, 2)
    assert {k for *_, k, _ in L.base_receivers(DEFAULT, s)} == {'pocket'}
    slot = DFParams(receiver_mode='pocket-slot')
    kinds = {(x, y): k for x, y, k, _ in L.base_receivers(slot, S.tower(slot, 2))}
    assert kinds[(0., 0.)] == 'pocket' and kinds[(2*slot.U, 0.)] == 'slot'
