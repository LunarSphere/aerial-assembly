import pytest

from brixzle.config import Physics
from brixzle.trials import ErrorModel, TrialConfig

from biasbrick import lattice as L, structures as S, trials as T
from biasbrick.params import DEFAULT

IDEAL = TrialConfig(physics=Physics(timestep=0.001, friction=0.35, contact_timeconst=0.004, iterations=50),
                    error=ErrorModel(ideal=True, aim_bias=0.))


def test_parts_alternate_and_poses_stack_by_course():
    p = DEFAULT
    a, b = L.Placement(0, 0), L.Placement(1, 0)
    assert (a.part, b.part) == ('A', 'B')
    assert L.pose(p, b)[0][2] - L.pose(p, a)[0][2] == pytest.approx(p.H0)
    assert L.pose(p, L.Placement(0, 1))[0][0] - L.pose(p, a)[0][0] == pytest.approx(p.pitch)


def test_validate_rejects_overlaps_and_floating_bricks():
    with pytest.raises(ValueError):
        L.validate(S.Structure('x', [L.Placement(0, 0), L.Placement(0, 1)]))
    with pytest.raises(ValueError):
        L.validate(S.Structure('x', [L.Placement(0, 0), L.Placement(1, 2)]))
    L.validate(S.overhang(3))  # one supported socket is enough


def test_structures():
    wall = S.wall(4, 4)
    assert [sum(b.course == k for b in wall.bricks) for k in range(4)] == [4, 3, 2, 1]
    assert all(len(L.supported_sockets(wall, b)) == 2 for b in wall.bricks if b.course)
    bridge = S.bridge(2)
    order = L.assembly_order(bridge)
    assert order[-1].name == bridge.key and len(L.supported_sockets(bridge, order[-1])) == 2
    assert all(len(L.supported_sockets(S.overhang(4, d), b)) == 1 for d in (1, -1) for b in S.overhang(4, d).bricks[1:])


@pytest.fixture(scope='module')
def bricks():
    return T.bundles(DEFAULT)


def test_ideal_tower_completes(bricks):
    """Simulated, uncalibrated contact: dropped at the aim with no error, a 3-tower stands."""
    result, _ = T.assemble(DEFAULT, bricks, S.tower(3), IDEAL)
    assert result['status'] == 'complete' and result['stable_bricks'] == 3


def test_a_placed_overhang_brick_hangs_on_its_hook(bricks):
    """Set at its seat (a carried placement), one overhanging brick stays: its tooth holds it."""
    result, _ = T.assemble(DEFAULT, bricks, S.overhang(2, +1), IDEAL, place=True)
    assert result['status'] == 'complete'
