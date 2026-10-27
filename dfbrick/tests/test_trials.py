import pytest

from brixzle.trials import ErrorModel, TrialConfig

from dfbrick import cad, structures as S, trials as T
from dfbrick.params import DEFAULT

IDEAL = TrialConfig(error=ErrorModel(ideal=True, aim_bias=0.))


@pytest.fixture(scope='module')
def brick():
    return cad.build_brick(DEFAULT)


@pytest.mark.parametrize('on_brick', [False, True])
def test_aligned_drop_seats(brick, on_brick):
    rows, _ = T.drop_trials(DEFAULT, brick, IDEAL, samples=1, on_brick=on_brick)
    assert rows[0]['outcome'] == 'seated', rows[0]
    assert rows[0]['settled']


def test_capture_covers_ten_mm_in_x_and_y(brick):
    offsets = [dict(dx=d) for d in (-10., -5., 5., 10.)] + [dict(dy=d) for d in (-10., 10.)]
    rows, _ = T.drop_trials(DEFAULT, brick, IDEAL, offsets=offsets)
    assert all(r['outcome'] == 'seated' for r in rows), [(r['dx'], r['dy'], r['outcome']) for r in rows]


def test_miss_beyond_the_ridge_is_wrong_receiver(brick):
    rows, _ = T.drop_trials(DEFAULT, brick, IDEAL, offsets=[dict(dx=DEFAULT.U)])
    assert rows[0]['outcome'] == 'wrong_receiver', rows[0]


@pytest.mark.parametrize('name,build', [('tower', lambda p: S.tower(p, 4)), ('wall', lambda p: S.wall(p, 4, 3)),
                                        ('bridge', lambda p: S.bridge(p, 2))])
def test_ideal_structures_complete(brick, name, build):
    result, _ = T.assemble(DEFAULT, brick, build(DEFAULT), IDEAL)
    assert result['status'] == 'complete', (name, result['stages'][-1])


def test_ideal_ring_seats(brick):
    result, _ = T.assemble(DEFAULT, brick, S.ring(DEFAULT, 7, 2), IDEAL)
    assert result['status'] == 'complete', result['stages'][-1]
    assert result['stable_bricks'] == result['total'] == 16


def test_push_holds_ideal_tower(brick):
    res = T.push_test(DEFAULT, brick, S.tower(DEFAULT, 3), IDEAL, limit=0.5)
    assert res['status'] in ('held', 'moved') and res['break_force_N'] > 0.


def test_cup_feet_capture_y_and_yaw_onto_a_brick(brick):
    """The nesting-cup foot slides on the ridge instead of rolling on its tip (the peg foot jams here)."""
    offsets = [dict(dy=d) for d in (-12., -6., 6., 12.)] + [dict(yaw=d) for d in (-8., 8.)]
    rows, _ = T.drop_trials(DEFAULT, brick, IDEAL, offsets=offsets, on_brick=True)
    assert all(r['outcome'] == 'seated' for r in rows), [(r['dy'], r['yaw'], r['outcome']) for r in rows]
