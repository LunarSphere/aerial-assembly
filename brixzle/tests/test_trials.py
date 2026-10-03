import pytest

from brixzle import cad, structures as S, trials as T
from brixzle.params import V0


@pytest.fixture(scope='module')
def brick():
    return cad.build_brick(V0)


def test_aligned_drop_seats(brick):
    rows, _ = T.drop_trials(V0, brick, T.TrialConfig(error=T.ErrorModel(ideal=True)), samples=1)
    assert rows[0]['outcome'] == 'seated'
    assert rows[0]['settled']


def test_capture_covers_ten_mm_around_aim_in_x(brick):
    cfg = T.TrialConfig(error=T.ErrorModel(ideal=True))
    aim = cfg.error.aim_bias
    rows, _ = T.drop_trials(V0, brick, cfg, offsets=[aim + d for d in (-10., -5., 0., 5., 10.)])
    assert all(r['outcome'] == 'seated' for r in rows), [(r['dx'], r['outcome']) for r in rows]


def test_ideal_tower_completes(brick):
    result, _ = T.assemble(V0, brick, S.tower(4), T.TrialConfig(error=T.ErrorModel(ideal=True)))
    assert result['status'] == 'complete'


def test_ideal_bridge_completes_with_key(brick):
    result, _ = T.assemble(V0, brick, S.bridge(3), T.TrialConfig(error=T.ErrorModel(ideal=True)))
    assert result['status'] == 'complete'
    assert result['stages'][-1]['brick'] == 'b3_3'
