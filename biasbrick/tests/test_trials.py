"""Physics smoke tests (simulated, uncalibrated contact): ideal drops at the aim seat; a drop past the
lip or off the socket does not. Experiment results live in run directories, not here."""
import pytest

from brixzle.config import Physics
from brixzle.trials import ErrorModel, TrialConfig

from biasbrick import geometry as G, trials as T
from biasbrick.params import DEFAULT

CFG = TrialConfig(physics=Physics(timestep=0.001, friction=0.35, contact_timeconst=0.004, iterations=50),
                  error=ErrorModel(ideal=True, aim_bias=0.))


@pytest.fixture(scope='module')
def bricks():
    return T.bundles(DEFAULT)


@pytest.mark.parametrize('on_brick', [False, True])
def test_ideal_drop_at_the_aim_seats(bricks, on_brick):
    rows, _ = T.drop_trials(DEFAULT, bricks, CFG, samples=1, on_brick=on_brick)
    assert rows[0]['outcome'] == 'seated', rows[0]['err_mm']
    assert rows[0]['dx'] == pytest.approx(G.aim(DEFAULT, 'A')[0]) and rows[0]['dy'] == pytest.approx(DEFAULT.aim_y)


def test_drops_off_the_catchment_do_not_seat(bricks):
    p = DEFAULT
    off = [{'dx': p.U/2 + 4}, {'dy': p.D/2 + p.flange + 4}]
    rows, _ = T.drop_trials(p, bricks, CFG, offsets=off, on_brick=True)
    assert [r['outcome'] for r in rows] != ['seated', 'seated']


def test_capture_band_is_the_seated_run_through_the_aim():
    rows = [{'dx': x, 'outcome': o} for x, o in ((-2, 'jammed'), (-1, 'seated'), (0, 'seated'), (1, 'seated'),
                                                (2, 'fell'), (3, 'seated'))]
    assert T.capture_band(rows, 'dx', 0.) == [-1, 1]
    assert T.capture_band(rows, 'dx', 2.) is None
