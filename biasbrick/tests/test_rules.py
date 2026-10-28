from dataclasses import replace
import json

import pytest

from biasbrick import cad, geometry as G, rules as R
from biasbrick.cli import load_params
from biasbrick.params import DEFAULT

ERROR = '../brixzle/configs/error-exp3-r4.json'


def _cfg(name):
    return load_params(f'configs/{name}.json')


def test_aim_lands_the_error_ring_on_the_y_slopes():
    p = DEFAULT
    assert G.aim(p) == (p.aim_x, p.aim_y)
    assert G.y_band(p)[0] <= p.aim_y - p.aim_r  # without yaw the nearest landing is on the nest face
    assert p.aim_x > 0  # slightly toward the stroke side: the leaning tooth enters from +x


def test_longer_rod_costs_y_catchment():
    """The rod's yaw lever: each end moves (pitch/2) sin(yaw), so the sized flange grows with the rod."""
    short, long_ = (R.size_for_tolerance(14.8, DEFAULT.__class__(rod_L=L)) for L in (30., 90.))
    assert long_.yaw_shift[1] > short.yaw_shift[1]
    assert long_.flange > short.flange


def test_configs_match_sizing_for_exp3():
    """(The p95 config centres its aim: aim_x was calibrated by sweeps of the mean design only.)"""
    summary = json.load(open(ERROR))
    for name, stat in (('bb-mean', 'mean'), ('bb-p95', 'p95')):
        p = _cfg(name)
        sized = R.size_for_tolerance(summary[f'{stat}_mm'], replace(DEFAULT, aim_x=p.aim_x))
        assert (p.U, p.flange) == (sized.U, sized.flange)
        assert p.aim_r == pytest.approx(summary[f'{stat}_mm'], abs=0.01)


def test_mean_design_passes_every_rule():
    """Including full coverage of a ring of the mean error plus 5 deg yaw around the (off-centre) aim."""
    p = _cfg('bb-mean')
    margins = R.rules(p, cad.build_brick(p), R.settings_for_tolerance(p.aim_r))
    assert R.feasible(margins), {k: v for k, v in margins.items() if v < 0}


def test_p95_design_is_over_the_payload():
    p = _cfg('bb-p95')
    margins = R.rules(p, cad.build_brick(p), R.settings_for_tolerance(p.aim_r))
    assert R.feasible(margins, skip=('mass_max', 'y_on_slopes', 'hangs_stable')), \
        {k: v for k, v in margins.items() if v < 0}
    assert margins['mass_max'] < 0
