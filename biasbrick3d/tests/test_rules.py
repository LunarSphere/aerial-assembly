import json
import math

import pytest

from biasbrick3d import cad, rules as R
from biasbrick3d.cli import load_params
from biasbrick3d.params import DEFAULT, BBParams

ERROR = '../brixzle/configs/error-exp3-r4.json'


def _cfg(name):
    return load_params(f'configs/{name}.json')


def test_crease_is_shallower_than_the_ramps():
    p = DEFAULT
    assert R.crease_angle(p) < p.ramp
    assert math.tan(math.radians(R.crease_angle(p))) == pytest.approx(math.tan(math.radians(p.ramp))/math.sqrt(2))
    # 40 degree ramps leave a crease too shallow for the 29.8 degree friction screen once margin is in.
    assert R.rules(BBParams(ramp=38.), {'mass_g': 0, 'com': [0, 0, 0], 'channels': [(0, 1), (1, 1)],
                                        'cavity_y': (0, 1)})['crease_slides'] < 0


def test_sized_design_is_one_sided_and_captures_exactly():
    for tol in (14.8, 27.8):
        p = R.size_for_tolerance(tol)
        near, far = R.landing(p, tol)
        assert min(near) == pytest.approx(p.aim_margin)
        assert min(R.capture(p) - f for f in far) >= 0
        smaller = BBParams(**{**p.to_dict(), 'U': p.U - 0.5})
        _, far = R.landing(smaller, tol)
        assert min(R.capture(smaller) - f for f in far) < 0


def test_sizing_is_monotone_and_p95_is_larger():
    sizes = [R.size_for_tolerance(t).U for t in (5., 10., 14.8, 20., 27.8)]
    assert sizes == sorted(sizes)
    assert R.size_for_tolerance(27.8).U > R.size_for_tolerance(14.8).U


def test_configs_match_sizing_for_exp3():
    summary = json.load(open(ERROR))
    for name, stat in (('bb-mean', 'mean'), ('bb-p95', 'p95')):
        p = _cfg(name)
        assert p.U == R.size_for_tolerance(summary[f'{stat}_mm']).U
        assert p.aim_r >= round(summary[f'{stat}_mm'], 1) - 1e-9


def test_mean_design_passes_every_rule_but_mass():
    p = _cfg('bb-mean')
    b = cad.build_brick(p)
    margins = R.rules(p, b, R.settings_for_tolerance(p.aim_r))
    assert R.feasible(margins, skip=('mass_max',)), {k: v for k, v in margins.items() if v < 0}
    # v0 is over the 21.6 g brick budget (30 g payload minus an 8.4 g fork); see the README.
    assert margins['mass_max'] < 0
    (x0, z0), (x1, z1) = b['channels']
    assert z0 == pytest.approx(z1) and x0 < b['com'][0] < x1
