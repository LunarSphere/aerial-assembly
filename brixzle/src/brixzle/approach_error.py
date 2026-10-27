"""Measured drone approach error (Exp 3) -> trial error model and minimum design tolerance.

Exp 3 (``flight-cal approach-metrics``) writes ``metrics.json`` with the positioning error
of each approach: the camera-measured deck centre minus the surveyed tag centre, X-Y only.
``load`` reduces it to an ``approach-error`` summary (mm), which is also the format of the
committed ``configs/error-exp3-*.json``, so trials never need the git-ignored ``runs/``.

The error is used zero-mean: only its magnitudes, in a uniformly random direction. The
measured mean vector is recorded but not applied, since a constant marker-to-deck offset
shows up as bias (see the metrics' known limitations). Yaw, tilt, release height and
residual velocity were not measured; they keep the ErrorModel defaults and are labelled so.
"""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from .config import read_json
from .trials import ErrorModel

CONFIG_DIR = Path(__file__).resolve().parents[2]/'configs'
STATISTICS = ('mean', 'max', 'empirical')
UNMEASURED = ('sigma_yaw', 'sigma_tilt', 'clearance', 'sigma_vel')
PERCENTILE = 95  # required tolerance for the empirical model


def _finite(values, what):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError(f'{what} must be non-empty, finite and >= 0')
    return values


def from_metrics(metrics, path=''):
    """``approach-error`` summary (mm) from an Exp 3 metrics.json dict."""
    if metrics.get('kind') != 'approach-accuracy-metrics':
        raise ValueError("expected an Exp 3 metrics.json (kind 'approach-accuracy-metrics')")
    radii = [a['positioning_error_m'] for a in metrics.get('approaches', [])
             if a.get('positioning_error_m') is not None]
    radii = [1e3*r for r in _finite(radii, 'positioning errors')]
    overall = metrics['summary']['positioning']['overall']
    bias = [1e3*float(b) for b in overall.get('bias_xy_m', [0., 0.])]
    path = Path(path) if path else None
    return {
        'version': 1, 'kind': 'approach-error', 'units': 'mm', 'error': 'positioning',
        'source': {'run': str(path.parent) if path else '', 'metrics': str(path) if path else '',
                   'positioning_source': metrics.get('positioning_source', 'unknown'),
                   'trials_included': list(metrics.get('trials_included', [])),
                   'trials_excluded': list(metrics.get('trials_excluded', []))},
        'n': len(radii),
        'mean_mm': float(np.mean(radii)),
        'rmse_mm': float(np.sqrt(np.mean(np.square(radii)))),
        'std_mm': float(np.std(radii, ddof=1)) if len(radii) > 1 else 0.,
        f'p{PERCENTILE}_mm': float(np.percentile(radii, PERCENTILE)),
        'max_mm': float(np.max(radii)),
        'bias_xy_mm': bias,
        'radii_mm': radii,
    }


def load(path):
    """An ``approach-error`` summary from a metrics.json or a saved summary config."""
    d = read_json(path)
    if d.get('kind') == 'approach-error':
        _finite(d['radii_mm'], 'radii_mm')
        if d['n'] != len(d['radii_mm']):
            raise ValueError('n does not match radii_mm')
        return d
    return from_metrics(d, path)


def _check_stat(statistic):
    if statistic not in STATISTICS:
        raise ValueError(f'statistic must be one of {STATISTICS}')


def required_tolerance(summary, statistic):
    """X-Y error (mm) a pickup or placement must tolerate under this statistic."""
    _check_stat(statistic)
    if statistic == 'empirical':
        return float(np.percentile(summary['radii_mm'], PERCENTILE))
    return float(summary[f'{statistic}_mm'])


def error_model(summary, statistic, base: ErrorModel = ErrorModel(), config=''):
    """Trial error model: measured X-Y magnitude, zero-mean; unmeasured terms from ``base``."""
    _check_stat(statistic)
    source = {**summary['source'], 'statistic': statistic, 'config': str(config),
              'required_tolerance_mm': required_tolerance(summary, statistic),
              'bias_xy_mm_not_applied': summary['bias_xy_mm'], 'unmeasured': list(UNMEASURED)}
    if statistic == 'empirical':
        return replace(base, radial_mode='bootstrap', radii_mm=tuple(summary['radii_mm']),
                       aim_bias=0., source=source)
    return replace(base, radial_mode='ring', radius_mm=float(summary[f'{statistic}_mm']), aim_bias=0.,
                   source=source)
