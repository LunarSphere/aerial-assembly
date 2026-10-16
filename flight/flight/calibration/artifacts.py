"""Versioned, finite JSON artifacts and calibration provenance."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

VERSION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if value.get('version') != VERSION:
        raise ValueError(f'{path}: unsupported artifact version')
    fingerprint(value)  # reject NaN/Infinity, including Python's permissive JSON input
    return value


def write(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def session(path, kind, rig):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    write(path / 'rig.json', rig)
    write(path / 'report.json', dict(version=VERSION, kind=kind, accepted=False,
                                   created_utc=datetime.now(timezone.utc).isoformat(),
                                   rig_fingerprint=fingerprint(rig)))
    return path


def report(path, **fields):
    target = Path(path) / 'report.json'
    value = read(target)
    value.update(fields)
    write(target, value)
    return value


def accepted(path, kind, rig, allow_unverified=False):
    value = read(Path(path) / 'report.json')
    skipped_fit = (allow_unverified and kind in ('camera', 'lighthouse') and value.get('verification_skipped') is True and
                   (value.get('fit_valid') is True if kind == 'camera'
                    else value.get('solution', {}).get('fit_valid') is True))
    if value.get('kind') != kind or (value.get('accepted') is not True and not skipped_fit):
        raise ValueError(f'{path}: no accepted {kind} calibration')
    if value.get('rig_fingerprint') != fingerprint(rig):
        raise ValueError('Survey changed; repeat Lighthouse and camera calibration')
    return value
