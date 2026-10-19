"""Versioned, finite JSON artifacts and calibration provenance."""
import hashlib
import json
import shutil
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
    if path.exists():
        # Replace only a previous session; never wipe an arbitrary directory.
        if not path.is_dir() or not (path / 'report.json').is_file():
            raise ValueError(f'{path} exists and is not a previous calibration or flight session; choose another output')
        shutil.rmtree(path)
    path.mkdir(parents=True)
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


def accepted(path, kind, rig):
    value = read(Path(path) / 'report.json')
    if value.get('kind') != kind or value.get('accepted') is not True:
        raise ValueError(f'{path}: no accepted {kind} calibration')
    if value.get('rig_fingerprint') != fingerprint(rig):
        raise ValueError('Survey changed; repeat Lighthouse and camera calibration')
    return value
