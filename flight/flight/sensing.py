"""Overhead-camera brick poses, kept separate from flying.

The planner may use only this (brick poses between placements) and the drone's own
Lighthouse estimate. ``poses()`` returns {name: Pose} in the build (Lighthouse) frame.
"""
import json
from pathlib import Path
from typing import Protocol

from .frames import Pose


class BrickPoseSource(Protocol):
    def poses(self) -> dict:
        ...


class JsonPoseSource:
    """Manual fallback: poses typed into a JSON file {name: {"pos": [m], "quat": [w, x, y, z]}}.

    The file is re-read on every call, so an operator can update it between placements.
    """

    def __init__(self, path):
        self.path = Path(path)

    def poses(self):
        data = json.loads(self.path.read_text())
        return {k: Pose.from_dict(v) for k, v in data.items()}


class RealSensePoseSource:
    """The overhead Intel RealSense D455 (``brixzle/src/testflight``): poses() senses the brick and
    the platform by colour + depth before every pick and check.

    ``rig``: path to ``testflight/rig.json`` (its ``calib/`` holds the camera pose). ``capture``:
    saved capture directories to replay instead of the live camera. Imported lazily, like ``--sim``:
    the ``testflight`` package lives in the brixzle environment.
    """

    def __init__(self, rig, capture=None):
        try:
            from testflight.sense import make_source
        except ImportError as e:
            raise ImportError('RealSensePoseSource needs the brixzle environment with the testflight extra '
                              '(cd brixzle && uv sync --extra testflight)') from e
        self.source = make_source(rig, capture)

    def poses(self):
        return self.source.poses()

    def close(self):
        self.source.close()
