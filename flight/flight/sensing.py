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
    """Placeholder for the overhead Intel RealSense pipeline (out of scope for experiment 2)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('The RealSense brick-pose pipeline is not implemented; use JsonPoseSource '
                                  '(--poses poses.json) to enter brick poses by hand.')

    def poses(self):
        raise NotImplementedError
