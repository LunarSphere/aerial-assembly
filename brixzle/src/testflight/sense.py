"""The overhead camera as the ``flight`` mission's pose source (``flight.sensing.BrickPoseSource``).

``poses()`` takes ``stable.checks`` bursts, detects the scene in each, and returns
{brick: Pose, platform: Pose, fixture: Pose} in the Lighthouse frame (names from ``rig.json``),
leaving out objects it does not see. It raises ``SceneError`` (the mission then lands in place)
when a capture is ambiguous or the captures disagree.

The platform pose is cached from its last full view: once the brick sits on it, only its two
ends are visible, and the cached pose is kept while the partial view agrees with it.
"""
import json
import math
import time

import numpy as np

from .calib import CameraPose
from .model import load_model
from .rig import Rig
from .vision import Detector, SceneError, overlay, wrap


class CameraPoseSource:
    def __init__(self, rig: Rig, camera, cam_pose=None, model=None, out=print):
        self.rig, self.camera, self.out = rig, camera, out
        self.cam = cam_pose or CameraPose(rig.load_calib('camera_pose.json'))
        self.model = model or load_model()
        self.det = Detector(self.model, rig, self.cam)
        self.names = rig['names']
        self.platform_cache = None
        self.calls = 0
        self.last = None
        self.log_dir = rig.log_dir
        if self.log_dir is not None:
            self.log_dir = self.log_dir/time.strftime('%Y%m%d-%H%M%S')
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---- one consistent look at the scene ------------------------------------------------------
    def snapshot(self):
        st = self.rig['stable']
        results = []
        for k in range(max(1, st['checks'])):
            frame = self.camera.capture()
            res = self.det.detect(frame)
            self._log(frame, res, k)
            if res.ambiguous:
                raise SceneError('ambiguous scene: ' + '; '.join(res.issues))
            results.append(res)
        for kind in ('brick', 'platform'):
            dets = [getattr(r, kind) for r in results]
            seen = sum(d is not None for d in dets)
            if 0 < seen < len(dets):
                raise SceneError(f'{kind} seen in {seen} of {len(dets)} captures')
            if seen > 1:
                dp = max(np.linalg.norm(d.pos - dets[0].pos) for d in dets)*1e3
                dy = max(abs(math.degrees(wrap(d.yaw - dets[0].yaw))) for d in dets)
                if dp > st['pos_mm'] or dy > st['yaw_deg']:
                    raise SceneError(f'{kind} moved between captures ({dp:.1f} mm, {dy:.1f} deg)')
        self.calls += 1
        self.last = results[-1]
        return results[-1]

    def _platform(self, plat):
        """Fresh full view -> cache it; partial view consistent with the cache -> the cached pose."""
        acc = self.rig['accept']
        cache = self.platform_cache
        if plat is not None and plat.coverage >= acc['platform_fresh_coverage']:
            self.platform_cache = plat
            return plat
        if cache is None:
            return plat
        if plat is None:
            return cache
        st = self.rig['stable']
        close = (np.linalg.norm(plat.pos - cache.pos)*1e3 <= 2*st['pos_mm'] and
                 abs(math.degrees(wrap(plat.yaw - cache.yaw))) <= 2*st['yaw_deg'])
        return cache if close else plat

    def poses(self):
        res = self.snapshot()
        out = {}
        if res.brick is not None:
            out[self.names['brick']] = res.brick.pose()
        plat = self._platform(res.platform)
        if plat is not None:
            out[self.names['platform']] = plat.pose()
        if res.fixture is not None:
            out[self.names['fixture']] = res.fixture.pose()
        return out

    # ---- debug log -----------------------------------------------------------------------------
    def _log(self, frame, res, k):
        if self.log_dir is None:
            return
        import cv2
        stem = self.log_dir/f'{self.calls:04d}_{k}'
        cv2.imwrite(str(stem.with_suffix('.png')), overlay(frame, res, self.cam, self.model))
        stem.with_suffix('.json').write_text(json.dumps({'t': time.time(), **res.to_dict()}, indent=1, default=float))

    def close(self):
        self.camera.close()


def make_source(rig=None, capture=None, out=print):
    """What ``fly --realsense rig.json`` constructs: the live D455 (or a replayed capture).

    ``rig``: a ``Rig`` or a path to rig.json."""
    from .camera import open_camera
    rig = rig if isinstance(rig, Rig) else Rig(rig)
    cam_pose = CameraPose(rig.load_calib('camera_pose.json'))     # fail before opening the camera
    return CameraPoseSource(rig, open_camera(rig, capture), cam_pose=cam_pose, out=out)
