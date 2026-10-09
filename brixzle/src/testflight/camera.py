"""Colour + depth frames: a live RealSense D455, or captures replayed from disk.

A ``Frame`` holds the colour image (BGR uint8), the depth aligned to it (metres, 0 = no data),
and the colour intrinsics. Depth is the per-pixel median over a short burst, which cuts the
D455's per-pixel noise (about 5 mm RMS at 2 m) several times.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time
import warnings

import numpy as np


@dataclass
class Frame:
    color: np.ndarray          # (H, W, 3) BGR uint8
    depth: np.ndarray          # (H, W) float32 metres, 0 where invalid
    K: np.ndarray              # (3, 3)
    dist: np.ndarray           # (5,) OpenCV k1 k2 p1 p2 k3
    stamp: float = 0.

    def intrinsics(self):
        h, w = self.depth.shape
        return {'width': w, 'height': h, 'fx': float(self.K[0, 0]), 'fy': float(self.K[1, 1]),
                'ppx': float(self.K[0, 2]), 'ppy': float(self.K[1, 2]), 'coeffs': [float(c) for c in self.dist]}


def K_from(intr):
    return np.array([[intr['fx'], 0, intr['ppx']], [0, intr['fy'], intr['ppy']], [0, 0, 1.]])


def save_frame(frame: Frame, out):
    import cv2
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out/'color.png'), frame.color)
    np.save(out/'depth.npy', frame.depth.astype(np.float32))
    (out/'intrinsics.json').write_text(json.dumps({**frame.intrinsics(), 'stamp': frame.stamp}, indent=1))
    return out


def load_frame(d):
    import cv2
    d = Path(d)
    intr = json.loads((d/'intrinsics.json').read_text())
    color = cv2.imread(str(d/'color.png'), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(d/'color.png')
    return Frame(color, np.load(d/'depth.npy'), K_from(intr), np.asarray(intr['coeffs'], float), intr.get('stamp', 0.))


class ReplayCamera:
    """Replays saved captures (directories from ``save_frame``) in order, looping."""

    def __init__(self, dirs):
        self.frames = [load_frame(d) for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs)]
        self.i = 0

    def capture(self, n=None):
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return f

    def close(self):
        pass


class RealSenseCamera:
    """D455 colour + aligned depth. Auto exposure/white balance are turned off when values are given."""

    def __init__(self, cfg):
        import pyrealsense2 as rs
        self.rs, self.cfg = rs, cfg
        w, h, fps = cfg['width'], cfg['height'], cfg['fps']
        # macOS: the first claim of the USB interface can fail and leave the device half-open in
        # that context, so each retry resets the camera and starts from a fresh context.
        for attempt in range(3):
            self.pipe = rs.pipeline(rs.context())
            c = rs.config()
            if cfg.get('serial'):
                c.enable_device(str(cfg['serial']))
            c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            c.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            try:
                self.profile = self.pipe.start(c)
                break
            except RuntimeError as e:
                print(f'RealSense start failed (attempt {attempt + 1}/3): {e}')
                del self.pipe
                if attempt == 2:
                    raise
                try:
                    for dev in rs.context().query_devices():
                        dev.hardware_reset()
                except RuntimeError as e2:
                    print(f'  hardware reset failed: {e2}')
                time.sleep(4.0)
        dev = self.profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1 if cfg.get('emitter', True) else 0)
        if cfg.get('preset') == 'high_accuracy' and depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, float(rs.rs400_visual_preset.high_accuracy))
        color_sensor = next(s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == 'RGB Camera')
        if cfg.get('exposure') is not None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, float(cfg['exposure']))
        if cfg.get('white_balance') is not None:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.white_balance, float(cfg['white_balance']))
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
        # The D455 colour stream reports (inverse) Brown-Conrady; its coefficients are tiny, and
        # OpenCV's k1 k2 p1 p2 k3 order matches.
        self.dist = np.asarray(list(intr.coeffs)[:5], float)
        for _ in range(cfg.get('warmup_frames', 30)):       # let auto exposure and the laser settle
            self.pipe.wait_for_frames()

    def _drain(self):
        while self.pipe.poll_for_frames():
            pass

    def capture(self, n=None):
        n = n or self.cfg.get('frames', 8)
        self._drain()                     # never use frames buffered before the call
        depths, color = [], None
        for _ in range(n):
            fs = self.align.process(self.pipe.wait_for_frames())
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not d or not c:
                continue
            d = self.spatial.process(d)
            depths.append(np.asanyarray(d.get_data()).astype(np.float32)*self.depth_scale)
            color = np.asanyarray(c.get_data()).copy()
        if not depths:
            raise RuntimeError('RealSense returned no frames')
        stack = np.stack(depths)
        stack[stack <= 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN pixels stay NaN -> 0
            depth = np.nanmedian(stack, axis=0)
        return Frame(color, np.nan_to_num(depth, nan=0.).astype(np.float32), self.K, self.dist, time.time())

    def close(self):
        self.pipe.stop()


def open_camera(rig, capture=None):
    """``capture``: a saved capture directory (or list) to replay instead of the live camera."""
    if capture:
        return ReplayCamera(capture)
    return RealSenseCamera(rig['camera'])
