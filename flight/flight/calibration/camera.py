"""RealSense D455 RGB-D camera adapter and lens-model helpers."""
from dataclasses import dataclass, field
import hashlib
import time

import numpy as np


def cv():
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError('Install flight[vision] to use cameras and AprilTags') from e
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('OpenCV with aruco support is required (flight[vision])')
    return cv2


def validate_intrinsics(data, profile=None):
    k = np.asarray(data['matrix'], float)
    d = np.asarray(data['distortion'], float)
    if (k.shape != (3, 3) or not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or d.ndim != 1 or d.size not in (4, 5, 8, 12, 14)
            or not np.all(np.isfinite(d))):
        raise ValueError('Invalid camera intrinsics')
    if profile is not None and data['profile'] != profile:
        raise ValueError('Camera device/profile differs from intrinsics; recalibrate at this resolution')
    if data.get('distortion_model', '') not in ('', 'none', 'distortion.none', 'distortion.brown_conrady',
                                                'distortion.inverse_brown_conrady', 'distortion.modified_brown_conrady'):
        raise ValueError('Unsupported camera distortion model')
    return data


@dataclass
class Frame:
    color: np.ndarray
    timestamp: float
    depth_m: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class RealSenseCamera:
    def __init__(self, device='', width=1280, height=720, fps=30, depth_width=848):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError('Install flight[vision,realsense] to use the RealSense camera') from e
        self.rs = rs
        self._restore_emitter = None
        self.pipeline = rs.pipeline()
        config = rs.config()
        if device:
            config.enable_device(device)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, depth_width, 480, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        try:
            camera = profile.get_device()
            self.profile = dict(backend='realsense', device=camera.get_info(rs.camera_info.serial_number),
                                width=width, height=height, fps=fps)
            sensor = camera.first_depth_sensor()
            option = rs.option.emitter_enabled
            if not sensor.supports(option):
                raise RuntimeError('Cannot disable the depth IR emitter for Lighthouse use')
            original = sensor.get_option(option)
            self._restore_emitter = (sensor, option, original)
            # Repeated stationary comparisons showed that the D455 projector
            # interferes with Lighthouse. Passive stereo depth still runs.
            sensor.set_option(option, 0)
            if sensor.get_option(option) != 0:
                raise RuntimeError('RealSense IR emitter did not switch off; cannot run alongside Lighthouse')
            self.profile['ir_emitter_enabled'] = False
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            if intr.model not in (rs.distortion.none, rs.distortion.brown_conrady,
                                   rs.distortion.modified_brown_conrady, rs.distortion.inverse_brown_conrady):
                raise ValueError(f'Unsupported color distortion model: {intr.model}')
            self.intrinsics = validate_intrinsics(dict(
                version=1, profile=self.profile, matrix=[[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                distortion=[0.] * 5 if intr.model == rs.distortion.none else list(intr.coeffs),
                distortion_model=str(intr.model)))
            self.align = rs.align(rs.stream.color)
            self.scale = camera.first_depth_sensor().get_depth_scale()
            self._last_color_number = None
            self._last_color_digest = None
            for sensor in camera.query_sensors():
                if sensor.supports(rs.option.frames_queue_size):
                    try:
                        sensor.set_option(rs.option.frames_queue_size, 1)
                    except RuntimeError:
                        pass  # Some firmware exposes a fixed queue size.
        except BaseException:
            self.close()
            raise

    def frame(self):
        deadline = time.monotonic()+5
        last_issue = 'no fresh color frame'
        # A new frameset/depth frame does not guarantee that its color is new.
        # Copy original color before alignment and track sensor metadata/content.
        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames(max(1, int((deadline-time.monotonic())*1000)))
            color = frames.get_color_frame()
            if not color:
                continue
            number = color.get_frame_number()
            image = np.asanyarray(color.get_data()).copy()
            digest = hashlib.sha256(image).hexdigest()
            if self._last_color_number is not None and number <= self._last_color_number:
                last_issue = 'repeated or reset color frame number'
                continue
            if digest == self._last_color_digest:
                last_issue = 'identical color image content'
                continue
            metadata = dict(color_frame_number=number, color_timestamp_ms=color.get_timestamp(),
                            color_timestamp_domain=str(color.get_frame_timestamp_domain()), color_sha256=digest)
            original_depth = frames.get_depth_frame()
            if not original_depth:
                last_issue = 'missing depth frame'
                continue
            metadata.update(depth_frame_number=original_depth.get_frame_number(),
                            depth_timestamp_ms=original_depth.get_timestamp(),
                            depth_timestamp_domain=str(original_depth.get_frame_timestamp_domain()))
            if metadata['color_timestamp_domain'] != metadata['depth_timestamp_domain']:
                last_issue = 'color/depth timestamp domains differ'
                continue
            delta = abs(metadata['color_timestamp_ms']-metadata['depth_timestamp_ms'])
            metadata['color_depth_delta_ms'] = delta
            if delta > 2000/self.profile['fps']:
                # Device clocks can settle/reset independently at startup.
                # A transient mismatch must not abort the warm-up itself.
                last_issue = f'color/depth timestamp offset {delta:.1f} ms'
                continue
            break
        else:
            raise RuntimeError(f'RealSense did not deliver a fresh synchronized frame within 5 seconds: {last_issue}. Reconnect the camera and retry.')
        source = self.align.process(frames).get_depth_frame()
        if not source:
            raise RuntimeError('No RealSense depth frame')
        depth = np.asanyarray(source.get_data()).astype(float) * self.scale
        depth[depth <= 0] = np.nan
        self._last_color_number, self._last_color_digest = number, digest
        return Frame(image, time.monotonic(), depth, metadata)

    def close(self):
        try:
            if self._restore_emitter is not None:
                sensor, option, value = self._restore_emitter
                sensor.set_option(option, value)
                self._restore_emitter = None
        finally:
            self.pipeline.stop()


def open_camera(args):
    return RealSenseCamera(args.device or '', args.width, args.height,
                           getattr(args, 'fps', 30), getattr(args, 'depth_width', 848))


def pixel_rays(pixels, intrinsics):
    """Unit-Z rays, including RealSense inverse Brown, without requiring its SDK.

    Inverse/modified Brown follow librealsense rs2_deproject_pixel_to_point's
    ten-iteration inversion: https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp
    """
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    k, d = np.asarray(intrinsics['matrix'], float), np.asarray(intrinsics['distortion'], float)
    model = intrinsics.get('distortion_model', '')
    if model in ('distortion.inverse_brown_conrady', 'distortion.modified_brown_conrady'):
        xo, yo = (pixels[:, 0]-k[0, 2])/k[0, 0], (pixels[:, 1]-k[1, 2])/k[1, 1]
        x, y = xo.copy(), yo.copy()
        for _ in range(10):
            r2 = x*x + y*y
            f = 1 + d[0]*r2 + d[1]*r2*r2 + d[4]*r2*r2*r2
            xq, yq = x*f, y*f
            dx = 2*d[2]*xq*yq + d[3]*(r2 + 2*xq*xq)
            dy = 2*d[3]*xq*yq + d[2]*(r2 + 2*yq*yq)
            x, y = (xo-dx)/f, (yo-dy)/f
        normalized = np.column_stack([x, y])
    else:
        normalized = cv().undistortPoints(pixels.reshape(-1, 1, 2), k, d).reshape(-1, 2)
    return np.column_stack([normalized, np.ones(len(pixels))])


def ideal_pixels(pixels, intrinsics):
    k = np.asarray(intrinsics['matrix'], float)
    return np.ascontiguousarray((pixel_rays(pixels, intrinsics) @ k.T)[:, :2])


def project_points(points, rvec, tvec, intrinsics):
    c = cv()
    k, d = np.asarray(intrinsics['matrix'], float), np.asarray(intrinsics['distortion'], float)
    if intrinsics.get('distortion_model') in ('distortion.inverse_brown_conrady', 'distortion.modified_brown_conrady'):
        xyz = np.asarray(points, float) @ c.Rodrigues(np.asarray(rvec, float))[0].T + np.asarray(tvec).reshape(3)
        x, y = xyz[:, 0]/xyz[:, 2], xyz[:, 1]/xyz[:, 2]
        r2 = x*x+y*y
        f = 1+d[0]*r2+d[1]*r2*r2+d[4]*r2*r2*r2
        x, y = x*f, y*f
        xx = x + 2*d[2]*x*y + d[3]*(r2+2*x*x)
        yy = y + 2*d[3]*x*y + d[2]*(r2+2*y*y)
        return np.column_stack([xx*k[0, 0]+k[0, 2], yy*k[1, 1]+k[1, 2]])
    return c.projectPoints(np.asarray(points, float), np.asarray(rvec, float), np.asarray(tvec, float), k, d)[0].reshape(-1, 2)


def intrinsics_compatibility(saved, live, max_drift_px=.5):
    """Compare lens mappings numerically; small SDK focal changes need no hash match."""
    validate_intrinsics(saved)
    validate_intrinsics(live)
    if saved['profile'] != live['profile']:
        raise ValueError('Camera device/profile differs from saved intrinsics')
    if saved.get('distortion_model', '') != live.get('distortion_model', ''):
        raise ValueError('Camera distortion model differs from saved intrinsics')
    width, height = saved['profile']['width'], saved['profile']['height']
    xx, yy = np.meshgrid(np.linspace(0, width-1, 9), np.linspace(0, height-1, 7))
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    # Forward and reverse comparisons avoid silently accepting a mismatch that
    # happens to cancel in only one of the two models.
    forward = project_points(pixel_rays(pixels, saved), np.zeros(3), np.zeros(3), live)
    reverse = project_points(pixel_rays(pixels, live), np.zeros(3), np.zeros(3), saved)
    errors = np.r_[np.linalg.norm(forward-pixels, axis=1), np.linalg.norm(reverse-pixels, axis=1)]
    maximum = float(errors.max())
    return dict(compatible=bool(np.all(np.isfinite(errors)) and maximum <= max_drift_px),
                max_pixel_drift=maximum, rms_pixel_drift=float(np.sqrt(np.mean(errors**2))),
                allowed_pixel_drift=max_drift_px, comparison_points=len(pixels),
                geometry_uses='saved_calibration_intrinsics')
