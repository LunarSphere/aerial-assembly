"""Visually prompted YOLOE detectors and camera-relative drone observations."""
from pathlib import Path
import os
import json
import time

import numpy as np

from .camera import cv, pixel_rays
from . import artifacts


def model_path(size=None):
    root = Path(__file__).resolve().parents[3]/'runs/models'
    if size is not None:
        return root/('drone-nano.onnx' if size == 'n' else f'drone-yoloe-{size}.onnx')
    large = root/'drone-yoloe-l.onnx'
    return large if large.exists() else root/'drone-nano.onnx'


def prepare(output=None, size='l', image_size=None):
    """One-time YOLOE visual prompt/export. Inference uses ONNX Runtime on CPU."""
    try:
        from ultralytics import YOLOE
    except ImportError as e:
        raise RuntimeError('Install flight[yolo] to prepare the nano detector; runtime only needs flight[detector]') from e
    if size not in ('n', 's', 'm', 'l', 'x'):
        raise ValueError('YOLOE size must be n, s, m, l or x')
    image_size = image_size or (960 if size == 'n' else 640)
    if image_size <= 0 or image_size % 32:
        raise ValueError('Detector image size must be a positive multiple of 32')
    target = Path(output) if output else model_path(size)
    target.parent.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(__file__).resolve().parents[2]/'configs'
    reference = artifacts.read(reference_dir/'drone-reference.json')
    weights = target.parent/f'yoloe-26{size}-seg.pt'
    import torch
    references = reference.get('references', [reference])
    embeddings = []
    for view in references:
        image = str(reference_dir/view['image'])
        prompted = YOLOE(str(weights))
        prompted.predict(image, refer_image=image,
                         visual_prompts=dict(bboxes=np.array([view['bbox_xyxy']]), cls=np.array([0])),
                         imgsz=image_size, conf=.4, device='cpu', verbose=False)
        embeddings.append(prompted.model.pe.detach().clone())
    classes = [f'Crazyflie reference {i}' for i in range(len(references))]
    model = YOLOE(str(weights))
    model.set_classes(classes, torch.cat(embeddings, dim=1))
    exported = Path(model.export(format='onnx', imgsz=image_size, simplify=False, opset=17, device='cpu'))
    if exported != target:
        import shutil
        shutil.copyfile(exported, target)
    artifacts.write(target.with_suffix('.json'), dict(version=1, architecture=f'YOLOE-26{size}',
        classes=classes, reference=reference, input_size=image_size,
        model_fingerprint=__import__('hashlib').sha256(target.read_bytes()).hexdigest()))
    return target


class NanoDetector:
    def __init__(self, path=None, confidence=.4):
        path = Path(path) if path else model_path()
        if not path.exists():
            raise RuntimeError('Drone detector is not prepared. Run flight-cal prepare-detector once with the yolo extra.')
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError('Install flight[detector] for lightweight CPU drone detection') from e
        metadata = path.with_suffix('.json')
        metadata = artifacts.read(metadata) if metadata.exists() else {}
        self.class_count = len(metadata.get('classes', ['drone']))
        self.architecture = metadata.get('architecture', 'YOLOE-26n')
        options = ort.SessionOptions()
        # Larger models need more CPU parallelism to meet the flight monitor's
        # one-second freshness budget (measured 0.25 s here at six threads).
        options.intra_op_num_threads = min(6 if self.architecture.lower().endswith(('l', 'x')) else 2,
                                          os.cpu_count() or 1)
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inp = self.session.get_inputs()[0]
        self.input_name, self.size = inp.name, int(inp.shape[-1])
        self.confidence = confidence
        self.path = str(path.resolve())

    def detect(self, image):
        h, w = image.shape[:2]
        scale = self.size/max(h, w)
        width, height = round(w*scale), round(h*scale)
        xpad, ypad = (self.size-width)//2, (self.size-height)//2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[ypad:ypad+height, xpad:xpad+width] = cv().resize(image, (width, height))
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255
        result = self.session.run(None, {self.input_name: tensor})[0]
        rows = np.asarray(result)[0]
        if rows.ndim == 2 and rows.shape[0] == 36+self.class_count and rows.shape[1] > 128:
            # xywh + one score per visual reference + 32 mask coefficients.
            # All references identify the same drone; suppress overlaps across classes.
            raw = rows.T
            xywh, scores = raw[:, :4], raw[:, 4:4+self.class_count].max(axis=1)
            self.last_max_confidence = float(scores.max()) if len(scores) else 0.
            candidates = np.flatnonzero(scores >= self.confidence)
            boxes = np.column_stack([xywh[candidates, :2]-xywh[candidates, 2:]/2, xywh[candidates, 2:]])
            indices = cv().dnn.NMSBoxes(boxes.tolist(), scores[candidates].tolist(), self.confidence, .5)
            indices = np.asarray(indices, int).reshape(-1)
            xywh = xywh[candidates[indices]]
            rows = np.column_stack([xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2,
                                    scores[candidates[indices]], np.zeros(len(indices))])
        if rows.ndim != 2 or rows.shape[1] < 6 or rows.shape[1] > 128:
            raise ValueError('Expected an end-to-end YOLOE nano export with [xyxy, score, class, ...] rows')
        if len(rows):
            self.last_max_confidence = float(rows[:, 4].max(initial=0.))
        rows = rows[(rows[:, 4] >= self.confidence) & (rows[:, 5] >= 0) & (rows[:, 5] < self.class_count)]
        if len(rows) == 0:
            score = getattr(self, 'last_max_confidence', 0.)
            raise RuntimeError(f'Drone not confidently visible (best {score:.2f}, required {self.confidence:.2f}); clear hands/occlusions and improve lighting')
        rows = rows[np.argsort(-rows[:, 4])]
        best = rows[0]
        box = (best[:4]-np.array([xpad, ypad, xpad, ypad]))/scale
        box[::2] = np.clip(box[::2], 0, w-1)
        box[1::2] = np.clip(box[1::2], 0, h-1)
        if box[0] <= 3 or box[1] <= 3 or box[2] >= w-4 or box[3] >= h-4:
            raise RuntimeError('Drone detection touches the image edge; move it fully into view or reposition the camera before restarting calibration')
        if len(rows) > 1 and rows[1, 4] > best[4]*.8:
            second = rows[1, :4]
            intersection = np.maximum(0, np.minimum(best[2:4], second[2:4])-np.maximum(best[:2], second[:2])).prod()
            area = np.maximum(1, best[2:4]-best[:2]).prod()
            if intersection/area < .5:
                raise RuntimeError('Multiple drone-like objects detected; cannot identify the connected drone visually')
        return dict(bbox_xyxy=box.tolist(), center_px=((box[:2]+box[2:])/2).tolist(),
                    confidence=float(best[4]), detector=self.architecture.lower()+'-onnx')


def observe(frame, intrinsics, detector, plane=None, reference_height=None):
    detection = detector.detect(frame.color)
    center = np.array(detection['center_px'])
    ray = pixel_rays([center], intrinsics)[0]
    if frame.depth_m is not None:
        x, y = np.rint(center).astype(int)
        patch = frame.depth_m[max(0, y-3):y+4, max(0, x-3):x+4]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid) < 8:
            raise RuntimeError('Insufficient depth on the drone body; reposition until the top deck is visible')
        # Centre patch prevents the support/nearby floor from defining the drone's depth.
        z = float(np.median(valid))
        if np.median(np.abs(valid-z)) > .01:
            raise RuntimeError('Depth around drone centre is inconsistent; keep the deck unobstructed')
        point = ray*z
        metric_source = 'measured_depth'
    elif plane is not None and reference_height is not None:
        normal, distance = np.array(plane['normal_camera']), plane['distance_m']
        # RGB cannot measure unknown drone height. Only static equal-height references
        # use the explicitly supplied support-height profile.
        point = ray*((distance+reference_height)/np.dot(ray, normal))
        metric_source = 'known_reference_height'
    else:
        point = None
        metric_source = 'bearing_only'
    return dict(**detection, point_camera_m=None if point is None else point.tolist(), ray_camera=ray.tolist(),
                metric_source=metric_source, timestamp=frame.timestamp, frame_metadata=frame.metadata)
