"""Lab configuration (``rig.json``): tags, camera, colour and height windows, field layout.

Values in ``rig.json`` override ``DEFAULTS`` key by key. Relative paths resolve against the
rig file's directory. Measured lab values (tag survey, camera pose, launch-stand pose, geofence)
live in ``calibration_dir`` and are written by ``testflight survey`` / ``testflight calibrate``.
"""
import copy
import json
from pathlib import Path

RIG_PATH = Path(__file__).with_name('rig.json')

DEFAULTS = {
    'calibration_dir': 'calib',
    'log_dir': None,                       # None: no debug images; else a directory for per-call overlays
    'names': {'brick': 'b0_0', 'platform': 'base0', 'fixture': 'fixture0'},
    'tags': {'dictionary': 'DICT_APRILTAG_25h9', 'ids': [0, 1, 2, 3, 4, 5]},
    'camera': {'serial': None, 'width': 1280, 'height': 720, 'fps': 30, 'frames': 8, 'warmup_frames': 30,
               'preset': 'high_accuracy', 'emitter': True, 'exposure': None, 'white_balance': None},
    # OpenCV HSV (H 0-179). Tuned on the lab photos; re-check with `testflight sense --capture ... --debug`.
    'colors': {'pink': [[[0, 80, 120], [12, 255, 255]], [[165, 80, 120], [179, 255, 255]]],
               'blue': [[[105, 90, 50], [135, 255, 255]]]},
    # Height above the floor (mm). Brick top in a cradle/on the platform: 23-47 mm (83% above 30); cradle/platform
    # top: 7-27 mm, so its sawtooth peaks stay ~3 mm (2 sigma of depth noise) below the brick window.
    'heights_mm': {'brick': [30., 55.], 'low': [3., 32.]},
    'workspace_margin_m': 0.5,             # objects must lie within the tags' bounding box grown by this
    'yaw_prior_deg': {'brick': 0., 'platform': 0., 'fixture': 0.},
    'yaw_ambiguity_deg': 20.,              # refuse a yaw within this of prior +- 90 deg
    'accept': {'brick_coverage': 0.45, 'partial_coverage': 0.2, 'platform_coverage': 0.3,
               'fixture_coverage': 0.25, 'max_outside': 0.5, 'min_cluster': 0.25, 'platform_fresh_coverage': 0.7,
               'brick_height_tol_mm': [-5., 8.]},
    # A brick this close to a holder's seat counts as seated in it and takes its yaw.
    'holder': {'along_m': 0.006, 'across_m': 0.006, 'yaw_deg': 8.},
    'stable': {'checks': 2, 'pos_mm': 2.5, 'yaw_deg': 2.0},
    'estimate_tilt': False,                # depth noise makes roll/pitch noisy (~2 deg); see README
    'field': {'camera_sigma_mm': 1.5, 'camera_sigma_deg': 1.0, 'home_yaw_deg': 0., 'fence_margin_m': 0.25},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k.startswith('_'):
            continue
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Rig:
    def __init__(self, path=None, overrides=None):
        self.path = Path(path or RIG_PATH).resolve()
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.cfg = _merge(_merge(DEFAULTS, data), overrides)
        self.root = self.path.parent

    def __getitem__(self, k):
        return self.cfg[k]

    def resolve(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root/p

    @property
    def calib_dir(self):
        return self.resolve(self.cfg['calibration_dir'])

    @property
    def log_dir(self):
        return None if self.cfg['log_dir'] is None else self.resolve(self.cfg['log_dir'])

    def calib_file(self, name):
        return self.calib_dir/name

    def load_calib(self, name, required=True):
        f = self.calib_file(name)
        if not f.exists():
            if required:
                raise FileNotFoundError(f'{f} is missing; see src/testflight/README.md (calibration)')
            return None
        return json.loads(f.read_text())

    def save_calib(self, name, data):
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        f = self.calib_file(name)
        f.write_text(json.dumps(data, indent=1))
        return f
