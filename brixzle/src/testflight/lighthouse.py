"""Check the cfclient Lighthouse geometry input (``testflight_lighthouse.yaml``) with cflib's own solver.

The frame this defines is the one every testflight pose is in:
* origin = the drone's Lighthouse deck at the "origin" sample, so z = 0 is the deck's sensor plane
  there, not the floor (the tag survey measures the floor);
* +x towards the "x-axis" sample, which cfclient scales to exactly 1.000 m, so the whole frame's
  scale is only as good as that 1 m placement;
* the "xy-plane" sample fixes the floor's tilt.
Re-running the geometry estimation changes the frame: survey the tags and calibrate the camera again.
"""
import math
from pathlib import Path

import numpy as np

YAML_PATH = Path(__file__).with_name('testflight_lighthouse.yaml')


def solve(path=YAML_PATH):
    import yaml
    from scipy.spatial.transform import Rotation
    from cflib.localization.lighthouse_geo_estimation_manager import LhGeoEstimationManager
    data = yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)['data']
    sol = LhGeoEstimationManager.estimate_geometry(data)
    bs = {}
    for k, p in sol.bs_poses.items():
        yaw, pitch, roll = Rotation.from_matrix(p.rot_matrix).as_euler('ZYX', degrees=True)
        bs[int(k)] = {'pos_m': [float(v) for v in p.translation], 'yaw_deg': float(yaw), 'pitch_deg': float(pitch),
                      'roll_deg': float(roll)}
    err = sol.error_stats
    report = {'ok': bool(sol.progress_is_ok), 'base_stations': bs,
              'samples': {'x_axis': data.x_axis_sample_count, 'xy_plane': data.xy_plane_sample_count,
                          'xyz_space': data.xyz_space_sample_count, 'verification': len(data.verification)},
              'error_mm': None if err is None else {'mean': float(err.mean)*1e3, 'max': float(err.max)*1e3},
              'xy_plane_sample_m': [float(v) for v in sol.samples[2].pose.translation] if len(sol.samples) > 2 else None}
    report['warnings'] = warnings(report)
    return report


def warnings(r):
    out = []
    if not r['ok']:
        out.append('cflib could not solve the geometry')
    s = r['samples']
    if s['xyz_space'] == 0:
        out.append('no xyz-space samples: add 10-20 (drone carried around the pick/place area at 0-30 cm) '
                   'so the solver is not fitting 3 poses')
    if s['verification'] == 0:
        out.append('no verification samples: add a few to get an independent error estimate')
    out.append('scale comes from the x-axis sample being exactly 1.000 m from the origin: tape-check it '
               '(1 cm off = 1 % scale = 5 mm at 0.5 m)')
    for k, b in r['base_stations'].items():
        if b['pos_m'][2] < 1.0:
            out.append(f'base station {k} is only {b["pos_m"][2]:.2f} m above the origin')
    if r['error_mm'] and r['error_mm']['max'] > 10:
        out.append(f'solution error up to {r["error_mm"]["max"]:.1f} mm')
    return out


def angle_between_stations(r):
    """Angle (deg) at the origin between the two base stations: near 0 or 180 is poorly conditioned."""
    p = [np.asarray(b['pos_m']) for b in r['base_stations'].values()]
    if len(p) < 2:
        return float('nan')
    a, b = p[0], p[1]
    return math.degrees(math.acos(np.clip(a @ b/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))
