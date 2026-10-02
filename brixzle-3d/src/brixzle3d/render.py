"""Offscreen rendering of recorded qpos frames to PNG/MP4."""
from pathlib import Path

import mujoco
import numpy as np


def _camera(model, frames, active):
    cam = mujoco.MjvCamera()
    data = mujoco.MjData(model)
    pts = []
    for q in frames[::max(1, len(frames)//20)] + frames[-1:]:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        pts += [data.xpos[b] for b in active]
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            pts.append(data.geom_xpos[g])
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    cam.lookat[:] = (lo + hi)/2
    cam.distance = max(0.15, float(np.max(hi - lo))*2.0)
    cam.azimuth, cam.elevation = 90, -12
    return cam


def render(model, frames, out, active=None, width=960, height=720, fps=60, azimuth=90, elevation=-12):
    """Write frames to ``out`` (.mp4 or .png of the last frame). Returns the path."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    if active is None:
        data.qpos[:] = frames[-1]
        mujoco.mj_kinematics(model, data)
        active = [b for b in range(1, model.nbody) if data.xpos[b][2] < 1.5]
    cam = _camera(model, frames, active)
    cam.azimuth, cam.elevation = azimuth, elevation
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    with mujoco.Renderer(model, height, width) as r:
        images = []
        for q in (frames if out.suffix == '.mp4' else frames[-1:]):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            r.update_scene(data, cam, opt)
            images.append(r.render())
    if out.suffix == '.mp4':
        import imageio
        imageio.mimsave(out, images, fps=fps, macro_block_size=8)
    else:
        from PIL import Image
        Image.fromarray(images[-1]).save(out)
    return out
