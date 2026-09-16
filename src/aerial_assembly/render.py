"""Headless video of a recorded drop (never re-simulates the experiment)."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def frame_camera(camera, directory, trace):
    import numpy as np
    from .config import read_json, rotation
    from .geometry import bounds_points
    bundle = read_json(Path(directory)/'geometry.json')
    points = bounds_points(bundle)
    low, high = points.min(axis=0), points.max(axis=0)
    corners = np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
    all_points = np.vstack([corners, *[rotation(row[4:8]).apply(corners)+row[1:4] for row in trace]])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    camera.lookat[:] = (low+high)/2
    camera.distance = max(.35,float(max(high-low))*2.2)
    camera.azimuth, camera.elevation = 125, -20


def render_video(directory, output, trial=0):
    env = os.environ.copy()
    env.setdefault('MUJOCO_GL', 'egl')
    env.setdefault('MESA_SHADER_CACHE_DISABLE', 'true')
    try:
        subprocess.run([sys.executable, '-m', 'aerial_assembly.render', str(directory),
                        '--out', str(output), '--trial', str(trial)], env=env, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Video rendering failed; recorded simulation is preserved in {directory}') from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('MESA_SHADER_CACHE_DISABLE', 'true')
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from .config import read_json
    directory, output = Path(args.directory), Path(args.out)
    if output.exists():
        raise ValueError('Video output already exists; choose a new path')
    trace = np.load(directory/f'trial_{args.trial:05d}'/'trajectory.npz')['state']
    result = read_json(directory/f'trial_{args.trial:05d}'/'result.json')
    model = mujoco.MjModel.from_xml_path(str(directory/'scene.xml'))
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    frame_camera(camera, directory, trace)
    model.vis.headlight.ambient[:] = [.4,.4,.4]
    model.vis.headlight.diffuse[:] = [.7,.7,.7]
    option = mujoco.MjvOption()
    option.geomgroup[2], option.geomgroup[3] = 1, 0
    if not any(model.geom_group == 2):
        option.geomgroup[3] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    # Half speed for inspecting engagement, then hold the final frame for a second.
    times = np.r_[np.linspace(trace[0,0],trace[-1,0],max(2,int(trace[-1,0]*fps*2))),
                  np.full(fps,trace[-1,0])]
    frames = []
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        with imageio.get_writer(output, fps=fps, macro_block_size=16) as writer:
            for t in times:
                index = min(np.searchsorted(trace[:,0],t), len(trace)-1)
                row = trace[index]
                data.qpos[:], data.qvel[:], data.time = row[1:8], row[8:14], row[0]
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0,0,960,65),fill=(20,25,35))
                draw.text((16,10),f"EXPORTED CAD DROP | t={row[0]:.3f} s | half-speed replay",fill='white')
                draw.text((16,32),f"Result: {result['status']} | final seating gap: {result['final']['max_seating_gap']*1000:.2f} mm",fill='white')
                writer.append_data(np.asarray(frame))
                if len(frames) < 1:
                    frame.save(output.with_name(output.stem+'-start.png'))
                    frames.append(True)
            frame.save(output.with_name(output.stem+'-end.png'))
    print(output)


if __name__ == '__main__':
    main()
