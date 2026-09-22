"""Load local MJCF geometry with MuJoCo's transforms and mesh scaling."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .config import quaternion


def _load(path):
    model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _parts(model, data, bodies, origin, axes):
    result = {'collision': [], 'visual': []}
    for i in range(model.ngeom):
        if int(model.geom_bodyid[i]) not in bodies:
            continue
        group = 'collision' if model.geom_contype[i] or model.geom_conaffinity[i] else 'visual'
        name = model.geom(i).name or f'{model.body(int(model.geom_bodyid[i])).name}_geom_{i}'
        R = axes.T @ data.geom_xmat[i].reshape(3, 3)
        pos = axes.T @ (data.geom_xpos[i] - origin)
        kind = int(model.geom_type[i])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[i]
            v, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            f, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            vertices = model.mesh_vert[v:v+nv].astype(float) @ R.T + pos
            # Weld duplicate STL vertices for topology checks, without changing surfaces.
            mesh = trimesh.Trimesh(vertices=vertices, faces=model.mesh_face[f:f+nf], process=True)
            part = {'name': name, 'type': 'mesh', 'vertices': mesh.vertices.tolist(),
                    'faces': mesh.faces.tolist()}
        elif kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            box = kind == mujoco.mjtGeom.mjGEOM_BOX
            part = {'name': name, 'type': 'box' if box else 'cylinder',
                    'size': model.geom_size[i, :3 if box else 2].tolist(),
                    'pos': pos.tolist(), 'quat': quaternion(Rotation.from_matrix(R))}
        else:
            raise ValueError(f'Unsupported {group} geom {name}: type {kind}; export meshes, boxes or cylinders')
        result[group].append(part)
    return result
