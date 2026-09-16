"""Import one rigid CAD assembly from onshape-to-robot's robot.xml.

MuJoCo resolves defaults, includes, mesh scaling and principal-axis transforms.
We retain the compiled triangle surfaces, never substitute their convex hulls.
No exporter pickle is loaded or executed.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .config import quaternion, read_json
from .geometry import bounds_points, save_bundle


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


def inspect_mjcf(path):
    """Report export problems without requiring experiment-specific metadata."""
    from .geometry import part_mesh
    model, data = _load(path)
    roots = [i for i in range(1, model.nbody) if model.body_parentid[i] == 0]
    problems = []
    if len(roots) != 1:
        problems.append('Expected one rigid block root; fasten components with fix_ mates in Onshape')
    if model.neq or model.nu:
        problems.append('Export contains constraints or actuators; export a passive rigid block')
    allowed = {int(model.body_jntadr[i]) for i in roots if model.body_jntnum[i] == 1
               and int(model.jnt_type[model.body_jntadr[i]]) == mujoco.mjtJoint.mjJNT_FREE}
    if any(i not in allowed for i in range(model.njnt)):
        problems.append('Internal joints found; all components of a block must be rigidly connected')
    placeholder = [model.body(i).name for i in range(1, model.nbody)
                   if 0 < model.body_mass[i] <= 1e-8]
    if placeholder or sum(model.body_mass) <= 1e-8:
        problems.append('Missing/placeholder mass properties; assign CAD materials and re-export')
    if not np.isfinite(model.mat_rgba).all() or not np.isfinite(model.geom_rgba).all():
        problems.append('Nonfinite material colors; set a finite color in the export configuration')
    parts = _parts(model, data, set(range(1, model.nbody)), np.zeros(3), np.eye(3))
    collision = []
    for part in parts['collision']:
        mesh = part_mesh(part)
        valid = bool(mesh.is_volume and mesh.is_watertight and mesh.is_convex)
        collision.append({'name': part['name'], 'closed_convex': valid})
        if not valid:
            problems.append(f"Collision {part['name']} is not a closed convex solid; decompose before import")
    if not collision:
        problems.append('No collision geometry found')
    points = bounds_points(parts) if collision else None
    return {'ready_for_metadata': not problems, 'problems': problems,
            'root_bodies': [model.body(i).name for i in roots],
            'mass_kg': float(sum(model.body_mass)), 'placeholder_mass_bodies': placeholder,
            'world_extents_m': np.ptp(points, axis=0).tolist() if points is not None else None,
            'collision': collision, 'sites': [model.site(i).name for i in range(model.nsite)]}


def prepare_mjcf(path, metadata, output, body=None):
    """Convert a single rigid root and SI seating metadata to a portable bundle.

The root's reference frame defines block coordinates. Its placement in the
export scene is removed. Metadata vectors must use this frame.
"""
    if Path(output).exists():
        raise ValueError('Output already exists; choose a new bundle path')
    report = inspect_mjcf(path)
    if report['problems']:
        raise ValueError('Export is not ready: ' + '; '.join(report['problems']))
    model, data = _load(path)
    root_name = report['root_bodies'][0]
    if body is not None and body != root_name:
        raise ValueError(f'Expected root body {root_name!r}, got {body!r}')
    root = model.body(root_name).id
    origin, axes = data.xpos[root], data.xmat[root].reshape(3, 3)
    bodies = set(range(1, model.nbody))  # inspector requires exactly one rigid tree
    parts = _parts(model, data, bodies, origin, axes)
    sites = {}
    for i in range(model.nsite):
        if model.site_bodyid[i] in bodies:
            sites[model.site(i).name] = {
                'pos': (axes.T @ (data.site_xpos[i]-origin)).tolist(),
                'quat': quaternion(Rotation.from_matrix(axes.T @ data.site_xmat[i].reshape(3, 3)))}

    def resolve(value):
        if isinstance(value, dict):
            if 'site' in value:
                if set(value) - {'site', 'field'} or value.get('field', 'pos') not in ('pos', 'quat'):
                    raise ValueError('Site reference requires site and optional field pos/quat')
                if value['site'] not in sites:
                    raise ValueError(f"Unknown block site {value['site']!r}; available: {list(sites)}")
                return sites[value['site']][value.get('field', 'pos')]
            return {k: resolve(v) for k, v in value.items()}
        return [resolve(v) for v in value] if isinstance(value, list) else value

    bundle = deepcopy(read_json(metadata))
    if set(bundle) & {'prepared', 'asset_hash', 'collision', 'visual', 'inertial', 'mesh_units', 'mesh_to_body'}:
        raise ValueError('Metadata must not override imported geometry, inertia or transforms')
    bundle = resolve(bundle)
    masses = model.body_mass[1:]
    mass = float(sum(masses))
    centers = (data.xipos[1:] - origin) @ axes
    com = np.average(centers, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for i in range(1, model.nbody):
        R = axes.T @ data.ximat[i].reshape(3, 3)
        delta = centers[i-1] - com
        inertia += R @ np.diag(model.body_inertia[i]) @ R.T
        inertia += model.body_mass[i] * (np.dot(delta, delta)*np.eye(3)-np.outer(delta, delta))
    bundle.update(parts)
    bundle['inertial'] = {'mass': mass, 'com': com.tolist(), 'matrix': inertia.tolist(),
                          'provenance': 'Compiled MJCF body inertias, aggregated about COM'}
    if not bundle.get('source', {}).get('revision'):
        raise ValueError('Metadata source.revision must identify the CAD export revision')
    bundle['source'].update(kind='onshape_to_robot', root_body=root_name,
                            xml_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            mujoco_version=mujoco.__version__)
    bundle['sites'] = sites
    bundle['prepared'] = True
    expected = np.asarray(bundle.get('expected_extents', []))
    actual = np.ptp(bounds_points(bundle), axis=0)
    if expected.shape != (3,) or not np.allclose(actual, expected, rtol=.005, atol=1e-5):
        raise ValueError(f'Geometry bounds {actual} disagree with expected_extents {expected}')
    return save_bundle(bundle, output)
