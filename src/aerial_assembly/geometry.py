"""Mesh surfaces and bounds shared by local CAD preparation and simulation."""
import numpy as np
import trimesh

from .config import rotation


def part_mesh(part):
    kind = part['type']
    if kind == 'mesh':
        return trimesh.Trimesh(vertices=part['vertices'], faces=part['faces'], process=False)
    if kind == 'box':
        mesh = trimesh.creation.box(extents=2*np.asarray(part['size']))
    elif kind == 'cylinder':
        mesh = trimesh.creation.cylinder(radius=part['size'][0], height=2*part['size'][1], sections=64)
    else:
        raise ValueError(f"Unsupported geometry type: {kind}")
    transform = np.eye(4)
    transform[:3, :3] = rotation(part.get('quat', [1, 0, 0, 0])).as_matrix()
    transform[:3, 3] = part.get('pos', [0, 0, 0])
    mesh.apply_transform(transform)
    return mesh


def bounds_points(bundle):
    return np.concatenate([part_mesh(p).vertices for p in bundle['collision']])
