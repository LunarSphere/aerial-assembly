"""Prepare explicit convex parts; never silently convexify a concave CAD part."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math

import numpy as np
import trimesh

from .config import digest, read_json, rotation, write_json


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


def convex_part(name, vertices):
    mesh = trimesh.convex.convex_hull(np.asarray(vertices))
    return {'name': name, 'type': 'mesh', 'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist()}


def bounds_points(bundle):
    return np.concatenate([part_mesh(p).vertices for p in bundle['collision']])


def validate_bundle(bundle):
    if bundle.get('schema_version') != 1:
        raise ValueError('Unsupported geometry schema; expected schema_version 1')
    # Enforce finite numbers everywhere, including feature metadata.
    digest(bundle)
    if not bundle.get('collision') or len(bundle.get('sockets', [])) != 2 or len(bundle.get('legs', [])) != 2:
        raise ValueError('Need convex collision parts, two sockets, and two corresponding legs')
    rotation(bundle['target']['quat'])
    if np.asarray(bundle['target']['pos']).shape != (3,):
        raise ValueError('Target position must have three components')
    if len(bundle.get('seating_pairs', [])) < 3 or not bundle.get('source'):
        raise ValueError('Source metadata and at least three seating point pairs are required')
    seating = np.asarray([p['upper'] for p in bundle['seating_pairs']])
    if seating.ndim != 2 or seating.shape[1] != 3 or np.linalg.matrix_rank(seating-seating[0], tol=1e-10) < 2:
        raise ValueError('Seating points must be noncollinear 3D points')
    if bundle['clearance'] <= 0 or bundle['max_penetration'] <= 0:
        raise ValueError('Clearance and penetration budget must be positive')
    if bundle['max_penetration'] > bundle['clearance'] * 0.2:
        raise ValueError('Penetration budget must be <= 20% of functional clearance')
    mass = bundle['inertial']['mass']
    inertia = np.asarray(bundle['inertial']['matrix'])
    if mass <= 0 or inertia.shape != (3, 3) or not np.allclose(inertia, inertia.T, atol=1e-12):
        raise ValueError('Need positive mass and a symmetric 3x3 COM inertia in kg m^2')
    eig = np.linalg.eigvalsh(inertia)
    if eig[0] <= 0 or eig[-1] > sum(eig[:-1]) + 1e-12:
        raise ValueError('Inertia is not physically valid')
    if np.asarray(bundle['inertial']['com']).shape != (3,):
        raise ValueError('Center of mass must have three components')
    names = set()
    for part in bundle['collision']:
        if part['name'] in names:
            raise ValueError('Collision part names must be unique')
        names.add(part['name'])
        mesh = part_mesh(part)
        if not mesh.is_watertight or not mesh.is_volume or not mesh.is_convex:
            raise ValueError(f"Collision part {part['name']} is not a closed convex solid; split it in CAD")
    for socket, leg in zip(bundle['sockets'], bundle['legs']):
        rotation(socket.get('quat', [1, 0, 0, 0]))
        if not 0 < socket['throat_radius'] <= socket['opening_radius'] or not 0 <= socket['lead_depth'] < socket['depth']:
            raise ValueError('Invalid circular socket dimensions')
        probes = np.asarray(leg['surface_probes'])
        if probes.ndim != 2 or probes.shape[1] != 3 or len(probes) < 12:
            raise ValueError('Leg needs surface probes along its shaft, root, and tip')
        if leg['target_depth'] <= 0 or leg['target_depth'] > socket['depth']:
            raise ValueError('Leg would bottom out before flush seating')
        if np.asarray(leg['tip']).shape != (3,) or np.asarray(socket['mouth']).shape != (3,):
            raise ValueError('Socket mouth and leg tip must be 3D points')
    R = rotation(bundle['target']['quat'])
    t = np.asarray(bundle['target']['pos'])
    for pair in bundle['seating_pairs']:
        normal = np.asarray(pair.get('normal', [0,0,1]))
        if normal.shape != (3,) or np.linalg.norm(normal) < 1e-10:
            raise ValueError('Seating normal must be a nonzero 3D vector')
        if np.linalg.norm(R.apply(pair['upper']) + t - pair['lower']) > 1e-8:
            raise ValueError('Seating point pair does not coincide at declared target')
    return bundle


def save_bundle(bundle, output):
    validate_bundle(bundle)
    bundle = deepcopy(bundle)
    bundle.pop('asset_hash', None)
    bundle['asset_hash'] = digest(bundle)
    write_json(output, bundle)
    return bundle


def load_bundle(path):
    bundle = read_json(path)
    expected = bundle.pop('asset_hash', None)
    if expected != digest(bundle):
        raise ValueError('Bundle hash mismatch; prepare geometry again rather than editing a prepared bundle')
    bundle['asset_hash'] = expected
    return validate_bundle(bundle)


def prepare_geometry(source, output):
    """Import a manifest. Metadata is SI; mesh coordinates use explicit mesh_units."""
    source = Path(source).resolve()
    bundle = read_json(source)
    if bundle.get('prepared'):
        raise ValueError('Input is already prepared')
    scale = {'m': 1.0, 'mm': 0.001}.get(bundle.pop('mesh_units', None))
    if scale is None:
        raise ValueError('Set mesh_units to mm or m; all non-mesh metadata must be SI')
    transform = bundle.pop('mesh_to_body', {'pos': [0, 0, 0], 'quat': [1, 0, 0, 0]})
    R = rotation(transform['quat'])
    file_hashes = {}
    for group in ('visual', 'collision'):
        prepared = []
        for item in bundle.get(group, []):
            if 'file' not in item:
                prepared.append(item)
                continue
            path = (source.parent / item['file']).resolve()
            file_hashes[item['file']] = hashlib.sha256(path.read_bytes()).hexdigest()
            mesh = trimesh.load_mesh(path, process=True)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError('Export individual STL parts, not mesh scenes')
            vertices = R.apply(mesh.vertices * scale) + np.asarray(transform['pos'])
            prepared.append({'name': item['name'], 'type': 'mesh', 'vertices': vertices.tolist(),
                             'faces': mesh.faces.tolist()})
        bundle[group] = prepared
    bundle['source']['mesh_sha256'] = file_hashes
    bundle['prepared'] = True
    points = bounds_points(bundle)
    if 'expected_extents' not in bundle:
        raise ValueError('expected_extents in meters is required as an independent units check')
    expected = np.asarray(bundle['expected_extents'])
    if not np.allclose(np.ptp(points, axis=0), expected, rtol=0.005, atol=1e-5):
        raise ValueError(f"Geometry bounds {np.ptp(points, axis=0)} disagree with expected_extents {expected}")
    return save_bundle(bundle, output)


def demo_geometry(output, ramp_angle=50.0, lead_angle=30.0, facets=48):
    """Synthetic test fixture, NOT a reconstruction of the user's Onshape design.

    lead_angle is the funnel's angle from vertical. All dimensions are SI.
    """
    if not 20 <= ramp_angle <= 75 or not 10 <= lead_angle <= 60 or facets < 16 or facets % 8:
        raise ValueError('Demo requires ramp 20..75 deg, lead 10..60 deg, facets >=16 divisible by 8')
    half = 0.02
    rise = 0.04
    run = rise / math.tan(math.radians(ramp_angle))
    centers = [-half, run + half]
    throat, opening, depth = 0.005, 0.0065, 0.034
    lead = (opening - throat) / math.tan(math.radians(lead_angle))
    radius, leg_length, tip_length = 0.00475, 0.03, 0.00475
    parts, sockets, legs, seating = [], [], [], []
    for idx, (cx, bottom) in enumerate(zip(centers, [0.0, rise])):
        top = bottom + rise
        parts.append({'name': f'floor_{idx}', 'type': 'box', 'size': [half, half, (rise-depth)/2],
                      'pos': [cx, 0, bottom+(rise-depth)/2]})
        # Broad seating strips keep support contacts off the many bore-sector seams.
        # A documented 5 um recess in sector tops stays below the 25 um surface budget.
        ring_half = 0.008
        for k, (dx, dy, sx, sy) in enumerate([
                (-(half+ring_half)/2, 0, (half-ring_half)/2, half),
                ((half+ring_half)/2, 0, (half-ring_half)/2, half),
                (0, -(half+ring_half)/2, ring_half, (half-ring_half)/2),
                (0, (half+ring_half)/2, ring_half, (half-ring_half)/2)]):
            parts.append({'name': f'seat_{idx}_{k}', 'type': 'box', 'size': [sx, sy, depth/2],
                          'pos': [cx+dx, dy, top-depth/2]})
        angles = np.linspace(0, 2*math.pi, facets+1)
        for k, (a, b) in enumerate(zip(angles[:-1], angles[1:])):
            for label, z0, z1, r0, r1 in [('bore', top-depth, top-lead, throat, throat),
                                         ('lead', top-lead, top-0.000005, throat, opening)]:
                verts = []
                for z, r in [(z0, r0), (z1, r1)]:
                    for theta in [a, b]:
                        c, s = math.cos(theta), math.sin(theta)
                        outer = ring_half / max(abs(c), abs(s))
                        verts.extend([[cx+r*c, r*s, z], [cx+outer*c, outer*s, z]])
                parts.append(convex_part(f'{label}_{idx}_{k}', verts))
        parts.append({'name': f'leg_{idx}', 'type': 'cylinder',
                      'size': [radius, (leg_length-tip_length)/2],
                      'pos': [cx, 0, bottom-(leg_length-tip_length)/2]})
        tip_ring = [[cx+radius*math.cos(a), radius*math.sin(a), bottom-leg_length+tip_length]
                    for a in angles[:-1]]
        parts.append(convex_part(f'tip_{idx}', tip_ring + [[cx, 0, bottom-leg_length]]))
        probes = [[cx, 0, bottom-leg_length]]
        for z in [bottom, bottom-(leg_length-tip_length)/2, bottom-leg_length+tip_length]:
            probes.extend([[cx+radius*math.cos(a), radius*math.sin(a), z] for a in angles[:-1]])
        sockets.append({'mouth': [cx, 0, top], 'depth': depth, 'lead_depth': lead,
                        'throat_radius': throat, 'opening_radius': opening})
        legs.append({'surface_probes': probes, 'tip': [cx, 0, bottom-leg_length], 'target_depth': leg_length})
        for dx, dy in [(-0.015,-0.015), (0.015,-0.015), (-0.015,0.015), (0.015,0.015)]:
            seating.append({'upper': [cx+dx, dy, bottom], 'lower': [cx+dx, dy, top]})
    parts.append(convex_part('ramp', [[x, y, z] for x, z0 in [(0, 0), (run, rise)]
                                   for y in [-half, half] for z in [z0, z0+rise]]))
    masses, coms, inertias = [], [], []
    for part in parts:
        mesh = part_mesh(part)
        mesh.density = 600  # explicitly provisional effective print density
        prop = mesh.mass_properties
        masses.append(prop.mass)
        coms.append(prop.center_mass)
        inertias.append(prop.inertia)
    mass = sum(masses)
    com = np.average(coms, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for m, c, I in zip(masses, coms, inertias):
        d = c-com
        inertia += I + m*(np.dot(d, d)*np.eye(3)-np.outer(d, d))
    points = np.concatenate([part_mesh(p).vertices for p in parts])
    bundle = {'schema_version': 1, 'prepared': True, 'name': 'SYNTHETIC_two_socket_fixture',
              'source': {'kind': 'synthetic', 'revision': 'fixture-v2',
                         'warning': 'NOT the Onshape block. No guide lips. Uniform effective density 600 kg/m^3. Socket sector tops recessed 5 um to avoid redundant support contacts.'},
              'parameters': {'ramp_angle_deg': ramp_angle, 'lead_angle_from_vertical_deg': lead_angle, 'facets': facets},
              'collision': parts, 'visual': [], 'clearance': throat-radius, 'max_penetration': 0.000025,
              'expected_extents': np.ptp(points, axis=0).tolist(),
              'inertial': {'mass': mass, 'com': com.tolist(), 'matrix': inertia.tolist(),
                           'provenance': 'Synthetic nonoverlapping solids, effective density 600 kg/m^3'},
              'target': {'pos': [0, 0, rise], 'quat': [1, 0, 0, 0]},
              'seating_pairs': seating, 'sockets': sockets, 'legs': legs}
    return save_bundle(bundle, output)
