"""Local drop preparation for supported two-peg block exports.

Feature recognition is deliberately limited to two vertical circular pegs and
blind coaxial tapered sockets. Unsupported topology fails instead of reusing
dimensions from the previous CAD revision.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import trimesh

from .config import Physics, Release, TrialSettings, digest, read_json, write_json
from .geometry import bounds_points, part_mesh
from .mjcf import _load, _parts
from .model import build_model, initial_state
from .simulation import run_drop, validate_geometry


def recognize_features(meshes):
    """Measure the current export, including designs that cannot fully seat."""
    if len(meshes) == 1:
        from .cad_single import recognize_single
        return recognize_single(meshes[0])
    if len(meshes) != 3:
        raise ValueError('Block profile expects one supported solid or three unmerged solids: two pegs and the body')
    ordered = sorted(meshes, key=lambda m: m.volume)
    pegs, body = ordered[:2], ordered[2]
    pegs.sort(key=lambda m: m.bounds[:, 0].mean())
    sockets, legs, seating, shifts = [], [], [], []
    for index, peg in enumerate(pegs):
        low, high = peg.bounds
        center = (low[:2]+high[:2])/2
        radius = float((high[0]-low[0])/2)
        if not peg.is_convex or abs(high[1]-low[1]-2*radius) > 1e-6:
            raise ValueError('Block profile requires convex vertical circular pegs')
        radial = np.linalg.norm(peg.vertices[:, :2]-center, axis=1)
        if abs(radial.max()-radius) > 1e-6 or high[2]-low[2] < 2*radius:
            raise ValueError('Peg shape/orientation is unsupported by the block profile')
        # Each socket has floor and throat rings of equal radius and a wider mouth.
        candidates = []
        for height in np.unique(np.round(body.vertices[:, 2], 6)):
            points = body.vertices[abs(body.vertices[:, 2]-height) < 1e-6]
            distances = np.linalg.norm(points[:, :2]-center, axis=1)
            for r in np.unique(np.round(distances, 6)):
                ring = points[abs(distances-r) < 1e-6]
                if len(ring) < 16 or r <= radius or r > 4*radius:
                    continue
                angles = np.sort(np.arctan2(ring[:,1]-center[1], ring[:,0]-center[0]))
                if np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])) > .5:
                    continue
                candidates.append((float(np.mean(ring[:,2])), float(np.mean(np.linalg.norm(ring[:,:2]-center, axis=1)))))
        candidates.sort()
        if len(candidates) != 3:
            raise ValueError(f'Could not recognize floor/throat/mouth rings for peg {index+1}: {candidates}')
        (floor, bore), (throat, r2), (mouth, opening) = candidates
        if abs(bore-r2) > 1e-6 or not floor < throat < mouth or opening <= bore:
            raise ValueError('Socket is not a blind cylindrical bore with a conical lead-in')
        shifts.append(mouth-float(high[2]))
        sockets.append({'mouth': [*center.tolist(), mouth], 'quat': [1,0,0,0],
                        'depth': mouth-floor, 'lead_depth': mouth-throat,
                        'throat_radius': bore, 'opening_radius': opening})
        legs.append({'tip': [*center.tolist(), float(low[2])],
                     'target_depth': float(high[2]-low[2]),
                     'surface_probes': peg.vertices.tolist()})
        # On this CAD family the flat land extends from the opening to Y=±depth/2.
        # Select points beyond the opening, inside the measured extrusion width.
        edge = min(abs(body.bounds[0,1]-center[1]), abs(body.bounds[1,1]-center[1]))
        if edge <= opening+1e-4:
            raise ValueError('No flat seating strip outside socket opening')
        y = (opening+edge)/2
        for dx, dy in [(-radius, -y), (radius, -y), (-radius, y), (radius, y)]:
            seating.append({'upper': [float(center[0]+dx), float(center[1]+dy), float(high[2])],
                            'lower': [float(center[0]+dx), float(center[1]+dy), mouth], 'normal': [0,0,1]})
    if abs(shifts[0]-shifts[1]) > 1e-6 or min(shifts) <= 0:
        raise ValueError('Socket mouths and peg roots do not define a common vertical seated transform')
    # Float STL coordinates can differ by a few nanometers; use one shared datum.
    shift = float(np.mean(shifts))
    for pair in seating:
        pair['lower'][2] = pair['upper'][2]+shift
    clearance = min(s['throat_radius']-(p.bounds[1,0]-p.bounds[0,0])/2 for s,p in zip(sockets,pegs))
    if clearance <= 0:
        raise ValueError('Peg radius is not smaller than the socket throat radius')
    return {'target': {'pos': [0,0,shift], 'quat': [1,0,0,0]},
            'sockets': sockets, 'legs': legs, 'seating_pairs': seating,
            'clearance': float(clearance), 'max_penetration': float(clearance*.1)}


def _inertia(meshes, density):
    masses = np.array([m.volume*density for m in meshes])
    centers = np.array([m.center_mass for m in meshes])
    com = np.average(centers, axis=0, weights=masses)
    inertia = np.zeros((3,3))
    for mesh, mass, center in zip(meshes, masses, centers):
        delta = center-com
        inertia += mesh.moment_inertia*density + mass*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    return {'mass': float(sum(masses)), 'com': com.tolist(), 'matrix': inertia.tolist()}


def prepare_local_export(directory, *, density=600., mass_grams=None, collision_error=.0000005,
                     cache='assets/cad-cache', progress=print):
    """Freeze a local robot export into one block; decompose collision geometry."""
    if density <= 0 or not np.isfinite(density) or (mass_grams is not None and (mass_grams <= 0 or not np.isfinite(mass_grams))):
        raise ValueError('Density/mass must be finite and positive')
    if collision_error <= 0 or not np.isfinite(collision_error):
        raise ValueError('Collision error must be finite and positive')
    directory = Path(directory).resolve()
    model, data = _load(directory/'robot.xml')
    import mujoco
    if model.neq or model.nu or any(int(t) != mujoco.mjtJoint.mjJNT_FREE for t in model.jnt_type):
        raise ValueError('Local block export must contain only rigid parts/root free joints, without actuators or equality constraints')
    parts = _parts(model, data, set(range(1,model.nbody)), np.zeros(3), np.eye(3))
    visual = parts['visual'] or parts['collision']
    meshes = [part_mesh(p) for p in visual]
    if any(not m.is_volume for m in meshes):
        raise ValueError('Source visual solids must be watertight and consistently oriented')
    features = recognize_features(meshes)
    if len(meshes) == 1 and collision_error < np.sqrt(3)*.5e-7:
        raise ValueError('Single-solid collision error budget must cover the 0.1 micrometer vertex grid')
    if collision_error > features['clearance']*.11:
        raise ValueError('Collision approximation budget exceeds 11% of measured functional clearance')
    # Export placeholders and collision partitions never determine mass properties.
    effective_density = density if mass_grams is None else mass_grams/1000/sum(m.volume for m in meshes)
    inertial = _inertia(meshes, effective_density)
    inertial['provenance'] = ('Provisional uniform effective density' if mass_grams is None else
                              'Measured total mass, uniform density distribution assumed')
    inertial['density_kg_m3'] = effective_density
    source_files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.xml','.stl','.obj','.json')}
    collision = []
    from .cad_collision import partition_body, partition_solid
    options = {'method': 'extruded-profile-minus-socket-frusta', 'version': 1, 'plane_tolerance_m': 1e-8,
               'vertex_snap_m': 1e-7, 'implementation_sha256': hashlib.sha256(Path(__file__).with_name('cad_collision.py').read_bytes()).hexdigest()}
    if len(meshes) == 1:
        import tetgen
        options = {'method': 'constrained-tetrahedra-convex-unions', 'version': 1,
                   'tetgen_version': tetgen.__version__, 'switches': 'pQ',
                   'vertex_snap_m': 1e-7,
                   'implementation_sha256': options['implementation_sha256']}
    for i, mesh in enumerate(meshes):
        if mesh.is_convex:
            pieces = [mesh]
        else:
            key = digest({'mesh': visual[i], 'options': options, 'sockets': features['sockets']})
            cached = Path(cache)/f'{key}.json'
            if cached.exists():
                saved = read_json(cached)
                if saved['hash'] != digest(saved['parts']):
                    raise ValueError('Collision cache hash mismatch; remove the affected cache file')
                pieces = [trimesh.Trimesh(vertices=p['vertices'], faces=p['faces'], process=False) for p in saved['parts']]
            else:
                progress(f'Partitioning CAD solid {i+1}, preserving exported socket facets…')
                pieces = partition_solid(mesh) if len(meshes) == 1 else partition_body(mesh, features['sockets'])
                if any(not p.is_volume or not p.is_convex for p in pieces):
                    raise ValueError('Decomposition produced a nonconvex or invalid collision solid')
                saved = [{'vertices': p.vertices.tolist(), 'faces': p.faces.tolist()} for p in pieces]
                write_json(cached, {'parts': saved, 'hash': digest(saved)})
        for j, piece in enumerate(pieces):
            if not piece.is_volume or not piece.is_convex:
                raise ValueError('Decomposition produced a nonconvex or invalid collision solid')
            collision.append({'name': f'part_{i}_convex_{j}', 'type': 'mesh',
                              'vertices': piece.vertices.tolist(), 'faces': piece.faces.tolist()})
    bundle = {'schema_version': 1, 'name': 'two_peg_block', 'visual': visual, 'collision': collision,
              'inertial': inertial, **features,
              'source': {'kind': 'local_robot_export',
                         'files_sha256': source_files, 'provisional_mass': mass_grams is None,
                         'assumptions': ['All exported parts form one rigid block', 'Uniform effective material density'],
                         'collision': options}}
    bundle['expected_extents'] = np.ptp(bounds_points(bundle), axis=0).tolist()
    bundle['asset_hash'] = digest(bundle)
    return bundle


def save_trial(directory, result, trace):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'result.json', result)
    np.savez_compressed(directory/'trajectory.npz', state=trace,
                        columns=np.array(['time','x','y','z','qw','qx','qy','qz',
                                          'vx','vy','vz','wx_local','wy_local','wz_local']))


def cad_drop(directory, output, *, density=600., mass_grams=None,
             physics=Physics(), settings=TrialSettings(duration=1., dwell=.1),
             release=Release(), collision_error=.0000005, cache='assets/cad-cache', progress=print):
    """Record one local drop, including diagnostic drops of infeasible geometry."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = prepare_local_export(directory, density=density, mass_grams=mass_grams,
                              collision_error=collision_error, cache=cache, progress=progress)
    write_json(output/'geometry.json', bundle)
    model, xml = build_model(bundle, physics)
    (output/'scene.xml').write_text(xml)
    gate = validate_geometry(model, bundle)
    bottoming = [max(0., leg['target_depth']-socket['depth'])
                 for leg, socket in zip(bundle['legs'], bundle['sockets'])]
    target_infeasible = (max(bottoming) > bundle['max_penetration'] or
                         gate['target']['penetration'] > bundle['max_penetration'])
    gate['peg_bottoming_m'] = bottoming
    write_json(output/'geometry_validation.json', gate)
    write_json(output/'experiment.json', {'release': asdict(release), 'physics': asdict(physics),
                                         'settings': asdict(settings)})
    write_json(output/'code_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in Path(__file__).parent.glob('*.py')})
    if not gate['passed']:
        progress('Geometry checks failed: recording a diagnostic drop; inspect geometry_validation.json.')
    progress(f'Drop: simulating {settings.duration:g} seconds…')
    result, trace = run_drop(model, bundle, initial_state(bundle, release), settings,
                             record=True, progress=progress)
    if not gate['passed'] and result['status'] == 'success':
        result['status'] = 'geometry_rejected'
        result['settling_time'] = None
    save_trial(output/'trial_00000', result, trace)
    valid = (gate['socket_probes']['passed'] and result['status'] != 'invalid'
             and (gate['passed'] or target_infeasible))
    summary = {'status': result['status'], 'invalid_reason': result['invalid_reason'],
               'valid': valid, 'geometry_feasible': gate['passed'],
               'collision_probes_passed': gate['socket_probes']['passed'],
               'target_infeasible': target_infeasible, 'peg_bottoming_m': bottoming,
               'provisional_mass': mass_grams is None, 'geometry_hash': bundle['asset_hash'],
               'max_penetration': result['max_penetration'],
               'final': result['final'], 'settling_time': result['settling_time']}
    write_json(output/'summary.json', summary)
    progress(f"Drop: {result['status']}; gap {result['final']['max_seating_gap']*1000:.3f} mm")
    return summary
