"""Synthetic geometry for physics regression tests, not a CAD import route."""
import math

import numpy as np
import trimesh

from aerial_assembly.config import digest, write_json
from aerial_assembly.geometry import part_mesh


def convex_part(name, vertices):
    mesh = trimesh.convex.convex_hull(np.asarray(vertices))
    return {'name': name, 'type': 'mesh', 'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist()}


def demo_geometry(output, ramp_angle=50.0, lead_angle=30.0, facets=48):
    """Synthetic test fixture, NOT a reconstruction of the user's CAD design.

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
                         'warning': 'NOT the supplied block. No guide lips. Uniform effective density 600 kg/m^3. Socket sector tops recessed 5 um to avoid redundant support contacts.'},
              'parameters': {'ramp_angle_deg': ramp_angle, 'lead_angle_from_vertical_deg': lead_angle, 'facets': facets},
              'collision': parts, 'visual': [], 'clearance': throat-radius, 'max_penetration': 0.000025,
              'expected_extents': np.ptp(points, axis=0).tolist(),
              'inertial': {'mass': mass, 'com': com.tolist(), 'matrix': inertia.tolist(),
                           'provenance': 'Synthetic nonoverlapping solids, effective density 600 kg/m^3'},
              'target': {'pos': [0, 0, rise], 'quat': [1, 0, 0, 0]},
              'seating_pairs': seating, 'sockets': sockets, 'legs': legs}
    bundle['asset_hash'] = digest(bundle)
    write_json(output, bundle)
    return bundle
