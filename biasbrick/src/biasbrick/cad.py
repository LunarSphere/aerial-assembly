"""CadQuery solid, print exports and mass properties of a biasbrick.

Each socket body is brixzle's profile with its lightening holes, extruded through the Y sawtooth's
two bands (as ``brixzle.cad._swept``); the flange, rod and sleeves are the union of their convex
pieces (dfbrick's hull-and-fuse). Collision uses the outer pieces only (``geometry.brick_pieces``).
"""
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from brixzle.cad import _mesh
from dfbrick.cad import union

from . import geometry as G
from .params import PLA_DENSITY, BBParams


def _wire(coords, dx, y, dz):
    pts = [cq.Vector(float(x) + dx, float(y), float(z) + dz) for x, z in list(coords)[:-1]]
    return cq.Wire.makePolygon(pts, close=True)


def socket_solid(p: BBParams, dx, part=None):
    prof = G.socket_profile(p, part)
    solids = []
    for poly, y0, y1 in G.body_bands(p, prof):
        z0, z1 = float(G.rise(p, y0)), float(G.rise(p, y1))
        outer = _wire(poly.exterior.coords, dx, y0, z0)
        holes = [_wire(h.exterior.coords, dx, y0, z0) for h in prof['holes']] if p.lighten else []
        solids.append(cq.Solid.extrudeLinear(outer, holes, cq.Vector(0, y1 - y0, z1 - z0)))
    shape = solids[0]
    for other in solids[1:]:
        shape = shape.fuse(other)
    return shape.clean()


def extra_pieces(p: BBParams, part=None):
    """Everything but the socket bodies: flanges, rod and sleeves."""
    prof = G.socket_profile(p, part)
    pieces = []
    for i in (0, 1):
        dx = p.socket_x(i)
        if p.flange > 0:
            pieces += G.sweep(p, [G.flange_profile(p, prof['slots'], part)], p.D/2, p.D/2 + p.flange, dx)
    return pieces + G.rod_and_sleeve_pieces(p, part)


def build_brick(p: BBParams, solid=True, part=None):
    """Geometry bundle (brick frame, mm and g) in the shape brixzle.scene expects."""
    part = part or p.part
    pieces = G.brick_pieces(p, part)
    prof = G.socket_profile(p, part)
    bundle = {'params': p.to_dict(), 'part': part, 'collision': pieces, 'collision_bored': pieces,
              'channels': G.channels(p, part), 'sleeve_y': G.sleeve_y(p), 'derived': p.derived(),
              'keels': [(p.socket_x(i) + prof['keel'][0], prof['keel'][1]) for i in (0, 1)],
              'valleys': [(p.socket_x(i) + prof['valley'][0], prof['valley'][1]) for i in (0, 1)],
              'profile_fragments': prof['fragments']}
    if solid:
        shape = union(extra_pieces(p, part))
        for i in (0, 1):
            shape = shape.fuse(socket_solid(p, p.socket_x(i), part))
        shape = shape.clean()
        mesh = _mesh(shape)
        density = PLA_DENSITY*p.infill
        mesh.density = density
        bundle.update(shape=shape, mesh=mesh, volume=mesh.volume, mass_g=mesh.volume*density,
                      com=mesh.center_mass.tolist(), inertia=mesh.moment_inertia.tolist(),
                      fragments=len(mesh.split(only_watertight=False)))
    else:
        vols = [trimesh.convex.convex_hull(v).volume for v in pieces]
        cents = [trimesh.convex.convex_hull(v).center_mass for v in pieces]
        bundle.update(volume=sum(vols), mass_g=sum(vols)*PLA_DENSITY*p.infill,
                      com=(np.average(cents, axis=0, weights=vols)).tolist())
    return bundle


def collision_volume(bundle):
    return sum(trimesh.convex.convex_hull(v).volume for v in bundle['collision'])


def export(bundle, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle['mesh'].export(out/'brick.stl')
    cq.exporters.export(bundle['shape'], str(out/'brick.step'))
    return out/'brick.stl', out/'brick.step'
