"""CadQuery solid, print exports and mass properties of a DF brick.

The solid is the union of the convex collision pieces (each built as a CadQuery solid from its
hull) minus the two tine bores, so the collision volume equals the printed outer volume.
"""
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from brixzle.cad import _mesh

from . import geometry as G
from .params import PLA_DENSITY, DFParams


def hull_solid(points):
    hull = trimesh.convex.convex_hull(np.asarray(points, float))
    faces = []
    for tri in hull.faces:
        pts = [cq.Vector(*map(float, hull.vertices[i])) for i in tri]
        faces.append(cq.Face.makeFromWires(cq.Wire.makePolygon(pts, close=True)))
    return cq.Solid.makeSolid(cq.Shell.makeShell(faces))


def union(pieces):
    shape = hull_solid(pieces[0])
    for piece in pieces[1:]:
        shape = shape.fuse(hull_solid(piece))
    return shape.clean()


def build_brick(p: DFParams, solid=True):
    """Geometry bundle (brick frame, mm and g) in the shape brixzle.scene expects."""
    pieces = G.brick_pieces(p)
    bundle = {'params': p.to_dict(), 'part': 'DF', 'collision': pieces,
              'collision_bored': G.brick_pieces(p, bored=True), 'channels': G.channels(p),
              'feet_x': G.feet_x(p), 'derived': p.derived()}
    if solid:
        shape = union(pieces)
        for x, z in bundle['channels']:
            bore = cq.Solid.makeCylinder(G.bore_radius(p), p.U + 2, cq.Vector(x, -p.U/2 - 1, z),
                                         cq.Vector(0, 1, 0))
            shape = shape.cut(bore)
        mesh = _mesh(shape)
        density = PLA_DENSITY*p.infill
        mesh.density = density
        bundle.update(shape=shape, mesh=mesh, volume=mesh.volume, mass_g=mesh.volume*density,
                      com=mesh.center_mass.tolist(), inertia=mesh.moment_inertia.tolist(),
                      fragments=len(mesh.split(only_watertight=False)))
    else:
        vols = [trimesh.convex.convex_hull(v).volume for v in pieces]
        cents = [trimesh.convex.convex_hull(v).center_mass for v in pieces]
        mass = sum(vols)*PLA_DENSITY*p.infill  # overlaps counted twice: an upper bound
        bundle.update(volume=sum(vols), mass_g=mass,
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
