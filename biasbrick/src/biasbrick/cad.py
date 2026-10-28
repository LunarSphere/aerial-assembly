"""CadQuery solid, print exports and mass properties of a biasbrick.

The solid is the union of the convex collision pieces (dfbrick's hull-and-fuse); the fork
cavities are built hollow from pieces, so nothing is cut and the collision volume equals the
printed outer volume.
"""
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from brixzle.cad import _mesh
from dfbrick.cad import union

from . import geometry as G
from .params import PLA_DENSITY, BBParams


def build_brick(p: BBParams, solid=True):
    """Geometry bundle (brick frame, mm and g) in the shape brixzle.scene expects."""
    pieces = G.brick_pieces(p)
    bundle = {'params': p.to_dict(), 'part': p.part, 'collision': pieces, 'collision_bored': pieces,
              'channels': G.channels(p), 'feet': G.feet_seats(p), 'cavity_y': G.cavity_y(p),
              'derived': p.derived()}
    if solid:
        shape = union(pieces)
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
