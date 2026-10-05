"""CadQuery solid, print exports, mass properties, and convex collision pieces.

The printable solid is the XZ profile (with lightening holes) swept through the
piecewise-affine Y bands of ``profile.y_bands`` (flat centre, sheared gutter
sides), unioned, minus two straight Y tine bores. Collision uses the outer
profile only (internal through-holes are never touched by other bricks): it is
decomposed into convex polygons, and each polygon swept through one band is
convex because the sweep is affine. No tetrahedralization is needed.
"""
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from . import profile as pr
from .convex import convex_parts
from shapely.ops import unary_union
from .profile import _largest
from .params import PLA_DENSITY, BrickParams


def _wire(coords, y=0., dz=0.):
    pts = [cq.Vector(float(x), float(y), float(z) + dz) for x, z in list(coords)[:-1]]
    return cq.Wire.makePolygon(pts, close=True)


def _swept(polygon, holes, p: BrickParams):
    solids = []
    for y0, y1, z0, z1 in pr.y_bands(p):
        outer = _wire(polygon.exterior.coords, y0, z0)
        inner = [_wire(h.exterior.coords, y0, z0) for h in holes]
        solids.append(cq.Solid.extrudeLinear(outer, inner, cq.Vector(0, y1 - y0, z1 - z0)))
    shape = solids[0]
    for other in solids[1:]:
        shape = shape.fuse(other)
    return shape.clean()


def _pieces(polygon, p: BrickParams):
    """Convex 3D vertex sets (mm) covering the banded sweep of polygon."""
    coords = np.asarray(polygon.simplify(1e-4).exterior.coords)[:-1]
    pieces = []
    for q in convex_parts(coords):
        for y0, y1, z0, z1 in pr.y_bands(p):
            near = np.column_stack([q[:, 0], np.full(len(q), y0), q[:, 1] + z0])
            far = np.column_stack([q[:, 0], np.full(len(q), y1), q[:, 1] + z1])
            pieces.append(np.vstack([near, far]))
    return pieces


def _mesh(shape, tolerance=0.05):
    vertices, faces = shape.tessellate(tolerance, 0.2)
    mesh = trimesh.Trimesh(np.array([v.toTuple() for v in vertices]), np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def build_brick(p: BrickParams, solid=True, part='A'):
    """Geometry bundle for one brick part (unrotated frame), in millimetres and grams."""
    parts = pr.brick_parts(p, part)
    channels = pr.fork_channels(p, parts)
    holes = pr.lightening_holes(p, parts, channels)
    outer = parts['outer']
    barbs = parts.get('barbs', [])
    printed = _largest(unary_union([outer, *barbs])) if barbs else outer
    net = printed
    for h in holes:
        net = net.difference(h)
    bundle = {
        'params': p.to_dict(), 'part': part,
        'outer': outer, 'holes': holes, 'channels': channels,
        'collision': _pieces(outer, p),
        'soft_collision': [q for b in barbs for q in _pieces(b, p)],
        'keels': [k.tolist() for k in parts['keels']],
        'valleys': [v.tolist() for v in parts['valleys']],
        'profile_area': outer.area, 'net_area': net.area, 'fragments': parts['fragments'],
    }
    if solid:
        shape = _swept(printed, holes, p)
        for x, z in channels:
            bore = cq.Solid.makeCylinder(p.fork_d/2 + p.fork_clearance, p.D + 2,
                                         cq.Vector(x, -p.D/2 - 1, z), cq.Vector(0, 1, 0))
            shape = shape.cut(bore)
        mesh = _mesh(shape)
        density = PLA_DENSITY*p.infill
        mesh.density = density
        bundle.update(shape=shape, mesh=mesh, volume=mesh.volume, mass_g=mesh.volume*density,
                      com=mesh.center_mass.tolist(), inertia=mesh.moment_inertia.tolist())
    else:
        c = net.centroid
        lift = sum(abs(y1 - y0)*(z0 + z1)/2 for y0, y1, z0, z1 in pr.y_bands(p))/p.D
        bundle.update(volume=net.area*p.D, mass_g=net.area*p.D*PLA_DENSITY*p.infill,
                      com=[c.x, 0., c.y + lift])
    return bundle


def build_bricks(p: BrickParams, solid=True):
    """{part: bundle}: just A, or A and B in 'ab' lean mode."""
    names = ('A', 'B') if p.lean_mode == 'ab' else ('A',)
    return {n: build_brick(p, solid, n) for n in names}


def build_base(p: BrickParams, voxels, thickness=None, direction=1):
    """Anchored base plate (receives course 0) as collision pieces."""
    thickness = thickness or p.tooth_L + p.slot_extra + 4
    outer, valleys = pr.base_outer(p, voxels, thickness, direction)
    return {'outer': outer, 'collision': _pieces(outer, p), 'valleys': [v.tolist() for v in valleys],
            'voxels': voxels}


def export(bundle, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle['mesh'].export(out/'brick.stl')
    cq.exporters.export(bundle['shape'], str(out/'brick.step'))
    return out/'brick.stl', out/'brick.step'


def collision_volume(bundle):
    return sum(trimesh.convex.convex_hull(v).volume for v in bundle['collision'])
