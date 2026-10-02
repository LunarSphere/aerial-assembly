"""Printable mesh, STEP, mass model, and fork bores from the convex pieces."""
from pathlib import Path

import numpy as np
import trimesh

from . import geometry as G
from .params import PLA_DENSITY, BrickParams


def fork_bores(p: BrickParams, com=None, step=0.25):
    """Straight X tine bores, one per occupied row, offset delta from the row centre line.

    Returns (bores, delta) with bores as [(y, z, x0, x1)], or ([], None) when
    nothing fits. Each bore keeps ``wall`` below the keel ridges and above the
    funnel along its whole run, and clears the socket mouths.
    """
    r = p.fork_d/2 + p.fork_clearance
    s_mouth = p.peg_w/2 + p.clearance + p.mouth_chamfer
    rows = sorted({j for _, j in p.cells})
    best = None
    edge = p.a - p.gap/2 - r - p.wall
    for delta in np.arange(-edge, edge + 1e-9, step):
        if abs(delta) - r - p.wall < s_mouth:
            continue
        # Along the line the keel/funnel offset P ranges over [h|delta|/a, h].
        z_lo = p.h + r + p.wall
        z_hi = p.H + p.h*abs(delta)/p.a - r - p.wall
        if z_hi < z_lo:
            continue
        cand = (z_hi, delta)
        if best is None or cand[0] > best[0] + 1e-9:
            best = cand
    if best is None:
        return [], None
    z, delta = best
    bores = []
    for j in rows:
        xs = [i for i, jj in p.cells if jj == j]
        bores.append(((j + .5)*p.U + delta, z, min(xs)*p.U, (max(xs) + 1)*p.U))
    return bores, delta


def _union(meshes):
    import manifold3d as mf
    result = None
    for m in meshes:
        man = mf.Manifold(mf.Mesh(np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.uint32)))
        result = man if result is None else result + man
    return result


def _to_trimesh(man):
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def build_brick(p: BrickParams, solid=True):
    pieces = G.brick_pieces(p)
    bundle = {'params': p.to_dict(), 'collision': pieces}
    bores, delta = fork_bores(p)
    bundle['bores'] = bores
    bundle['bore_offset'] = delta
    if not solid:
        return bundle
    import manifold3d as mf
    hulls = [trimesh.convex.convex_hull(v) for v in pieces]
    # Grow each piece by ~1 micron for the print mesh so face-to-face seams fuse.
    grown = [trimesh.convex.convex_hull(v + 1e-3*np.sign(v - v.mean(0))) for v in pieces]
    man = _union(grown)
    r = p.fork_d/2 + p.fork_clearance
    for y, z, x0, x1 in bores:
        cyl = mf.Manifold.cylinder(x1 - x0 + 2, r, r, 48).rotate([0, 90, 0]).translate([x0 - 1, y, z])
        man = man - cyl
    mesh = _to_trimesh(man)
    solid_volume = mesh.volume
    shell_volume = min(solid_volume, mesh.area*p.shell)
    printed = shell_volume + (solid_volume - shell_volume)*p.infill
    mass = printed*PLA_DENSITY
    mesh.density = PLA_DENSITY
    inertia = mesh.moment_inertia*(mass/(solid_volume*PLA_DENSITY))
    bundle.update(mesh=mesh, volume=solid_volume, printed_volume=printed, mass_g=mass,
                  com=mesh.center_mass.tolist(), inertia=inertia.tolist(),
                  collision_volume=float(sum(h.volume for h in hulls)))
    return bundle


def build_base(p: BrickParams, cells):
    return {'collision': G.base_pieces(p, cells), 'cells': list(cells)}


def to_step(bundle, path):
    """STEP via CadQuery: each convex piece as a planar-faced solid, fused, minus bores."""
    import cadquery as cq
    p = BrickParams.from_dict(bundle['params'])
    shape = None
    for v in bundle['collision']:
        hull = trimesh.convex.convex_hull(v)
        faces = []
        for facet_tris in [[t] for t in range(len(hull.faces))]:
            tri = hull.vertices[hull.faces[facet_tris[0]]]
            faces.append(cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(*q) for q in tri], close=True)))
        solid = cq.Solid.makeSolid(cq.Shell.makeShell(faces))
        shape = solid if shape is None else shape.fuse(solid)
    shape = shape.clean()
    r = p.fork_d/2 + p.fork_clearance
    for y, z, x0, x1 in bundle['bores']:
        shape = shape.cut(cq.Solid.makeCylinder(r, x1 - x0 + 2, cq.Vector(x0 - 1, y, z), cq.Vector(1, 0, 0)))
    cq.exporters.export(shape, str(path))
    return path


def export(bundle, out, step=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle['mesh'].export(out/'brick.stl')
    if step:
        to_step(bundle, out/'brick.step')
    return out
