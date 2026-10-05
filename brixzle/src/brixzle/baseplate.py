"""n x n baseplate: the anchored base that receives course 0, as a printable solid.

The top is the course-0 seat (sawtooth with slots) from ``profile.base_outer``,
swept through the same Y gutter as a brick, so a brick's keel and gutter nest
into it. Each of the n rows along Y is one swept strip; the strips are fused and
the bottom is then cut flat (the gutter shear would otherwise ridge it).
"""
import cadquery as cq

from . import cad, profile as pr
from .params import BrickParams


def build_baseplate(p: BrickParams, n: int, thickness=None, direction=1):
    """CadQuery solid and mesh for an n x n baseplate. X spans [0, nU], Y spans [0, nD], seats at z = 0."""
    if n < 1:
        raise ValueError('n must be >= 1')
    thickness = thickness or p.tooth_L + p.slot_extra + 4
    drop = pr.shear_drop(p) if p.gutter > 0 else 0.
    # Extra depth so the sheared bottom still reaches the flat cut plane.
    outer, _ = pr.base_outer(p, n, thickness + drop + 1., direction)
    strip = cad._swept(outer, [], p)
    shape = None
    for j in range(n):
        s = strip.translate(cq.Vector(0, (j + .5)*p.D, 0))
        shape = s if shape is None else shape.fuse(s)
    box = cq.Solid.makeBox(n*p.U + 2, n*p.D + 2, thickness + p.amplitude + p.tooth_L + 10,
                           cq.Vector(-1, -1, -thickness))
    shape = shape.intersect(box).clean()
    mesh = cad._mesh(shape)
    mesh.density = 1.24e-3*p.infill
    return {'shape': shape, 'mesh': mesh, 'mass_g': mesh.volume*mesh.density, 'n': n, 'thickness': thickness}


def export(bundle, out, step=False):
    from pathlib import Path
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = f'baseplate_{bundle["n"]}x{bundle["n"]}'
    bundle['mesh'].export(out/f'{name}.stl')
    if step:
        cq.exporters.export(bundle['shape'], str(out/f'{name}.step'))
    return out/f'{name}.stl'
