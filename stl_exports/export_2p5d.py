"""Export print STLs for the 2.5D brixzle parts (run with brixzle's uv env)."""
from dataclasses import replace
from pathlib import Path

import cadquery as cq
from shapely import affinity

from brixzle import cad, profile as pr, ring as Rg
from brixzle.params import V0

OUT = Path(__file__).parent


def base_mesh(p, voxels, direction=1):
    outer, _ = pr.base_outer(p, voxels, p.tooth_L + p.slot_extra + 4, direction)
    return cad._mesh(cad._swept(outer, [], p))


def save(mesh, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    print(f'{path.relative_to(OUT)}: {mesh.volume*1.24e-3:.1f} g solid, watertight={mesh.is_watertight}')


# AB pair (pegs lean against the build direction)
ab = replace(V0, lean_mode='ab')
for name, b in cad.build_bricks(ab).items():
    save(b['mesh'], OUT/'2p5d_ab'/f'brick_{name}.stl')
save(base_mesh(ab, 4, +1), OUT/'2p5d_ab'/'base_plate_build_plus_x_4vox.stl')
save(base_mesh(ab, 4, -1), OUT/'2p5d_ab'/'base_plate_build_minus_x_4vox.stl')

# Ring / turn system (alternate mode): L3 turn plus the other ring parts
parts = Rg.build_parts(V0)
for name in ('L3', 'I3', 'K2', 'S2'):
    save(parts[name]['mesh'], OUT/'2p5d_ring'/f'{name}.stl')
save(base_mesh(V0, 4), OUT/'2p5d_ring'/'base_plate_straight_4vox.stl')

# Anchored base for a 6x6 ring: one oriented base cell per course-0 cell
outer, _ = pr.base_outer(V0, 1, V0.tooth_L + V0.slot_extra + 4)
cell = cad._swept(affinity.translate(outer, -V0.U/2, 0.), [], V0)
shape = None
for (ci, cj), yaw in Rg.course0_yaws(Rg.ring(6, 1)).items():
    s = cell.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 90*yaw).translate(
        cq.Vector((ci + .5)*V0.U, (cj + .5)*V0.U, 0))
    shape = s if shape is None else shape.fuse(s)
save(cad._mesh(shape.clean()), OUT/'2p5d_ring'/'base_plate_ring_6x6.stl')
