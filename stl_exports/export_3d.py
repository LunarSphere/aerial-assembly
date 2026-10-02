"""Export print STLs for the 3D polycube L brick (run with the repo-root .venv)."""
from pathlib import Path

import trimesh

from brixzle3d import cad, geometry as G
from brixzle3d.params import V0

OUT = Path(__file__).parent/'3d_L'
OUT.mkdir(parents=True, exist_ok=True)

b = cad.build_brick(V0)
b['mesh'].export(OUT/'L_brick_v1_hooked.stl')
print(f'L_brick_v1_hooked.stl: {b["mass_g"]:.1f} g printed (shell+infill), watertight={b["mesh"].is_watertight}')

for n in (4,):
    cells = [(i, j) for i in range(n) for j in range(n)]
    pieces = G.base_pieces(V0, cells)
    mesh = cad._to_trimesh(cad._union([trimesh.convex.convex_hull(v) for v in pieces]))
    mesh.export(OUT/f'base_plate_{n}x{n}.stl')
    print(f'base_plate_{n}x{n}.stl: {mesh.volume*1.24e-3:.1f} g solid, watertight={mesh.is_watertight}')
