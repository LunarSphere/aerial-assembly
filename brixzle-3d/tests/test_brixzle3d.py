from dataclasses import replace
import math

import numpy as np
import pytest
import trimesh

from brixzle3d import cad, geometry as G, lattice as L, rules as R, structures as S, trials as T
from brixzle3d.params import V0


@pytest.fixture(scope='module')
def brick():
    return cad.build_brick(V0)


def test_rotations_land_on_lattice_cells():
    assert G.rotate_cells([(0, 0), (1, 0), (0, 1)], 1) == [(-1, 0), (-1, 1), (-2, 0)]
    assert G.rotate_cells([(0, 0)], 4) == [(0, 0)]


def test_cell_body_has_constant_thickness_volume():
    p = replace(V0, peg_w=0.01, peg_taper=0., clearance=0., mouth_chamfer=0.01, gap=0., peg_lean=0.)
    pieces = [v for hs in G.cell_pieces(p, (0, 0), {(0, 0)}, top=False) for v in [G.vertices(hs)]]
    vol = sum(trimesh.convex.convex_hull(v).volume for v in pieces)
    assert math.isclose(vol, p.U**2*p.H, rel_tol=1e-3)


@pytest.mark.parametrize('lean', [0., 15.])
def test_brick_mesh_watertight_and_in_mass_window(lean):
    b = cad.build_brick(replace(V0, peg_lean=lean))
    assert b['mesh'].is_watertight
    assert 1 <= b['mass_g'] <= 25


def test_v1_rules_pass(brick):
    margins = R.rules(V0, brick)
    assert R.feasible(margins), {k: v for k, v in margins.items() if v < 0}


def test_tines_above_com(brick):
    assert all(z > brick['com'][2] for _, z, _, _ in brick['bores'])


def test_support_graph_and_overlap():
    g = L.connectivity(V0, S.pillar(4))
    assert not L.unsupported(g)
    with pytest.raises(ValueError):
        L.connectivity(V0, L.Structure('x', [L.Placement(0, 0, 0, 0), L.Placement(0, 1, 0, 0)], [[(0, 0)]]))


def test_tiler_rejects_3x3_and_tiles_2x3():
    base = {(i, j) for i in range(-1, 4) for j in range(-1, 4)}
    com = [20., 20.]
    two_by_three = {(i, j) for i in range(2) for j in range(3)}
    assert len(L.tile(V0, {0: two_by_three}, base, com).bricks) == 2
    with pytest.raises(ValueError):
        L.tile(V0, {0: {(i, j) for i in range(3) for j in range(3)}}, base, com)


def test_aligned_drop_seats(brick):
    rows, _ = T.drop_trials(V0, brick, T.TrialConfig(error=T.ErrorModel(ideal=True)), offsets=[(0, 0)])
    assert rows[0]['outcome'] == 'seated'


def test_capture_square_8mm(brick):
    offs = [(x, y) for x in (-8., 0., 8.) for y in (-8., 0., 8.)]
    rows, _ = T.drop_trials(V0, brick, T.TrialConfig(), offsets=offs)
    assert all(r['outcome'] == 'seated' for r in rows), [(r['dx'], r['dy'], r['outcome']) for r in rows]


def test_hooked_pillar_completes(brick):
    r, _ = T.assemble(V0, brick, S.pillar(6), T.TrialConfig(error=T.ErrorModel(ideal=True)))
    assert r['status'] == 'complete'
