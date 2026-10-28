import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull
import trimesh

from biasbrick import cad, geometry as G, rules as R
from biasbrick.params import DEFAULT, BBParams


@pytest.fixture(scope='module')
def brick():
    return cad.build_brick(DEFAULT)


def test_params_validate():
    with pytest.raises(ValueError):
        BBParams.from_dict({'U': 49.5, 'flare': 35.})
    with pytest.raises(ValueError):
        BBParams(part='Z')
    with pytest.raises(ValueError):
        BBParams(U=12.)
    with pytest.raises(ValueError):
        BBParams(ramp=float('nan'))
    with pytest.raises(ValueError):
        BBParams(alpha=65.)
    assert BBParams.from_dict({**DEFAULT.to_dict(), 'derived': {}}) == DEFAULT


def test_seat_is_in_the_minus_x_minus_y_corner():
    p = DEFAULT
    assert p.seat < 0
    # The side wall's outer face sits half an end gap inside the voxel's -x-y edges.
    assert p.seat - p.throat - p.wall_t == pytest.approx(-p.U/2 + p.end_gap/2)
    assert p.reach == pytest.approx(p.U/2 - p.end_gap/2 - p.seat)
    assert p.seat_xy(0) == (pytest.approx(-p.U + p.seat), pytest.approx(p.seat))


def test_ramps_reach_the_ridge_and_the_rail_clears_it():
    p = DEFAULT
    assert G.ramp_z(p, p.reach, 0.) == pytest.approx(p.guide_h)
    assert G.ramp_z(p, p.throat + p.lean_shift, 0.) == pytest.approx(p.slot_h)
    assert p.foot_L - p.guide_h == pytest.approx(p.top_gap)


def test_aim_covers_error_and_yaw_on_both_axes():
    p = DEFAULT
    near, far = R.landing(p, p.aim_r)
    assert near == (pytest.approx(p.aim_margin), pytest.approx(p.aim_margin))
    assert p.aim[1] > p.aim[0]  # yaw moves the feet across the brick far more than along it


def test_pieces_are_convex_and_solid(brick):
    for v in brick['collision']:
        hull = trimesh.convex.convex_hull(v)
        assert hull.volume > 0
        assert len(np.unique(np.round(v, 6), axis=0)) == len(np.unique(np.round(hull.vertices, 6), axis=0))


def test_solid_is_one_watertight_piece(brick):
    assert brick['fragments'] == 1
    assert brick['mesh'].is_watertight
    assert brick['volume'] == pytest.approx(cad.collision_volume(brick), rel=0.1)


def _hulls(pieces, dz=0.):
    return [ConvexHull(np.asarray(v) + [0., 0., dz]).equations for v in pieces]


def _inside(hulls, pts, depth=0.):
    """Points more than ``depth`` inside the union of convex pieces (the solid, by construction)."""
    pts = np.atleast_2d(pts)
    h = np.c_[pts, np.ones(len(pts))]
    return np.any([np.all(h @ eq.T <= -depth, axis=1) for eq in hulls], axis=0)


def _sample(piece, n, rng):
    """Points inside a convex piece: random convex combinations of its vertices."""
    return rng.dirichlet(np.ones(len(piece)), n) @ np.asarray(piece)


def test_cup_foot_nests_in_the_hopper_below(brick):
    """Two courses of a tower do not overlap: the cup sits clearance off every hopper face."""
    p = DEFAULT
    lower = _hulls(brick['collision'])
    rng = np.random.default_rng(0)
    sx, sy = p.seat_xy(0)
    feet = [q + [0., 0., p.H0] for seat in G.feet_seats(p) for q in G.foot_pieces(p, seat)]
    pts = np.vstack([_sample(q, 4000, rng) for q in feet] + [q for q in feet])
    # No part of the upper brick's feet is inside the lower brick (0.05 mm tolerance).
    assert not _inside(lower, pts, depth=0.05).any()
    # The seated tooth is in the slot, on its axis just above its tip.
    h = p.slot_extra + 0.5
    tip = [sx + h*p.lean_k, sy + h*p.lean_k, p.bar_h + h]
    assert _inside(_hulls(feet), tip).all()
    assert not _inside(lower, tip).any()


def test_slot_leans_alpha_along_the_diagonal():
    p = DEFAULT
    plus_x = G.hopper_pieces(p, (0., 0.), 0.)[0]  # the +x slot plate
    bottom = plus_x[plus_x[:, 2] == 0.]
    top = plus_x[np.isclose(plus_x[:, 2], p.slot_h)]
    shift = top[:, :2].min(0) - bottom[:, :2].min(0)
    assert shift == pytest.approx([p.lean_shift, p.lean_shift])
    assert math.degrees(math.atan(np.hypot(*shift)/p.slot_h)) == pytest.approx(p.alpha)


def test_seated_tooth_hooks_and_leaves_along_its_axis():
    """Straight up, the tooth hits the slot's overhanging walls; backwards along its axis it is free."""
    p = DEFAULT
    slot = _hulls(G.hopper_pieces(p, (0., 0.), 0.)[:4])
    tooth = G.foot_pieces(p)[0] + [0., 0., p.foot_L]
    rng = np.random.default_rng(0)
    pts = np.vstack([_sample(tooth, 4000, rng), tooth])
    pts = pts[pts[:, 2] < p.slot_h]  # the part inside the slot
    assert not _inside(slot, pts, depth=0.02).any()
    # The tip stops slot_extra above the floor: the brick bears on its cups, not on the tooth.
    assert pts[:, 2].min() == pytest.approx(p.slot_extra)
    lift = (p.clearance + p.tooth_taper/2)/p.lean_k + 0.5
    assert _inside(slot, pts + [0., 0., lift], depth=0.1).any()
    axis = np.array([p.lean_k, p.lean_k, 1.])
    for h in (1., 3., 6.):
        assert not _inside(slot, pts + h*axis, depth=0.02).any()


def test_cavity_is_a_shelled_square_with_a_v_roof(brick):
    p = DEFAULT
    mesh = _hulls(brick['collision'])
    y0, y1 = brick['cavity_y']
    ym = (y0 + y1)/2
    h = G.cavity_half(p)
    assert len(brick['channels']) == 2
    for x, z in brick['channels']:
        # The tine as carried, and the whole square inside, is empty space; the shell around it is solid.
        assert not _inside(mesh, [[x, ym, z], [x - h + 0.2, ym, z - 2*h + 2], [x + h - 0.2, ym, z - 2*h + 2]]).any()
        assert _inside(mesh, [[x - h - p.cav_wall/2, ym, z - h], [x + h + p.cav_wall/2, ym, z - h]]).all()
        # The V roof: just above the tine is solid, and the roof is lower at the sides than at the apex.
        assert _inside(mesh, [x, ym, z + p.fork_d/2/math.cos(math.radians(p.roof_angle)) + 0.3]).all()
        assert _inside(mesh, [x - h + 0.3, ym, z + 0.5]).all()
        # The fork enters from the -y face through an open mouth wider than the cavity.
        assert not _inside(mesh, [[x - h - 1., y0 + 0.2, z - h], [x, y0 + 0.2, z]]).any()


def test_y_part_mirrors_so_its_seats_point_world_minus_x_minus_y():
    p = DEFAULT
    x_pieces = G.brick_pieces(p)
    y_pieces = G.brick_pieces(BBParams(**{**p.to_dict(), 'part': 'Y'}))
    assert all(np.allclose(a*[1, -1, 1], b) for a, b in zip(x_pieces, y_pieces))
    # Placed at yaw +90, (x, y) -> (-y, x): each Y-part seat lands in the -x-y corner of its world voxel.
    rot = lambda x, y: (-y, x)  # noqa: E731
    for i in range(p.n_vox):
        sx, sy = p.seat_xy(i)
        vx, vy = rot(p.voxel_x(i), 0.)
        wx, wy = rot(sx, -sy)
        assert (wx - vx, wy - vy) == (pytest.approx(p.seat), pytest.approx(p.seat))


def test_base_has_a_hopper_per_voxel():
    p = DEFAULT
    pieces = G.base_pieces(p, [(0., 0.), (p.U, 0.)])
    assert len(pieces) == 2*(1 + len(G.hopper_pieces(p)))
