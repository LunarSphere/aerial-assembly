import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull
import trimesh

from brixzle import profile as pr
from biasbrick import cad, geometry as G
from biasbrick.params import DEFAULT, BBParams


@pytest.fixture(scope='module')
def bricks():
    return {k: cad.build_brick(DEFAULT, part=k) for k in ('A', 'B')}


def _hulls(pieces, shift=(0., 0., 0.)):
    return [ConvexHull(np.asarray(v) + shift).equations for v in pieces]


def _inside(hulls, pts, depth=0.):
    """Points more than ``depth`` inside the union of convex pieces (the solid, by construction)."""
    pts = np.atleast_2d(pts)
    h = np.c_[pts, np.ones(len(pts))]
    return np.any([np.all(h @ eq.T <= -depth, axis=1) for eq in hulls], axis=0)


def _sample(pieces, n=150, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.dirichlet(np.ones(len(v)), n) @ v for v in pieces] + list(pieces))


def _overlap(a, b, shift, depth=0.05):
    """Does piece set ``b`` moved by ``shift`` overlap piece set ``a``? Checked both ways."""
    b = [np.asarray(v) + shift for v in b]
    return _inside(_hulls(a), _sample(b), depth).any() or _inside(_hulls(b), _sample(a), depth).any()


def test_params_validate():
    with pytest.raises(ValueError):
        BBParams.from_dict({'U': 36., 'ramp': 41.})
    with pytest.raises(ValueError):
        BBParams(part='X')
    with pytest.raises(ValueError):
        BBParams(slope=float('nan'))
    with pytest.raises(ValueError):
        BBParams(rod_L=-1.)
    with pytest.raises(ValueError):
        BBParams(nest_w=30.)
    assert BBParams.from_dict({**DEFAULT.to_dict(), 'derived': {}}) == DEFAULT
    assert DEFAULT.profile().gutter == 0.  # the Y shear is biasbrick's own, one-sided


@pytest.mark.parametrize('part', ['A', 'B'])
def test_socket_is_one_voxel_of_the_brixzle_part(part):
    p = DEFAULT
    prof = G.socket_profile(p, part)
    whole = pr.brick_parts(p.profile(), part)['outer']
    assert prof['fragments'] == 1 and len(prof['slots']) == 1 and len(prof['teeth']) == 1
    assert prof['outer'].area == pytest.approx(whole.area/2, rel=0.05)
    lo, hi = prof['outer'].bounds[0], prof['outer'].bounds[2]
    assert (lo, hi) == (pytest.approx(-p.U/2 + p.end_gap/2), pytest.approx(p.U/2 - p.end_gap/2))


def test_courses_register_b_keel_on_a_valley():
    p = DEFAULT
    assert G.socket_profile(p, 'B')['keel'][0] == pytest.approx(G.seat_x(p, 'A'))
    assert G.socket_profile(p, 'A')['keel'][0] == pytest.approx(G.seat_x(p, 'B'))


@pytest.mark.parametrize('part', ['A', 'B'])
def test_pieces_are_convex_and_the_solid_is_one_watertight_piece(bricks, part):
    b = bricks[part]
    for v in b['collision']:
        hull = trimesh.convex.convex_hull(v)
        assert hull.volume > 0
        assert len(np.unique(np.round(v, 6), axis=0)) == len(np.unique(np.round(hull.vertices, 6), axis=0))
    assert b['fragments'] == 1 and b['mesh'].is_watertight


@pytest.mark.parametrize('lower,upper', [('A', 'B'), ('B', 'A')])
def test_tower_courses_nest_without_overlap(lower, upper):
    """Seated one course up, nothing overlaps: sockets nest, and the lip, rod and sleeves fit in a course
    (the rod once poked out of the end face's falling top into the keel above)."""
    p = DEFAULT
    assert not _overlap(G.brick_pieces(p, lower), G.brick_pieces(p, upper), [0., 0., p.H0])


def test_course_zero_nests_on_the_base():
    p = DEFAULT
    base = G.base_pieces(p, [p.socket_x(0), p.socket_x(1)], 'A')
    assert not _overlap(base, G.brick_pieces(p, 'A'), [0., 0., 0.])


def _tooth_sockets(p, part):
    return [q for i in (0, 1) for q in G.sweep(p, [G.socket_profile(p, part)['outer']], -p.D/2, p.D/2, p.socket_x(i))]


def test_seated_tooth_hooks_and_leaves_along_its_stroke():
    """Straight up, a seated B's teeth hit A's slots; backwards along the stroke (up and toward +x) they are free."""
    p = DEFAULT
    low, up = _tooth_sockets(p, 'A'), _tooth_sockets(p, 'B')
    assert _overlap(low, up, [0., 0., p.H0 + 3.])
    d = -np.array([math.cos(math.radians(90 - p.alpha)), 0., math.sin(math.radians(90 - p.alpha))])
    for t in (1., 3., 6.):
        assert not _overlap(low, up, np.array([0., 0., p.H0]) - t*d)


def test_brick_slides_down_the_y_slope_into_its_nest():
    p = DEFAULT
    low, up = G.brick_pieces(p, 'A'), G.brick_pieces(p, 'B')
    for dy in (5., 15., 30.):  # resting on the main slope (and the flange) at any landing offset
        assert not _overlap(low, up, [0., dy, p.H0 + p.shear_k*dy + 0.05])
    # It cannot slide on past the seat toward -y: the keel meets the nest face, and the hooked tooth
    # meets the slot channel's bend at the valley (a stop it cannot ride over, unlike a lip).
    for dz in (0.05, p.nest_k + 0.05):
        assert _overlap(low, up, [0., -1., p.H0 + dz])


def test_rise_is_an_asymmetric_sawtooth_in_y():
    p = DEFAULT
    assert G.rise(p, p.valley_y) == 0.
    assert G.rise(p, p.valley_y + 10.) == pytest.approx(10*p.shear_k)
    assert G.rise(p, -p.D/2) == pytest.approx(p.nest_w*p.nest_k)


def test_tines_pass_through_the_sleeves_and_hang_in_the_v_roof(bricks):
    p = DEFAULT
    b = bricks['A']
    solid = _hulls(b['collision'])
    y0, y1 = b['sleeve_y']
    ys = np.linspace(y0 - 2, y1 + 2, 60)
    r = p.fork_d/2
    for x, z in b['channels']:
        for dx, dz in ((0., 0.), (r, 0.), (-r, 0.), (0., -r)):
            assert not _inside(solid, np.column_stack([np.full_like(ys, x + dx), ys, np.full_like(ys, z + dz)]), 0.02).any()
        # The V roof sits right over the tine, and the whole square inside is open for a coarse tine.
        apex_gap = r/math.cos(math.radians(p.roof_angle)) - r
        assert _inside(solid, [x, (y0 + y1)/2, z + r + apex_gap + 0.4]).all()
        h = G.cavity_half(p)
        floor = z - 2*h + 2
        assert not _inside(solid, [[x - h + 0.2, (y0 + y1)/2, floor], [x + h - 0.2, (y0 + y1)/2, floor]], 0.02).any()


def test_sleeves_straddle_the_com_and_cross_the_rod(bricks):
    p = DEFAULT
    b = bricks['A']
    (x0, z0), (x1, z1) = b['channels']
    assert z0 == z1 and x0 < b['com'][0] < x1
    y0, y1 = b['sleeve_y']
    assert y0 < p.rod_y - p.rod_w/2 and p.rod_y + p.rod_w/2 < y1
