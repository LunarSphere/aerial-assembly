"""Concept genes (capture lip, snap barb, scarfed ends): seated neighbours never overlap."""
from dataclasses import replace

import pytest
from shapely import affinity

from brixzle import cad, optimize as O, profile as pr
from brixzle.params import V0

AB = replace(V0, lean_mode='ab', clearance=0.05)
CONCEPTS = {
    'lip': dict(lip_L=5., lip_w=4.),
    'barb': dict(barb_h=0.4),
    'scarf+': dict(end_scarf=0.3, end_step=1.5),
    'scarf-': dict(end_scarf=0.3, end_step=-1.5),
    'all': dict(lip_L=5., barb_h=0.4, end_scarf=0.3, end_step=1.5),
}


@pytest.mark.parametrize('name', CONCEPTS)
def test_seated_neighbours_do_not_overlap(name):
    p = replace(AB, **CONCEPTS[name])
    parts = {k: pr.brick_parts(p, k) for k in 'AB'}
    assert all(v['fragments'] == 1 for v in parts.values())

    def at(cx, course):
        return affinity.translate(parts['A' if course % 2 == 0 else 'B']['outer'], cx, course*p.H0)
    U = p.U
    pairs = [(at(U, 0), at(2*U, 1)),   # overhang step / running bond
             (at(U, 0), at(3*U, 0)),   # course neighbours
             (at(2*U, 1), at(4*U, 1)),
             (at(U, 0), at(U, 1))]     # stack bond
    for a, b in pairs:
        assert a.intersection(b).area < 1e-6


def test_defaults_leave_the_brick_unchanged():
    assert not pr.brick_parts(AB)['barbs']
    assert O.course_insertable(AB) == 1.


def test_barb_is_soft_and_printed():
    p = replace(AB, barb_h=0.4)
    b = cad.build_brick(p)
    assert b['soft_collision'] and b['mesh'].is_watertight
    assert b['mass_g'] > cad.build_brick(AB)['mass_g']


def test_scarf_parallel_to_insertion_axis_is_not_insertable():
    assert O.course_insertable(replace(AB, end_scarf=1.0, end_step=0.)) == -1.
    assert O.course_insertable(replace(AB, end_scarf=0.3, end_step=1.5)) == 1.
