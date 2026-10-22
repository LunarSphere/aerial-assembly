"""3D structures from the A/B bricks: walls along X and Y joined by turn pieces.

Everything is built from *cells*: one voxel of the A or B part profile (``profile.brick_parts``)
swept through the depth with the Y gutter. A cell is placed with a yaw in quarter turns. The
straight 2.5D brick is two cells; a Y-running wall is the same brick rotated 90 degrees. Courses
alternate parts (A even, B odd) exactly as in the 2.5D lattice, so a cell nests on the cell
below when it has the *same* yaw and the other part: every cell column keeps one yaw.

A/B teeth trail toward local -x, so every brick seats with a stroke along local -x, the same
way round the ring on every course.

Parts (built in both A and B variants, named e.g. ``L3A``):
  S2  the straight 2.5D brick (cad.build_brick).
  L3  turn, even courses: corner + 1 cell along local +x + 1 cell along local +y. Teeth only
      on the corner and x-arm cells: a rigid brick seats along one stroke, so teeth leaning
      along x and along y would fight. The y-arm cell has yaw 270 (its profile runs along the
      wall it belongs to) and nests by keel and gutter only.
  I3  three straight cells along local +x that own the corner on odd courses, so the corner
      alternates between the two walls (masonry corner bond) and straights shift one cell per
      course (running bond).
  K2  closer: a straight whose upstream (+x) end is cut back by the stroke's sideways travel.
      Every stroke points back into its neighbour all the way round the closed ring, so on
      every course some brick must go in last next to an already placed upstream neighbour;
      masonry calls it the closer.

The ring has n x n cells (n even, >= 6). Even course: L3 at each corner, S2 fill. Odd course:
I3 owns each corner, S2 fill shifted by one. The last straight of the last wall is the K2.
"""
from dataclasses import dataclass
import math

import cadquery as cq
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import box
from shapely.ops import unary_union

from . import cad, lattice as L, profile as pr, trials as T
from .params import PLA_DENSITY, BrickParams
from .scene import MM, activate, body_pose, build_scene

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


# ---------------------------------------------------------------- cell geometry

def stroke_travel(p: BrickParams):
    """Sideways travel of the slanted final seating stroke."""
    return (p.tooth_L + p.slot_extra)*math.cos(math.radians(p.phi))


def cell_profile(p: BrickParams, part='A', teeth=True, trim_lo=False, trim_hi=False, lighten=True, extra_hi=0.):
    """One voxel of ``part`` (the right voxel of the brick, x in [0, U]), centred on x = 0: (outer polygon, holes)."""
    from shapely import affinity
    U = p.U
    top_m, bottom_m, top_rev, bottom_rev = pr.PART_FLAGS[part]
    x0 = p.end_gap/2 if trim_lo else 0.
    x1 = (U - p.end_gap/2 if trim_hi else U) - extra_hi
    body = pr._strip(p, x0, x1, top_mirrored=top_m, bottom_mirrored=bottom_m, bottom_offset=0.)
    feats = [body]
    if teeth:
        keel = U - p.valley_u if bottom_m else p.valley_u
        feats.append(pr.tooth(p, np.array([keel, 0.]), mirrored=bottom_m, reversed=bottom_rev))
    valley = U - p.valley_u if top_m else p.valley_u
    outer = unary_union(feats).difference(pr.slot(p, np.array([valley, p.H0]), mirrored=top_m, reversed=top_rev))
    outer = pr._largest(outer.buffer(-0.01, join_style=2).buffer(0.01, join_style=2))
    holes = []
    if lighten:
        parts = pr.brick_parts(p, part)
        voxel = box(0., outer.bounds[1] - 1, U, outer.bounds[3] + 1)
        for h in pr.lightening_holes(p, parts, pr.fork_channels(p)):
            # The brick's right voxel spans x in [0, U]: same profile as this cell.
            g = h.intersection(voxel).intersection(outer.buffer(-p.wall))
            for q in ([g] if g.geom_type == 'Polygon' else getattr(g, 'geoms', [])):
                if q.geom_type == 'Polygon' and q.area > 6.:
                    holes.append(q)
    shift = lambda g: affinity.translate(g, -U/2, 0.)
    return shift(outer), [shift(h) for h in holes]


def _yaw_matrix(q):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][q % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@dataclass(frozen=True)
class Cell:
    i: int          # local cell coordinates (corner cell at 0, 0)
    j: int
    yaw: int        # quarter turns of the cell profile
    teeth: bool = True
    lighten: bool = True
    extra_hi: float = 0.   # extra cut at the local +x (upstream) end (closer)


def _exposed(cell, cells):
    """Which local profile ends (lo = -x, hi = +x in the cell's own frame) face no sibling."""
    occupied = {(c.i, c.j) for c in cells}
    out = []
    for sign in (-1, 1):
        dx, dy = DIRS[cell.yaw % 4]
        nb = (cell.i + sign*dx, cell.j + sign*dy)
        out.append(nb not in occupied)
    return out


def build_part(p: BrickParams, cells, name, part='A'):
    """Bundle (collision pieces, mesh, mass, COM, inertia) for a multi-cell part made of ``part`` cells."""
    pieces, shape = [], None
    for c in cells:
        lo, hi = _exposed(c, cells)
        outer, holes = cell_profile(p, part, c.teeth, lo, hi, c.lighten and p.lighten, c.extra_hi)
        R = _yaw_matrix(c.yaw)
        t = np.array([c.i*p.U, c.j*p.U, 0.])
        for v in cad._pieces(outer, p):
            pieces.append(v @ R.T + t)
        solid = cad._swept(outer, holes, p)
        solid = solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 90*(c.yaw % 4)).translate(cq.Vector(*t))
        shape = solid if shape is None else shape.fuse(solid)
    shape = shape.clean()
    mesh = cad._mesh(shape)
    density = PLA_DENSITY*p.infill
    mesh.density = density
    return {'name': name, 'collision': pieces, 'mesh': mesh, 'shape': shape, 'mass_g': mesh.volume*density,
            'com': mesh.center_mass.tolist(), 'inertia': mesh.moment_inertia.tolist(),
            'cells': [(c.i, c.j) for c in cells]}


PART_CELLS = {
    'L3': [Cell(0, 0, 0), Cell(1, 0, 0), Cell(0, 1, 3, teeth=False)],
    'I3': [Cell(0, 0, 0), Cell(1, 0, 0), Cell(2, 0, 0)],
}
BODIES = ('L3A', 'S2A', 'K2A', 'I3B', 'S2B', 'K2B')   # the parts ``ring`` uses


def build_parts(p: BrickParams, bodies=BODIES):
    """{body name: bundle}, e.g. 'L3A' (turn of A cells), 'S2B' (the B straight)."""
    cells = dict(PART_CELLS)
    cells['K2'] = [Cell(0, 0, 0), Cell(1, 0, 0, extra_hi=stroke_travel(p) + 0.5)]
    parts = {}
    for body in bodies:
        kind, part = body[:-1], body[-1]
        if kind == 'S2':
            parts[body] = cad.build_brick(p, part=part)
            parts[body]['cells'] = [(-0.5, 0), (0.5, 0)]   # cell centres in units of U, frame at the voxel boundary
        else:
            parts[body] = build_part(p, cells[kind], body, part)
    return parts


def base_cell_pieces(p: BrickParams, thickness=None):
    """Anchored base for one cell (top matches an A cell's bottom: course 0 is A), centred on x = 0."""
    thickness = thickness or p.tooth_L + p.slot_extra + 4
    outer, _ = pr.base_outer(p, 1, thickness)
    from shapely import affinity
    return cad._pieces(affinity.translate(outer, -p.U/2, 0.), p)


# ---------------------------------------------------------------- placements

@dataclass(frozen=True)
class Brick:
    kind: str       # 'S2', 'L3', 'I3', 'K2'
    course: int
    x: float        # frame position in units of U (cell centre grid; S2 sits on a cell boundary)
    y: float
    yaw: int

    @property
    def part(self):
        return 'AB'[self.course % 2]

    @property
    def body(self):
        return self.kind + self.part

    @property
    def name(self):
        return f'{self.kind}_{self.course}_{self.x:g}_{self.y:g}'


def cells_of(b: Brick, local_cells):
    R = _yaw_matrix(b.yaw)[:2, :2]
    out = []
    for c in local_cells:
        w = R @ np.asarray(c, dtype=float) + np.array([b.x, b.y])
        out.append((int(round(w[0])), int(round(w[1]))))
    return out


LOCAL_CELLS = {'L3': [(0, 0), (1, 0), (0, 1)], 'I3': [(0, 0), (1, 0), (2, 0)], 'S2': [(-0.5, 0), (0.5, 0)],
               'K2': [(0, 0), (1, 0)]}


def pose(p: BrickParams, b: Brick):
    pos = np.array([(b.x + .5)*p.U, (b.y + .5)*p.U, b.course*p.H0])
    half = math.pi/4*(b.yaw % 4)
    return pos, np.array([math.cos(half), 0., 0., math.sin(half)])


def stroke(b: Brick):
    """World grid direction the brick moves along while seating (local -x: A/B teeth trail -x)."""
    dx, dy = DIRS[b.yaw % 4]
    return (-dx, -dy)


def ring(n=6, courses=6):
    """Square ring of n x n cells (n even >= 6), as a list of bricks."""
    if n % 2 or n < 6:
        raise ValueError('ring needs an even side of at least 6 cells')
    corners = [(0, 0), (n - 1, 0), (n - 1, n - 1), (0, n - 1)]
    bricks = []
    for k in range(courses):
        odd = k % 2
        for r, c in enumerate(corners):
            e = np.array(DIRS[r])
            bricks.append(Brick('I3' if odd else 'L3', k, c[0], c[1], r))
            ts = range(3, n - 1, 2) if odd else range(2, n - 2, 2)
            for t in ts:
                if r == 3 and t == ts[-1]:
                    # Closer: K2's frame sits on its first cell; its +x (upstream) end is cut back.
                    first = np.array(c) + t*e
                    bricks.append(Brick('K2', k, float(first[0]), float(first[1]), r))
                    continue
                mid = np.array(c) + (t + 0.5)*e
                bricks.append(Brick('S2', k, float(mid[0]), float(mid[1]), r))
    return bricks


def course0_yaws(bricks):
    """Cell -> yaw of the course-0 cell above it (the base must match it)."""
    yaws = {}
    for b in bricks:
        if b.course != 0:
            continue
        local = LOCAL_CELLS[b.kind]
        for lc, wc in zip(local, cells_of(b, local)):
            yaw = b.yaw
            if b.kind == 'L3' and tuple(lc) == (0, 1):
                yaw = (b.yaw + 3) % 4
            yaws[wc] = yaw
    return yaws


def check_columns(bricks):
    """Every cell is used once per course and every column keeps its yaw (parts alternate by course)."""
    grid = {}
    for b in bricks:
        local = LOCAL_CELLS[b.kind]
        for lc, wc in zip(local, cells_of(b, local)):
            yaw = (b.yaw + 3) % 4 if b.kind == 'L3' and tuple(lc) == (0, 1) else b.yaw % 4
            key = (b.course, wc)
            if key in grid:
                raise ValueError(f'cell {key} used twice')
            grid[key] = yaw
    for (k, c), yaw in grid.items():
        below = grid.get((k - 1, c))
        if k > 0 and (below is None or yaw != below):
            raise ValueError(f'cell {c} course {k}: yaw {yaw} over {below}')
    return grid


def _blocked(b, placed_cells):
    if b.kind == 'K2':
        return False   # its upstream end is cut back by the stroke travel
    sx, sy = stroke(b)
    mine = set(cells_of(b, LOCAL_CELLS[b.kind]))
    return any((c[0] - sx, c[1] - sy) in placed_cells and (c[0] - sx, c[1] - sy) not in mine for c in mine)


def assembly_order(bricks):
    """Course by course; a brick waits while a placed sibling occupies the cell
    upstream of any of its cells along its seating stroke (vertical end faces
    would collide during the slanted final stroke). Depth-first per course."""
    order = []
    for k in sorted({b.course for b in bricks}):
        todo = [b for b in bricks if b.course == k]

        def dfs(remaining, placed, seq):
            if not remaining:
                return seq
            for b in sorted(remaining, key=lambda x: x.kind == 'K2'):
                if _blocked(b, placed):
                    continue
                out = dfs([x for x in remaining if x is not b],
                          placed | set(cells_of(b, LOCAL_CELLS[b.kind])), seq + [b])
                if out is not None:
                    return out
            return None
        seq = dfs(todo, frozenset(), [])
        if seq is None:
            raise ValueError(f'no insertion order for course {k}')
        order += seq
    return order


# ---------------------------------------------------------------- simulation

def _rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _release(p, b, sample, seat=None):
    pos, quat = seat if seat is not None else pose(p, b)
    R = _rot(quat)
    lift = p.amplitude + p.tooth_L + p.slot_extra + sample['clear']
    offset = R.apply([sample['dx'], sample['dy'], 0.])   # aim bias upstream of the stroke (local +x)
    err = Rotation.from_euler('xyz', [sample['roll'], sample['pitch'], sample['yaw']], degrees=True)
    q = (err*R).as_quat()
    return pos + offset + np.array([0, 0, lift]), np.array([q[3], q[0], q[1], q[2]])


def _expected(p, data, info, b, ref):
    """Seat of b carried with supporter ref=(index, brick)'s measured pose."""
    tpos, tquat = pose(p, b)
    if ref is None:
        return tpos, tquat
    k, rb = ref
    rpos, rquat = body_pose(data, info, k)
    ipos, iquat = pose(p, rb)
    delta = _rot(rquat)*_rot(iquat).inv()
    q = (delta*_rot(tquat)).as_quat()
    return rpos + delta.apply(tpos - ipos), np.array([q[3], q[0], q[1], q[2]])


def assemble(p: BrickParams, parts, bricks, cfg: T.TrialConfig, seed=0, stop_on_fail=False, record=None):
    order = assembly_order(bricks)
    yaws = course0_yaws(bricks)
    piece = base_cell_pieces(p)
    static = []
    for (ci, cj), yaw in yaws.items():
        R = _yaw_matrix(yaw)
        t = np.array([(ci + .5)*p.U, (cj + .5)*p.U, 0.])
        static += [v @ R.T + t for v in piece]
    model, xml, info = build_scene(p, parts, [], len(order), cfg.physics, visual=record is not None,
                                   body_parts=[b.body for b in order], static=static)
    data = mujoco.MjData(model)
    owner = T._brick_geoms(model, info)
    rng = np.random.default_rng(seed)
    where = {}
    stages, status = [], 'complete'
    for j, b in enumerate(order):
        before = [body_pose(data, info, i) for i in range(j)]
        cells = cells_of(b, LOCAL_CELLS[b.kind])
        sup = [where[(b.course - 1, c)] for c in cells if (b.course - 1, c) in where]
        ref = (max(set(sup), key=sup.count), order[max(set(sup), key=sup.count)]) if sup else None
        sample = cfg.error.sample(rng)
        seat = _expected(p, data, info, b, ref) if cfg.error.track_supporter else None
        pos, quat = _release(p, b, sample, seat)
        activate(model, data, info, j, pos, quat, linvel=sample['vel'])
        m = T.run_until_settled(model, data, info, j, cfg.tolerance, owner, watch=range(j), frames=record)
        epos, equat = _expected(p, data, info, b, ref)
        apos, aquat = body_pose(data, info, j)
        err = _rot(equat).inv().apply(apos - epos)        # in the brick frame
        angle = math.degrees((_rot(equat).inv()*_rot(aquat)).magnitude())
        outcome = T.classify(p, err, angle, cfg.tolerance, m['floor'])
        moved = 0.
        for i in range(j):
            pi, qi = body_pose(data, info, i)
            turn = math.degrees((_rot(before[i][1]).inv()*_rot(qi)).magnitude())
            moved = max(moved, float(np.linalg.norm(pi - before[i][0])), turn*p.U*math.pi/180)
        collapsed = moved > cfg.tolerance.drift or m['floor']
        stages.append({'brick': b.name, 'kind': b.body, **m, 'outcome': outcome, 'err_mm': err.tolist(),
                       'angle_deg': angle, 'stage_motion_mm': moved, 'collapsed': bool(collapsed)})
        for c in cells:
            where[(b.course, c)] = j
        if collapsed:
            status = 'collapse'
            break
        if outcome != 'seated' and stop_on_fail:
            status = f'placement_{outcome}'
            break
    seated = sum(s['outcome'] == 'seated' for s in stages)
    return {'structure': f'ring', 'status': status, 'placed': len(stages), 'total': len(order),
            'seated': seated, 'stages': stages}, (model, data, info, xml)
