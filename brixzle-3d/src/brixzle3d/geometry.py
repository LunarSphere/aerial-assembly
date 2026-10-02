"""Convex-by-construction brick geometry from half-space intersections.

Each cell is split into four quadrants by its diagonals. In a quadrant the
keel and funnel faces are single planes, so the body is a sheared triangular
prism (convex). The socket is a tapered square hole: within a quadrant the
material outside it is one more half-space, and the slab under the socket
floor is a second convex piece. The peg is a chamfered square frustum. The
same half-space pieces feed MuJoCo collision (exactly) and the printable mesh.

With ``peg_lean`` > 0 every peg is sheared toward body -x (a hook: it can only
leave along its own axis), and every socket becomes a cross of four slanted
lobes so a peg leaning any of the four lattice directions finds its lobe.
Per quadrant the material outside the cross is still a handful of convex
pieces: beyond the own lobe (A), beside it (E, F), between centre and lobe
(C), the centre core once the lobes have separated (D), and under the floor (B).

Brick frame: cell (i, j) spans [iU, (i+1)U] x [jU, (j+1)U]; z = 0 at the keel
tips, the funnel bottoms sit at z = H.
"""
import math

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from .params import BrickParams

DIRS = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


def vertices(halfspaces):
    """Vertices of {x : A x + b <= 0} for rows [A | b] (scipy convention)."""
    H = np.asarray(halfspaces, dtype=float)
    A, b = H[:, :3], H[:, 3]
    norm = np.linalg.norm(A, axis=1)
    # Chebyshev centre: deepest interior point.
    res = linprog(np.r_[0, 0, 0, -1], A_ub=np.column_stack([A, norm]), b_ub=-b,
                  bounds=[(None, None)]*3 + [(0, None)], method='highs')
    if res.status != 0 or res.x[3] < 1e-6:
        return None
    hs = HalfspaceIntersection(H, res.x[:3])
    pts = hs.intersections
    return pts[ConvexHull(pts).vertices]


def _rot(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    return np.array([[c, -s], [s, c]], dtype=float)


def _plane(au, av, az, b, center, k):
    """Half-space au*u + av*v + az*z + b <= 0 in quadrant-local (u, v) -> world row."""
    R = _rot(k)
    # (u, v) = R^T (xy - c)  =>  n_xy = R @ (au, av)
    n = R @ np.array([au, av])
    return [n[0], n[1], az, b - n @ np.asarray(center)]


def cell_pieces(p: BrickParams, cell, occupied, top=True, bottom=True, base_depth=None):
    """Half-space sets for one cell.

    ``occupied`` is the brick's cell set (internal sides get no gap).
    ``base_depth``: flat bottom at z = -base_depth instead of a keel (anchored base).
    """
    a, h, H = p.a, p.h, p.H
    i, j = cell
    center = ((i + .5)*p.U, (j + .5)*p.U)
    slope = h/a
    ext = {d: (a if (i + d[0], j + d[1]) in occupied else a - p.gap/2) for d in DIRS}
    # Socket: half-width s(z) = s_top - tau*(H - z) at depth; floor at z_f.
    tau = (p.peg_taper/2)/p.peg_L
    s_top = p.peg_w/2 + p.clearance
    z_f = H - p.peg_L - p.socket_extra
    pieces = []
    for d, k in DIRS.items():
        left = ext[tuple(_rot(1) @ np.array(d))]   # +v side
        right = ext[tuple(_rot(3) @ np.array(d))]  # -v side
        quad = [
            _plane(-1, 1, 0, 0, center, k),        # v <= u
            _plane(-1, -1, 0, 0, center, k),       # -v <= u
            _plane(1, 0, 0, -ext[d], center, k),   # u <= ext
            _plane(0, 1, 0, -left, center, k),     # v <= left
            _plane(0, -1, 0, -right, center, k),   # -v <= right
        ]
        if base_depth is None:
            quad.append(_plane(slope, 0, -1, 0, center, k))          # z >= slope*u (keel)
        else:
            quad.append([0, 0, -1, -base_depth])                     # z >= -base_depth
        quad.append(_plane(-slope, 0, 1, -H, center, k))             # z <= H + slope*u (funnel)
        if not top:
            pieces.append(quad)
            continue
        lam = math.tan(math.radians(abs(p.peg_lean)))  # lobes are symmetric; the sign is the peg's
        # Lobe along +u: |u - lam*t| < s(z), |v| < s(z), t = H - z, s = s_top - tau*t.
        # A: beyond the lobe's far wall, u >= s_top + (lam - tau)*t.
        far = _plane(-1, 0, tau - lam, s_top + (lam - tau)*H, center, k)
        # Mouth chamfer from (s_top, H - m) on the socket wall to the funnel face
        # at u = s_top + m. Steeper than the funnel, so it only removes the rim.
        m = p.mouth_chamfer
        du, dz = m, m + slope*(s_top + m)
        mouth = _plane(-1, 0, du/dz, s_top - (H - m)*du/dz, center, k)
        floor = [0, 0, -1, z_f]                                      # z >= z_f
        pieces.append(quad + [far, mouth])
        pieces.append(quad + [[0, 0, 1, -z_f]])                      # B: slab under the floor
        if lam > 0:
            out_wall = _plane(-1, 0, tau, s_top - tau*H, center, k)  # u >= s(z)
            near = _plane(1, 0, lam + tau, s_top - (lam + tau)*H, center, k)   # u <= lam*t - s
            pieces.append(quad + [out_wall, near, floor])            # C
            core = [_plane(1, 0, -tau, tau*H - s_top, center, k),    # u <= s(z)
                    near,
                    _plane(-1, 0, lam + tau, s_top - (lam + tau)*H, center, k),  # u >= s - lam*t
                    _plane(0, 1, lam + tau, s_top - (lam + tau)*H, center, k),   # v <= lam*t - s
                    _plane(0, -1, lam + tau, s_top - (lam + tau)*H, center, k),  # -v <= lam*t - s
                    floor]
            pieces.append(quad + core)                               # D
            for sv in (1, -1):                                       # E, F: |v| >= s(z)
                pieces.append(quad + [_plane(0, -sv, tau, s_top - tau*H, center, k), mouth])
    return pieces


def peg_vertices(p: BrickParams, cell):
    cx, cy = (cell[0] + .5)*p.U, (cell[1] + .5)*p.U
    w_root, w_tip = p.peg_w/2, (p.peg_w - p.peg_taper)/2
    ch = min(p.tip_chamfer, 0.45*w_tip*2, 0.45*p.peg_L)
    levels = [(1.0, w_root), (-p.peg_L + ch, w_tip + (w_root - w_tip)*ch/p.peg_L), (-p.peg_L, w_tip - ch)]
    lam = math.tan(math.radians(p.peg_lean))
    # Shear with depth: positive lean puts the tip toward body -x (hook).
    pts = [(cx + sx*w + lam*z, cy + sy*w, z) for z, w in levels for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(pts)


def brick_pieces(p: BrickParams):
    """Convex vertex arrays (mm) for an assembled brick in its own frame."""
    occupied = set(p.cells)
    out = []
    for cell in p.cells:
        for hs in cell_pieces(p, cell, occupied):
            v = vertices(hs)
            if v is not None and len(v) >= 4:
                out.append(v)
        out.append(peg_vertices(p, cell))
    return out


def base_pieces(p: BrickParams, cells, depth=None):
    """Anchored plate: funnel + socket on every cell, flat bottom."""
    depth = depth if depth is not None else p.peg_L + p.socket_extra + 4
    occupied = set(cells)
    out = []
    for cell in cells:
        for hs in cell_pieces(p, cell, occupied, base_depth=depth):
            v = vertices(hs)
            if v is not None and len(v) >= 4:
                out.append(v)
    return out


def rotate_cells(cells, r):
    """Rotate lattice cells by r*90 degrees about the brick origin corner."""
    out = []
    for i, j in cells:
        for _ in range(r % 4):
            i, j = -j - 1, i
        out.append((i, j))
    return out
