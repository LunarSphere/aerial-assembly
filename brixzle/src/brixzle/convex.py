"""Convex decomposition of simple polygons: ear clipping + Hertel-Mehlhorn."""
import numpy as np


def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def _ccw(points):
    pts = np.asarray(points, dtype=float)
    area = 0.5*np.sum(pts[:, 0]*np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1)*pts[:, 1])
    return pts if area > 0 else pts[::-1]


def _clean(pts, eps=1e-9):
    out = []
    for p in pts:
        if not out or np.linalg.norm(p - out[-1]) > 1e-7:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= 1e-7:
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i-1], out[i], out[(i+1) % len(out)]
            if abs(_cross(a, b, c)) < eps:
                out.pop(i)
                changed = True
                break
    return np.array(out)


def _in_triangle(p, a, b, c):
    return _cross(a, b, p) >= -1e-12 and _cross(b, c, p) >= -1e-12 and _cross(c, a, p) >= -1e-12


def triangulate(points):
    pts = _clean(_ccw(points))
    idx = list(range(len(pts)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k-1], idx[k], idx[(k+1) % n]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if _cross(a, b, c) <= 1e-12:
                continue
            if any(_in_triangle(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)
                   and not (np.allclose(pts[j], a) or np.allclose(pts[j], b) or np.allclose(pts[j], c))):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise ValueError('Ear clipping failed; polygon is not simple')
    tris.append(tuple(idx))
    return pts, tris


def _is_convex(poly, pts):
    n = len(poly)
    return all(_cross(pts[poly[i-1]], pts[poly[i]], pts[poly[(i+1) % n]]) >= -1e-9 for i in range(n))


def convex_parts(points):
    """Hertel-Mehlhorn: merge ear-clipped triangles across diagonals while convex."""
    pts, tris = triangulate(points)
    polys = [list(t) for t in tris]
    merged = True
    while merged:
        merged = False
        for i in range(len(polys)):
            for j in range(i+1, len(polys)):
                a, b = polys[i], polys[j]
                shared = [v for v in a if v in b]
                if len(shared) != 2:
                    continue
                u, v = shared
                ia, ib = a.index(u), a.index(v)
                # Orient so the shared edge is a[k] -> a[k+1].
                if (ia + 1) % len(a) == ib:
                    k, e0, e1 = ia, u, v
                elif (ib + 1) % len(a) == ia:
                    k, e0, e1 = ib, v, u
                else:
                    continue
                jb = b.index(e1)
                if b[(jb + 1) % len(b)] != e0:
                    continue
                rest_b = [b[(jb + 1 + t) % len(b)] for t in range(1, len(b) - 1)]
                candidate = a[:k+1] + rest_b + a[k+1:]
                if _is_convex(candidate, pts):
                    polys[i] = candidate
                    polys.pop(j)
                    merged = True
                    break
            if merged:
                break
    return [pts[p] for p in polys]
