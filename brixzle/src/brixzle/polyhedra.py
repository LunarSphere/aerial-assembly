"""Convex polyhedral CSG for collision geometry without closing cavities.

All generated surfaces are planar. Subtracting a convex cutter partitions each
convex source into convex pieces; no voxelization or external hull decomposition
is used. Coordinates remain in millimeters until MJCF export.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .models import ConvexPiece


def hull(vertices: np.ndarray) -> ConvexHull | None:
    vertices = np.unique(np.round(vertices, 9), axis=0)
    if len(vertices) < 4:
        return None
    try:
        result = ConvexHull(vertices)
    except QhullError:
        return None
    return result if result.volume > 1e-7 else None


def box(center, size) -> np.ndarray:
    c, s = np.asarray(center), np.asarray(size) / 2
    return np.array([c + s * np.array([x, y, z]) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def prism(points_xy, bottom, top, upper_xy=None) -> np.ndarray:
    upper_xy = points_xy if upper_xy is None else upper_xy
    return np.array([[x, y, bottom] for x, y in points_xy] + [[x, y, top] for x, y in upper_xy], dtype=float)


def clip(vertices: np.ndarray, plane: np.ndarray, inside: bool = True) -> np.ndarray | None:
    """Clip a hull by a plane; including all vertex chords is safe and simple."""
    distances = vertices @ plane[:3] + plane[3]
    if not inside:
        distances = -distances
    kept = list(vertices[distances <= 1e-8])
    for i, j in combinations(range(len(vertices)), 2):
        if (distances[i] < -1e-8 and distances[j] > 1e-8) or (distances[j] < -1e-8 and distances[i] > 1e-8):
            t = distances[i] / (distances[i] - distances[j])
            kept.append(vertices[i] + t * (vertices[j] - vertices[i]))
    if not kept:
        return None
    out = hull(np.asarray(kept))
    return None if out is None else out.points[out.vertices]


def subtract(vertices: np.ndarray, cutter: np.ndarray) -> list[np.ndarray]:
    ch = hull(cutter)
    if ch is None:
        return [vertices]
    if np.any(vertices.max(axis=0) < cutter.min(axis=0) - 1e-8) or np.any(cutter.max(axis=0) < vertices.min(axis=0) - 1e-8):
        return [vertices]
    # If a separating cutter plane exists, the shapes do not intersect.
    if any(np.all(vertices @ e[:3] + e[3] >= -1e-8) for e in ch.equations):
        return [vertices]
    result = []
    remainder = vertices
    for equation in np.unique(np.round(ch.equations, 10), axis=0):
        outside = clip(remainder, equation, inside=False)
        if outside is not None:
            result.append(outside)
        remainder = clip(remainder, equation)
        if remainder is None:
            break
    return result


def csg(sources: list[np.ndarray], cutters: list[np.ndarray]) -> list[ConvexPiece]:
    # Remove overlap between source solids to avoid duplicate contact surfaces.
    disjoint: list[np.ndarray] = []
    for source in sources:
        fragments = [source]
        for existing in disjoint:
            fragments = [piece for fragment in fragments for piece in subtract(fragment, existing)]
        disjoint.extend(fragments)
    for cutter in cutters:
        disjoint = [piece for fragment in disjoint for piece in subtract(fragment, cutter)]
    output = []
    for vertices in disjoint:
        h = hull(vertices)
        if h is not None:
            # Faces point outward, which is also required for CAD shells.
            faces = []
            for indices, eq in zip(h.simplices, h.equations):
                a, b, c = h.points[indices]
                if np.dot(np.cross(b - a, c - a), eq[:3]) < 0:
                    indices = indices[::-1]
                faces.append(indices.tolist())
            output.append(ConvexPiece(h.points.tolist(), faces))
    return output
