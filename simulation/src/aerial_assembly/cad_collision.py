"""Convex partition of an extruded body with two blind circular sockets.

Extrude the CAD side-face triangles, then subtract socket frusta by successive
half-space cuts. All boundaries come from the exported triangles/rings. This
retains guide lips and socket facets without a stochastic decomposition pass.
"""
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull, QhullError
import trimesh


def partition_solid(mesh):
    """Constrained tetrahedra, merged only when their union is convex.

    Preserve the input surface and its cavities; do not remesh or convexify the
    complete part. The original mesh still supplies visuals and mass properties.
    """
    import tetgen
    # Remove STL float noise below the existing 0.5 um approximation budget.
    # Each coordinate moves <= 0.05 um; visuals and mass stay on the source mesh.
    surface = trimesh.Trimesh(vertices=np.round(mesh.vertices, 7), faces=mesh.faces, process=True)
    surface.update_faces(surface.nondegenerate_faces())
    surface.remove_unreferenced_vertices()
    if not surface.is_volume:
        raise ValueError('Collision vertex cleanup damaged the single-solid surface')
    nodes, elements = tetgen.TetGen(surface.vertices, surface.faces).tetrahedralize(switches='pQ')[:2]
    cells = nodes[elements[:, :4]]
    volumes = abs(np.linalg.det(cells[:, 1:]-cells[:, :1]))/6
    if not np.isclose(volumes.sum(), mesh.volume, rtol=2e-5, atol=1e-12):
        raise ValueError('Single-solid tetrahedralization changed source volume')
    groups = {i: set(map(int, cell)) for i, cell in enumerate(elements[:, :4])}
    owners = np.arange(len(elements))
    members = {i: {i} for i in groups}
    neighbors = {i: set() for i in groups}
    faces = {}
    edge_stars = {}
    for i, cell in enumerate(elements[:, :4]):
        for edge in combinations(sorted(cell), 2):
            edge_stars.setdefault(edge, []).append(i)
        for skip in range(4):
            face = tuple(sorted(np.delete(cell, skip)))
            other = faces.pop(face, None)
            if other is None:
                faces[face] = i
            else:
                neighbors[i].add(other)
                neighbors[other].add(i)
    def merge(candidates):
        candidates = set(candidates)
        if len(candidates) < 2:
            return False
        ids = set().union(*(groups[i] for i in candidates))
        if len(ids) > 128:
            return False
        try:
            convex = ConvexHull(nodes[sorted(ids)])
        except QhullError:
            return False
        volume = sum(volumes[i] for i in candidates)
        if not np.isclose(convex.volume, volume, rtol=1e-8, atol=1e-18):
            # Fill a candidate hull only with existing material cells. This can
            # close a nonconvex intermediate union without filling a CAD void.
            checked = set(candidates)
            pending = set().union(*(neighbors[i] for i in candidates))-checked
            while pending:
                k = pending.pop()
                checked.add(k)
                points = nodes[sorted(groups[k])]
                if np.all(points @ convex.equations[:, :3].T + convex.equations[:, 3] <= 1e-10):
                    candidates.add(k)
                    ids.update(groups[k])
                    volume += volumes[k]
                    pending.update(neighbors[k]-checked)
        if not np.isclose(convex.volume, volume, rtol=1e-8, atol=1e-18):
            return False
        keep = min(candidates)
        adjacent = set().union(*(neighbors[i] for i in candidates))-candidates
        combined = set().union(*(members[i] for i in candidates))
        for k in adjacent:
            neighbors[k].difference_update(candidates)
            neighbors[k].add(keep)
        for i in candidates-{keep}:
            del groups[i], neighbors[i], members[i]
        groups[keep], neighbors[keep], members[keep] = ids, adjacent, combined
        volumes[keep] = volume
        owners[list(combined)] = keep
        return True

    # An edge-star can form a convex cell even when no pairwise intermediate
    # union is convex. Merge those first, then adjacent pairs, until stable.
    changed = True
    while changed:
        changed = False
        for star in edge_stars.values():
            changed |= merge(owners[star])
        for i in sorted(groups):
            if i not in groups:
                continue
            for j in sorted(neighbors[i]):
                if j not in groups:
                    continue
                if merge([i, j]):
                    changed = True
                    if i not in groups:
                        break
    pieces = [trimesh.convex.convex_hull(nodes[sorted(ids)]) for ids in groups.values()]
    if not np.isclose(sum(p.volume for p in pieces), mesh.volume, rtol=2e-5, atol=1e-12):
        raise ValueError('Single-solid convex merging changed source volume')
    return pieces


def hull(vertices):
    vertices = np.unique(np.round(vertices, 12), axis=0)
    if len(vertices) < 4:
        return None
    try:
        result = trimesh.convex.convex_hull(vertices)
    except QhullError:
        return None
    if result.volume <= 1e-15:
        return None
    return result


def clip(mesh, normal, distance):
    """Keep n.x <= d, preserving existing vertices and crossing-edge points."""
    signed = mesh.vertices @ normal-distance
    if signed.max() <= 1e-8:
        return mesh
    if signed.min() >= -1e-8:
        return None
    edges = mesh.edges_unique
    a, b = edges[:,0], edges[:,1]
    crossing = (signed[a] < 0) != (signed[b] < 0)
    a, b = a[crossing], b[crossing]
    t = signed[a]/(signed[a]-signed[b])
    intersections = mesh.vertices[a] + t[:,None]*(mesh.vertices[b]-mesh.vertices[a])
    return hull(np.vstack([mesh.vertices[signed <= 0], intersections]))


def subtract(pieces, void):
    planes = np.unique(np.round(ConvexHull(void.vertices).equations, 10), axis=0)
    result = []
    for piece in pieces:
        if np.any(piece.bounds[1] <= void.bounds[0]+1e-8) or np.any(piece.bounds[0] >= void.bounds[1]-1e-8):
            result.append(piece)
            continue
        remaining = piece
        for plane in planes:
            if remaining is None:
                break
            n, d = plane[:3], -plane[3]
            outside = clip(remaining, -n, -d)
            if outside is not None:
                result.append(outside)
            remaining = clip(remaining, n, d)
        # The remainder is inside every void half-space: socket empty space.
    return result


def partition_body(body, sockets):
    lower, upper = body.bounds[:,1]
    side = body.triangles[np.all(abs(body.triangles[:,:,1]-upper) < 1e-7, axis=1)]
    if not len(side):
        raise ValueError('Extruded block collision profile requires planar side faces parallel to XZ')
    pieces = []
    for triangle in side:
        opposite = triangle.copy()
        opposite[:,1] = lower
        piece = hull(np.vstack([triangle, opposite]))
        if piece is not None:
            pieces.append(piece)
    # Isolate bore and lead-in height bands before radial cuts, so a bore's
    # radial partition does not propagate into the wider conical region.
    levels = sorted({s['mouth'][2]-d for s in sockets for d in (s['depth'],s['lead_depth'],0)})
    for z in levels:
        split = []
        for piece in pieces:
            if piece.bounds[0,2]+1e-8 < z < piece.bounds[1,2]-1e-8:
                split.extend(p for p in (clip(piece, np.array([0,0,1]), z),
                                          clip(piece, np.array([0,0,-1]), -z)) if p is not None)
            else:
                split.append(piece)
        pieces = split
    for socket in sockets:
        center = np.asarray(socket['mouth'])[:2]
        mouth = socket['mouth'][2]
        floor, throat = mouth-socket['depth'], mouth-socket['lead_depth']
        radial = np.linalg.norm(body.vertices[:,:2]-center, axis=1)
        rings = []
        ring_grid = None
        for z,r in [(floor,socket['throat_radius']), (throat,socket['throat_radius']), (mouth,socket['opening_radius'])]:
            ring = body.vertices[(abs(body.vertices[:,2]-z) < 1e-6) & (abs(radial-r) < 1e-6)]
            if len(ring) < 16:
                raise ValueError('Socket ring missing in source mesh')
            # Restore the regular CAD ring from float STL coordinates. Adjacent
            # frustum triangles then share a plane instead of nanometer slivers.
            n = len(ring)
            angles = np.arctan2(ring[:,1]-center[1], ring[:,0]-center[0])
            phase = np.angle(np.mean(np.exp(1j*angles*n)))/n
            if ring_grid is None:
                ring_grid = (n, phase)
            elif n != ring_grid[0]:
                raise ValueError('Socket rings need matching facet counts')
            phase = ring_grid[1]
            a = phase+np.arange(n)*2*np.pi/n
            regular = np.column_stack([center[0]+r*np.cos(a), center[1]+r*np.sin(a), np.full(n,z)])
            if np.max(np.min(np.linalg.norm(ring[:,None,:]-regular[None,:,:],axis=2),axis=1)) > 2e-7:
                raise ValueError('Socket ring is not a regular circular CAD tessellation')
            rings.append(regular)
        for a,b in zip(rings[:-1], rings[1:]):
            void = hull(np.vstack([a,b]))
            if void is None:
                raise ValueError('Invalid socket void')
            pieces = subtract(pieces, void)
    # This detects unsupported non-extruded features and unrecognized cavities.
    total = sum(p.volume for p in pieces)
    if not np.isclose(total, body.volume, rtol=2e-5, atol=1e-12):
        raise ValueError(f'CAD partition volume mismatch: {total:g} versus {body.volume:g}; unsupported body topology')
    # STL float noise creates near-zero-area faces at tangencies. Snap only the
    # collision vertices to 0.1 micrometer before rebuilding their convex hulls.
    # This is far below the 25 micrometer contact budget; visuals stay untouched.
    cleaned = []
    for piece in pieces:
        vertices = np.unique(np.round(piece.vertices, 7), axis=0)
        if len(vertices) < 4 or np.linalg.matrix_rank(vertices-vertices[0], tol=1e-8) < 3:
            continue
        convex = ConvexHull(vertices)
        faces = convex.simplices.copy()
        triangles = vertices[faces]
        normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        reverse = np.einsum('ij,ij->i', normals, convex.equations[:,:3]) < 0
        faces[reverse] = faces[reverse, ::-1]
        clean = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        clean.remove_unreferenced_vertices()
        clean.fix_normals()
        if clean.volume > 1e-15:
            cleaned.append(clean)
    if not np.isclose(sum(p.volume for p in cleaned), body.volume, rtol=5e-5, atol=1e-12):
        raise ValueError('Collision vertex cleanup changed body volume beyond the allowed tolerance')
    return cleaned
