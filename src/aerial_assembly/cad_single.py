"""Measure the two vertical peg/socket profiles in a single watertight GOAT solid."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def circular_rings(mesh):
    """Find closed horizontal feature edges; distinguish inner and outer surfaces."""
    edges = mesh.face_adjacency_edges
    horizontal = abs(np.diff(mesh.vertices[edges, 2], axis=1)[:, 0]) < 1e-7
    edges = edges[horizontal & (mesh.face_adjacency_angles > 1e-3)]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(mesh.vertices),)*2)
    count, labels = connected_components(graph, directed=False)
    rings = []
    for label in range(count):
        ids = np.flatnonzero(labels == label)
        if len(ids) < 16:
            continue
        points = mesh.vertices[ids]
        xy = points[:, :2]
        a, b, c = np.linalg.lstsq(np.column_stack([2*xy, np.ones(len(xy))]),
                                 (xy*xy).sum(axis=1), rcond=None)[0]
        center = np.array([a, b])
        radius = np.sqrt(c+a*a+b*b)
        if np.ptp(points[:, 2]) > 1e-7 or np.max(abs(np.linalg.norm(xy-center, axis=1)-radius)) > 1e-7:
            continue
        angles = np.sort(np.arctan2(xy[:, 1]-b, xy[:, 0]-a))
        if np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])) > .5:
            continue
        touching = np.any(np.isin(mesh.faces, ids), axis=1)
        radial = mesh.triangles_center[:, :2]-center
        radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-20)
        signs = np.einsum('ij,ij->i', mesh.face_normals[:, :2], radial)[touching]
        inner = signs.min() < -.1 and signs.max() < .1
        outer = signs.max() > .1 and signs.min() > -.1
        if inner or outer:
            rings.append({'center': center, 'z': float(points[:, 2].mean()),
                          'radius': float(radius), 'inner': inner, 'points': points, 'ids': ids})
    return rings


def _on_faces(points, triangles):
    """Point inclusion in horizontal triangles, without a spatial-index dependency."""
    inside = np.zeros(len(points), dtype=bool)
    for triangle in triangles:
        a, b, c = triangle[:, :2]
        matrix = np.column_stack([b-a, c-a])
        if abs(np.linalg.det(matrix)) < 1e-16:
            continue
        uv = np.linalg.solve(matrix, (points-a).T).T
        inside |= (uv.min(axis=1) >= -1e-8) & (uv.sum(axis=1) <= 1+1e-8)
    return inside


def recognize_single(mesh):
    rings = circular_rings(mesh)
    centers = []
    for ring in rings:
        if not any(np.linalg.norm(ring['center']-c) < 1e-6 for c in centers):
            centers.append(ring['center'])
    if len(centers) != 2:
        raise ValueError('Single-solid GOAT profile requires exactly two vertical circular peg/socket axes')
    sockets, legs, seating, shifts, clearances = [], [], [], [], []
    for center in sorted(centers, key=lambda c: c[0]):
        axis = [r for r in rings if np.linalg.norm(r['center']-center) < 1e-6]
        outer = sorted((r for r in axis if not r['inner']), key=lambda r: r['z'])
        inner = sorted((r for r in axis if r['inner']), key=lambda r: r['z'])
        if len(outer) < 2 or len(inner) < 3:
            raise ValueError('Missing peg shaft or socket profile rings')
        shaft = outer[0]
        matching = [r for r in outer if abs(r['radius']-shaft['radius']) < 1e-7]
        if len(matching) != 2:
            raise ValueError('Expected one cylindrical peg shaft with a pointed tip')
        root = matching[-1]
        axial = mesh.vertices[np.linalg.norm(mesh.vertices[:, :2]-center, axis=1) < 1e-7]
        tips = sorted(axial[axial[:, 2] < shaft['z'], 2])
        if not len(tips):
            raise ValueError('Expected a pointed outer peg tip')
        tip = float(tips[0])
        mouth = inner[-1]
        # Follow the actual inward-facing cavity surface. A hollow peg can have
        # a separate opening below the socket; it must not extend socket depth.
        radial = mesh.triangles_center[:, :2]-center
        radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-20)
        inward = np.einsum('ij,ij->i', mesh.face_normals[:, :2], radial) < -.1
        adjacency = [[] for _ in mesh.faces]
        for a, b in mesh.face_adjacency:
            if inward[a] and inward[b]:
                adjacency[a].append(b)
                adjacency[b].append(a)
        pending = list(np.flatnonzero(inward & np.any(np.isin(mesh.faces, mouth['ids']), axis=1)))
        cavity = set(pending)
        while pending:
            for neighbor in adjacency[pending.pop()]:
                if neighbor not in cavity:
                    cavity.add(neighbor)
                    pending.append(neighbor)
        cavity_vertices = np.unique(mesh.faces[sorted(cavity)])
        inner = [r for r in inner if np.isin(r['ids'], cavity_vertices).all()]
        floor = float(mesh.vertices[cavity_vertices, 2].min())
        lowest = mesh.vertices[cavity_vertices][abs(mesh.vertices[cavity_vertices, 2]-floor) < 1e-7]
        floor_faces = mesh.triangles[(mesh.face_normals[:, 2] > .999) &
                                     np.all(abs(mesh.triangles[:, :, 2]-floor) < 1e-7, axis=1)]
        pointed_floor = np.all(np.linalg.norm(lowest[:, :2]-center, axis=1) < 1e-7)
        if len(inner) < 3 or not (pointed_floor or _on_faces(center[None, :], floor_faces)[0]):
            raise ValueError('Expected a closed planar or conical blind socket floor')
        if any(a['radius'] > b['radius']+1e-7 for a, b in zip(inner, inner[1:])):
            raise ValueError('Socket radius must increase monotonically toward its mouth')
        mouth = inner[-1]
        throat = inner[-2]
        clearance = throat['radius']-root['radius']
        if clearance <= 0:
            raise ValueError('Peg radius is not smaller than the socket throat radius')
        shift = mouth['z']-root['z']
        if shift <= 0:
            raise ValueError('Socket mouth must be above the peg root')
        profile = [[mouth['z']-r['z'], r['radius']] for r in reversed(inner)]
        if pointed_floor:
            profile.append([mouth['z']-floor, 0.])
        sockets.append({'mouth': [*center.tolist(), mouth['z']], 'quat': [1, 0, 0, 0],
                        'depth': mouth['z']-floor, 'lead_depth': mouth['z']-throat['z'],
                        'opening_radius': mouth['radius'], 'throat_radius': throat['radius'],
                        'radius_profile': profile})
        # Include points partway along the cone, where a shallow socket can
        # interfere even if its axis and mouth are clear.
        cone_mid = shaft['points'].copy()
        cone_mid[:, :2] = center+(cone_mid[:, :2]-center)/2
        cone_mid[:, 2] = (shaft['z']+tip)/2
        probes = np.vstack([root['points'], shaft['points'], cone_mid, [*center, tip]])
        legs.append({'tip': [*center.tolist(), tip], 'target_depth': root['z']-tip,
                     'surface_probes': probes.tolist()})
        # Pair actual planar material on the underside and the receiver's rim.
        down = mesh.triangles[(mesh.face_normals[:, 2] < -.999) &
                              np.all(abs(mesh.triangles[:, :, 2]-root['z']) < 1e-7, axis=1)]
        up = mesh.triangles[(mesh.face_normals[:, 2] > .999) &
                            np.all(abs(mesh.triangles[:, :, 2]-mouth['z']) < 1e-7, axis=1)]
        grid = np.linspace(-1.3*mouth['radius'], 1.3*mouth['radius'], 25)
        points = np.array([[x, y] for x in grid for y in grid])+center
        points = points[_on_faces(points, down) & _on_faces(points, up)]
        if len(points) < 3 or np.linalg.matrix_rank(points-points[0], tol=1e-8) < 2:
            raise ValueError('Could not identify matching planar seating surfaces')
        for point in points[np.linspace(0, len(points)-1, min(8, len(points)), dtype=int)]:
            seating.append({'upper': [*point.tolist(), root['z']],
                            'lower': [*point.tolist(), mouth['z']], 'normal': [0, 0, 1]})
        shifts.append(shift)
        clearances.append(clearance)
    if abs(shifts[0]-shifts[1]) > 1e-6:
        raise ValueError('Peg roots and socket mouths do not define one vertical seating transform')
    shift = float(np.mean(shifts))
    for pair in seating:
        pair['lower'][2] = pair['upper'][2]+shift
    return {'target': {'pos': [0, 0, shift], 'quat': [1, 0, 0, 0]},
            'sockets': sockets, 'legs': legs, 'seating_pairs': seating,
            'clearance': min(clearances), 'max_penetration': min(clearances)*.1}
