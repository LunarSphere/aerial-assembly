"""Static figures of a biasbrick3d: views, a two-course stack, and sections through a hopper and the cavities."""
import math

import numpy as np

from . import geometry as G


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _iso(ax, meshes, title, elev=25, azim=-60):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    lo = np.min([m.bounds[0] for m, _ in meshes], axis=0)
    hi = np.max([m.bounds[1] for m, _ in meshes], axis=0)
    for mesh, color in meshes:
        ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor=color, edgecolor='k',
                                             linewidths=0.05, alpha=0.6))
    c, r = (lo + hi)/2, (hi - lo).max()/2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)


def _tines(ax, p, bundle):
    plt = _plt()
    for x, z in bundle['channels']:
        ax.add_patch(plt.Circle((x, z), p.fork_d/2, color='tab:red'))


def views(p, bundle, out):
    """Side, end and top projections plus an isometric view; tines (red) as carried."""
    plt = _plt()
    mesh = bundle['mesh']
    tris = mesh.vertices[mesh.faces]
    fig = plt.figure(figsize=(15, 9))
    for k, (title, axes) in enumerate((('side (XZ)', (0, 2)), ('end (YZ)', (1, 2)), ('top (XY)', (0, 1)))):
        ax = fig.add_subplot(2, 3, k + 1)
        for t in tris:
            ax.fill(t[:, axes[0]], t[:, axes[1]], color='tab:blue', alpha=0.05, lw=0)
        if axes == (0, 2):
            _tines(ax, p, bundle)
        if axes == (0, 1):
            for i in range(p.n_vox):
                sx, sy = p.seat_xy(i)
                sy = -sy if p.part == 'Y' else sy
                ax.plot(sx, sy, 'k+', ms=10)
                ax.annotate('', xy=(sx, sy), xytext=(sx + p.aim[0], sy + (p.aim[1] if p.part == 'X' else -p.aim[1])),
                            arrowprops=dict(arrowstyle='->', color='tab:orange'))
        ax.set_title(title + (', seats + and aim arrows' if axes == (0, 1) else ''))
        ax.set_aspect('equal')
        ax.grid(True, alpha=.4)
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    _iso(ax, [(mesh, 'tab:blue')], f'biasbrick3d {p.part}, U = {p.U} mm, {bundle["mass_g"]:.1f} g, '
                                   f'aim ({p.aim[0]:.1f}, {p.aim[1]:.1f}) mm for {p.aim_r} mm error')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def stack(p, bundle, out, lift=25.):
    """Two courses of a tower: seated (left) and the upper brick lifted by ``lift`` mm (right)."""
    plt = _plt()
    lower = bundle['mesh']
    fig = plt.figure(figsize=(15, 7))
    for k, (dz, title) in enumerate(((0., 'seated, course pitch H0 = %.1f mm' % p.H0),
                                     (lift, 'upper brick lifted %.0f mm: cup feet over the corner hoppers' % lift))):
        upper = lower.copy()
        upper.apply_translation([0., 0., p.H0 + dz])
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        _iso(ax, [(lower, 'tab:blue'), (upper, 'tab:orange')], title, elev=20, azim=-50)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def sections(p, bundle, out):
    """Diagonal section through an outer voxel's seat (crease, throat and side walls, with the
    upper brick's cup foot seated), and an XZ section through the fork cavities with the tines."""
    plt = _plt()
    mesh = bundle['mesh']
    upper = mesh.copy()
    upper.apply_translation([0., 0., p.H0])
    sx, sy = p.seat_xy(0)
    sy = -sy if p.part == 'Y' else sy
    d = np.array([1., 1. if p.part == 'X' else -1., 0.])/math.sqrt(2)
    n = np.array([-d[1], d[0], 0.])
    origin = np.array([sx, sy, 0.])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7))
    along = lambda pts: (pts - origin) @ d  # noqa: E731
    up = lambda pts: pts[:, 2]  # noqa: E731
    for m, color, label in ((mesh, 'tab:blue', 'course 0'), (upper, 'tab:orange', 'course 1 (cup foot)')):
        s = m.section(plane_origin=origin, plane_normal=n)
        if s is not None:
            for k, seg in enumerate(s.discrete):
                a1.plot(along(seg), up(seg), color=color, lw=1, label=label if k == 0 else None)
    a1.axvline(0, color='k', lw=0.5, ls=':')
    a1.set_title(f'diagonal section through voxel 0 seat: crease {math.degrees(math.atan(math.tan(math.radians(p.ramp))/math.sqrt(2))):.1f} deg, '
                 f'tooth and slot lean {p.alpha} deg')
    a1.set_xlabel('distance from seat along +x+y diagonal (mm)')
    a1.set_ylabel('z (mm)')
    a1.legend(loc='upper left')
    y0, y1 = bundle['cavity_y']
    ym = (y0 + y1)/2*(1 if p.part == 'X' else -1)
    s = mesh.section(plane_origin=[0., ym, 0.], plane_normal=[0., 1., 0.])
    if s is not None:
        for seg in s.discrete:
            a2.plot(seg[:, 0], seg[:, 2], color='tab:blue', lw=1)
    _tines(a2, p, bundle)
    a2.plot(*np.asarray(bundle['com'])[[0, 2]], 'kx', ms=10, label='COM')
    a2.set_title(f'XZ section at y = {ym:.1f} mm: shelled cavities ({2*G.cavity_half(p):.0f} mm square, '
                 f'{p.roof_angle:.0f} deg V roof), tines as carried')
    a2.legend(loc='lower left')
    for ax in (a1, a2):
        ax.set_aspect('equal')
        ax.grid(True, alpha=.4)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
