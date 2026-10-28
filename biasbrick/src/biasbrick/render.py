"""Static figures of a biasbrick: views, a two-course stack, and sections through a socket and the sleeves."""
import math

import numpy as np

from . import cad, geometry as G


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
    for k, (title, axes) in enumerate((('side (XZ), tines as carried', (0, 2)), ('end (YZ): one-sided Y slope', (1, 2)),
                                       ('top (XY): seats + and aim arrows for a brick on part A', (0, 1)))):
        ax = fig.add_subplot(2, 3, k + 1)
        for t in tris:
            ax.fill(t[:, axes[0]], t[:, axes[1]], color='tab:blue', alpha=0.05, lw=0)
        if axes == (0, 2):
            _tines(ax, p, bundle)
        if axes == (0, 1):
            ax_, ay = G.aim(p, 'A')
            for i in (0, 1):
                sx = p.socket_x(i) + G.seat_x(p, p.part)
                ax.plot(sx, 0., 'k+', ms=10)
                ax.annotate('', xy=(sx, 0.), xytext=(sx + ax_, ay), arrowprops=dict(arrowstyle='<-', color='tab:orange'))
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=.4)
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    _iso(ax, [(mesh, 'tab:blue')], f'biasbrick {p.part}: U = {p.U} mm sockets, rod {p.rod_L} mm, flange {p.flange} mm, '
                                   f'{bundle["mass_g"]:.1f} g, sized for {p.aim_r} mm')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _pair(p, lower):
    """The lower brick's mesh and an upper brick of the other part, seated one course up."""
    upper = cad.build_brick(p, part='B' if lower['part'] == 'A' else 'A')['mesh'].copy()
    upper.apply_translation([0., 0., p.H0])
    return lower['mesh'], upper


def stack(p, bundle, out, lift=25.):
    """Two courses of a tower (A then B): seated (left) and the upper brick lifted by ``lift`` mm (right)."""
    plt = _plt()
    lower, upper = _pair(p, bundle)
    fig = plt.figure(figsize=(15, 7))
    for k, (dz, title) in enumerate(((0., f'seated, course pitch H0 = {p.H0:.0f} mm'),
                                     (lift, f'upper brick lifted {lift:.0f} mm'))):
        up = upper.copy()
        up.apply_translation([0., 0., dz])
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        _iso(ax, [(lower, 'tab:blue'), (up, 'tab:orange')], title, elev=20, azim=-50)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _cut(ax, mesh, origin, normal, axes, color, label=None):
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    if s is None:
        return
    for k, seg in enumerate(s.discrete):
        ax.plot(seg[:, axes[0]], seg[:, axes[1]], color=color, lw=1, label=label if k == 0 else None)


def sections(p, bundle, out):
    """Seated A (blue) and B (orange): XZ through a socket (sawtooth, slot, leaning tooth), YZ through the
    socket's valley (Y sawtooth: main slope, nest face, flange), and XZ through the sleeves with the tines."""
    plt = _plt()
    lower, upper = _pair(p, bundle)
    x0 = p.socket_x(0)
    xv = x0 + G.seat_x(p, bundle['part'])
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'width_ratios': [1, 1.6, 1.3]})
    a1, a2, a3 = axes
    for m, color, label in ((lower, 'tab:blue', f'course 0 ({bundle["part"]})'), (upper, 'tab:orange', 'course 1')):
        _cut(a1, m, [0., 0., 0.], [0., 1., 0.], (0, 2), color, label)
        _cut(a2, m, [xv + 0.3, 0., 0.], [1., 0., 0.], (1, 2), color, label)
    a1.set_xlim(x0 - p.U/2 - 3, x0 + p.U/2 + 3)
    a1.set_title(f'XZ at y = 0, socket 0: tooth leans {p.alpha:.0f} deg into the slot')
    a2.set_title(f'YZ through the valley: {p.slope:.0f} deg slope, {p.nest_angle:.0f} deg nest face, '
                 f'{p.flange:.0f} mm flange')
    a1.legend(loc='upper left')
    y0, y1 = bundle['sleeve_y']
    _cut(a3, lower, [0., (y0 + y1)/2, 0.], [0., 1., 0.], (0, 2), 'tab:blue')
    _tines(a3, p, bundle)
    a3.plot(*np.asarray(bundle['com'])[[0, 2]], 'kx', ms=10, label='COM')
    a3.set_xlim(-p.pitch/2, p.pitch/2)
    a3.set_title(f'XZ at y = {(y0 + y1)/2:.1f}: rod and sleeves ({2*G.cavity_half(p):.1f} mm square, '
                 f'{p.roof_angle:.0f} deg V roof)')
    a3.legend(loc='lower left')
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, alpha=.4)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def capture_map(p, cells, best, out):
    """Seated (green), jammed (orange) and fell (grey) cells of a dx x dy capture map, with the current
    aim (black +) and the best aim for each error radius (red x, with its ring)."""
    plt = _plt()
    colors = {'seated': 'tab:green', 'jammed': 'tab:orange', 'fell': '0.6'}
    fig, ax = plt.subplots(figsize=(8, 9))
    for o, c in colors.items():
        pts = np.array([(r['dx'], r['dy']) for r in cells if r['outcome'] == o])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=40, marker='s', color=c, label=o)
    ax.plot(*G.aim(p, 'A'), 'k+', ms=14, mew=2, label='current aim')
    for radius, b in best.items():
        a = b['aim_mm']
        ax.plot(*a, 'rx', ms=12, mew=2)
        ax.add_patch(plt.Circle(a, float(radius), fill=False, color='tab:red', ls='--'))
        ax.annotate(f'r {radius}: {100*b["ring_score"]:.0f}% (current {100*b["current_aim_score"]:.0f}%)', a,
                    textcoords='offset points', xytext=(8, 8), color='tab:red')
    ax.set_xlabel('dx from seat (mm, +x = stroke side)')
    ax.set_ylabel('dy from seat (mm, +y = up the main slope)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('capture map (simulated, uncalibrated contact)')
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
