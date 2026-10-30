"""Static figures: three orthographic views and an isometric view of a DF brick."""
import numpy as np


def views(p, bundle, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = bundle['mesh']
    tris = mesh.vertices[mesh.faces]
    fig = plt.figure(figsize=(15, 9))
    for k, (title, axes) in enumerate((('side (XZ)', (0, 2)), ('end (YZ)', (1, 2)), ('top (XY)', (0, 1)))):
        ax = fig.add_subplot(2, 3, k + 1)
        for t in tris:
            ax.fill(t[:, axes[0]], t[:, axes[1]], color='tab:blue', alpha=0.05, lw=0)
        for x, z in bundle['channels']:
            if axes == (0, 2):
                ax.add_patch(plt.Circle((x, z), p.fork_d/2, color='tab:red'))
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=.4)
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.add_collection3d(Poly3DCollection(tris, facecolor='tab:blue', edgecolor='k', linewidths=0.05, alpha=0.6))
    lo, hi = mesh.bounds
    c, r = (lo + hi)/2, (hi - lo).max()/2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=25, azim=-60)
    ax.set_title(f'DF brick, U = {p.U} mm, {bundle["mass_g"]:.1f} g, {p.foot_kind} feet, {p.receiver_mode}')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return np.asarray(lo), np.asarray(hi)
