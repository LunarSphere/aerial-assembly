"""Exp 3 approach accuracy (issue #24): raw data, tables and figures for discussion.

    # once: copy the session's raw logs from runs/ (git-ignored) into data/
    uv run --project flight --extra plot python figures/exp3-approach-accuracy/make_figures.py --import runs/exp3-s2
    # any time: rebuild the tables and figures from data/
    uv run --project flight --extra plot python figures/exp3-approach-accuracy/make_figures.py

With data/trial-NN/masks.json from segment_frames.py (#26), it also compares the box centre with the
SAM mask centroid on the same saved frames (methods.csv, positioning-methods.png).

Errors are computed with the same functions as `flight-cal approach-metrics`
(flight/flight/calibration/accuracy.py), from the raw event logs.
"""
import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from flight.calibration import accuracy

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'
CALIBRATION_COLOURS = {1: '#2a78d6', 2: '#eb6834'}
CALIBRATION_MARKERS = {1: 'o', 2: '^'}
METHOD_COLOURS = {'box': '#52514e', 'mask': '#7a3fc4'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e8e7e3', '#fcfcfb'


def import_session(source):
    source = Path(source)
    if DATA.exists():
        raise SystemExit(f'{DATA} already exists; remove it to re-import')
    DATA.mkdir(parents=True)
    for path in sorted(source.glob('*.json')):
        if path.name != 'metrics.json':  # derived; rebuilt here
            shutil.copy(path, DATA/path.name)
    for trial in sorted(source.glob('trial-*')):
        (DATA/trial.name).mkdir()
        if (trial/'report.json').exists():  # route.json is regenerable and unused here
            shutil.copy(trial/'report.json', DATA/trial.name/'report.json')
        with (trial/'events.jsonl').open('rb') as src, gzip.open(DATA/trial.name/'events.jsonl.gz', 'wb') as dst:
            shutil.copyfileobj(src, dst)


def load():
    session = json.loads((DATA/'session.json').read_text())
    surveys = {c['index']: json.loads((DATA/c['survey']).read_text()) for c in accuracy.calibrations(session)}
    approaches, trials, methods = [], [], []
    for trial in session['trials']:
        folder = DATA/trial['dir']
        report = json.loads((folder/'report.json').read_text())
        calibration = trial.get('calibration', 1)
        events = [json.loads(line) for line in gzip.open(folder/'events.jsonl.gz', 'rt')]
        launch = accuracy.launch_offset(events, trial['index']) if trial['accepted'] else None
        check = report.get('station_consistency') or {}
        trials.append(dict(trial=trial['index'], calibration=calibration, accepted=trial['accepted'],
                           reason=trial.get('reason', ''), preflight_battery_v=(report.get('preflight') or {}).get('vbat_V'),
                           station_residual_mrad=check.get('worst_rms_mrad'),
                           launch_offset_mm=None if not launch or launch['offset_m'] is None else launch['offset_m']*1000))
        if trial['accepted']:
            centres = accuracy.centres(surveys[calibration])
            for row in accuracy.approaches(events, centres, trial['index']):
                approaches.append(dict(row, calibration=calibration))
            if (folder/'masks.json').exists():
                masks = json.loads((folder/'masks.json').read_text())['frames']
                centroids = {name: m['centroid_px'] for name, m in masks.items()}
                intrinsics = surveys[calibration]['intrinsics']
                for method, shift in (('box', False), ('mask', True)):
                    for row in accuracy.approaches(accuracy.recentre(events, centroids, intrinsics, shift),
                                                   centres, trial['index']):
                        methods.append(dict(row, calibration=calibration, method=method))
    return session, approaches, trials, methods


def method_rows(approaches, methods):
    """One row per approach: positioning error from the box centre (all frames, and the saved frames only)
    and from the mask centroid on those same saved frames."""
    key = lambda r: (r['trial'], r['tag_id'])
    by = {m: {key(r): r for r in methods if r['method'] == m} for m in ('box', 'mask')}
    rows = []
    for r in approaches:
        box, mask = by['box'].get(key(r)), by['mask'].get(key(r))
        if not box or box['positioning_error_m'] is None:
            continue
        row = dict(trial=r['trial'], calibration=r['calibration'], tag=r['tag_id'],
                   all_frames=r['camera_frames'], saved_frames=box['camera_frames'],
                   box_all_frames_cm=r['positioning_error_m']*100, box_cm=box['positioning_error_m']*100,
                   mask_cm=mask['positioning_error_m']*100)
        shift = np.subtract(mask['camera_xy_m'], box['camera_xy_m'])*100
        row.update(mask_minus_box_dx_cm=shift[0], mask_minus_box_dy_cm=shift[1],
                   mask_minus_box_cm=float(np.linalg.norm(shift)))
        rows.append(row)
    return rows


def method_summary(rows):
    groups = [('pooled', rows)]+[(f'calibration {c}', [r for r in rows if r['calibration'] == c])
                                 for c in sorted({r['calibration'] for r in rows})]
    out = []
    for label, subset in groups:
        for column in ('box_all_frames_cm', 'box_cm', 'mask_cm', 'mask_minus_box_cm'):
            values = np.array([r[column] for r in subset])
            out.append(dict(group=label, measure=column[:-3], n=len(values), mean_cm=float(values.mean()),
                            rmse_cm=float(np.sqrt(np.mean(values**2))), max_cm=float(values.max())))
    return out


def write_csv(path, rows, columns):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(['' if row.get(c) is None else (f'{row[c]:.4f}' if isinstance(row[c], float) else row[c])
                             for c in columns])


def flat_approaches(approaches):
    out = []
    for r in approaches:
        row = dict(trial=r['trial'], calibration=r['calibration'], tag=r['tag_id'],
                   camera_frames=r['camera_frames'], lighthouse_samples=r['lighthouse_samples'])
        for kind in ('positioning', 'control'):
            vector = r[f'{kind}_error_xy_m']
            row.update({f'{kind}_dx_cm': None if vector is None else vector[0]*100,
                        f'{kind}_dy_cm': None if vector is None else vector[1]*100,
                        f'{kind}_error_cm': None if vector is None else r[f'{kind}_error_m']*100})
        for name in ('tag_centre_xy_m', 'commanded_xy_m', 'lighthouse_xy_m', 'camera_xy_m'):
            row.update({f'{name[:-5]}_x_m': r[name][0] if r[name] else None,
                        f'{name[:-5]}_y_m': r[name][1] if r[name] else None})
        out.append(row)
    return out


def summary_rows(approaches):
    rows = []
    groups = [('pooled', approaches)]+[(f'calibration {c}', [r for r in approaches if r['calibration'] == c])
                                        for c in sorted({r['calibration'] for r in approaches})]
    for label, subset in groups:
        summary = accuracy.summarize(subset)
        for kind in accuracy.ERROR_TYPES:
            for tag, stats in [('all', summary[kind]['overall'])]+sorted(summary[kind]['per_tag'].items()):
                if stats['n']:
                    rows.append(dict(group=label, error=kind, tag=tag, n=stats['n'],
                                     mean_cm=stats['mean_error_m']*100, rmse_cm=stats['rmse_m']*100,
                                     std_cm=stats['std_error_m']*100, max_cm=stats['max_error_m']*100,
                                     bias_x_cm=stats['bias_xy_m'][0]*100, bias_y_cm=stats['bias_xy_m'][1]*100))
    return rows


def style(axis):
    axis.grid(color=GRID, linewidth=.8)
    axis.set_axisbelow(True)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)


def plot_by_tag(approaches, path):
    """Error vectors in cm, one small panel per tag, both error types, coloured by calibration."""
    import matplotlib.pyplot as plt
    tags = sorted({r['tag_id'] for r in approaches})
    figure, axes = plt.subplots(2, len(tags), figsize=(2.3*len(tags), 5.2), sharex=True, sharey=True,
                                constrained_layout=True)
    limit = 1.1*max(abs(v)*100 for r in approaches for k in accuracy.ERROR_TYPES
                    for v in (r[f'{k}_error_xy_m'] or [0, 0]))
    for row, kind in enumerate(accuracy.ERROR_TYPES):
        for column, tag in enumerate(tags):
            axis = axes[row, column]
            style(axis)
            axis.axhline(0, color=MUTED, linewidth=.8)
            axis.axvline(0, color=MUTED, linewidth=.8)
            for calibration in sorted(CALIBRATION_COLOURS):
                vectors = np.array([r[f'{kind}_error_xy_m'] for r in approaches if r['tag_id'] == tag
                                    and r['calibration'] == calibration and r[f'{kind}_error_xy_m']]).reshape(-1, 2)*100
                axis.scatter(vectors[:, 0], vectors[:, 1], s=40, marker=CALIBRATION_MARKERS[calibration],
                             color=CALIBRATION_COLOURS[calibration], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                             label=f'calibration {calibration}')
            errors = [r[f'{kind}_error_m']*100 for r in approaches if r['tag_id'] == tag and r[f'{kind}_error_m'] is not None]
            axis.text(.04, .96, f'mean {np.mean(errors):.1f} cm', transform=axis.transAxes, va='top',
                      fontsize=8, color=INK)
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
            axis.set_aspect('equal')
            if row == 0:
                axis.set_title(f'tag {tag}', fontsize=9, color=INK)
            if row == 1:
                axis.set_xlabel('X error (cm)', fontsize=8, color=MUTED)
            if column == 0:
                axis.set_ylabel(f'{kind.capitalize()}\nY error (cm)', fontsize=8, color=MUTED)
    axes[0, -1].legend(frameon=False, fontsize=8, loc='lower right')
    figure.suptitle('Exp 3 approach error per tag (5 trials, 25 approaches)', fontsize=10, color=INK)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def plot_by_trial(approaches, trials, path):
    """Error per approach in flight order, plus the Lighthouse-vs-camera pad offset per trial."""
    import matplotlib.pyplot as plt
    accepted = [t for t in trials if t['accepted']]
    order = {t['trial']: i for i, t in enumerate(accepted)}
    figure, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True, constrained_layout=True)
    panels = [('positioning', 'Positioning error (cm)'), ('control', 'Control error (cm)')]
    for axis, (kind, label) in zip(axes[:2], panels):
        style(axis)
        for calibration in sorted(CALIBRATION_COLOURS):
            rows = [r for r in approaches if r['calibration'] == calibration and r[f'{kind}_error_m'] is not None]
            x = [order[r['trial']]+(r['tag_id']-2)*.06 for r in rows]
            axis.scatter(x, [r[f'{kind}_error_m']*100 for r in rows], s=36, marker=CALIBRATION_MARKERS[calibration],
                         color=CALIBRATION_COLOURS[calibration], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                         label=f'calibration {calibration}')
        means = [np.mean([r[f'{kind}_error_m']*100 for r in approaches if r['trial'] == t['trial']]) for t in accepted]
        axis.plot(range(len(accepted)), means, color=MUTED, linewidth=1.5, marker='_', markersize=14,
                  label='trial mean', zorder=2)
        axis.set_ylabel(label, fontsize=9, color=INK)
        top = max(r[f'{kind}_error_m']*100 for r in approaches if r[f'{kind}_error_m'] is not None)
        axis.set_ylim(0, top*1.3)  # headroom for the legend
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc='upper left')
    axis = axes[2]
    style(axis)
    for t in accepted:
        axis.bar(order[t['trial']], t['launch_offset_mm'], width=.5, color=CALIBRATION_COLOURS[t['calibration']])
        axis.text(order[t['trial']], t['launch_offset_mm']+.4, f'{t["launch_offset_mm"]:.1f}', ha='center',
                  fontsize=8, color=INK)
    axis.set_ylabel('Pad offset, Lighthouse\nvs camera (mm)', fontsize=9, color=INK)
    axis.set_xticks(range(len(accepted)), [f'trial {t["trial"]}\ncal {t["calibration"]}' for t in accepted], fontsize=8)
    figure.suptitle('Exp 3 error per approach in flight order, with the launch-pad offset per trial', fontsize=10, color=INK)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def plot_methods(rows, path):
    """Positioning error per approach on the same saved frames: box centre vs SAM mask centroid."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    style(axis)
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        axis.plot([i, i], [r['box_cm'], r['mask_cm']], color=GRID, linewidth=2, zorder=1)
    axis.scatter(x, [r['box_cm'] for r in rows], s=40, facecolor=SURFACE, edgecolor=METHOD_COLOURS['box'],
                 linewidth=1.6, zorder=3, label='YOLOE box centre')
    axis.scatter(x, [r['mask_cm'] for r in rows], s=40, color=METHOD_COLOURS['mask'], edgecolor=SURFACE,
                 linewidth=1.2, zorder=3, label='SAM mask centroid')
    trials = [r['trial'] for r in rows]
    starts = [i for i in range(len(rows)) if i == 0 or trials[i] != trials[i-1]]
    for i in starts[1:]:
        axis.axvline(i-.5, color=MUTED, linewidth=.6)
    axis.set_xticks([(a+b-1)/2 for a, b in zip(starts, starts[1:]+[len(rows)])],
                    [f'trial {trials[i]}\ncal {rows[i]["calibration"]}' for i in starts], fontsize=8)
    axis.set_xlim(-.6, len(rows)-.4)
    axis.set_ylim(0, 1.25*max(max(r['box_cm'], r['mask_cm']) for r in rows))
    axis.set_ylabel('Positioning error (cm)', fontsize=9, color=INK)
    axis.legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    figure.suptitle('Exp 3 positioning error: box centre vs SAM mask centroid, same saved frames',
                    fontsize=10, color=INK)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--import', dest='source', help='copy a session directory from runs/ into data/ first')
    args = parser.parse_args()
    if args.source:
        import_session(args.source)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
    session, approaches, trials, methods = load()
    flat = flat_approaches(approaches)
    write_csv(HERE/'approaches.csv', flat, list(flat[0]))
    write_csv(HERE/'trials.csv', trials, list(trials[0]))
    summary = summary_rows(approaches)
    write_csv(HERE/'summary.csv', summary, list(summary[0]))
    plot_by_tag(approaches, HERE/'errors-by-tag.png')
    plot_by_trial(approaches, trials, HERE/'errors-by-trial.png')
    for row in summary:
        if row['tag'] == 'all':
            print(f'{row["group"]:14s} {row["error"]:11s} n={row["n"]:2d} mean {row["mean_cm"]:.2f} cm, '
                  f'RMSE {row["rmse_cm"]:.2f} cm, max {row["max_cm"]:.2f} cm')
    if methods:
        rows = method_rows(approaches, methods)
        write_csv(HERE/'methods.csv', rows, list(rows[0]))
        compared = method_summary(rows)
        write_csv(HERE/'methods-summary.csv', compared, list(compared[0]))
        plot_methods(rows, HERE/'positioning-methods.png')
        for row in compared:
            print(f'{row["group"]:14s} {row["measure"]:15s} n={row["n"]:2d} mean {row["mean_cm"]:.2f} cm, '
                  f'RMSE {row["rmse_cm"]:.2f} cm, max {row["max_cm"]:.2f} cm')


if __name__ == '__main__':
    main()
