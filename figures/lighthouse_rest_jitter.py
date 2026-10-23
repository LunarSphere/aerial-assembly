"""Lighthouse position jitter at rest versus battery voltage and calibration (Exp 3, issue #24).

Reads the local flight logs in runs/ (git-ignored) and writes the checked-in table and figure:

    uv run --project flight --extra plot python figures/lighthouse_rest_jitter.py

Jitter is the standard deviation of the Lighthouse X-Y estimate (stateEstimate, 20 Hz) while
the drone rests on its launch pad between preflight and takeoff (about 1.2 s, ~25 samples).
Voltage is the at-rest preflight reading. Attempts that failed the preflight stability check
never reached takeoff, so they have no jitter value; their voltage is bounded above by the
previous flight's landing voltage.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RUNS = [
    # (series, Lighthouse calibration, battery charges, each a list of run directories in flight order)
    ('Session 1: 3-placement calibration', '3 floor placements',
     [[f'runs/exp3-s1/trial-0{i}' for i in range(1, 5)]]),
    ('Session 2, calibration 1 (stations drifting)', '3 placements + 13 hand-held',
     [[f'runs/exp3-s2/trial-0{i}' for i in range(1, 6)]]),
    ('Session 2, calibration 2 (fresh)', '3 placements + 13 hand-held, recalibrated',
     [['runs/exp3-s2/trial-07', 'runs/exp3-s2/trial-08']]),
]
# Failed attempts that are not a battery effect.
EXCLUDED = {'runs/exp3-s2/trial-06': 'a base station moved (station check, 3.9 V battery); recalibrated before trial 7'}
COLOURS = dict(zip([r[0] for r in RUNS], ['#2a78d6', '#eb6834', '#1baf7a']))
MARKERS = dict(zip([r[0] for r in RUNS], ['o', 's', '^']))


def rest_stats(directory):
    report = json.loads((directory/'report.json').read_text())
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    telemetry = [e for e in events if e.get('event') == 'telemetry']
    targets = [e for e in events if e.get('event') == 'target']
    battery = [e['values']['pm.vbat'] for e in telemetry if 'pm.vbat' in e['values']]
    row = dict(run=str(directory.relative_to(ROOT)), outcome='flew' if targets else 'preflight failed',
               reason=report.get('reason') or '', preflight_v=(report.get('preflight') or {}).get('vbat_V'),
               landing_v=battery[-1] if battery else None)
    if targets:
        rest = np.array([[e['values']['stateEstimate.x'], e['values']['stateEstimate.y']] for e in telemetry
                         if 'stateEstimate.x' in e['values'] and e['host_time'] < targets[0]['t']])
        std = rest.std(axis=0)*1000
        row.update(samples=len(rest), std_x_mm=std[0], std_y_mm=std[1], std_xy_mm=float(np.hypot(*std)),
                   ptp_x_mm=np.ptp(rest[:, 0])*1000, ptp_y_mm=np.ptp(rest[:, 1])*1000)
    return row


def collect():
    rows = []
    for session, geometry, charges in RUNS:
        for charge, directories in enumerate(charges, 1):
            previous_landing = None
            for name in directories:
                row = dict(session=session, geometry=geometry, charge=charge, **rest_stats(ROOT/name))
                if row['outcome'] == 'preflight failed':
                    if 'Lighthouse' not in row['reason']:
                        continue  # e.g. the camera-mismatch abort: not a Lighthouse result
                    row['voltage_upper_bound_v'] = previous_landing
                previous_landing = row['landing_v'] or previous_landing
                rows.append(row)
    return rows


def write_table(rows, path):
    columns = ['session', 'geometry', 'charge', 'run', 'outcome', 'preflight_v', 'voltage_upper_bound_v', 'landing_v',
               'samples', 'std_x_mm', 'std_y_mm', 'std_xy_mm', 'ptp_x_mm', 'ptp_y_mm']
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(['' if row.get(c) is None else (f'{row[c]:.2f}' if isinstance(row[c], float) else row[c])
                             for c in columns])


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    figure, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ceiling = max(r['std_xy_mm'] for r in rows if 'std_xy_mm' in r)*1.25
    for session, *_ in RUNS:
        flown = [r for r in rows if r['session'] == session and 'std_xy_mm' in r]
        for charge in sorted({r['charge'] for r in flown}):  # lines join flights on one charge only
            same = [r for r in flown if r['charge'] == charge]
            axis.plot([r['preflight_v'] for r in same], [r['std_xy_mm'] for r in same],
                      color=COLOURS[session], linewidth=1, alpha=.5)
        x = [r['preflight_v'] for r in flown]
        y = [r['std_xy_mm'] for r in flown]
        axis.scatter(x, y, s=48, marker=MARKERS[session], color=COLOURS[session], edgecolor='#fcfcfb',
                     linewidth=1.5, zorder=3, label=session)
        failed = [r for r in rows if r['session'] == session and r['outcome'] == 'preflight failed']
        if failed:
            # One row per session; the arrow reads "at or below this voltage".
            level = ceiling*(.95 if session == RUNS[0][0] else .87)
            bound = failed[0]['voltage_upper_bound_v']
            axis.annotate('', xy=(bound-.035, level), xytext=(bound, level),
                          arrowprops=dict(arrowstyle='->', color=COLOURS[session], linewidth=1.5))
            axis.scatter([bound], [level], s=60, marker='x', color=COLOURS[session], linewidth=2, zorder=3)
            axis.text(bound+.006, level, f'×{len(failed)}', color='#52514e', fontsize=8, va='center')
    axis.text(3.625, ceiling*.80, 'preflight failed: no stable\nposition (voltage ≤ previous landing)',
              color='#52514e', fontsize=8, va='top')
    axis.axvline(3.8, color='#52514e', linewidth=1, linestyle='--')
    axis.text(3.803, ceiling*.05, '3.8 V preflight minimum', color='#52514e', fontsize=8, rotation=90, va='bottom')
    axis.set_xlim(3.62, 4.2)
    axis.set_ylim(0, ceiling)
    axis.set_xlabel('Battery at preflight (V)')
    axis.set_ylabel('Lighthouse X-Y jitter at rest (mm, std)')
    axis.grid(color='#e8e7e3', linewidth=.8)
    axis.set_axisbelow(True)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)
    axis.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, .8), fontsize=8)
    figure.savefig(path, dpi=200, facecolor='#fcfcfb')
    plt.close(figure)


if __name__ == '__main__':
    rows = collect()
    write_table(rows, OUT/'lighthouse-rest-jitter.csv')
    plot(rows, OUT/'lighthouse-rest-jitter.png')
    for row in rows:
        print(row['session'], row['run'], row['outcome'], row.get('preflight_v'), row.get('voltage_upper_bound_v'),
              row.get('std_xy_mm'))
