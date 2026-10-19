"""Real Crazyflie backend: unchanged cflib. Dry run by default; arming needs ``--arm``.

Modes:
* dry run (default): offline plan checks, then the whole mission against a recording fake
  Crazyflie with a scripted camera; prints the cflib calls that would be sent. No radio.
* ``--check-only``: opens the radio link and runs the preflight (deck, estimator, battery,
  pose on the stand, geofence). Never arms.
* ``--arm``: preflight, then the mission; Ctrl+C or any exception lands in place and stops.
"""
import json

from .checks import check_plan
from .clock import RealClock
from .fake import FakeLogConfig, FakeSyncCrazyflie, PublishingClock, RecordingCrazyflie, ScriptedPoseSource
from .mission import Mission


def dry_run(plan, out=print):
    """Offline checks + recorded call sequence. Returns (summary, calls)."""
    summary = check_plan(plan)
    clock = PublishingClock()
    cf = RecordingCrazyflie(clock=clock, home=plan['home']['pos'])
    clock.cf = cf
    poses = ScriptedPoseSource(plan)
    with FakeSyncCrazyflie('dry-run://', cf=cf) as scf:
        mission = Mission(scf, plan, clock, poses, FakeLogConfig, on_event=poses.on_event)
        result = mission.run()
    summary['dry_run_status'] = result.status
    summary['dry_run_duration_s'] = clock.time()
    summary['calls'] = len(cf.rec.calls)
    out(json.dumps(summary, indent=1))
    return summary, cf.rec.calls


def connect(uri):
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    cflib.crtp.init_drivers()
    return SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./.cache'))


def fly(plan, uri, poses, arm=False, keep_going=False, out=print, on_event=None):
    """Real hardware. Only with ``arm=True`` are motors armed; otherwise preflight only."""
    from cflib.crazyflie.log import LogConfig
    from .preflight import run_preflight
    check_plan(plan)
    clock = RealClock()
    with connect(uri) as scf:
        report = run_preflight(scf, plan, LogConfig, clock)
        out(f'preflight ok: {report}')
        if not arm:
            out('--check-only: not arming')
            return report

        def event(e):
            out(json.dumps(e, default=str))
            if on_event is not None:
                on_event(e)
        mission = Mission(scf, plan, clock, poses, LogConfig, keep_going=keep_going, arm=True, on_event=event)
        result = mission.run()   # Ctrl+C / exceptions: lands in place and stops, then re-raises
        out(f'mission: {result.status}')
        return result
