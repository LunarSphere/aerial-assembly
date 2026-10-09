"""End-to-end check without hardware: synthetic camera -> calibration -> sensing -> field plan ->
the real ``flight`` mission against a recording fake Crazyflie, with the synthetic world reacting to
the mission (the brick disappears from view when lifted and appears on the seat it was placed on).

Nothing here talks to a radio. It shows that the pieces fit: the pick targets the sensed brick, the
place targets the sensed platform, and both verifications pass on what the camera sees.
"""
import json
import math
from pathlib import Path

import numpy as np

from . import calib as Ca, fieldplan as FP, synth as Sy
from .model import load_model
from .rig import Rig
from .sense import CameraPoseSource


class SynthWorld:
    """A ``capture()``-able camera over a scene that follows the mission's events."""

    def __init__(self, objects, R, t, brick='b0_0', seed=0, place_error=(0., 0., 0.)):
        self.objects = list(objects)
        self.R, self.t, self.seed, self.brick = R, t, seed, brick
        self.place_error = place_error          # (dx m, dy m, dyaw deg) added where the brick lands
        self.seat = None
        self._render()

    def _render(self):
        self.cam = Sy.SynthCamera(Sy.lab_scene(self.objects), self.R, self.t, seed=self.seed)
        self.seed += 1

    def _brick(self):
        return next(i for i, o in enumerate(self.objects) if o.kind == 'brick')

    def on_event(self, e):
        if e.get('brick') != self.brick:
            return
        phase = e.get('phase')
        if phase == 'LIFT':
            i = self._brick()
            o = self.objects[i]
            if o.z_rel is None:                     # first LIFT event: the brick leaves the cradle
                self.objects[i] = Sy.Placed('brick', o.x, o.y, o.yaw_deg, z_rel=0.09)
                self._render()
        elif phase == 'PREPLACE_TARGET':
            self.seat = e['seat']
        elif phase == 'RETREAT' and self.seat is not None:
            from flight.frames import Pose
            seat = Pose.from_dict(self.seat)
            dx, dy, dyaw = self.place_error
            self.objects[self._brick()] = Sy.Placed('brick', seat.pos[0] + dx, seat.pos[1] + dy,
                                                     math.degrees(seat.yaw) + dyaw)
            self.seat = None
            self._render()

    def capture(self, n=None):
        return self.cam.capture(n)

    def close(self):
        pass


def calibrate_synthetic(rig, R, t):
    scene = Sy.lab_scene()
    report = Ca.calibrate(Sy.SynthCamera(scene, R, t), scene.survey(), rig['tags']['dictionary'], frames=3)
    return report


def run_demo(out=None, brick=(0.25, -0.35, 15.), platform=(-0.20, 0.30, -25.), seed=0, plan=None,
             place_error=(0., 0., 0.), log=print):
    """Returns a summary dict. ``brick``/``platform``: (x m, y m, yaw deg) footprint centres."""
    from flight.fake import FakeLogConfig, FakeSyncCrazyflie, PublishingClock, RecordingCrazyflie
    from flight.checks import check_plan
    from flight.mission import Mission
    rig = Rig(overrides={'log_dir': None if out is None else str(Path(out).resolve()/'camera')})
    R, t = Sy.look_at([0.05, -0.15, 2.0], [0., 0.05, 0.])
    cam_pose = Ca.CameraPose(calibrate_synthetic(rig, R, t))
    objects = [Sy.Placed('fixture', *brick), Sy.Placed('brick', *brick), Sy.Placed('platform', *platform)]
    world = SynthWorld(objects, R, t, rig['names']['brick'], seed=seed, place_error=place_error)
    source = CameraPoseSource(rig, world, cam_pose=cam_pose)
    sensed = source.poses()
    model = load_model()
    field = {'home': [-0.25, -0.60, 0.0693], 'home_yaw_deg': 0.,
             'geofence': {'min': [-0.55, -0.85], 'max': [0.55, 0.85]}}
    plan = FP.adapt_plan(plan or FP.compile_plan(), sensed, field, cam_pose.floor_z, model.platform_origin_height,
                         rig['names'])
    checks = check_plan(plan)
    clock = PublishingClock()
    cf = RecordingCrazyflie(clock=clock, home=plan['home']['pos'])
    clock.cf = cf
    with FakeSyncCrazyflie('dry-run://', cf=cf) as scf:
        result = Mission(scf, plan, clock, source, FakeLogConfig, on_event=world.on_event).run()
    final = source.poses()
    b, p = final.get(rig['names']['brick']), final.get(rig['names']['platform'])
    summary = {'mission_status': result.status, 'bricks': result.bricks,
               'sensed_before': {k: v.to_dict() for k, v in sensed.items()},
               'truth': {'brick': brick, 'platform': platform},
               'camera_calls': source.calls, 'offline_checks': checks,
               'events': [e for e in result.events if e.get('phase') in ('PICK_REQUEST', 'VERIFY_PICK',
                                                                          'PREPLACE_TARGET', 'VERIFY_PLACE')]}
    if b is not None and p is not None:
        from flight.frames import Pose
        seat = p.compose(Pose.from_dict(plan['bricks'][0]['place']['frame']['rel']))
        summary['final_brick_in_seat_frame_mm'] = (seat.inverse().apply(b.pos)*1e3).tolist()
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out/'plan.json').write_text(json.dumps(plan, indent=1, default=float))
        (out/'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    log(json.dumps({k: summary[k] for k in ('mission_status', 'camera_calls')}, default=float))
    return summary
