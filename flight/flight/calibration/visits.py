"""Inward tag visits, checked in the calibrated overhead camera view."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..frames import Pose, wxyz_to_xyzw
from .camera import cv, project_points
from .hover import hover_height, route as flight_route
from .rig import corners


def route(rig, calibration, intrinsics, offset_m=.15, demo=False):
    """Find the first inward waypoint whose projected envelope clears every tag.

    Visibility is predicted at each dwell, at zero attitude, with 2 cm envelope
    inflation and a 4 px tag border. Tracking error and transit occlusion are
    measured during flight; this prediction is not a visibility guarantee.
    """
    if not math.isfinite(offset_m) or not 0 < offset_m <= 1:
        raise ValueError('--tag-offset-m must be greater than zero and at most 1 metre')
    c = cv()
    inv = Pose.from_dict(calibration['T_lighthouse_camera']).inverse()
    rv = Rotation.from_quat(wxyz_to_xyzw(inv.quat)).as_rotvec()
    tv = np.asarray(inv.pos)
    low, high = (np.asarray(rig['geofence'][key], float) for key in ('min', 'max'))
    centre = (low[:2]+high[:2])/2
    lo, hi = (np.asarray(rig['envelope_body_m'][key], float) for key in ('min', 'max'))
    lo, hi = lo-.02, hi+.02
    envelope = np.array([[x, y, z] for x in (lo[0], hi[0])
                         for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    width, height = (intrinsics['profile'][key] for key in ('width', 'height'))

    def project(points):
        if np.any(inv.apply(points)[:, 2] <= 0):
            raise ValueError('Route or tag lies behind the calibrated camera')
        return project_points(points, rv, tv, intrinsics).astype(np.float32)

    def hull(points):
        return c.convexHull(np.asarray(points, np.float32)).reshape(-1, 2)

    tags = {tag['id']: project(corners(tag)) for tag in rig['tags']}
    if any(np.any(p < 0) or np.any(p >= [width, height]) for p in tags.values()):
        raise ValueError('Saved tag map projects outside the camera view; check registration')
    border = np.array([[-4, -4], [-4, 4], [4, -4], [4, 4]])
    padded = {key: hull((points[:, None, :]+border).reshape(-1, 2)) for key, points in tags.items()}
    visits = []
    for tag in rig['tags']:
        pos = np.asarray(tag['pose']['pos'], float)
        delta = centre-pos[:2]
        distance_to_centre = float(np.linalg.norm(delta))
        if distance_to_centre < 1e-6:
            delta = np.asarray(rig['home']['pos'][:2])-centre
            if np.linalg.norm(delta) < 1e-6:
                delta = np.array([1., 0.])
            distance_to_centre = float(np.linalg.norm(delta))
        direction = delta/distance_to_centre
        # A central tag may require continuing across the centre to clear its
        # image. Keep the direction fixed and bound automatic extension to 40 cm.
        maximum = max(offset_m, .40)
        candidates = np.append(np.arange(offset_m, maximum, .01), maximum)
        for distance in candidates:
            target = np.array([*(pos[:2]+direction*distance), hover_height(rig)])
            if np.any(target+lo < low) or np.any(target+hi > high):
                continue
            projected = project(envelope+target)
            if np.any(projected < 0) or np.any(projected >= [width, height]):
                continue
            body_hull = hull(projected)
            overlaps = [key for key, polygon in padded.items()
                        if c.intersectConvexConvex(body_hull, polygon)[0] > 0]
            if overlaps:
                continue
            visits.append(dict(name=f'tag-{tag["id"]}', tag_id=tag['id'],
                               tag_centre_m=pos.tolist(), target_body_m=target.tolist(), dwell_s=5.,
                               inward_offset_m=float(distance), projected_body_px=body_hull.tolist(),
                               predicted_occluded_tag_ids=[]))
            break
        else:
            raise ValueError(f'No unobscured inward waypoint fits for tag {tag["id"]}; '
                             'adjust the camera view or tag layout')
    if demo:
        visits = nearest_order(visits, rig['home']['pos'])
    else:
        order = {tag['id']: i for i, tag in enumerate(sorted(rig['tags'], key=lambda t:
                 ({'origin': 0, 'x': 1, 'xy': 2, 'check': 3}[t['role']], t['id'])))}
        visits.sort(key=lambda v: order[v['tag_id']])
    result = flight_route(rig, demo=demo, visits=visits)
    result.update(offset_toward='geofence_centre', centre_xy_m=centre.tolist(),
                  minimum_tag_offset_m=offset_m, tag_border_px=4,
                  projected_tags_px={str(key): points.tolist() for key, points in tags.items()},
                  camera_profile=intrinsics['profile'],
                  T_lighthouse_camera=calibration['T_lighthouse_camera'],
                  visibility_scope='nominal dwell positions; transit may occlude tags')
    return result


def nearest_order(visits, start):
    remaining, ordered, here = list(visits), [], np.asarray(start[:2], float)
    while remaining:
        visit = min(remaining, key=lambda v: np.linalg.norm(np.asarray(v['target_body_m'][:2])-here))
        remaining.remove(visit)
        ordered.append(visit)
        here = np.asarray(visit['target_body_m'][:2])
    return ordered


def centre_route(rig, calibration, intrinsics, centres=None, demo=False, dwell_s=5.):
    """Hover with the Lighthouse deck directly over each tag centre (Exp 3).

    The drone covers its target tag by design, so no occlusion test is made; every
    dwell envelope must still project inside the camera image. ``centres`` maps tag id
    to a surveyed Lighthouse-frame centre; it defaults to the rig.json tag poses.
    """
    if not math.isfinite(dwell_s) or not 0 < dwell_s <= 30:
        raise ValueError('--dwell-s must be greater than zero and at most 30 seconds')
    inv = Pose.from_dict(calibration['T_lighthouse_camera']).inverse()
    rv = Rotation.from_quat(wxyz_to_xyzw(inv.quat)).as_rotvec()
    tv = np.asarray(inv.pos)
    lo, hi = (np.asarray(rig['envelope_body_m'][key], float) for key in ('min', 'max'))
    envelope = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    width, height = (intrinsics['profile'][key] for key in ('width', 'height'))
    offset = np.asarray(rig['body_to_sample_m'], float)

    def project(points, outside_message):
        points = np.asarray(points, float).reshape(-1, 3)
        if np.any(inv.apply(points)[:, 2] <= 0):
            raise ValueError('Route or tag lies behind the calibrated camera')
        projected = project_points(points, rv, tv, intrinsics)
        if np.any(projected < 0) or np.any(projected >= [width, height]):
            raise ValueError(outside_message)
        return projected

    centres = {int(k): np.asarray(v, float) for k, v in (centres or {}).items()}
    tags = {}
    visits = []
    for tag in rig['tags']:
        centre = centres.get(tag['id'], np.asarray(tag['pose']['pos'], float))
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError(f'Tag {tag["id"]}: invalid surveyed centre')
        tags[tag['id']] = project(corners(tag), 'Saved tag map projects outside the camera view; check registration')
        # Body target that puts the deck sensor centre (the Lighthouse reference) over the tag.
        target = np.array([*(centre[:2]-offset[:2]), hover_height(rig)])
        message = f'Tag {tag["id"]}: drone over the tag centre would leave the camera view'
        body = project(envelope+target, message)
        deck = project(target+offset, message)[0]
        visits.append(dict(name=f'tag-{tag["id"]}', tag_id=tag['id'], tag_centre_m=centre.tolist(),
                           target_body_m=target.tolist(), dwell_s=float(dwell_s),
                           projected_body_px=cv().convexHull(body.astype(np.float32)).reshape(-1, 2).tolist(),
                           projected_target_px=deck.tolist()))
    result = flight_route(rig, demo=demo, visits=nearest_order(visits, rig['home']['pos']))
    result.update(kind='approach-accuracy', mode='over_centre',
                  centre_source='survey' if centres else 'rig',
                  projected_tags_px={str(key): points.tolist() for key, points in tags.items()},
                  camera_profile=intrinsics['profile'],
                  T_lighthouse_camera=calibration['T_lighthouse_camera'],
                  visibility_scope='target tag is covered at its dwell by design')
    return result


def preview(plan, path):
    """Save a camera-coordinate diagram of tags and planned drone footprints."""
    c = cv()
    profile = plan['camera_profile']
    image = np.full((profile['height'], profile['width'], 3), 245, np.uint8)
    for key, points in plan['projected_tags_px'].items():
        polygon = np.rint(points).astype(np.int32)
        c.polylines(image, [polygon], True, (20, 120, 20), 2)
        c.putText(image, f'tag {key}', tuple(polygon.mean(axis=0).astype(int)),
                  c.FONT_HERSHEY_SIMPLEX, .45, (20, 100, 20), 1)
    for segment in plan['segments']:
        if 'tag_id' not in segment:
            continue
        polygon = np.rint(segment['projected_body_px']).astype(np.int32)
        centre = tuple(polygon.mean(axis=0).astype(int))
        c.polylines(image, [polygon], True, (160, 70, 10), 1)
        if 'projected_target_px' in segment:
            target = tuple(np.rint(segment['projected_target_px']).astype(int))
            c.drawMarker(image, target, (10, 10, 200), c.MARKER_CROSS, 18, 2)
            c.putText(image, f'{segment["tag_id"]}: over centre', (centre[0]+12, centre[1]-12),
                      c.FONT_HERSHEY_SIMPLEX, .45, (160, 70, 10), 1)
            continue
        tag_centre = tuple(np.mean(plan['projected_tags_px'][str(segment['tag_id'])], axis=0).astype(int))
        c.arrowedLine(image, tag_centre, centre, (160, 70, 10), 1, tipLength=.12)
        c.putText(image, f'{segment["tag_id"]}: {segment["inward_offset_m"]*100:.0f} cm', centre,
                  c.FONT_HERSHEY_SIMPLEX, .45, (160, 70, 10), 1)
    title = ('Commanded deck centres (cross) and dwell footprints, in visit order' if plan.get('mode') == 'over_centre'
             else 'Predicted dwell footprints; 2 cm envelope margin + 4 px tag border')
    c.putText(image, title, (15, 25), c.FONT_HERSHEY_SIMPLEX, .55, (40, 40, 40), 1)
    if plan.get('mode') == 'over_centre':
        c.putText(image, 'Crosses are at hover height: their image offset from each tag is camera perspective',
                  (15, 48), c.FONT_HERSHEY_SIMPLEX, .55, (40, 40, 40), 1)
    if not c.imwrite(str(path), image):
        raise OSError(f'Could not save route preview: {path}')
