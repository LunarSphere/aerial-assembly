"""Guided RGB chessboard intrinsics, saved with the exact capture profile."""
import time

import numpy as np

from . import artifacts
from .camera import cv, open_camera


def calibrate(args):
    c = cv()
    if min(args.board) < 3 or not np.isfinite(args.square_size) or args.square_size <= 0:
        raise ValueError('Use a measured chessboard with at least 3×3 internal corners and positive square size')
    out = artifacts.session(args.out, 'intrinsics', {})
    camera = None
    try:
        camera = open_camera(args, need_intrinsics=False)
        object_points = np.zeros((args.board[0] * args.board[1], 3), np.float32)
        object_points[:, :2] = np.mgrid[0:args.board[0], 0:args.board[1]].T.reshape(-1, 2) * args.square_size
        images, objects, previous = [], [], None
        print('Move/tilt a measured chessboard across the view. Space captures; q finishes after ≥15 views.')
        print('Keep focus/zoom fixed. Include corners and edges of the image.')
        while True:
            frame = camera.frame()
            gray = c.cvtColor(frame.color, c.COLOR_BGR2GRAY)
            found, points = c.findChessboardCornersSB(gray, tuple(args.board))
            preview = frame.color.copy()
            if found:
                c.drawChessboardCorners(preview, tuple(args.board), points, found)
            c.putText(preview, f'{len(images)} / 15+ views', (20, 30), c.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            c.imshow('flight-cal intrinsics', preview)
            key = c.waitKey(20) & 255
            if key == ord('q'):
                if len(images) < 15:
                    raise ValueError('At least 15 distinct chessboard views are required')
                break
            if key == 32 and found:
                if previous is not None and np.median(np.linalg.norm(points - previous, axis=2)) < 15:
                    print('Move or tilt the board more before capturing another view.')
                    continue
                if not c.imwrite(str(out / f'board-{len(images):02d}.png'), frame.color):
                    raise OSError('Could not save chessboard image')
                images.append(points.astype(np.float32))
                objects.append(object_points.copy())
                previous = points.copy()
        size = (camera.profile['width'], camera.profile['height'])
        rms, matrix, distortion, _, _ = c.calibrateCamera(objects, images, size, None, None)
        if not np.isfinite(rms) or rms > 1.:
            raise ValueError(f'Intrinsics rejected: RMS {rms:.3f} px > 1 px; repeat with better board coverage')
        data = dict(version=1, matrix=matrix.tolist(), distortion=distortion.flatten().tolist(),
                    profile=camera.profile, rms_px=rms, views=len(images), square_size_m=args.square_size,
                    board_internal_corners=args.board)
        from .camera import validate_intrinsics
        validate_intrinsics(data)
        artifacts.write(out / 'intrinsics.json', data)
        artifacts.report(out, accepted=True, rms_px=rms, views=len(images))
    except BaseException as e:
        artifacts.report(out, accepted=False, reason=str(e))
        raise
    finally:
        if camera is not None:
            camera.close()
        c.destroyAllWindows()
    return out
