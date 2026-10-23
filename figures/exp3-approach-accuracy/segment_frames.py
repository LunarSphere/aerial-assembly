"""Exp 3 follow-up (issue #26): SAM mask centroids of the drone in the saved hover frames.

    uv run --project flight --extra vision --extra yolo python figures/exp3-approach-accuracy/segment_frames.py runs/exp3-s2

Writes data/trial-NN/masks.json (one centroid per saved image) and mask-examples.png; make_figures.py
reads the centroids. The frames themselves stay in runs/ (git-ignored). They have the YOLOE box drawn
on them (#29), and SAM segments that rectangle, so the exact drawn pixels are inpainted first.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'
MODEL = 'sam2.1_b.pt'


def drawn_box(shape, bbox):
    """The pixels hover.camera_worker drew: a 2 px rectangle on the rounded box corners."""
    box = np.rint(bbox).astype(int)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(mask, tuple(box[:2]), tuple(box[2:]), 255, 2)
    return mask


def segment(model, image, bbox):
    drawn = drawn_box(image.shape, bbox)
    clean = cv2.inpaint(image, cv2.dilate(drawn, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
    result = model.predict(clean, bboxes=[bbox], verbose=False)[0]
    mask = result.masks.data[0].cpu().numpy().astype(bool)
    yy, xx = np.nonzero(mask)
    x0, y0, x1, y1 = bbox
    return clean, mask, dict(centroid_px=[float(xx.mean()), float(yy.mean())], area_px=int(mask.sum()),
                             box_area_px=float((x1-x0)*(y1-y0)), score=float(result.boxes.conf[0]))


def tile(image, mask, bbox, centroid, pad=30, size=140):
    overlay = image.copy()
    overlay[mask] = (.5*overlay[mask]+[0, 0, 127]).astype(np.uint8)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    crop = overlay[max(y0-pad, 0):y1+pad, max(x0-pad, 0):x1+pad].copy()
    cv2.circle(crop, (round(centroid[0])-max(x0-pad, 0), round(centroid[1])-max(y0-pad, 0)), 1, (255, 0, 255), -1)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('session', type=Path, help='the approach-accuracy session directory in runs/')
    parser.add_argument('--model', type=Path, default=Path('runs/models')/MODEL)
    args = parser.parse_args()
    from ultralytics import SAM
    model = SAM(str(args.model))
    session = json.loads((args.session/'session.json').read_text())
    tiles = []
    for trial in session['trials']:
        if not trial['accepted']:
            continue
        folder = args.session/trial['dir']
        frames = {}
        for line in (folder/'events.jsonl').open():
            event = json.loads(line)
            if event.get('event') == 'camera' and event.get('image') and 'drone' in event:
                frames.setdefault(event['image'], event['drone'])
        masks = {}
        for name, drone in sorted(frames.items()):
            clean, mask, row = segment(model, cv2.imread(str(folder/name)), drone['bbox_xyxy'])
            masks[name] = dict(row, box_centre_px=drone['center_px'])
            if len(tiles) < 24 and len(masks) % 10 == 1:
                tiles.append(tile(clean, mask, drone['bbox_xyxy'], row['centroid_px']))
        out = dict(version=1, kind='drone-mask-centroids', model=MODEL, prompt='YOLOE box, drawn box inpainted',
                   source=str(folder), frames=masks)
        (DATA/trial['dir']/'masks.json').write_text(json.dumps(out, indent=1)+'\n')
        print(f'{trial["dir"]}: {len(masks)} frames')
    rows = [np.hstack(tiles[i:i+8]) for i in range(0, len(tiles)-len(tiles) % 8, 8)]
    cv2.imwrite(str(HERE/'mask-examples.png'), np.vstack(rows))


if __name__ == '__main__':
    main()
