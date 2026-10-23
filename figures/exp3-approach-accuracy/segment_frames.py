"""Exp 3 follow-up (issue #26): SAM 2.1 mask centroids of the drone in the saved hover frames.

    uv run --project flight --extra vision --extra yolo python figures/exp3-approach-accuracy/segment_frames.py runs/exp3-r2

SAM is prompted with each frame's logged YOLOE box and the mask centroid replaces the box centre as the
deck position. Writes data/trial-NN/masks-sam.json (one centroid per saved image) and mask-examples.png;
make_figures.py then compares box centre and mask centroid on the same frames. The frames stay in runs/
(git-ignored). They have the YOLOE box drawn on them (#29), and SAM segments that rectangle, so the
exact drawn pixels are inpainted first.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'
MODEL = Path('runs/models/sam2.1_b.pt')  # ultralytics downloads it if missing


def clean(image, bbox):
    """Inpaint the 2 px rectangle hover.camera_worker drew on the rounded box corners."""
    box = np.rint(bbox).astype(int)
    drawn = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(drawn, tuple(box[:2]), tuple(box[2:]), 255, 2)
    return cv2.inpaint(image, cv2.dilate(drawn, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)


def segment(model, image, bbox):
    image = clean(image, bbox)
    result = model.predict(image, bboxes=[bbox], verbose=False)[0]
    mask = result.masks.data[0].cpu().numpy().astype(bool)
    if not mask.any():
        return image, mask, None
    yy, xx = np.nonzero(mask)
    return image, mask, dict(centroid_px=[float(xx.mean()), float(yy.mean())], area_px=int(mask.sum()),
                             score=float(result.boxes.conf[0]))


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
    args = parser.parse_args()
    from ultralytics import SAM
    model = SAM(str(MODEL))
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
        masks, missed = {}, []
        for name, drone in sorted(frames.items()):
            image, mask, found = segment(model, cv2.imread(str(folder/name)), drone['bbox_xyxy'])
            if found is None:
                missed.append(name)
                continue
            masks[name] = dict(found, box_centre_px=drone['center_px'])
            if len(tiles) < 24 and len(masks) % 10 == 1:
                tiles.append(tile(image, mask, drone['bbox_xyxy'], found['centroid_px']))
        out = dict(version=1, kind='drone-mask-centroids', model=MODEL.name, prompt='YOLOE box, drawn box inpainted',
                   source=str(folder), frames=masks, missed=missed)
        (DATA/trial['dir']/'masks-sam.json').write_text(json.dumps(out, indent=1)+'\n')
        print(f'{trial["dir"]}: {len(masks)} frames, {len(missed)} without a mask')
    rows = [np.hstack(tiles[i:i+8]) for i in range(0, len(tiles)-len(tiles) % 8, 8)]
    cv2.imwrite(str(HERE/'mask-examples.png'), np.vstack(rows))


if __name__ == '__main__':
    main()
