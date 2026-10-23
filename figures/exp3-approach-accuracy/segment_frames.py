"""Exp 3 follow-up (issue #26): SAM mask centroids of the drone in the saved hover frames.

    uv run --project flight --extra vision --extra yolo python figures/exp3-approach-accuracy/segment_frames.py runs/exp3-s2
    ... segment_frames.py runs/exp3-s2 --model runs/models/sam3.pt                  # gated: hf download facebook/sam3 sam3.pt
    ... segment_frames.py runs/exp3-s2 --model runs/models/sam3.pt --deck-exemplar  # deck only; wants a CUDA GPU

Writes data/trial-NN/masks-<method>.json (one centroid per saved image) and mask-examples-<method>.png;
make_figures.py compares every mask set it finds. The frames stay in runs/ (git-ignored). They have the
YOLOE box drawn on them (#29), and SAM segments that rectangle, so the exact drawn pixels are inpainted first.

Default: the whole drone, prompted with the logged YOLOE box. --deck-exemplar: SAM 3 concept segmentation of
the Lighthouse deck only. SAM 3 takes exemplar boxes in the image it segments, so each target crop is placed
beside a reference crop with one hand-drawn deck box (DECK_EXEMPLAR). The best-scoring detection in the
target half whose centroid lies inside the logged YOLOE box is kept: the deck has to be on the drone.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'
# Reference frame (relative to the session) and its deck box in full-frame pixels, drawn by hand.
DECK_EXEMPLAR = dict(trial='trial-07', image='hover-0074.png', box_xyxy=[490, 568, 514, 592])
CROP_HALF_PX = 60


def drawn_box(shape, bbox):
    """The pixels hover.camera_worker drew: a 2 px rectangle on the rounded box corners."""
    box = np.rint(bbox).astype(int)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(mask, tuple(box[:2]), tuple(box[2:]), 255, 2)
    return mask


def clean(image, bbox):
    drawn = drawn_box(image.shape, bbox)
    return cv2.inpaint(image, cv2.dilate(drawn, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)


def row(mask, bbox, score):
    yy, xx = np.nonzero(mask)
    x0, y0, x1, y1 = bbox
    return dict(centroid_px=[float(xx.mean()), float(yy.mean())], area_px=int(mask.sum()),
                box_area_px=float((x1-x0)*(y1-y0)), score=float(score))


def segment(model, image, bbox):
    image = clean(image, bbox)
    result = model.predict(image, bboxes=[bbox], verbose=False)[0]
    mask = result.masks.data[0].cpu().numpy().astype(bool)
    return image, mask, row(mask, bbox, result.boxes.conf[0])


def crop(image, centre):
    """A 2*CROP_HALF_PX square around centre, edge-padded, and its top-left corner in the frame."""
    x, y = [int(round(v)) for v in centre]
    padded = cv2.copyMakeBorder(image, *[CROP_HALF_PX]*4, cv2.BORDER_REPLICATE)
    return padded[y:y+2*CROP_HALF_PX, x:x+2*CROP_HALF_PX], (x-CROP_HALF_PX, y-CROP_HALF_PX)


class DeckExemplar:
    def __init__(self, model, session):
        import torch
        from ultralytics.models.sam import SAM3SemanticPredictor, build_sam3
        # Memory-map the 3.4 GB checkpoint so it is not held in RAM beside the built model.
        build_sam3.torch_load = lambda f: torch.load(f.name, map_location='cpu', mmap=True, weights_only=False)
        gpu = torch.cuda.is_available()
        self.predictor = SAM3SemanticPredictor(overrides=dict(model=str(model), conf=.2, verbose=False, save=False,
                                                              half=gpu, device=0 if gpu else 'cpu'))
        events = map(json.loads, (session/DECK_EXEMPLAR['trial']/'events.jsonl').open())
        drone = next(e['drone'] for e in events if e.get('image') == DECK_EXEMPLAR['image'] and 'drone' in e)
        reference = clean(cv2.imread(str(session/DECK_EXEMPLAR['trial']/DECK_EXEMPLAR['image'])), drone['bbox_xyxy'])
        box = np.asarray(DECK_EXEMPLAR['box_xyxy'], float)
        self.reference, corner = crop(reference, (box[:2]+box[2:])/2)
        self.box = (box-[*corner, *corner]).tolist()

    def __call__(self, image, bbox, centre):
        image = clean(image, bbox)
        target, (x0, y0) = crop(image, centre)
        width = self.reference.shape[1]
        result = self.predictor(source=np.hstack([self.reference, target]), bboxes=[self.box], labels=[1])[0]
        best = None
        if result.masks is not None:
            for mask, box, score in zip(result.masks.data.cpu().numpy().astype(bool), result.boxes.xyxy.tolist(),
                                        result.boxes.conf.tolist()):
                yy, xx = np.nonzero(mask[:, width:])
                inside = len(xx) and bbox[0] <= xx.mean()+x0 <= bbox[2] and bbox[1] <= yy.mean()+y0 <= bbox[3]
                if box[0] >= width and inside and (best is None or score > best[1]):
                    best = mask[:, width:], score, box
        if best is None:
            return image, None, None
        full = np.zeros(image.shape[:2], bool)
        full[y0:y0+target.shape[0], x0:x0+target.shape[1]] = best[0]
        box = [best[2][0]-width+x0, best[2][1]+y0, best[2][2]-width+x0, best[2][3]+y0]
        return image, full, row(full, box, best[1])


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
    parser.add_argument('--model', type=Path, default=Path('runs/models/sam2.1_b.pt'))
    parser.add_argument('--deck-exemplar', action='store_true', help='SAM 3 deck-only concept masks (see above)')
    args = parser.parse_args()
    if args.deck_exemplar:
        if 'sam3' not in args.model.stem:
            parser.error('--deck-exemplar needs a SAM 3 model')
        method, prompt = f'{args.model.stem}-deck', f'deck exemplar {DECK_EXEMPLAR}, drawn box inpainted'
        deck = DeckExemplar(args.model, args.session)
        segmenter = lambda image, drone: deck(image, drone['bbox_xyxy'], drone['center_px'])
    else:
        from ultralytics import SAM
        model = SAM(str(args.model))
        method, prompt = args.model.stem, 'YOLOE box, drawn box inpainted'
        segmenter = lambda image, drone: segment(model, image, drone['bbox_xyxy'])
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
            clean_image, mask, found = segmenter(cv2.imread(str(folder/name)), drone)
            if found is None:
                missed.append(name)
                continue
            masks[name] = dict(found, box_centre_px=drone['center_px'])
            if len(tiles) < 24 and len(masks) % 10 == 1:
                tiles.append(tile(clean_image, mask, drone['bbox_xyxy'], found['centroid_px']))
        out = dict(version=1, kind='drone-mask-centroids', model=args.model.name, prompt=prompt,
                   source=str(folder), frames=masks, missed=missed)
        (DATA/trial['dir']/f'masks-{method}.json').write_text(json.dumps(out, indent=1)+'\n')
        print(f'{trial["dir"]}: {len(masks)} frames, {len(missed)} without a mask')
    rows = [np.hstack(tiles[i:i+8]) for i in range(0, len(tiles)-len(tiles) % 8, 8)]
    cv2.imwrite(str(HERE/f'mask-examples-{method}.png'), np.vstack(rows))


if __name__ == '__main__':
    main()
