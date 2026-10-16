"""Reviewable hardware assumptions for the already selected CF21B steel-pin fork."""
import json
from pathlib import Path
import struct

import numpy as np


def stl_bounds(path):
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from('<I', data, 80)[0]
        if len(data) == 84+50*count:
            dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
            vertices = np.frombuffer(data, dtype=dtype, offset=84, count=count)['vertices'].reshape(-1, 3)
            return np.array([vertices.min(axis=0), vertices.max(axis=0)], float)/1000
    raise ValueError(f'Expected a binary CAD STL: {path}')


def default_hardware():
    root = Path(__file__).resolve().parents[3]
    base = root/'brixzle'
    # The flight package is also installable without the CAD project; packaged constants
    # below are the conservative envelope derived from the checked-in exports.
    body = dict(body_mass_g=43.4, deck_mass_g=2.7)
    fork = dict(mass_g=6.323699822011288)
    meshes = list((base/'src/brixzle/cf/meshes').glob('*.stl'))
    gripper = base/'docs/exp2/gripper-fork-msh2-30/gripper.stl'
    if gripper.exists():
        meshes.append(gripper)
        body = json.loads((base/'configs/drone-cf21B.json').read_text())
        fork = json.loads((gripper.parent/'gripper.json').read_text())
    bounds = np.array([stl_bounds(p) for p in meshes]) if meshes else None
    lo = bounds[:, 0].min(axis=0) if bounds is not None else np.array([-.070, -.070, -.039])
    hi = bounds[:, 1].max(axis=0) if bounds is not None else np.array([.070, .070, .020])
    hi[2] = max(hi[2], .025)  # cover the top-mounted Lighthouse deck
    return dict(name='cf21b-steel-fork', mass_kg=(body['body_mass_g']+body['deck_mass_g']+fork['mass_g'])/1000,
                body_to_sample_m=[0., 0., .020],
                envelope_body_m=dict(min=lo.tolist(), max=hi.tolist()),
                assumptions=['CF21B with Lighthouse deck and MSH2-30 fork, no brick',
                             'Deck sensor centre 20 mm above motor plane; envelope has a separate 20 mm runtime margin',
                             'Mass uses the existing hardware/CAD preset, not an automatic weight measurement'])
