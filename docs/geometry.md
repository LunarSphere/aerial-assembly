# Preparing the actual block

## 1. Establish the intended pose

In a simulation copy of the document, rigidly position two identical blocks with both legs inserted and the intended faces flush. Inspect interference and section the sockets. The screenshot's 38 mm leg versus 34 mm socket values may prevent this, depending on their datums and inclusion of the tip. Do not compensate with soft collision settings. Record an interference-free target transform or revise the CAD first.

Use a common block frame: Z up, X along the ramp, Y along extrusion depth. Save the transform from original Part Studio coordinates to this frame. Export all three components in their original relative positions; do not independently recenter each STL.

## 2. Export appearance and convex material pieces

Export binary STL in mm. Keep the original body and legs as visual meshes. In a separate CAD tab/copy, partition the solid material into convex pieces: ramp prisms, landing regions around the openings, tapered socket sectors, bore-wall sectors, blind floors, guide wedges, shafts, and tips. Every collision mesh must be a closed convex solid. MuJoCo uses each piece's hull.

Avoid collision pieces spanning socket openings or crossing recesses. Prefer common-boundary partitions; overlapping interiors are acceptable for contact only if they do not change exposed surfaces. Preserve planar continuity. Control approximation around mating surfaces to below 10% of minimum functional clearance. Inspect the actual union in section and with collision-only replay.

## 3. Describe the geometry

The importer takes JSON, structurally illustrated below. Values shown are **illustrative synthetic dimensions**, not verified dimensions of your model. Supply your own measured data. All metadata is SI even when STL vertices are mm. `mesh_to_body.pos` is in meters; mesh vertices are scaled before this transform.

```json
{
  "schema_version": 1,
  "name": "goat_actual",
  "source": {"kind": "onshape_manual", "revision": "YOUR_IMMUTABLE_VERSION", "document_url": "YOUR_URL"},
  "mesh_units": "mm",
  "mesh_to_body": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]},
  "expected_extents": [0.113564, 0.04, 0.11],
  "clearance": 0.00025,
  "max_penetration": 0.000025,
  "inertial": {
    "mass": 0.1,
    "com": [0.016, 0, 0.04],
    "matrix": [[0.00003, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]],
    "provenance": "REPLACE with measured/CAD source"
  },
  "target": {"pos": [0, 0, 0.04], "quat": [1, 0, 0, 0]},
  "visual": [{"name": "body", "file": "body.stl"}, {"name": "legs", "file": "legs.stl"}],
  "collision": [{"name": "ramp", "file": "collision/ramp.stl"}],
  "seating_pairs": [],
  "sockets": [],
  "legs": []
}
```

The abbreviated arrays intentionally fail validation until completed. List **all** collision pieces. `expected_extents` is the XYZ bounding size of the complete block including legs and guides, measured independently in CAD; this catches wrong scaling. `clearance` is the minimum radial/functional clearance, not diameter difference. Use a penetration limit no greater than 20% of clearance (10% recommended initially).

`inertial.matrix` is the inertia about the COM, expressed in block axes, in kg m²; off-diagonal entries are actual tensor entries, not unsigned products of inertia. For an assembled rigid body combine component inertias using the parallel-axis theorem. Replace homogeneous CAD mass assumptions with print measurements before calibration.

Each seating pair has local coordinates `{"upper": [x,y,z], "lower": [x,y,z], "normal": [0,0,1]}`. These points must coincide under the target transform. Provide at least three noncollinear points per intended seating face. Normals are expressed in the lower block frame and point outward.

Each of the two `sockets` has:

```json
{"mouth": [-0.02,0,0.04], "quat": [1,0,0,0], "depth": 0.034,
 "lead_depth": 0.002598, "throat_radius": 0.005, "opening_radius": 0.0065}
```

The socket frame's +Z points out of the mouth; the floor is at local `z=-depth`. Radius interpolates linearly through the lead-in and is constant below it. The order of sockets corresponds to the order of legs.

Each leg has `tip`, `target_depth`, and `surface_probes`, all in upper-block coordinates except the scalar target insertion depth. Include the tip and circumferential samples at the root, middle of the shaft, and tip transition. At least 12 probes are required; 16 or more points per ring is recommended. Example probe generation for a Z-aligned shaft in Python:

```python
angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
probes = [[cx, cy, tip_z]]
for z in [root_z, midshaft_z, tip_transition_z]:
    probes.extend([[cx + radius*np.cos(a), cy + radius*np.sin(a), z] for a in angles])
```

The prepared demo bundle provides a full, machine-readable example of all these arrays. Visual and collision meshes are embedded in prepared bundles with content hashes, making runs independent of later source file changes.

## 4. Validate before collecting success rates

`aerial validate` checks target interference, an 81-pose aligned insertion path, and tiny-sphere socket probes. Probes test bore/lead empty space, surrounding walls, and blind floors. These are discrete checks; they do not certify all facets or tilted trajectories.

Run an aligned drop, a seated-at-rest check (covered by the Python tests), an intentional miss, and an offset sweep. Use `converge` and a finer collision partition. Do not loosen success thresholds to accommodate a collider that blocks insertion. Keep the target, material model, tolerances, and release envelope fixed during candidate comparisons.
