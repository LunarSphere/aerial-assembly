# Importing onshape-to-robot exports

For the supplied GOAT block, use the automated [CAD iteration workflow](cad-iteration.md)
and `aerial cad-drop`. It locally assembles the parts, estimates provisional mass,
partitions the collider and measures features without a manual metadata file.
The instructions below describe the stricter general-purpose importer.

Use `onshape-to-robot` to export one complete rigid block to MuJoCo, then use
`aerial prepare-mjcf` to turn it into our portable geometry bundle. The existing
experiment builder makes a fixed lower copy and a freely falling upper copy.
Exported joints, actuators, scene floors, contact defaults and solver settings
do not become the experiment; the existing experiment configuration controls
the physics. No Onshape credentials are needed to import local files.

## The supplied export

`export_example/robot.xml` and its meshes are readable, but are not ready for
physical experiments:

- `part_1`, `part_2`, and `part_3` each have their own free joint. A block needs
  one rigid root containing the two legs and the body.
- Every part has mass `1e-9` kg and diagonal inertia `1e-9` kg m², which are
  placeholders, not usable CAD mass properties.
- The main body collision mesh is concave. MuJoCo would use its convex hull,
  closing recesses. The two leg collision meshes are convex.
- All six material colors contain NaN values.

Run the inspector to reproduce this assessment (exit code 2 means issues):

```bash
source .venv/bin/activate
aerial inspect-mjcf export_example/robot.xml --out runs/export-inspection.json
```

The importer never opens `robot.pkl`. Supplied exports are not modified.

## Export setup

1. In a simulation assembly, rigidly connect the body and legs using the
   exporter's `fix_` mate naming convention. Export a single block, with one
   root body and either no joint or one root free joint. Internal moving joints
   and multiple roots are rejected.
2. Assign appropriate materials/density in CAD and confirm the mass properties
   of the complete block. Export with dynamics enabled (the default). Check
   the resulting mass against the printed block; CAD material density alone
   does not account for infill.
3. Use a stable block frame: Z up, X along the ramp, Y along extrusion depth.
   Add `frame_socket_1`, `frame_socket_2`, `frame_tip_1`, etc. mate connectors
   if you want exported sites for feature metadata. Inspect the exported site
   names before referencing them; the supplied example has no sites.
4. Copy [the export configuration](../examples/onshape-to-robot/config.json)
   into a new export directory. It starts with your assembly URL; update it to
   the simulation assembly and preferably an immutable CAD version. It disables
   STL merging and simplification, enables CoACD decomposition, and supplies a
   finite color. It does not fix missing mass or assembly mates.
5. Install `onshape-to-robot` in your exporter environment and configure its
   Onshape API credentials there. Run `onshape-to-robot PATH_TO_EXPORT_DIRECTORY`.
   Preserve the exporter version and configuration with each candidate. Follow
   the upstream installation instructions for any CoACD dependencies.
6. Run `aerial inspect-mjcf PATH_TO_EXPORT_DIRECTORY/robot.xml`. Prefer
   `robot.xml` to the demonstration `scene.xml`. Inspect the decomposed socket
   surfaces visually; convex decomposition alone does not guarantee accurate
   clearance. If necessary, replace collision geometry with explicit convex
   CAD pieces, preserving the original physical mass/inertia.

References: [design conventions](https://onshape-to-robot.readthedocs.io/en/latest/design.html),
[CoACD processor](https://onshape-to-robot.readthedocs.io/en/latest/processor_convex_decomposition.html),
[configuration](https://onshape-to-robot.readthedocs.io/en/latest/config.html),
[installation](https://onshape-to-robot.readthedocs.io/en/latest/).

## Seating metadata and preparation

Copy [metadata.template.json](../examples/onshape-to-robot/metadata.template.json)
outside the example directory and replace every null and placeholder. The
template deliberately contains no guessed mating dimensions. Follow the
socket, probe, seating and tolerance definitions in [geometry.md](geometry.md).
Provide at least three noncollinear seating points per intended seating face
and shaft/root/tip probes for both legs. All lengths are meters.

Coordinates are relative to the exported root body frame at its default pose;
the root's world position and orientation are removed during import. The target
is the upper block's seated transform relative to the lower block in that frame.
`expected_extents` must be measured independently in CAD in the same axes.

Any metadata position can reference an exported site as `{"site": "socket_1"}`.
A quaternion can use `{"site": "socket_1", "field": "quat"}`. Numeric arrays
are equally supported; sockets need their +Z axis pointing out of the mouth.
Sites specify frames, not dimensions, so socket radii, depth, probes and target
pose still need to be supplied. Mass, COM and inertia are imported automatically
and must not be copied into the metadata.

```bash
aerial prepare-mjcf assets/goat-export/robot.xml \
  --metadata assets/goat-metadata.json --out assets/goat.json
aerial validate assets/goat.json --out runs/goat-validation.json
# Proceed only after validation passes and collision surfaces are inspected.
aerial drop assets/goat.json --out runs/goat-first-drop
aerial replay runs/goat-first-drop --collisions
```

An optional `--body NAME` asserts the expected root name; it does not select or
silently discard pieces from a multi-root assembly. Meshes, boxes and cylinders
are supported. Other geom types fail explicitly. Collision surfaces must be
closed convex solids; the importer does not silently convexify a concave mesh.
Export colors are checked but the experiment uses its own lower/upper colors.

MuJoCo resolves includes, defaults, STL paths, mesh scale and mesh principal-axis
offsets. The importer transforms geometry and sites into the block frame and
combines rigid child inertias using the parallel-axis theorem. Prepared bundles
embed geometry and sites, include the top-level XML hash and CAD revision, and
hash the entire result. Later edits to source meshes cannot change a saved run.
The XML hash alone is not a hash of included files; the bundle hash covers the
resulting imported geometry and mass properties.

`inspect-mjcf` checks export structure, not whether a block can seat. Geometry
validation, aligned drop checks and convergence remain necessary. The legacy
`aerial prepare` STL route and Variable Studio adapter remain available.
