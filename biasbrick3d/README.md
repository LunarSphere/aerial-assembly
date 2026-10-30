# biasbrick3d: corner-hopper bricks (the 3D version of #45)

The first version of the aim-bias idea in issue #45, kept so the 3D and 2.5D designs can be compared.
The 2.5D design that replaced it is [`../biasbrick/`](../biasbrick/README.md).

A DF-style rail carries three **corner hoppers**: each voxel slopes from its +x+y side into a seat in its
-x-y corner. The drone aims diagonally off the seat by the error plus a yaw allowance, so every landing slides
one way into the corner. Two nesting-cup feet end in teeth leaning 38° into slanted slots (brixzle's hook).
Shelled square fork cavities with a V roof hang under the inner ridges.

```bash
cd biasbrick3d
uv sync --all-extras
uv run pytest          # about 17 s
uv run biasbrick3d export --params configs/bb-mean.json --out ../runs/bb3d-export
uv run biasbrick3d animate --params configs/bb-mean.json --out ../runs/bb3d-anim     # geometry only
```

Status, restored as it was at commit `23cb2f6`:

- **Geometry, rules and a kinematic stacking animation only.** No MuJoCo trials were run on it.
- **Too heavy.** Sized for the 14.8 mm mean error it is 38.7 g, against a 21.6 g budget. Each ramp must climb
  across the whole voxel, about 31 mm high, and the cup feet copy it.
- **Rule margins:** every rule passes except mass. The y catchment is tight (0.1 mm margin).
