# Aerial assembly

**Research question:** can a small, imprecise drone build structures if the
bricks do the precise work? We design interlocking bricks whose geometry funnels
a coarsely dropped brick into its seat (gravity does the final alignment), and a
passive gripper that a Crazyflie can pick up and release without actuators.

## Experiments

- **Experiment 1: drop-assembled bricks** ([`brixzle/`](brixzle/README.md)).
  A parametric 2.5D chevron brick in CadQuery, tested by MuJoCo drop and
  sequential-assembly trials under a sampled placement-error model, and tuned
  with NSGA-III. The drone is not modelled; each brick is a free body released
  above its seat.
- **Experiment 2: drone pick-and-place with a passive gripper**
  ([`brixzle/`](brixzle/README.md) and [`flight/`](flight/README.md)). A
  simulated Crazyflie 2.1 Brushless with a passive fork flies `plan.json`
  missions in the brixzle MuJoCo scene. The same mission code drives the real
  drone through cflib. The real setup calibrates Lighthouse positioning against
  an overhead RealSense D455 using AprilTags, then flies supervised hover and
  tag-visit checks. Decisions and results so far are in
  [`brixzle/docs/exp2/STATUS.md`](brixzle/docs/exp2/STATUS.md).

Results from simulation use uncalibrated contact and aerodynamics. They are not
hardware success rates.

## Repo map

| Path | What |
|---|---|
| [`brixzle/`](brixzle/README.md) | Brick geometry, MuJoCo trials, NSGA-III search, simulated drone and `fly-assemble` planner |
| [`flight/`](flight/README.md) | cflib flight program (`fly`) and Experiment 2 calibration (`flight-cal`) |
| [`stl_exports/`](stl_exports/) | CAD meshes, including the A/B bricks and the printable fork gripper |
| [`.claude/`](.claude/REPO_MAP.md) | Module map and Claude Code project skills |
| `.github/` | CI (pytest for `flight` and `brixzle`) and Dependabot |

Run outputs (`runs/`), generated assets, model weights and videos are
git-ignored and stay local.

## Getting started

```bash
cd brixzle && uv sync --all-extras && uv run pytest
cd .. && uv run --project flight --extra test pytest flight/tests -q
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR. Claude Code
follows [CLAUDE.md](CLAUDE.md).
