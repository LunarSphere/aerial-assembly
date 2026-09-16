# Optional Onshape integration

For geometry export, prefer the [onshape-to-robot workflow](onshape-to-robot.md),
which carries assembly transforms, mass properties and reference sites into
MuJoCo. The adapter below remains available for Variable Studio changes and
the legacy per-part STL route.

Manual export is sufficient for the initial experiment. The adapter is optional and never contacted by `drop`, `batch`, or `compare`. No live API credentials or CAD exports were supplied during implementation; transport behavior is tested with simulated responses only.

Copy `examples/onshape.template.json` and replace its IDs and exact part names. Use a dedicated experiment workspace. Keep credentials in `ONSHAPE_ACCESS_KEY` and `ONSHAPE_SECRET_KEY` environment variables, outside the configuration and repository. API requests use HMAC signatures with fresh nonces and dates.

```bash
# Read-only: resolve named parts, export at one microversion, cache locally.
aerial onshape-export my-onshape.json --cache assets/onshape

# changes.json example: {"ramp_angle": "45 deg", "socket_lead_angle": "25 deg"}
# Preview preserves other variables and derived expressions.
aerial onshape-variables my-onshape.json changes.json

# Explicit write to the configured experiment workspace, with before/after audit.
aerial onshape-variables my-onshape.json changes.json --apply --audit runs/candidate-01-variable-change
```

Only names in `independent_variables` may change. The adapter refuses to replace a `#`-referencing expression or a configured expression. Read/modify/write preserves unrelated definitions, descriptions, and configured fields. Do not list derived widths or depths as independent controls. No workspace restoration is performed automatically; the audit retains previous definitions for deliberate recovery.

The export cache key includes the configuration and current document microversion. Each named part is resolved again at that immutable microversion, avoiding stale part IDs. STL export settings and hashes are recorded. Existing cache content is verified before reuse. Exports use meters; set `mesh_units` accordingly when preparing them.

The variable read is bracketed by microversion checks. Variable writes also check that the workspace did not change during preparation. These checks detect many races but are not a transactional lock: keep one writer and avoid simultaneous browser edits in the experiment workspace. Serialize CAD candidate generation; parallelism, if later needed, belongs in local simulations after export.

HTTP 307/308 exports are followed with freshly signed headers. Credentials are only sent to the configured Onshape host or `cad.onshape.com`; an unexpected host stops with a diagnostic. HTTP 429 retries use bounded backoff. HTTP 402 stops on quota exhaustion. Audit/cache outputs contain geometry and variables, never authentication headers.

## Connecting a candidate search

1. Establish a flush baseline in CAD, including simulation collision parts driven by the same independent variables.
2. Select a small grid, initially `ramp_angle` and `socket_lead_angle` at baseline ±5°. Confirm the angle conventions from the CAD sketches; the synthetic demo's lead angle is measured from vertical and is not assumed to match your variable.
3. Apply one candidate to the experiment workspace and read back its evaluated variables. Check feature regeneration in CAD.
4. Export the immutable snapshot and supply current target, feature metadata, extents, and mass properties in a local manifest. These data must come from the same revision. The adapter deliberately does not reuse stale baseline mating metadata when geometry changes.
5. Prepare and validate the candidate. Reject geometry with impossible flush seating, blocked sockets, or invalid regeneration before spending a drop-test budget.
6. Use `aerial compare` on validated bundles. It uses common release samples and excludes numerically invalid candidates. Run finalists on independent release seeds and with numerical/mesh refinement.

There is no unattended CAD-to-optimization loop yet: automatically extracting the design-specific mating frames, dimensions, regeneration status, and print mass distribution requires your real model. The public `OnshapeClient.update_variables`, `OnshapeClient.snapshot`, `prepare_geometry`, and `compare_candidates` interfaces are the connection points. Full CAD automation should be added only after this explicit pipeline is verified on your document.

References: [Variables API](https://raw.githubusercontent.com/onshape-public/go-client/master/onshape/docs/VariablesApi.md), [export API](https://onshape-public.github.io/docs/api-adv/translation/), [configuration API](https://onshape-public.github.io/docs/api-adv/configs/), [authentication](https://onshape-public.github.io/docs/auth/apikeys/), [API limits](https://onshape-public.github.io/docs/auth/limits/).
