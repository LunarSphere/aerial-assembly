---
name: pick-up-issue
description: Pick up GitHub issue N in this repo. Reads the issue, branches from master as <N>-short-name, writes a failing test first, implements it, runs the tests and opens a PR with "Closes #N". Use when asked to "pick up", "work on" or "solve" issue N.
argument-hint: <issue number>
---

Pick up issue `$ARGUMENTS` following the workflow in `CLAUDE.md`.

1. **Read the issue.** Run `gh issue view $ARGUMENTS --comments`. Restate the goal and the acceptance criteria in two or three lines. If they are vague or the issue is an open-ended design question, stop and suggest `/grill-me` first. Don't guess at requirements.
2. **Branch.** Run `git fetch origin`, then `git switch -c $ARGUMENTS-<short-kebab-name> origin/master`. If the working tree has uncommitted changes, ask before switching. If a branch for this issue already exists, ask whether to reuse it.
3. **Find the code.** Use `.claude/REPO_MAP.md` to find the modules and their tests. Read them before you change anything.
4. **Write a failing test.** Add or extend a test in the project's `tests/`. Run only that test and confirm it fails for the expected reason. Show the failure.
5. **Make it pass.** Write the smallest change that passes, then clean up. Match the surrounding style. Follow the conventions in `CLAUDE.md`: SI units, `[w,x,y,z]` quaternions, a new `--out` per run.
6. **Run the tests** for each project you touched:
   - `cd brixzle && uv run pytest -q`
   - `uv run --project flight --extra test pytest flight/tests -q` (from the repo root)
   Report failures with their output. Don't open the PR while tests fail unless the user says to.
7. **Housekeeping.** Update `.claude/REPO_MAP.md` if modules moved or changed role. If an Experiment 2 decision changed, update the decision table in `brixzle/docs/exp2/STATUS.md`.
8. **Commit and open the PR.** Use imperative commit subjects. Push with `git push -u origin HEAD`, then `gh pr create --base master`. The body has: a summary, `Closes #$ARGUMENTS`, the tests run, and evidence for any number (run dir, config, seed). Label simulated results "simulated, uncalibrated contact". Ask the maintainer to review it. Merge only when they ask, once CI is green.

Never arm the drone, open a radio link, or run `--arm` or `--check-only` along the way. If the issue needs a real flight, hand the human the exact command.
