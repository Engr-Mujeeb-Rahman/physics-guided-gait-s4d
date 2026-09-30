# RULES — Constraints for Any AI Coding Agent Working on This Project

**Status:** LOCKED. This file governs what an AI coding agent (or any collaborator) is and is not
allowed to do on this project without Mujeeb's explicit, in-the-moment authorization.

## 1. Protected Documents

`prd.md`, `design.md`, `architecture.md`, `phases.md`, and this file (`rules.md`) are **locked scope
documents**. Do not modify their content — including "helpful" additions, scope expansions, or
architecture changes — without explicit instruction in that specific instance. This includes:
- Do not reintroduce raw ultrasound processing.
- Do not reintroduce a live differentiable multibody/muscle simulator.
- Do not reintroduce the 8-discrete-cadence framing.
- Do not silently swap the S4D backbone for `mamba-ssm` or any other architecture.
- Do not add new datasets beyond the primary (six-speed) and secondary (Fukuchi) ones without
  explicit approval.

`memory.md` is the one file in this set that IS expected to be updated regularly — it's the running
log, not a locked spec (see memory.md itself).

## 2. Allowed Files / Folders

- Full read/write access to `src/`, `data/processed/`, `data/opensim_processed/`, `experiments/`,
  `notebooks/`, `paper/`.
- `data/raw/` is read-only once populated — never modify or delete raw downloaded data.

## 3. Forbidden Actions (no exceptions without explicit real-time authorization)

- **Git:** never run `git commit`, `git push`, `git reset`, `git rebase`, or any history-altering
  command, and never run `git status`/`git diff` either, unless explicitly authorized in that specific
  message. This applies every time — prior authorization does not carry forward to future sessions.
- **Dependencies:** do not add, remove, or upgrade any package (PyTorch, OpenSim, ONNX, etc.) without
  explicit approval. If a task seems to require a new dependency, stop and ask.
- **Architecture:** do not change the model architecture, loss formulation, dataset choice, or
  conditioning mechanism from what's specified in design.md without explicit approval — this includes
  "small" changes like swapping FiLM for a different conditioning scheme.
- **Deletion:** never delete files, especially anything under `data/raw/`, `experiments/`, or `paper/`,
  without explicit confirmation for that specific deletion.
- **APIs/interfaces:** do not change function signatures or module interfaces defined in architecture.md
  without flagging the change and why first.

## 4. Working Style Expectations

- Match the structure in prd.md/design.md/architecture.md/phases.md exactly — file names, module
  responsibilities, and phase boundaries are fixed points, not suggestions to reinterpret.
- When a phases.md checkpoint fails, apply the documented fallback. Do not propose a novel fix that
  isn't in the plan without flagging it explicitly as a deviation and why.
- Never claim a result (accuracy number, latency measurement, convergence) without it being backed by
  an actual run logged under `experiments/`. No estimated numbers presented as measured ones — this
  applies especially to the embedded hardware latency requirement (see architecture.md §5).
- If scope pressure builds and something needs to be cut beyond what phases.md already accounts for,
  stop and surface the tradeoff explicitly rather than quietly dropping a PRD success criterion.

## 5. Escalation

If a task cannot be completed within these constraints (e.g., a checkpoint fails and no documented
fallback applies, or an action requires touching a protected file/action), stop and ask rather than
improvising a workaround.
