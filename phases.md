# PHASES — 3-Week Execution Plan

**Status:** LOCKED (see rules.md). Every checkpoint below has a fallback — if a checkpoint fails, take
the stated fallback immediately. Do not spend extra days trying to force the original approach to work;
that is exactly the over-scoping failure mode this plan exists to prevent.

## Phase 0 — Environment & Data Sanity (Day 0–1)

- Set up repo structure (architecture.md), Python env, OpenSim install, PyTorch (CUDA + CPU fallback).
- Download primary dataset (six-speed) and Fukuchi dataset.
- Run OpenSim Muscle Analysis on exactly ONE subject/trial from the primary dataset as a pipeline test.

**Checkpoint (end of Day 1):** Does the Muscle Analysis output produce sane, non-degenerate l_m/v_m
values for that one trial?
- **Pass →** proceed to Phase 1 with muscle-tendon channel included.
- **Fail →** invoke design.md §2 fallback: drop the muscle-tendon channel entirely, proceed to Phase 1
  with kinematics + GRF + speed only. Do not spend more than Day 0–1 on this — this decision must be
  made by end of Day 1, no exceptions.

## Phase 1 — Data Pipeline (Day 2–5)

- Extend Muscle Analysis pipeline to full primary dataset (all subjects/trials, if channel is in scope).
- Build windowing, normalization, subject-wise train/val/test split.
- Build leave-one-speed-out split generator.
- Confirm Fukuchi dataset loader produces compatible feature format for zero-shot cross-dataset testing.

**Checkpoint (end of Day 5):** Can you load a batch from the primary dataset's train split and produce
correctly shaped tensors end-to-end? If not, this blocks everything downstream — do not start model
work until this is true.

## Phase 2 — Backbone + Baseline (Day 5–9)

- Implement S4D layer (pure PyTorch) and the plain baseline model (no conditioning, no physics loss).
- Overfit the baseline on a small subset (2–3 subjects) to confirm the training loop is correct before
  scaling to the full dataset.
- Train baseline on full primary train split.

**Checkpoint (end of Day 9):** Does the baseline converge and produce reasonable joint-moment
predictions (visually plausible, roughly correct magnitude/timing vs. ground truth) on held-out
subjects?
- **Fail →** debug the backbone/training loop itself. Do NOT proceed to add FiLM conditioning or
  physics loss on top of a broken foundation — that only compounds the debugging surface. If not fixed
  by Day 10, reduce model size (d_model down, fewer layers) before trying anything else.

## Phase 3 — Conditioning + Physics Loss (Day 9–13)

- Add FiLM speed conditioning; train the "+conditioning" ablation config.
- Implement jerk penalty and energy-consistency residual losses; train the full "+conditioning+physics"
  config with a loss-weight ramp-up schedule.
- (Stretch, only if ahead of schedule) implement the OpenSim precomputed M/C/G residual term.

**Checkpoint (end of Day 13):** Does the full model train stably (loss doesn't diverge/NaN)?
- **Fail (instability) →** reduce physics-loss weight via the fixed ramp-up schedule (design.md §8).
  Do NOT redesign the loss function itself under time pressure — a working, lower-weighted physics
  term beats a broken more-ambitious one.

## Phase 4 — Generalization Evaluation (Day 13–16)

- Run leave-one-speed-out evaluation for all three ablation configs on the primary dataset.
- Run cross-dataset zero-shot evaluation (train on primary, test on Fukuchi, no fine-tuning) for all
  three configs.
- Produce the core generalization results table (PRD success criteria #2, #3).

**Checkpoint (end of Day 16):** Do you have a complete three-way ablation × two generalization tests
result table? This table is the paper's central evidence — if any cell is missing, resolve it before
moving to Phase 5, since Phase 5's writing depends on having final numbers.

## Phase 5 — Physical Consistency + Hardware Benchmark (Day 16–18)

- Compute jerk and energy-consistency-residual metrics for all three ablation configs on test data
  (PRD success criteria #4).
- Export final model to ONNX.
- Benchmark on Raspberry Pi 5 if acquired; otherwise run the CPU-profiled fallback and label it as an
  estimate (architecture.md §5).

**Checkpoint (end of Day 18):** All four PRD success-criteria experiments (accuracy, generalization,
physical consistency, latency) have final numbers. If not, this is the last point to cut scope — cut
the OpenSim M/C/G stretch term or reduce ablation configs to two (baseline vs. full model) before
cutting any of the four core PRD experiments themselves.

## Phase 6 — Writing (Day 18–21)

- Confirm J-BHI manuscript template/format before writing (do this on Day 18, not Day 20).
- Draft: Intro → Related Work → Methods (explicitly describing the physics-guidance-via-auxiliary-loss
  design choice and why, per design.md §4 — reviewers respond far better to an honest, well-justified
  simplification than to an unsupported claim of full differentiable simulation) → Experiments →
  Results → Discussion → Limitations (state plainly: speed conditions not discrete cadences; auxiliary
  physics loss not a live differentiable simulator; single embedded device or estimate).
- Internal review pass with fresh eyes (or ask for one) before submission — reserve at least half a day
  for this, do not submit the first complete draft untouched.

## Standing Rule Across All Phases

If a checkpoint fails and the fallback is unclear, stop and re-read prd.md's Non-Goals and design.md's
"explicitly out of scope" section before inventing a new fix. The plan already accounts for the likely
failure modes — most "new ideas" under deadline pressure are re-introductions of scope that was cut for
a reason.
