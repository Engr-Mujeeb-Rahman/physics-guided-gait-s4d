# PRD — Physics-Guided Neural State-Space Model for Generalizable Gait Kinetics

**Status:** LOCKED (see rules.md — do not modify scope without explicit sign-off)
**Target venue:** IEEE J-BHI Special Issue — "Emerging AI Paradigms for Next-Gen Medicine: Safety, Reliability, and Digital Twins"
**Deadline:** 30 September 2026
**Available time:** 3 weeks from project start

## 1. Problem Statement

Black-box deep learning models that predict joint kinetics (torques/moments) from kinematics
generalize poorly across walking speeds/cadences and can produce physically implausible outputs
(force spikes, energy-conservation violations, non-smooth torque trajectories). This makes them
unsafe for downstream use in robotic exoskeletons and smart prostheses, which require:
1. Generalization to speeds/cadences not seen during training (patients don't walk at fixed speed).
2. Physically consistent outputs (no torque spikes that would translate into unsafe actuator commands).
3. Real-time-plausible inference on embedded hardware (exoskeletons/prostheses run on embedded compute,
   not GPUs).

## 2. Goal

Build and validate a sequence model for joint-kinetics prediction that:
- Uses a state-space backbone (S4-family) instead of an LSTM/Transformer, conditioned on gait speed.
- Is constrained by physics-derived auxiliary signals (not a free-form black box).
- Demonstrably generalizes zero-shot to unseen speeds and to a completely different dataset/lab.
- Is measurably more physically consistent (lower jerk, lower energy-conservation residual) than an
  equivalent black-box model trained on the same data.
- Has a real, measured (not estimated) inference-latency number on at least one embedded device.

## 3. Non-Goals (explicit — do not attempt these; this is what makes the 3-week timeline real)

- **Not** collecting new ultrasound data. No IRB, no new data acquisition of any kind.
- **Not** building a live differentiable rigid-body + Hill-type muscle-tendon simulator inside the
  training loop. This is a multi-month systems problem and is out of scope entirely.
- **Not** claiming 8 discrete metronome cadence conditions (60–130 bpm) — no such public dataset
  exists combined with kinetics. We use continuous walking-speed conditions from existing public
  datasets and state this honestly in the paper.
- **Not** guaranteeing benchmarking on both Jetson Orin Nano AND Raspberry Pi 5 — one real embedded
  device is the target; a second device is a stretch goal only, never a blocking requirement.

## 4. Success Criteria (what "done" means for submission)

A submittable draft requires ALL of:
1. Model trained and evaluated on the primary dataset (see design.md) with reported joint-moment
   prediction accuracy (MAE/RMSE) comparable to or better than baseline.
2. Zero-shot leave-one-speed-out results showing the physics-guided model degrades less than the
   no-physics baseline as test speed moves away from training speeds.
3. Zero-shot cross-dataset generalization result (train on dataset A, test on dataset B, no fine-tuning).
4. Ablation table: baseline (no conditioning, no physics loss) vs. +conditioning vs. +conditioning+physics —
   showing the physics loss term measurably reduces jerk and energy-consistency violation metrics.
5. At least one real embedded-hardware latency measurement (not simulated), clearly labeled as such.
6. Full written manuscript sections: Intro, Related Work, Methods, Experiments, Results, Discussion,
   Limitations, formatted for J-BHI.

## 5. Deliverables

- Codebase (see architecture.md) — data pipeline, model, training, evaluation, ONNX export.
- Trained model checkpoints for baseline and full model.
- Results tables/figures for the four experiments above.
- Manuscript draft (LaTeX or Word, per J-BHI author guidelines — confirm template before Week 3).

## 6. Risks (see phases.md for fallback triggers tied to each)

- OpenSim musculotendon extraction pipeline fails to produce sane values → fallback: drop muscle-tendon
  channel, keep kinematics+GRF+speed only. Still physics-guided via jerk/energy loss.
- S4-family backbone fails to converge in pure PyTorch on available hardware → fallback: reduce model
  size / sequence length before switching architectures; do not switch to a new architecture family
  mid-timeline.
- No embedded hardware available in time → fallback: CPU-profiled latency estimate, explicitly labeled
  as an estimate, with real hardware validation named as future work in the paper.
- Physics loss destabilizes training (common failure mode for physics-informed losses) → fallback: reduce
  physics-loss weight via a fixed schedule; do not redesign the loss function under time pressure.
