# DESIGN — Physics-Guided SSM for Gait Kinetics

**Status:** LOCKED (see rules.md)

This document records the *why* behind every major decision. Each section states the decision, the
alternative that was rejected, and the reason — so nothing here reads as arbitrary later.

## 1. Datasets

**Primary (training/main results): six-speed treadmill dataset (2026)** — 45 subjects, 6 fixed
speeds (0.6–1.6 m/s), full 3D kinematics, GRF, joint moments/powers from inverse dynamics, already
processed through OpenSim 4.1 with the Rajagopal musculoskeletal model.
*Why:* the Rajagopal-model processing is already done — this is what makes musculotendon length/velocity
extraction (l_m, v_m) tractable in days rather than weeks, since we query an already-scaled model
instead of building a scaling+IK+muscle-analysis pipeline from raw markers ourselves.

**Secondary (cross-dataset zero-shot generalization only): Fukuchi, Fukuchi & Duarte (2018), PeerJ**
— 42 subjects, treadmill + overground, different speed range, different lab, different marker set.
*Why:* training on dataset A and testing zero-shot on dataset B (different lab, different equipment,
different subjects) is a much stronger generalization claim than held-out speeds from the same
recording session. This directly strengthens the "generalizes across arbitrary walking speeds" claim
in the PRD and is cheap to add since the dataset is public and speed-labeled.

**Rejected:** any dataset combining ultrasound + kinetics + discrete metronome cadences. Confirmed
not to exist publicly (see project history). Rebuilt the entire proposal around this constraint rather
than pretending it doesn't exist.

## 2. Muscle state (replacing "ultrasound calibration")

**Decision:** extract musculotendon length (l_m) and velocity (v_m) via OpenSim's Muscle Analysis
tool, run on the pre-scaled Rajagopal model already provided with the primary dataset, using measured
joint angles as input.
*Why not ultrasound:* no ultrasound data exists for us to use; acquiring it means a new human-subjects
study, which is out of scope (see prd.md Non-Goals).
*Fallback if this pipeline doesn't produce sane values in Phase 0:* drop the muscle-tendon channel
entirely. The model still qualifies as physics-guided via the jerk/energy-consistency loss (Section 4).
Do not spend more than the Phase 0 budget (see phases.md) trying to fix this — cut and move on.

## 3. Sequence backbone: S4D (Diagonal State Space), pure PyTorch — not `mamba-ssm`

**Decision:** implement a diagonal state-space (S4D) layer directly in PyTorch, no custom CUDA kernels.
**Rejected:** the `mamba-ssm` package (selective-scan Mamba). It depends on fused custom CUDA kernels
that are a real risk of multi-day environment/build failures on a GTX 1660 (6GB, Turing architecture) —
exactly the kind of failure that would eat a large fraction of a 3-week budget with nothing to show for
it. S4D gets ~90% of the modeling benefit (long-range dependencies, efficient sequence modeling) with
none of the build risk.
*If time remains after Phase 4:* swapping in `mamba-ssm` is a valid stretch goal, not a requirement.

**Model size guidance for 6GB VRAM:** start at d_model=64–128, 4 layers. Scale up only if GPU memory
and epoch time allow; do not scale up before the baseline (Phase 2) is confirmed converging.

## 4. "Physics-guided" mechanism: precomputed OpenSim residuals, not a live differentiable simulator

**Decision:** physics guidance is implemented as an **auxiliary loss term**, not as a differentiable
multibody dynamics layer inside the forward pass. Specifically:
- **Jerk penalty:** second time-derivative of predicted joint torque, penalized directly (encourages
  smooth, physiologically plausible torque trajectories).
- **Energy-consistency residual:** mechanical power implied by predicted torque × measured angular
  velocity is compared against the power reconstructed from measured joint kinematics/GRF (via
  standard, already-computed inverse dynamics from the dataset) — the residual is penalized.
- **(Optional, if time allows in Phase 3) Multibody consistency residual:** for a sampled subset of
  frames, query OpenSim's own dynamics engine (not reimplemented) for M(q), C(q,q̇), G(q) at the
  predicted joint state, offline/precomputed — not backpropagated through — and penalize deviation
  from the equations-of-motion identity. This gives physics grounding without requiring OpenSim itself
  to be differentiable.

**Why this instead of a live differentiable simulator:** the original proposal's differentiable
rigid-body + Hill-type muscle dynamics layer is a research problem on its own (see prd.md Non-Goals).
Precomputed/auxiliary physics residuals are the standard, tractable version of "physics-informed" deep
learning and are defensible to reviewers as long as the paper is explicit about the distinction — which
it will be, in the Methods and Limitations sections.

## 5. Speed conditioning: FiLM, not a bespoke "neural compliance module"

**Decision:** a small 2-layer MLP maps normalized walking speed to per-layer scale/shift (γ, β)
parameters (FiLM-style modulation) applied to the S4D backbone's hidden state.
*Why not a bespoke "compliance module":* an unnamed, undocumented new biomechanical mechanism is a
reviewer target — it invites the question "what is this, why does it work, where's the ablation
proving the specific form matters?" FiLM is an established, well-understood conditioning technique;
using it lets the paper focus its novelty claim on the physics-guidance + generalization result, not
on defending an invented architectural component.

## 6. Baselines and ablations

1. **Baseline:** S4D backbone, no speed conditioning, no physics loss — plain supervised regression
   on joint moments.
2. **+Conditioning:** baseline + FiLM speed conditioning, no physics loss.
3. **+Conditioning+Physics (full model):** baseline + FiLM + jerk/energy-consistency loss terms.

This three-way ablation is required — it isolates whether generalization gains come from conditioning
alone or from the physics constraint, and it isolates whether physical-consistency gains come from the
physics loss specifically. Skipping this ablation removes the evidence behind the paper's core claim.

## 7. Metrics

- **Accuracy:** MAE/RMSE on predicted joint moments (hip, knee, ankle — sagittal plane at minimum).
- **Generalization:** MAE/RMSE under leave-one-speed-out (within-dataset) and cross-dataset zero-shot
  (train on primary, test on Fukuchi).
- **Physical consistency:** mean jerk magnitude; energy-consistency residual (as defined in Section 4).
- **Deployment:** inference latency (ms/window) and parameter count/FLOPs. Measured via host x86 CPU profiling under the architecture.md §5 fallback convention (labeled latency_estimate=True). Physical validation on real embedded hardware (e.g., ARM Cortex / Raspberry Pi 5 / Jetson) is documented as a limitation and future work item.

## 8. Explicitly out of scope for this design (do not re-litigate — see rules.md)

- Raw ultrasound processing.
- Live differentiable multibody/muscle simulation.
- 8-condition discrete metronome cadence framing.
- Mandatory dual-device (Jetson + Pi 5) hardware benchmarking.

## 9. Modeling Limitations and Simplifications

- **Segment-Ratio Model Scaling:** OpenSim musculoskeletal model scaling uses direct segment-length-ratio scaling (via `ScaleSet` and `model.scale()`) based on static calibration marker distances (pelvis width, femur length, tibia length, foot length, torso height). The full iterative least-squares Marker Placer / IK-based marker adjustment step was skipped in favor of direct anatomical segment ratio scaling.
- **Dynamic GRF Body Mass Derivation:** Subject body mass is derived for all 44 primary dataset subjects directly from steady-state mean vertical ground reaction force averaged across complete gait cycles in actual walking trials ($M = \frac{1}{g T} \int_0^T F_z(t) dt$). For periodic steady-state gait, mean vertical GRF equals total body weight. This method is validated against the 21 subjects with usable quiet-standing static force-plate readings, demonstrating near-perfect agreement ($r = 0.9991, R^2 = 0.9981$, mean absolute difference of $1.39\text{ kg}$ / $2.16\%$, RMSE $= 1.48\text{ kg}$). This provides experimentally derived, dynamically consistent body masses across 100% of the cohort, entirely avoiding ad hoc volumetric approximations.
- **Inference Latency Characterization and Embedded Hardware Limitation:**
  Inference latency was profiled on the host x86 CPU (Intel Core i7-8565U @ 1.80 GHz) via ONNXRuntime with single-threaded execution, measuring 9.93 ms/window (throughput: 100.7 windows/s; parameter count: 172k; model size: 0.48 MB). Per `architecture.md §5` and `prd.md §6`, this measurement is explicitly flagged with `latency_estimate=True`. It does NOT represent a physical measurement on an embedded ARM-based SoC (e.g., Raspberry Pi 5, NVIDIA Jetson, or ARM Cortex wearable controllers), which feature distinct instruction sets (ARM NEON vs. x86 AVX2), memory bandwidths, cache hierarchies, and thermal envelopes. Unsupported claims of "Real-Time Feasibility for wearable embedded controllers" are strictly removed until backed by physical embedded silicon benchmarking. Real embedded-hardware deployment and profiling are explicitly documented as a limitation and future work.
- **Empirical Results Narrative and Physical Trade-Off Framing:**
  Evaluations across Leave-One-Speed-Out (LOSO) 6-fold cross-validation and cross-dataset zero-shot transfer (Fukuchi et al., 2018) reveal decoupled roles for speed conditioning and physics guidance:
  1. *FiLM Speed-Conditioning Drives Generalization Accuracy:* Model 2 (+Conditioning) is the best performer on raw accuracy across both generalization benchmarks (LOSO cross-speed mean MAE: 0.0664 N·m/kg vs. Baseline 0.0703 N·m/kg; Fukuchi cross-dataset MAE: 0.2479 N·m/kg vs. Baseline 0.2640 N·m/kg, Pearson $r$: 0.7492 vs. 0.6553). FiLM conditioning is responsible for the generalization-accuracy gains under speed shift and domain transfer.
  2. *Physics Guidance Acts as a Smoothness/Safety Regularizer at a Small Accuracy Cost:* Model 3 (+Conditioning+Physics) exhibits the highest raw MAE among the three models on both generalization tests (LOSO mean MAE: 0.0711 N·m/kg; Fukuchi MAE: 0.2701 N·m/kg). However, it consistently and decisively reduces jerk across every speed fold and dataset (~25% to 34% jerk reduction; LOSO mean Jerk RMS: 93.31 vs. Baseline 132.28 N·m/(kg·s²); held-out test Jerk RMS: 100.93 vs. 153.34 N·m/(kg·s²)). The auxiliary physics loss functions as a regularizer, suppressing high-frequency torque oscillations and enforcing energy plausibility at a minor accuracy trade-off (~1% higher MAE than baseline).
  3. *PRD Criterion #2 Partial Support:* PRD Criterion #2 ("physics-guided model degrades less... as speed moves away from training") is only **partially supported**: it holds under slow out-of-distribution conditions (0.6 m/s OOD: Model 3 MAE 0.0688 vs. Baseline 0.0735, a 6.4% error reduction), but not under fast out-of-distribution conditions (1.6 m/s OOD: Model 3 MAE 0.0934 vs. Baseline 0.0924), being roughly neutral overall (+1.1% error). This must be reported honestly as partial support.
  4. *1.4 m/s LOSO Speed Fold Anomaly:* Across all 6 LOSO folds, the 1.4 m/s fold is the sole condition where +Conditioning underperforms Baseline (MAE: 0.0728 vs. 0.0673 N·m/kg). In Discussion, this will be highlighted and discussed (e.g., potential localized MLP sensitivity near the upper-middle transition of the training speed range) rather than omitted.

## 10. Hyperparameter Selection and Evaluation Protocol (Anti-Leakage Principle)

To prevent test-set leakage via hyperparameter selection and guarantee unbiased generalization estimates:
1. **Validation-Based Selection Only:** All candidate hyperparameter choices (specifically loss weights $\lambda_{\text{jerk}}$ and $\lambda_{\text{energy}}$, loss ramp-up schedules, learning rates, or checkpoint epochs) must be selected strictly on validation split performance (task MSE on held-out validation subjects). Test set metrics must never be computed, inspected, or utilized to choose among candidate configurations.
2. **Single Unbiased Test Evaluation:** Once a single configuration is selected based strictly on validation loss, its frozen checkpoint is evaluated exactly once on the held-out test split. That single test evaluation constitutes the official reported result.
3. **Application to Phase 4 Generalization:**
   - **Leave-One-Speed-Out (LOSO) Zero-Speed-Leakage Protocol:** For each of the 6 folds where speed $S_{\text{held}} \in \{0.6, 0.8, 1.0, 1.2, 1.4, 1.6\}\text{ m/s}$ is held out for zero-shot testing:
     - The **Training Split** contains strictly the 30 training subjects evaluated only across the remaining 5 non-held-out speeds ($s \in \text{ALL\_SPEEDS} \setminus \{S_{\text{held}}\}$). Zero trials or frames from $S_{\text{held}}$ exist in the training split.
     - The **Validation Split** used for checkpoint selection, early stopping, and learning rate scheduling is drawn exclusively from the 7 validation subjects evaluated only across the remaining 5 non-held-out speeds ($s \in \text{ALL\_SPEEDS} \setminus \{S_{\text{held}}\}$). Zero trials or frames from $S_{\text{held}}$ exist in the validation split.
     - The **Held-Out Speed Test Split** ($S_{\text{held}}$) is completely firewalled during training and validation. It is evaluated exactly once at the end using the checkpoint chosen by the fold's within-split validation MSE. No hyperparameter adjustments or checkpoint re-selections based on test performance are permitted.
   - **Cross-Dataset Zero-Shot (Fukuchi):** Checkpoints are selected purely on primary dataset validation loss. No tuning, configuration selection, or threshold calibration against the Fukuchi dataset is permitted.

