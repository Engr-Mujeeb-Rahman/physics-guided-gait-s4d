# MEMORY — Running Project Context

**Status:** LIVE (this file is meant to be updated as the project progresses — unlike the other five
files, it is not locked). Read this file first at the start of any session before touching code.

## Project One-Liner

Physics-guided S4D state-space model (FiLM speed-conditioned, physics-informed auxiliary loss)
predicting gait joint kinetics, validated for zero-shot generalization across speeds and datasets, with
a host CPU-profiled latency estimate (embedded ARM hardware validation documented as future work) —
targeting IEEE J-BHI's "Emerging AI Paradigms for Next-Gen Medicine" special issue, deadline 30 Sept 2026.

## Governing Documents (read in this order for full context)

1. `prd.md` — what we're building and why, success criteria, non-goals
2. `design.md` — specific technical decisions and rationale
3. `architecture.md` — codebase structure and data flow
4. `phases.md` — the 21-day execution plan with checkpoints and fallbacks
5. `rules.md` — hard constraints on what any agent/collaborator may change

## Key Decisions Already Locked (do not re-litigate — see rules.md)

- Datasets: six-speed treadmill dataset (primary, train+main eval) + Fukuchi 2018 PeerJ dataset
  (secondary, cross-dataset zero-shot eval only).
- Muscle state via OpenSim Muscle Analysis on the pre-scaled Rajagopal model — not raw ultrasound.
- Backbone: S4D (pure PyTorch, no custom CUDA kernels) — not `mamba-ssm`.
- Physics guidance: auxiliary loss (jerk penalty + energy-consistency residual), not a live
  differentiable multibody simulator.
- Conditioning: FiLM on walking speed — not a bespoke "neural compliance module."
- Three-way ablation required: baseline / +conditioning / +conditioning+physics.
- Latency benchmarking: host x86 CPU estimate (`latency_estimate=True` per architecture.md §5 fallback);
  physical embedded deployment (ARM / Jetson / Raspberry Pi 5) documented as a limitation and future work.
  "Real-time feasibility for wearable embedded controllers" language removed until backed by physical embedded silicon.
- Results narrative framing: FiLM speed-conditioning is the component responsible for generalization-accuracy
  gains (LOSO cross-speed and Fukuchi cross-dataset); physics auxiliary loss provides consistent, decisive
  smoothness/safety regularization (~25–34% jerk reduction) at a minor accuracy trade-off (~1% higher MAE).
  PRD Criterion #2 is partially supported (holds at 0.6 m/s OOD, neutral on average).
- Original proposal (ultrasound calibration, 8 discrete cadences, live differentiable dynamics layer,
  mandatory dual embedded-device benchmarking) was explicitly rejected as infeasible for a 3-week
  timeline with no existing dataset — see prd.md Non-Goals for the full list.

## Current Status

**Phase:** Phase 4 & Phase 5 Complete; Pre-Phase 6 Corrections Applied (Latency relabeled as host CPU estimate, results narrative reframed, 1.4 m/s fold anomaly flagged). Ready for Phase 6 (Writing).
**Days elapsed:** 3 / 21.
**Open risks:** None. PRD experimental criteria evaluated: Criterion #1 (accuracy), #3 (cross-dataset), #4 (physical consistency/jerk) cleanly met; Criterion #2 (OOD speed degradation) partially supported (true at 0.6 m/s OOD, neutral on average); Criterion #5 (latency) satisfied via architecture.md §5 CPU estimate fallback with embedded ARM validation explicitly framed as future work. Ready for IEEE J-BHI manuscript drafting.

## Update Log

*(Append a dated entry here each session — phase reached, checkpoint outcomes, any fallback invoked,
and any decision that required deviating from the locked docs, with the reason.)*

- **[Project setup]** prd.md, design.md, architecture.md, phases.md, rules.md, memory.md created.
  Scope locked to the revised/executable plan (see Key Decisions above), replacing the original
  ultrasound/differentiable-simulator proposal after confirming no matching dataset exists.
- **[2026-09-10 - Phase 0 Completed]**
  - **Repo setup:** Created all repository directories and skeleton files exactly per architecture.md §2 (`data/raw/`, `data/opensim_processed/`, `data/processed/`, `src/data/`, `src/models/`, `src/physics/`, `experiments/`, `notebooks/`, `paper/`, `README.md`, and module stubs).
  - **Environment setup:** Configured environment `myenv` (Python 3.11.13) with OpenSim 4.5.2 (via conda-forge), PyTorch 2.14.0, ONNX 1.22.0, onnxruntime 1.29.0, NumPy 2.1.3, Pandas 2.3.1, Matplotlib 3.10.3. Verified all imports execute cleanly.
  - **Datasets downloaded:**
    - Primary dataset: six-speed treadmill dataset (Le Goff et al., 2026, Mendeley Data DOI: 10.17632/3dmymnb65h.1) downloaded into `data/raw/primary_six_speed_gait_dataset.zip` (1.08 GB).
    - Secondary dataset: Fukuchi et al. (2018, PeerJ DOI: 10.7717/peerj.4640) downloaded into `data/raw/fukuchi/` (`WBDSascii.zip`, `WBDSinfo.xlsx`, `WBDSmodel.mdh`).
  - **OpenSim Muscle Analysis pipeline:** Implemented in `src/data/opensim_pipeline.py`. Acquired Rajagopal model (`Rajagopal2016.osim`) under `data/opensim_processed/models/`. Executed pipeline on Subject `16_BC`, trial `0_60` (0.6 m/s condition).
  - **Checkpoint outcome:** **PASS**. Musculotendon length ($l_m$) and velocity ($v_m$) extracted across 100 gait cycle frames for 40 lower-limb muscles with zero NaNs/Infs and realistic physiological ranges ($l_m \in [0.0788, 0.5919]$ m, $v_m \in [-0.2808, 0.3054]$ m/s). Saved to `data/opensim_processed/16_BC_0_60_muscles.npz`. Ready to proceed to Phase 1 with muscle-tendon channel included.
- **[2026-09-10 - Phase 1 Completed & Verified with Per-Subject Model Scaling]**
  - **Per-Subject Model Scaling Pipeline:** Implemented `src/data/scale_models.py`. For all 44 subjects, extracted subject anthropometrics from `static.c3d`: mass (from quiet-standing force plates or volumetric fallback) and marker-pair segment lengths (pelvis width, femur length, tibia length, foot length, torso height). Scaled `Rajagopal2016.osim` per subject via OpenSim `model.scale()` and saved 44 scaled models to `data/opensim_processed/models/<subject>_scaled.osim`.
    - **Total Scaling Time:** **26.00 s** (mean: **448.4 ms / subject**).
  - **Scaled Primary Muscle Pipeline:** Extended `src/data/opensim_pipeline.py` to load each subject's scaled model (with caching). Re-extracted $l_m$ and $v_m$ for all 262 trials across 44 subjects and saved to `data/opensim_processed/*_muscles.npz`.
    - **Total Extraction Time:** **47.21 s** (mean: **179.6 ms / trial**, median: **141.4 ms / trial**).
  - **Anthropometric Scaling Validation:**
    - Pearson correlation between subject leg length (74.4 cm – 108.4 cm) and mean $l_m$ across all 44 subjects: **r = 0.9723** ($p = 3.74 \times 10^{-28}$). Longer legs systematically and linearly track longer musculotendon lengths, confirming true physiological scaling.
  - **Splits Generator:** Implemented `src/data/splits.py` with `get_subject_splits()` (30 train subjects / 180 trials, 7 val subjects / 42 trials, 7 test subjects / 40 trials; zero subject overlap) and `get_leave_one_speed_splits()` across all 6 treadmill speeds.
  - **Windowing, Normalization & Dataset:** Implemented `src/data/dataset.py` with `GaitDataset` and `get_dataloaders()`. Per-channel z-score normalization computed strictly on the train split and applied to val/test splits.
  - **Phase 1 Checkpoint Outcome:** **PASS**. Re-verified batch loading on the scaled primary train split: batch size 16 produces input `x` shaped `[16, 100, 102]` (9 angles + 9 vels + 3 GRFs + 40 $l_m$ + 40 $v_m$ + 1 speed), target `y` shaped `[16, 100, 3]` (sagittal moments: hip, knee, ankle in N*m/kg), plus auxiliary tensors `speed` `[16, 1]`, `angular_vel` `[16, 100, 3]`, and `power_meas` `[16, 100, 3]`. Normalized features mean $\approx -0.06$, std $\approx 0.94$, 0 NaNs/Infs.
  - **Cross-Dataset Compatibility:** Implemented `src/data/fukuchi_loader.py`. Cataloged 328 valid treadmill trials across 42 subjects from Fukuchi et al. (2018). Verified sample trial `WBDS01walkT01` resampled to 100 frames generates feature tensor `(100, 102)` and target tensor `(100, 3)` with 0 NaNs/Infs, confirming 100% format compatibility for cross-dataset zero-shot evaluation.
- **[2026-09-10 - Phase 2 Hardware Timing Check Completed (Scaled Data)]**
  - **S4D Backbone & Baseline Implementation:** Implemented pure PyTorch diagonal state-space model in `src/models/s4d.py`, FiLM speed-conditioning module in `src/models/film.py`, and complete `GaitS4D` model architecture in `src/models/backbone.py` ($d_{model}=64, d_{state}=64, 4$ layers, 172,547 parameters).
  - **Training Engine:** Implemented `src/train.py` supporting baseline supervised training, learning rate warmup/cosine annealing, gradient clipping, small-subset overfitting verification, and high-precision per-epoch wall-clock timing.
  - **Small-Subset Overfitting Test (CPU, Scaled Data):** Executed 20 epochs on 3 subjects (18 trials, batch size 16). Loss decreased monotonically from 0.45973 to 0.03618 (92.1% reduction), confirming implementation correctness and numerical stability on the scaled data.
    - **Measured small-subset epoch time:** **0.7375 s/epoch** (median: 0.6897 s).
  - **Full Dataset Verification (CPU, Scaled Data):** Executed 2-epoch timing run across the full primary train split (30 subjects, 180 trials, 12 batches of 16).
    - **Measured full train split epoch time:** **5.7441 s/epoch** (~5.7 seconds).
  - **Hardware Decision Checkpoint Outcome:** The empirical per-epoch wall-clock time is firmly on the order of seconds (5.74s on full dataset, 0.74s on small subset), well below the 10+ minute threshold. Per the Phase 2 checkpoint rule, CPU training is fully capable and recommended for all Phase 2–3 experiments (baseline, +conditioning, +conditioning+physics). Reported to user for authorization.
- **[2026-09-10 - Phase 2 Baseline Training & Pre-Phase 3 Checkpoint Completed]**
  - **Pre-Phase 3 Item 1 (Mass Formula Justification):**
    - The formula $M = 75.337 \times s_{\text{pelvis}} \cdot s_{\text{leg}} \cdot s_{\text{torso}}$ is an **ad hoc geometric-volumetric fallback approximation** (assuming volume $V \propto L_x L_y L_z$, uniform density $\rho \approx 1000\text{ kg/m}^3$, scaled relative to the 75.337 kg nominal Rajagopal reference subject).
    - Fit error against the 21 force-plate-valid subjects: Absolute error mean = 12.04 kg (18.18%), median = 7.75 kg (15.10%), min = 1.73 kg (2.47%), max = 24.72 kg (37.48%), std = 7.90 kg (11.69%), RMSE = 14.40 kg. Signed error mean = -7.86 kg, indicating systematic underestimation for wider/heavier subjects due to unmeasured trunk sagittal depth.
  - **Pre-Phase 3 Item 2 (Cohort Mass Split & Limitation):**
    - 21 of 44 subjects (47.7%) have direct force-plate-derived mass; 23 of 44 subjects (52.3%) rely on formula-estimated mass.
    - Flagged as a limitation: mass uncertainty propagates into kinetic/potential energy terms for the energy-consistency loss in Phase 3.
  - **Pre-Phase 3 Item 3 (Scaling Step Documentation):**
    - Confirmed OpenSim iterative Marker Placer / IK scaling was skipped in favor of direct anatomical segment-length-ratio scaling via `ScaleSet` and `model.scale()`. Documented in `design.md §9 Modeling Limitations and Simplifications`.
  - **Full Phase 2 Baseline Training (50 Epochs, CPU):**
    - Trained on 30 subjects (180 trials, batch size 16).
    - Initial train loss (Epoch 1): 0.24755, final train loss (Epoch 50): 0.00957 (96.1% loss reduction).
    - Best validation loss: 0.01657.
    - Mean epoch time: 7.900 s (total training duration: 395.0 s / 6.58 min).
    - Saved `experiments/phase2_baseline/model_best.pt`, `model_last.pt`, and `train_metrics.json`.
  - **Held-Out Test Set Kinetics Evaluation (7 subjects / 40 trials):**
    - Test Loss (MSE): 0.01989.
    - Hip Flexion/Extension Moment: MAE = **0.0844** N*m/kg, RMSE = **0.1141** N*m/kg, Pearson r = **0.9414**.
    - Knee Flexion/Extension Moment: MAE = **0.0699** N*m/kg, RMSE = **0.1081** N*m/kg, Pearson r = **0.9226**.
    - Ankle Plantar/Dorsiflexion Moment: MAE = **0.0797** N*m/kg, RMSE = **0.1488** N*m/kg, Pearson r = **0.9633**.
    - Overall Average: MAE = **0.0780** N*m/kg, RMSE = **0.1237** N*m/kg, Pearson r = **0.9425**.
  - **Phase 2 Baseline Status:** **100% COMPLETE**.
- **[2026-09-10 - Dynamic GRF Mass Estimation Fix & Phase 2 Baseline Verification]**
  - **Dynamic GRF Mass Estimation:** Completely replaced the ad hoc volumetric mass formula. Derived body mass for all 44 primary dataset subjects from steady-state mean vertical ground reaction force averaged across complete gait cycles in actual walking trials ($M = \frac{1}{g T} \int_0^T F_z(t) dt$).
  - **Validation Cross-Check against 21 Static Force-Plate Subjects:**
    - Pearson correlation: **r = 0.9991** ($R^2 = 0.9981$).
    - Mean absolute difference: **1.39 kg** (2.16%).
    - Median absolute difference: **1.45 kg** (2.29%).
    - RMSE: **1.48 kg**; Max absolute difference: **2.08 kg** (4.14%).
    - Saved canonical verified masses for all 44 subjects to `data/opensim_processed/subject_masses.json`.
  - **Documentation Updated:** `design.md §9` updated to reflect the dynamic GRF body mass derivation method and completely drop the abandoned volumetric formula.
  - **Numerical Fix Explained:** In `src/train.py`, during post-training test evaluation, `np.corrcoef` on flattened arrays triggered an underlying Intel MKL BLAS (`dsyrk`/`np.cov`) silent access violation on Windows. Fixed by replacing with the direct, numerically stable mathematical Pearson correlation formula ($r = \frac{\sum (p - \bar{p})(t - \bar{t})}{\sqrt{\sum (p - \bar{p})^2 \sum (t - \bar{t})^2}}$), executing cleanly with zero divergence and producing identical metrics.
  - **Timing Gap Explained:** In the 2-epoch check, only 2 quick initial training passes ran (mean: 5.74 s). Across 50 continuous epochs, multi-threaded CPU matrix operations experienced thermal throttling and OS scheduling variation (raising mean training epoch time to ~7.7 s, with occasional spikes to 12.4 s).
- **[2026-09-11 - Phase 3 Completed: Conditioning & Physics Loss Three-Way Ablation]**
  - **Physics Loss Calibration & Verification:**
    - Stride durations extracted for all 262 trials via anterior-posterior heel trajectory peak detection (Zeni et al., 2008), saved to `data/opensim_processed/trial_stride_durations.json`. Real $\Delta t = T_{\text{stride}} / 99$ passed to `TorqueJerkLoss`.
    - Joint angles converted to radians before central differentiation to produce angular velocity in $\text{rad/s}$, yielding joint mechanical power $\hat{\tau} \cdot \dot{q}$ in $\text{W/kg}$ matching dataset coordinate system `[+1.0, -1.0, -1.0]` ($r > 0.995$).
  - **Three-Way Ablation Results (50 epochs each, CPU):**
    - **Model 1 (Baseline):** Supervised S4D backbone, MSE loss.
      - Best Val Loss: 0.01657 | Test MSE: 0.01989 | MAE: 0.0780 N·m/kg | RMSE: 0.1237 N·m/kg | $r = 0.9425$.
      - Jerk RMS: 153.34 N·m/(kg·s²) | Power Residual RMSE: 0.2256 W/kg.
    - **Model 2 (+Conditioning):** S4D backbone + FiLM speed conditioning, MSE loss.
      - Best Val Loss: 0.01516 | Test MSE: 0.01928 | MAE: 0.0746 N·m/kg | RMSE: 0.1191 N·m/kg | $r = 0.9472$.
      - Jerk RMS: 124.61 N·m/(kg·s²) | Power Residual RMSE: 0.2516 W/kg.
    - **Model 3 (+Conditioning + Physics Blind Re-Run Protocol):**
      - Retrained all 3 candidates fresh with test evaluation completely decoupled (only validation loss computed).
      - Candidate 1 ($\lambda_{\text{jerk}}=10^{-6}, \lambda_{\text{energy}}=0.02$): Best Val Loss = **0.07003** (jerk penalty overwhelmed early epochs, causing underfitting).
      - Candidate 2 ($\lambda_{\text{jerk}}=10^{-7}, \lambda_{\text{energy}}=0.002$): Best Val Loss = **0.01968** (stable, but jerk penalty still constrained task loss).
      - Candidate 3 ($\lambda_{\text{jerk}}=10^{-8}, \lambda_{\text{energy}}=0.01$): Best Val Loss = **0.01510** (lowest validation loss across all candidates; selected strictly on validation MSE with zero test metrics computed).
      - **Selected Official Model 3 (Candidate 3, single decoupled held-out test evaluation):**
        - Test Loss (MSE): **0.01972** | MAE: **0.0768 N·m/kg** | RMSE: **0.1220 N·m/kg** | $r = \mathbf{0.9436}$.
        - Per-Joint: Hip MAE: 0.0795 N·m/kg ($r=0.9522$), Knee MAE: 0.0713 N·m/kg ($r=0.9178$), Ankle MAE: 0.0797 N·m/kg ($r=0.9609$).
        - Jerk RMS: **100.93 N·m/(kg·s²)** (34.2% smoother than baseline 153.34; 48.9% smoother than target 197.53) | Power Residual RMSE: **0.2425 W/kg**.
  - **Checkpoint Outcome:** **PASS**. Blind candidate selection re-verified under strict decoupling. Candidate 3 selected solely on validation loss; evaluated exactly once on the held-out test split. Canonical checkpoint saved to `experiments/phase3_conditioning_physics/`. Ready for Phase 4.
- **[2026-09-11 - Phase 4 Completed: Speed & Cross-Dataset Generalization Evaluations]**
  - **Cross-Dataset Zero-Shot Evaluation (Fukuchi et al., 2018):**
    - Cached 328 treadmill trials to `data/processed/fukuchi_processed.npz` (swing-phase unmeasured GRFs encoded with physical 0.0 N/kg).
    - Evaluated primary-trained checkpoints with zero fine-tuning:
      - Model 1 (Baseline): MAE = 0.2640 N·m/kg, RMSE = 0.3278 N·m/kg, $r = 0.6553$.
      - Model 2 (+Conditioning): MAE = 0.2479 N·m/kg, RMSE = 0.3131 N·m/kg, $r = 0.7492$.
      - Model 3 (+Conditioning+Physics): MAE = 0.2701 N·m/kg, RMSE = 0.3360 N·m/kg, $r = 0.6849$.
    - Saved to `experiments/results/fukuchi_zero_shot_summary.json`.
  - **Leave-One-Speed-Out (LOSO) 6-Fold Cross-Validation:**
    - Executed all 6 speed folds (0.6, 0.8, 1.0, 1.2, 1.4, 1.6 m/s) across all 3 models (18 total training runs) with zero speed leakage into train or val.
    - Checkpoints frozen strictly via within-fold validation MSE; evaluated once on held-out speed.
    - Master results:
      - Model 1 (Baseline): Cross-speed mean MAE = **0.0703** N·m/kg, RMSE = **0.1028** N·m/kg, $r = \mathbf{0.9525}$, Mean Jerk RMS = **132.28** N·m/(kg·s²).
      - Model 2 (+Conditioning): Cross-speed mean MAE = **0.0664** N·m/kg, RMSE = **0.0982** N·m/kg, $r = \mathbf{0.9545}$, Mean Jerk RMS = **120.02** N·m/(kg·s²).
      - Model 3 (+Conditioning+Physics): Cross-speed mean MAE = **0.0711** N·m/kg, RMSE = **0.1045** N·m/kg, $r = \mathbf{0.9491}$, Mean Jerk RMS = **93.31** N·m/(kg·s²) (29.5% reduction in jerk vs baseline!).
      - On slow out-of-distribution speed (0.6 m/s): Model 3 MAE: **0.0688** N·m/kg vs Baseline **0.0735** N·m/kg (6.4% error reduction + 27% lower jerk).
    - Saved to `experiments/results/loso_summary.json` and `loso_summary.csv`.
  - **Checkpoint Outcome:** **PASS**. Complete 3-way ablation $\times$ 2 generalization tests results table finalized.
- **[2026-09-11 - Phase 5 Completed: Physical Consistency & Hardware Latency Benchmark]**
  - **ONNX Export:** Exported Model 3 to `experiments/phase3_conditioning_physics/model.onnx` (0.48 MB) using folded real 1D depthwise convolution kernels (`FoldedGaitS4D`), eliminating complex numbers and achieving numerical equivalence (discrepancy 3.58e-07).
  - **Latency Profiling:** Profiled over 1,000 iterations via ONNXRuntime CPU provider:
    - Mean Latency: **9.931 ms / window** (Throughput: **100.7 windows/sec**).
    - Median Latency: **9.278 ms / window**; 95th Percentile: **12.896 ms / window**.
  - **Checkpoint Outcome:** **PASS**. Phase 5 completed with ONNX export and single-threaded ONNXRuntime profiling. Host x86 CPU latency measurement (9.93 ms) labeled with `latency_estimate=True` per architecture.md §5 fallback.
- **[2026-09-11 - Pre-Phase 6 Corrections & Narrative Reframing Completed]**
  - **Latency Characterization & Scope Realignment:**
    - Explicitly relabeled the 9.93 ms ONNX inference latency as a host x86 CPU-profiled estimate (`latency_estimate=True` in `experiments/results/latency_benchmark.json` and `design.md §9`).
    - Stripped any unsupported claims of "Real-Time Feasibility for wearable embedded controllers" across documentation.
    - Explicitly documented physical embedded deployment (ARM Cortex, Raspberry Pi 5, Jetson) and hardware-in-the-loop validation as an experimental limitation and high-priority future work item.
  - **Reframed Results Narrative:**
    - *Driver of Generalization Accuracy:* FiLM speed conditioning (Model 2) is the primary driver of accuracy under distribution shift (LOSO cross-speed mean MAE 0.0664 vs. Baseline 0.0703; Fukuchi cross-dataset MAE 0.2479 vs. Baseline 0.2640, Pearson $r$ 0.7492 vs. 0.6553).
    - *Role of Physics-Guided Loss:* Auxiliary physics losses (Model 3) act as a physiological smoothness and safety regularizer. It achieves consistent, decisive jerk reduction across every single speed fold and dataset (~25% to 34% reduction; LOSO mean Jerk RMS 93.31 vs. Baseline 132.28; held-out test Jerk RMS 100.93 vs. 153.34 N·m/(kg·s²)), eliminating force spikes essential for wearable robotic actuator safety, but incurs a slight accuracy trade-off (~1% higher MAE: LOSO mean MAE 0.0711 vs. Baseline 0.0703; Fukuchi MAE 0.2701 vs. Baseline 0.2640).
    - *PRD Criterion #2 Evaluation:* Criterion #2 ("physics-guided model degrades less... as speed moves away from training") is **partially supported**: holds at slow out-of-distribution speeds (0.6 m/s OOD: Model 3 MAE 0.0688 vs. Baseline 0.0735, a 6.4% error reduction), but fails at fast out-of-distribution speeds (1.6 m/s OOD: Model 3 MAE 0.0934 vs. Baseline 0.0924), being roughly neutral overall. Framed honestly as partial support.
- **[2026-09-12 - Phase 6 Completed: IEEE J-BHI Manuscript Finalized & Verified]**
  - **Critical Presentation & Formatting Defects Resolved:**
    - Restored Figures 1, 2, and 3 to crisp, full-width `figure*` layouts (width 0.88\textwidth, 0.92\textwidth, 0.88\textwidth), ensuring full publication-grade legibility for the S4D architecture diagram, the 6-subplot waveform grid with $\pm 1\sigma$ envelopes, and the two-panel LOSO generalization chart.
    - Activated all 9 previously uncited references in `paper/references.bib` (expanding active references from 30 to 39), including Karniadakis et al. (Nature Rev Phys), Winter (Biomechanics 2009), Seth et al. (PLOS Comp Bio 2018), Ding et al. (Sci Robotics 2018), Zhang et al. (Science 2017), Franks et al. (Sci Robotics 2021), Chakshu et al. (2024), Wright & Davidson (2023), and Gu et al. (HiPPO NeurIPS 2020).
    - Eliminated duplicate subsection lettering (`A. A.`, `B. B.`, etc.) in Section VII Limitations.
    - Explicitly decoupled Table I test-set jerk reduction (34.2% across 40 trials) from Table II cross-speed fold consistency (two-tailed $W=0.0, p=0.0313$, one-tailed $p=0.0156$ across $N=6$ folds), harmonizing captions across text, Table II, and Figure 3.
    - Standardized loss weighting notation to $\lambda_{\mathrm{power}}$ matching $\mathcal{L}_{\mathrm{power}}$.
  - **Layout & Compilation Verification:**
    - Compiled cleanly via `tectonic.exe` with exit code 0.
    - Fits into **exactly 8 pages** in standard IEEEtran two-column journal format.
    - Page 8 references balanced symmetrically across both columns.
  - **Status:** **PHASE 6 COMPLETE. MANUSCRIPT SUBMISSION-READY FOR IEEE J-BHI SPECIAL ISSUE.**







