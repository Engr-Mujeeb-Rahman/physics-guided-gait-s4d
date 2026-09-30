# ARCHITECTURE — Physics-Guided SSM for Gait Kinetics

**Status:** LOCKED (see rules.md)

## 1. Tech Stack

- **Language:** Python 3.10+
- **DL framework:** PyTorch (CUDA build, targeting GTX 1660 6GB locally; CPU-only laptop for
  non-training work — writing, analysis, plotting)
- **Biomechanics:** OpenSim 4.x Python API (`opensim` conda package) — Muscle Analysis tool, and
  (optionally, Phase 3 stretch) direct queries to the dynamics engine for M(q), C(q,q̇), G(q)
- **Data handling:** NumPy, Pandas
- **Plotting:** Matplotlib
- **Export/deployment:** ONNX + `onnxruntime` for embedded inference
- **No custom CUDA kernels** (see design.md §3 — this is a deliberate risk-avoidance decision)
- **Experiment tracking:** flat CSV/JSON run logs under `experiments/` — no external service
  (wandb/mlflow) unless explicitly requested; this is a 3-week solo project, not a team project, and
  extra infra is time not spent on the model.

## 2. Repository Structure

```
gait-kinetics-paper/
├── data/
│   ├── raw/                     # untouched downloaded datasets (primary + Fukuchi)
│   ├── opensim_processed/       # OpenSim Muscle Analysis outputs (l_m, v_m per subject/trial)
│   └── processed/                # final model-ready tensors (windowed, normalized)
├── src/
│   ├── data/
│   │   ├── opensim_pipeline.py   # runs Muscle Analysis via OpenSim API
│   │   ├── dataset.py            # PyTorch Dataset/DataLoader, windowing, normalization
│   │   └── splits.py             # subject-wise train/val/test + leave-one-speed-out splits
│   ├── models/
│   │   ├── s4d.py                # S4D layer (pure PyTorch, no custom kernels)
│   │   ├── film.py               # FiLM speed-conditioning module
│   │   └── backbone.py           # full model: S4D stack + FiLM + prediction head
│   ├── physics/
│   │   ├── jerk_loss.py
│   │   ├── energy_consistency_loss.py
│   │   └── opensim_residual.py   # optional Phase 3 stretch — precomputed M/C/G residual
│   ├── train.py                  # training loop, handles all 3 ablation configs via a flag
│   ├── evaluate.py               # accuracy, OOD, cross-dataset zero-shot, physical-consistency metrics
│   ├── export_onnx.py            # export trained checkpoint to ONNX
│   └── benchmark_latency.py      # run on embedded device, log latency/throughput
├── experiments/                  # run configs + result logs (CSV/JSON), one folder per run
├── notebooks/                    # exploratory analysis only — nothing here feeds the paper directly
├── paper/                        # manuscript source (LaTeX/Word per J-BHI template)
├── prd.md / design.md / architecture.md / phases.md / rules.md / memory.md
└── README.md
```

## 3. Data Flow

1. `opensim_pipeline.py` reads raw dataset kinematics + the dataset's pre-scaled Rajagopal model,
   runs OpenSim Muscle Analysis, writes l_m/v_m per subject/trial to `data/opensim_processed/`.
2. `dataset.py` merges kinematics + GRF + joint moments (labels) + l_m/v_m + speed label into
   fixed-length windows, normalizes per-channel, and exposes a PyTorch `Dataset`.
3. `splits.py` produces: (a) standard train/val/test subject split on the primary dataset, (b)
   leave-one-speed-out splits for OOD evaluation, (c) the full Fukuchi dataset held out entirely for
   cross-dataset zero-shot testing (never touched during training).
4. `train.py` trains one of the three ablation configs (baseline / +conditioning / +conditioning+physics)
   against the primary dataset's train split, logging to `experiments/<run_name>/`.
5. `evaluate.py` runs all four PRD-required evaluations against a trained checkpoint and writes result
   tables to `experiments/<run_name>/results/`.
6. `export_onnx.py` + `benchmark_latency.py` take the final chosen checkpoint and produce the embedded
   hardware latency number.

## 4. Model Architecture (default config — see design.md §3 for rationale)

- Input window: joint angles + angular velocities + GRF + (l_m, v_m if available) + speed scalar,
  fixed window length (set in Phase 1 based on sampling rate — typically 1–2 gait cycles).
- Backbone: 4× S4D blocks, d_model=64–128 (scale only after baseline convergence is confirmed).
- Conditioning: FiLM MLP (speed scalar → γ, β) applied at each S4D block.
- Head: linear projection to per-timestep joint moment predictions (hip/knee/ankle, sagittal plane).
- Loss: MSE on joint moments (all configs) + jerk penalty + energy-consistency residual (physics config
  only), with loss-weight ramp-up schedule (see design.md's instability fallback).

## 5. Embedded Deployment Path

- Train on GPU (or CPU fallback) → export to ONNX (`export_onnx.py`) → run on Raspberry Pi 5 via
  `onnxruntime` (`benchmark_latency.py`) → log latency (ms/window) and parameter count/FLOPs.
- If no device is available by Phase 5 (see phases.md), `benchmark_latency.py` still runs against
  CPU-only inference locally, and the output is labeled `latency_estimate=True` in the result log so
  it can never accidentally be reported as a real embedded measurement in the paper.

## 6. Reproducibility

- Every run under `experiments/<run_name>/` stores: config used, git commit hash (if git is in use —
  see rules.md for git permission policy), random seed, and full result table.
- No result goes into the paper without a corresponding `experiments/` folder backing it.
