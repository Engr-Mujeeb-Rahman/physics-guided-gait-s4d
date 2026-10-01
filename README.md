# Physics-Guided SSM for Gait Kinetics

Target Venue: IEEE Journal of Biomedical and Health Informatics (J-BHI) Special Issue on "Emerging AI Paradigms for Next-Gen Medicine: Safety, Reliability, and Digital Twins"  
Deadline: 30 September 2026

## Overview

A physics-guided diagonal state-space model (S4D) with FiLM speed conditioning and physics-derived auxiliary losses (jerk minimization and energy-consistency residual) for joint-kinetics prediction during human gait. Validated for zero-shot generalization across walking speeds and datasets, with CPU-profiled latency estimation and physical embedded-hardware validation deferred to future work.

## Repository Structure

- `data/`
  - `raw/`: Untouched downloaded raw datasets (primary six-speed treadmill dataset + secondary Fukuchi 2018 PeerJ dataset).
  - `opensim_processed/`: OpenSim Muscle Analysis outputs ($l_m, v_m$ per subject/trial).
  - `processed/`: Final windowed, normalized model-ready tensors.
- `src/`
  - `data/`: Data loading, splits, and OpenSim processing pipelines.
  - `models/`: S4D layers, FiLM conditioning, and backbone models.
  - `physics/`: Jerk penalty and energy-consistency residual losses.
  - `train.py`: Model training loop.
  - `evaluate.py`: Accuracy, OOD, cross-dataset zero-shot evaluation.
  - `export_onnx.py`: ONNX export pipeline.
  - `benchmark_latency.py`: Latency benchmarking script.
- `experiments/`: Run configurations and logged metrics.
- `notebooks/`: Exploratory analysis.
