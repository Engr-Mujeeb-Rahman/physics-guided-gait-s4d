# Implementation Execution Tracker

## Phase 1: Critical Fixes (P0 & P1)
- [x] **Task 1.2 (P0)**: Fix S4D discretization error in `paper/main.tex` (Tustin -> ZOH equations) & compile check <!-- id: 0 -->
- [x] **Task 1.5 (P3)**: Clean up unused `goel2022s` in `paper/references.bib` & cite or prune <!-- id: 1 -->
- [ ] **Task 1.1 (P0)**: Implement LSTM baseline (`src/models/lstm_baseline.py`) <!-- id: 2 -->
- [ ] **Task 1.1 (P0)**: Implement TCN baseline (`src/models/tcn_baseline.py`) <!-- id: 3 -->
- [ ] **Task 1.4 (P1)**: Support Model 1.5 (S4D + Physics, no FiLM) and baselines in `src/train.py` & `src/evaluate.py` <!-- id: 4 -->
- [ ] **Task 1.1 & 1.4 Execution**: Train LSTM, TCN, and Model 1.5 on subject split & LOSO folds <!-- id: 5 -->
- [ ] **Task 1.3 (P1)**: Upgrade statistical analysis (`src/statistical_tests.py`): effect sizes (Cohen's d / Cliff's delta), 95% bootstrap CIs, trial-level paired tests <!-- id: 6 -->

## Phase 2: Strengthening Manuscript Claims (P2)
- [ ] **Task 2.1**: Tone down "Digital Twin" framing across abstract, introduction, and conclusion <!-- id: 7 -->
- [ ] **Task 2.2**: Add Discussion subsection analyzing the 3.3x cross-dataset transfer gap on Fukuchi <!-- id: 8 -->
- [ ] **Task 2.3**: Trim abstract to ~180 words, focusing on core technical and clinical value <!-- id: 9 -->
- [ ] **Task 2.4**: Standardize and clarify power consistency loss nomenclature throughout `main.tex` <!-- id: 10 -->
- [ ] **Task 2.5**: Update Tables I, II, III in `main.tex` with baseline comparisons, Model 1.5, and effect sizes <!-- id: 11 -->

## Phase 3: Visual & Textual Polish (P3)
- [ ] **Task 3.1**: Generate Figure 4 (Bland-Altman analysis on test set) <!-- id: 12 -->
- [ ] **Task 3.2**: Generate Figure 5 (Torque second-derivative/jerk smoothing comparison) <!-- id: 13 -->
- [ ] **Task 3.3**: Expand Limitations section (pathological gait, single-plane, overground vs treadmill) <!-- id: 14 -->
- [ ] **Task 3.4**: Final LaTeX build verification via `tectonic.exe` and proofread <!-- id: 15 -->
