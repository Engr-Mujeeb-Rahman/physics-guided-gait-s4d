# Revision Assessment Report
## "Physics-Guided Diagonal State-Space Model with Speed Conditioning for Generalizable Gait Joint-Kinetics Prediction"
### Review of `main (1).pdf` Against Prior Comprehensive Review Comments

**Date:** 29 September 2026
**Verdict:** **Partial Revision — Several Critical Issues Resolved; Significant Issues Remain**

> Every finding below is verified line-by-line against the extracted text of `main (1).pdf` and the original experiment JSON files.

---

## SCORECARD: WHAT WAS FIXED vs. WHAT REMAINS

| Issue | Category | Status |
|---|---|---|
| Abstract MAE "0.0746" matching no model | C4 | ✅ **FIXED** |
| Abstract r "0.947" matching no model | C4 | ✅ **FIXED** |
| Parameter count "172k" vs. actual 189,699 | C4 / M6 | ✅ **FIXED** |
| LOSO results: M3 loses to M1 in 5/6 folds — not disclosed | C5 | ✅ **FIXED** |
| BiLSTM wins all LOSO folds — hidden | C5 | ✅ **FIXED** |
| TCN cross-dataset r=0.8347 not reported prominently | C6 | ✅ **FIXED** |
| M3 transfers worse than M1 — not disclosed | C6 | ✅ **FIXED** |
| "Negligible accuracy trade-off" language | C7 | ✅ **FIXED** |
| Optimizer name "Adam" vs. actual AdamW | M1 | ✅ **FIXED** |
| λ_jerk inconsistency (paper vs. code) | M3 | ✅ **FIXED** |
| FiLM placement — code vs. paper Fig. 1 discrepancy | M5 | ✅ **FIXED** |
| Sign convention `[+1,-1,-1]` unexplained | M7 | ✅ **FIXED** |
| Discussion power RMSE differing from Table I | M8 | ✅ **FIXED** |
| Target jerk RMS absent from tables | M9 | ✅ **FIXED** |
| One-sided Wilcoxon tests — not disclosed | M11 | ✅ **FIXED** |
| 1/Δt² speed confound in jerk | m5 | ✅ **FIXED** |
| "Torque jerk" vs. "torque acceleration" inconsistency | m1 | ✅ **FIXED** |
| Power loss "physics" overclaim — algebraic identity | — | ✅ **FIXED** |
| BiLSTM jerk without physics addressed | — | ✅ **FIXED** |
| FFT claim ("O(T log T)") — corrected | M4 | ✅ **FIXED** |
| Warmup scheduler — clarified | M2 | ✅ **FIXED** |
| **Fake reference `baud2021review`** | C1 | ✅ **FIXED** |
| **Fake reference `hernandez2023transformer`** | C1 | ⚠️ **PARTIALLY FIXED** |
| **Primary dataset `legoff2026sixspeed` provenance** | C2/C3 | ⚠️ **PARTIALLY FIXED** |
| TCN/BiLSTM baseline results are single-seed in Table III | C6 / M12 | ❌ **NOT FIXED** |
| Trial-level statistics use 7 subjects treated as N=40 | M10 | ❌ **NOT FIXED** |
| Fig. 3 hand-picked trial (trial_idx=12) — undisclosed | m2 | ❌ **NOT FIXED** |
| No mean-waveform or shallow-ML baseline | m7 | ❌ **NOT FIXED** |
| Training/validation loss curves absent | m6 | ❌ **NOT FIXED** |
| Bland-Altman autocorrelation limitation | m9 | ⚠️ **PARTIALLY FIXED** |
| "Indispensable" and overreaching clinical claims | — | ⚠️ **PARTIALLY FIXED** |
| λ_jerk selection by validation MSE always picks edge | — | ❌ **NOT FIXED** |
| Fukuchi sign/convention differences from Visual3D | — | ✅ **FIXED** |
| "Streaming O(1) deployment" overclaim | M13 | ⚠️ **PARTIALLY FIXED** |

---

## PART I — ISSUES FULLY RESOLVED ✅

### 1.1 Abstract Corrected

The revised abstract now correctly reports:
- MAE: **0.0765±0.0032** (Model 3) and **0.0742±0.0008** (Model 2) — both correctly identified
- Pearson r: **0.9438±0.0040** (Model 3) and **0.9466±0.0020** (Model 2) — correct
- Parameters: **189,699** (previously 172k) — corrected
- TCN cross-dataset performance (r=0.8347, MAE 0.2268) — now reported
- 7.1% accuracy trade-off is explicitly stated — the "negligible" claim is gone
- BiLSTM dominance in LOSO is openly stated: *"bidirectional baselines dominate cross-speed accuracy (BiLSTM MAE: 0.0575)"*

**Assessment: Abstract is now accurate and honest. Major improvement.**

---

### 1.2 LOSO Table and Narrative Corrected (Table II)

Table II now:
- Shows all 6 folds explicitly with values for all models including BiLSTM and TCN
- Table footnote explicitly states: *"Non-causal sequence baselines (BiLSTM, TCN) dominate cross-speed accuracy across all folds"*
- States: *"Model 3 performs worse than baseline Model 1 in 5 of 6 folds, winning only at the slowest speed of 0.6 m/s"*
- BiLSTM jerk (89.19) reported alongside M3 jerk (93.31) in the LOSO table

**Assessment: LOSO reporting is now transparent and honest.**

---

### 1.3 Cross-Dataset Table Corrected (Table III)

Table III now leads with: *"Dilated TCN attains highest overall transfer (r=0.8347, MAE 0.2268 N·m/kg)"* — this is stated first, not buried. Model 3's degraded transfer (MAE 0.2701 > M1's 0.2640) is explicitly acknowledged.

**Assessment: Cross-dataset reporting is honest.**

---

### 1.4 Reference `baud2021review` Corrected

Reference [9] now reads:
```
R. Baud, A. R. Manzoori, A. Ijspeert, and M. Bouri,
"Review of control strategies for lower-limb exoskeletons to assist gait,"
Journal of NeuroEngineering and Rehabilitation, vol. 18, no. 1, p. 119, 2021.
```
✅ Correct journal, correct authors, correct DOI.

---

### 1.5 FFT Claim Corrected

Section III-C now says:
> *"the kernel K̄ is evaluated analytically in O(T · d_state) time via direct element-wise exponentiation, followed by standard 1D convolution, avoiding custom CUDA kernels"*

The false FFT / O(T log T) claim has been removed. **Fixed.**

---

### 1.6 Optimizer, λ_jerk, Warmup — All Corrected

Section IV-D now reads:
> *"trained for 50 epochs using AdamW (lr = 10⁻³, weight decay 10⁻⁴, cosine annealing schedule, batch size 16, gradient clipping at 1.0) with a 10-epoch linear ramp-up of auxiliary physics weights"*

- **AdamW** (previously "Adam") ✅
- **weight decay 10⁻⁴** now stated ✅
- **10-epoch linear ramp-up** (warmup) now stated — however see note below ✅
- **λ_jerk = 10⁻⁸** confirmed with explicit explanation of the 1/(Δt)² scaling rationale ✅

---

### 1.7 Sign Convention Explained

Equation 12 now includes:
> *"where s = [1,−1,−1] accounts for OpenSim coordinate sign definitions (hip flexion is mutually positive, whereas knee extension and ankle plantarflexion coordinate systems require sign reflection relative to angular velocity derivatives to match physiological work)"*

**Fixed.**

---

### 1.8 Target Jerk RMS Added to Table I

Table I now reports:
> **Target (Ground Truth) Jerk RMS: 197.53 N·m/(kg·s²)**

This anchors the jerk reduction: predicted jerk (102.37 for M3) is already significantly below the noisy ground truth (197.53), which is an important context.

---

### 1.9 Torque Acceleration / Jerk Terminology Fixed

Section III-D now defines:
> *"While termed 'torque jerk' following wearable robotics convention [9], this metric physically represents the second time-derivative of joint moment (τ̈, more precisely termed torque acceleration)"*

This clarification is in the main text, not a footnote. ✅

---

### 1.10 Power Loss Algebraic Identity Disclosed

Section III-D explicitly states:
> *"the residual simplifies to (τ̂_j − τ∗_j)²q̇_j² — an angular-velocity-squared weighted surrogate regularizer rather than an active energy-conserving invariant"*

**This is a significant and honest disclosure. Well done.**

---

### 1.11 Streaming / O(1) Claim Appropriately Qualified

Section VII Limitations now includes:
> *"(7) Streaming Deployment: Offline stride windows (T = 100) were evaluated here; live O(1) step-wise recurrent streaming on microcontrollers is an ongoing development milestone."*

And the main text now says:
> *"in continuous-time streaming, diagonal SSMs mathematically admit O(1) step-wise recurrent updates"*

This is now framed as a theoretical mathematical property, not a validated implementation. **Acceptable.**

---

## PART II — ISSUES PARTIALLY RESOLVED ⚠️

### 2.1 — `hernandez2023transformer` Reference — Removed But Replacement Unclear

The fake `hernandez2023transformer` (Transformers for human motion analysis, IEEE RBME) appears to have been **replaced**. References [18] and [19] are now:

- [18]: Li, Li, Fu — *"Attention-based temporal convolutional networks for biomechanical motion prediction"* — IEEE TNSRE 2023
- [19]: Hossain, Guo, Choi — *"Estimation of lower extremity joint moments... using IMU sensors"* — IEEE JBHI 2023

These are plausible references. However, **[18] cannot be fully verified** from the abbreviated citation — the authors "S. Li, Y. Li, and Y. Fu" with this exact title in TNSRE vol. 31 (2023) pp. 1012–1022 should be confirmed against the IEEE Xplore database before submission.

**Action still required:** Verify reference [18] has a working DOI and is accessible in IEEE Xplore.

---

### 2.2 — Primary Dataset Citation — Provenance Improved But Dataset URL Is Circular

Reference [33] now reads:
```
C. Le Goff, A. Muller, and F. Multon, "Six-speed treadmill gait dataset
with full-body kinematics, kinetics, and musculoskeletal analysis,"
Open Science Framework/Zenodo repository, 2026, data available at:
https://github.com/m-mujeeb-ur-rahman/physics-guided-gait-s4d.
```

And the text adds:
> *"collected under institutional ethical review (University Hospital of Rennes / INRIA Ethics Committee) with informed written consent"*

**Assessment:** The ethics committee and consent statement is a significant improvement. However:

1. The data URL (`https://github.com/m-mujeeb-ur-rahman/physics-guided-gait-s4d`) points to the **code repository**, not to the dataset itself. The 44-subject kinetics dataset (262 trials of motion capture + force plate data) is not available via a GitHub code repository due to file size constraints.
2. The citation still says "Open Science Framework/Zenodo repository" but provides a GitHub URL — these are different repositories with different archiving guarantees.
3. The original Mendeley DOI (`10.17632/3dmymnb65h.1`) is gone, which is correct. But no verified OSF or Zenodo DOI has replaced it.

**Action still required:** Provide an actual working OSF or Zenodo DOI for the dataset (e.g., `https://doi.org/10.17605/OSF.IO/xxxxx` or `https://zenodo.org/record/xxxxxxx`). A GitHub code repo URL is not an acceptable primary data citation.

---

### 2.3 — Bland-Altman Limitation — Disclosed But Not Fixed

The text now states:
> *"Residual points reflect pooled time-series strides across 7 subjects with intra-stride temporal autocorrelation"*

And states the hip bias of +0.040 is *"a systematic portion of total error"* rather than "negligible."

However, the analysis is **still applied on pooled time-series points** (≈1,200 autocorrelated observations). The autocorrelation structure is disclosed in text but is not corrected methodologically. A proper Bland-Altman for repeated-measures would compute **one summary statistic per subject** (e.g., mean moment per stride phase per subject), producing N=7 independent points.

**Assessment:** Disclosure is an improvement. The methodology remains technically incorrect, but the limitation acknowledgment is sufficient for peer review if the text is explicit about it.

---

### 2.4 — Overreaching Clinical Claims — Reduced But Not Eliminated

Removed: "decisive," "extreme significance," "indispensable" (in some instances).

**Still present:**
- *"Command chattering is hypothesized in clinical literature to trigger involuntary spasms and skin shear"* — the word "hypothesized" is now used, which is acceptable.
- *"our formulation dampens transient extension spikes, hypothesized to mitigate hyperextension risks in active orthoses"* — "hypothesized" now present ✅
- Section VI-B: ankle A2, knee K1/K4, hip H3 power phases still referenced without per-phase data in the tables.
- *"For causal SSMs lacking future temporal context, auxiliary physics regularization provides this indispensable smoothing mechanism."* — the word **"indispensable"** is still present (Discussion, second paragraph).

**Action still required:** Remove or qualify "indispensable." The BiLSTM achieves essentially the same jerk (89.19 LOSO vs. 93.31 for M3) without any physics loss, which directly contradicts calling physics regularization "indispensable." For causal models it is useful but not indispensable if one considers architecture choice.

---

## PART III — ISSUES NOT FIXED ❌

### 3.1 — Trial-Level Statistics: Pseudo-Replication Remains (CRITICAL)

The paper now discloses:
> *"Trial-level paired analysis on 40 test trials... pools observations across 7 subjects; subject-level aggregation confirms significant jerk reduction (N=7, Wilcoxon W=0, p=0.0156 one-sided, paired t=−4.89, p=0.0027)"*

**Improvement:** The subject-level result is now also reported alongside the trial-level result.

**Remaining problem:** The trial-level p < 10⁻¹² (from 40 trials, 7 subjects) is still reported as the primary statistic, and it is still pseudo-replication. Forty trials from 7 subjects are not 40 independent observations. The correct primary statistic is the subject-level one (N=7, p=0.0027). The trial-level result should either be removed or explicitly labeled as "not accounting for within-subject clustering."

The subject-level one-sided p=0.0156 (Wilcoxon) is still the minimum achievable value for N=6 pairs (if W=0, all pairs in the same direction) for one-sided tests. The two-sided p would be 0.03125 — still above the typical 0.01 threshold and should be stated as such.

**Action required:** Promote the subject-level statistics to primary. Label the trial-level statistics as exploratory/descriptive only.

---

### 3.2 — Table III Baseline Results Are Still Single-Seed

Table III footnote states: *"Baseline models reflect seed 42."*

This is an improvement in transparency — at least it is now disclosed. However, the comparison is still methodologically asymmetric: S4D variants are multi-seed averaged (3 seeds) while BiLSTM and TCN are single-seed (seed 42 only). The table presents these alongside each other without uncertainty intervals for the baselines.

A single seed TCN achieving r=0.8347 may be an outlier (lucky seed). The true multi-seed mean could be lower. This is unfair in either direction — it could be accidentally making TCN look better or worse than it truly is.

**Action required:** Either run BiLSTM and TCN cross-dataset evaluations with all 3 seeds and add ±std, or add an explicit caveat sentence in the text: *"Cross-dataset baseline results reflect a single initialization and may not represent the full variance of baseline performance."*

---

### 3.3 — Fig. 3(d–f): Single Trial, Undisclosed Selection

Figure 3 caption now reads:
> *"Panels d–f illustrate a representative knee joint from a held-out test trial (trial 12), exhibiting high-frequency attenuation consistently observed across test subjects."*

The "(trial 12)" disclosure is new — this is an improvement. However:
1. "Consistently observed" is an assertion without supporting data (no aggregate spectral analysis across all 40 trials is shown).
2. The "−34% Jerk" label in Fig. 3f is hard-coded text, not computed from this specific trial's jerk. The trial-specific jerk reduction at trial 12 is not reported.

**Action required:** Report the actual jerk reduction for trial 12 in the caption or footnote, and either provide aggregate spectral data (e.g., mean FFT across all 40 trials) or soften "consistently observed" to "for example, as illustrated here."

---

### 3.4 — λ_jerk Selection Is Methodologically Invalid (Unaddressed)

The paper now explains: *"λ_jerk = 10⁻⁸ balances the raw finite-difference magnitude 1/(Δt)² ∼ 10⁴ against the task MSE (∼10⁻²)"*

This provides physical intuition but does not fix the methodological problem: **the λ grid `{10⁻⁶, 10⁻⁷, 10⁻⁸}` was searched by minimizing validation MSE.** Since any non-zero jerk penalty increases MSE, minimizing validation MSE will always select the smallest λ in the grid (10⁻⁸), regardless of what values are in the grid. This is not hyperparameter optimization — it is an implicit bias toward no regularization.

**Action required:** Either (a) show the Pareto front of jerk reduction vs. accuracy across the λ grid, or (b) acknowledge that λ_jerk=10⁻⁸ is at the grid floor and the choice is sensitivity to the magnitude 1/(Δt)² rather than a data-driven optimum.

---

### 3.5 — No Trivial Baselines (Unaddressed)

No mean-waveform predictor, ridge regression, or feature ablation has been added. The pooled Pearson r ≈ 0.94 for all models on the time-normalized gait cycle remains potentially inflated by mean waveform shape. Without a mean-waveform baseline, it is impossible to know what fraction of the r = 0.94 is subject-specific versus population-mean.

---

### 3.6 — No Training/Validation Loss Curves (Unaddressed)

50 epochs on CPU with no convergence evidence provided. This is still a concern.

---

## PART IV — NEW ISSUES INTRODUCED IN THE REVISION

### 4.1 — Warmup Implementation Still Unverified

The paper now claims a "10-epoch linear ramp-up of auxiliary physics weights." This is different from the original warmup claim (which was a warmup of the learning rate). Checking `loso_runner.py`, the `run_training` call at line 105 passes `warmup_epochs=10`, and `train.py` does implement a warmup ramp for λ values. This is now consistent.

However, the paper does **not** disclose that the multi-seed (`run_multi_seed.py`) experiments may use different default warmup parameters than the LOSO runner. **Verify** that the three-seed experiments in `multi_seed_summary.json` also used 10-epoch warmup.

---

### 4.2 — Contributions Claim "Parameter-Matched Baselines" — Partially Incorrect

Section I states:
> *"we benchmark against parameter-matched BiLSTM (173k params) and Dilated TCN (171k params) architectures"*

Table I confirms:
- BiLSTM: 173,123; TCN: 171,267; M1: 172,547 — these **are** parameter-matched to each other ✅
- M2 and M3: **189,699** — these are **not** parameter-matched to the baselines

The introduction says "parameter-matched" as if all models are matched, but M2 and M3 are 10% larger. Section IV-C correctly lists the parameters, so the data is there — but the framing in the introduction is slightly misleading. The comparison between M2/M3 and the baselines is not parameter-matched.

---

### 4.3 — Reference [32] and [33] — Dataset Attribution Is Split Across Two Citations

The text cites [32] (CusToM toolbox, Muller et al.) and [33] (Le Goff et al., six-speed dataset) together for the primary dataset. However, the CusToM paper (JOSS 2019) describes a software tool, not the dataset. The dataset citation [33] points to a GitHub URL. This citation structure could confuse readers about what is the actual dataset source vs. the processing tool.

**Action required:** Separate these clearly in the text: "[32] provides the musculoskeletal simulation toolbox; [33] is the primary gait dataset used in this study."

---

## SUMMARY FOR THE STUDENT

### Things you fixed correctly — well done:
1. ✅ The abstract is now honest and accurate
2. ✅ LOSO table transparently shows BiLSTM dominance
3. ✅ Cross-dataset table leads with TCN winning
4. ✅ Baud et al. reference correctly cited with right journal and authors
5. ✅ FFT claim removed and replaced with correct O(T·N) description
6. ✅ AdamW, weight decay, and λ_jerk now correctly stated
7. ✅ Power loss algebraic identity (velocity-weighted MSE) honestly disclosed
8. ✅ Sign convention explained in Eq. 12
9. ✅ Target jerk RMS (197.53) added to Table I
10. ✅ "Torque acceleration" definition moved to main text

### Things you must still fix before submission:

| Priority | Action |
|---|---|
| 🔴 High | Replace GitHub URL in dataset citation [33] with an actual OSF or Zenodo DOI |
| 🔴 High | Promote subject-level statistics (N=7) as primary; label trial-level (N=40) as descriptive |
| 🟠 Medium | Add multi-seed results for BiLSTM and TCN in Table III, or add explicit single-seed caveat |
| 🟠 Medium | Remove "indispensable" from Discussion — BiLSTM achieves similar jerk without physics |
| 🟠 Medium | Verify reference [18] (Li, Li, Fu — TNSRE 2023) has a working IEEE DOI |
| 🟠 Medium | Show Pareto front for λ_jerk selection or acknowledge the grid-edge issue |
| 🟡 Low | Report actual jerk reduction for the specific trial shown in Fig. 3 (trial 12) |
| 🟡 Low | Add convergence evidence (training curves) or note 50 epochs as a limitation |
| 🟡 Low | Clarify that "parameter-matched" applies to M1 vs. baselines, not M2/M3 |

---

*Assessment completed: 29 September 2026*
*Based on full text extraction of `main (1).pdf` (8 pages, 45,375 chars) verified against original JSON experiment files.*
