# COMPAS Assignment 4 – From Accuracy to Accountability: Stress Testing a Predictive Model

## Purpose of the Analysis

Hi Professor, this assignment implements a full end-to-end reliability audit of the COMPAS recidivism prediction pipeline. The goal is to move beyond measuring in-sample performance and stress-test the model across five audit dimensions: distribution drift, generalization, spurious-correlation probing, robustness, and slice-based evaluation. The audit asks not just *how well* the model performs, but *for whom*, *under what conditions*, and *for what reasons* — the three questions that separate responsible deployment from mere technical accuracy.

> **Note:** This notebook is cumulative. Assignments 1, 2, and 3 occupy cells 1–50. **Assignment 4 begins at cell 51.** All prior cells must be run first as they build the `df` object, feature engineering, and model pipeline that Assignment 4 depends on.

---

## What This Project Does

This project reproduces the Lecture 4 COMPAS pipeline and extends it into a structured reliability audit. Two models are evaluated — Logistic Regression (interpretable baseline) and Gradient-Boosted Trees (higher-capacity model) — across the following five audit dimensions:

| Part | Audit Dimension | Core Question |
|------|----------------|---------------|
| A | Distribution Drift | Has the data-generating process shifted between train and test? |
| B | Generalization | Does in-training performance transfer to unseen data? |
| C | Spurious-Correlation Probe | Does the model respond to protected attributes (race, gender) held in isolation? |
| D | Robustness | Does performance degrade gracefully when `priors_count` is perturbed? |
| E | Slice-Based Evaluation | Does aggregate performance mask failure modes for specific subgroups? |

---

## Python Libraries Used

- `pandas`, `numpy` — data manipulation
- `matplotlib` — visualization
- `scikit-learn` — model training, metrics, permutation importance
- `scipy` — KS test, statistical testing
- `shap` — feature importance (used in prior assignments, referenced here)

---

## Key Tasks Completed

**Part A — Distribution Drift**
- Computed PSI (Population Stability Index) for `priors_count`
- Ran KS tests for all numeric features (`priors_count`, `age`, `decile_score`)
- Computed MMD² (Maximum Mean Discrepancy) in the encoded feature space with permutation testing
- Compared train vs. test predicted score distributions visually and statistically

**Part B — Generalization**
- Compared train vs. test AUC, accuracy, and log loss for both LR and GBT
- Computed generalization gaps and diagnosed overfitting
- Ran permutation importance to identify feature shortcuts (including `race_factor`)

**Part C — Spurious-Correlation Probe**
- Ran counterfactual race swap: set all defendants to African-American vs. Caucasian, measured ΔP(High-Risk)
- Ran counterfactual gender swap: Male vs. Female, same methodology
- Measured both mean effect size and proportion of observations significantly affected

**Part D — Robustness**
- Global sensitivity sweep of `priors_count` from 0 to 30 for both models
- ICE (Individual Conditional Expectation) curves stratified by race group
- Computed sensitivity index Vⱼ for both models
- Noise injection test: added Gaussian noise to `priors_count` at σ = 0, 0.5, 1.0, 2.0, 5.0

**Part E — Slice-Based Evaluation**
- Computed AUC, Accuracy, FPR, and FNR by race, gender, and age slices
- Visualized FPR and FNR disparities across all subgroup slices
- Compared LR vs. GBT AUC per race slice to test whether model complexity benefits all groups equally

---

## Key Findings

### Part A — No Distribution Drift
All three drift tests confirmed IID train/test splits. PSI = 0.010 (well below the 0.10 stability threshold), KS p-values all > 0.20, and MMD² permutation p = 0.620. This rules out covariate shift as an explanation for any generalization gap.

### Part B — Minimal Overfitting, But a Critical Shortcut
LR shows a negative AUC gap (−0.004), meaning the model generalizes perfectly. GBT shows mild overfitting (AUC gap = +0.014). More importantly, `race_factor` ranks 3rd in permutation importance (0.029) — a governance red flag, since race carries no legitimate causal weight in recidivism prediction.

### Part C — Strong Evidence of Proxy Discrimination
The race counterfactual swap produces a mean ΔP = **+0.087**: setting a defendant's race to African-American (all else equal) raises the predicted high-risk probability by 8.7 percentage points on average, and 64% of the test set is affected by more than 5pp. This is direct causal evidence of racial encoding. The gender swap produces ΔP = −0.023, with females predicted slightly higher risk — a statistical artifact of how female defendants are distributed in the training data.

### Part D — Graceful but Unequal Degradation
Both models degrade gracefully under noise (AUC remains above 0.78 even at σ = 5.0). LR is more sensitive in terms of Vⱼ (0.065 vs. GBT 0.047), but GBT degrades faster under noise injection. ICE curves reveal racial heterogeneity: African-American defendants show steeper response slopes to `priors_count` increases than Caucasian defendants — the same criminal history increase generates a larger risk penalty for Black defendants.

### Part E — Severe Subgroup Failure Modes
The most critical finding: **African-American FPR = 0.364 vs. Caucasian FPR = 0.099** — a 3.7× ratio. 36% of Black defendants who will not recidivate are incorrectly labeled high-risk, vs. 10% of white defendants. The reverse holds for FNR (Caucasian FNR = 0.469 vs. African-American FNR = 0.228). The youngest defendants ("Less than 25") show an FPR of 0.541 — the most severe single-slice failure in the entire model. Notably, GBT performs *worse* than LR for African-Americans (AUC 0.816 vs. 0.820), demonstrating that model complexity does not uniformly benefit all groups.

---

## Instructions to Reproduce the Results

1. Open `COMPAS_Assignment4_HassanAlshamrani.ipynb` in Google Colab
2. Run **all cells from top to bottom** — cells 1–50 set up the data and models; Assignment 4 starts at **cell 51**
3. All five parts will execute and generate outputs automatically
4. No additional data downloads are required — data is fetched from the ProPublica GitHub repository

---

## Overall Audit Verdict

The model is technically stable (no drift, small generalization gap) but morally fragile. It encodes race as a predictive shortcut (+8.7pp counterfactual effect), applies disproportionate false alarms to Black defendants (3.7× FPR ratio) and young defendants (54% FPR for under-25), and the more complex GBT model worsens — not improves — performance for the most affected group.

Deployment in any consequential context requires at minimum: race-constrained retraining, threshold recalibration for FPR parity, and ongoing slice-level monitoring as a condition of responsible use. A model that performs well on average can still fail under shift, fail for subgroups, or fail for the wrong reasons — this audit provides evidence across all three dimensions.

---

Thanks, Professor.  
**Hassan Alshamrani**
