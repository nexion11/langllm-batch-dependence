# Pilot plan (fixed before any pilot result was computed)

Written 2026-11-06. This file is committed before `src/pilot.py` is run on the data. Rules below decide what the pilot means; they are not changed after seeing results.

## Question
Erlikhman et al. report 48.6% mean cross-language attribution accuracy (5 models, chance 20%). Their transfer z-scores every feature within the test language using all of that language's essays: a balanced batch of all five models, answering the same 12 prompts the classifier trained on. The pilot asks how much of that number depends on the batch and on the shared prompts.

## Data
Only `data/original/features.csv` from LangLLM commit 06109dc (839 essays, 21 features). No new essays and no API calls.

## Experiments
**Step 0, reproduction.** Rebuild their transfer in our own code: within-language median imputation, within-language z-score, StandardScaler and logistic regression (C=1). Train on all of language A, test on all of language B, for the 42 ordered pairs.

**Experiment A, test batch (original protocol, same prompts).** The training side is unchanged. The test language is normalised in one of three ways:
- **Batch:** statistics come from the test batch itself.
- **None:** statistics come from the source language, so nothing is used from the test language. This is the single-essay case.
- **Full:** statistics come from all of the test language, which is the original method.

Two sweeps, each with 200 random draws per language pair:
- **Mix sweep:** batches of 20, where the dominant model is 20%, 40%, 60% or 80% of the batch. The dominant model rotates over all five.
- **Size sweep:** balanced batches of 5, 10, 20 or 40.

Metric: balanced accuracy, meaning mean recall over the five models, pooled over draws.

**Experiment B, prompt-disjoint reference set.** 50 random splits of the 12 prompts into halves, H1 and H2. Train on language A's H1 essays and test on language B's H2 essays, so no test prompt is seen in training. Three arms:
- **Batch:** statistics from B-H2, the full balanced test batch.
- **Reference:** statistics from B-H1, unlabeled essays on other prompts.
- **None:** statistics from A-H1.

Metric: accuracy, averaged over pairs and splits.

**Experiment C, unknown models.** For each model u, train a 4-class classifier on A-H1 without u, and test on all of B-H2. The unknown score is 1 minus the maximum predicted probability. Report AUROC (u's essays against the known essays), pooled over pairs and splits, for the reference arm with statistics from B-H1 excluding u, and for the none arm.

## Decision rules
- **G2, reproduction:** mean off-diagonal accuracy within 0.02 of 0.4861. If it fails, stop and debug before reading anything else.
- **G3a, batch effect:** the effect "matters" if either holds:
  - Balanced accuracy at an 80% mix is at least 3 points below the 20% mix, averaged over pairs and dominant models.
  - In Experiment B, the none arm is at least 3 points below the batch arm.

  If neither holds, batch dependence is not the headline. Make rejection the headline, or switch to the backup.
- **G3b, recovery:** the reference set "recovers most" of the gap if (reference − none) / (batch − none) ≥ 0.5 in Experiment B. This is only computed when batch − none ≥ 3 points.
- **G3c, rejection:** a pooled AUROC of 0.55 or below in the reference arm means rejection is a brief negative result, not a headline.

## Limits
The pilot reuses one collection date and 12 prompts. Draws within a language pair share essays, so spreads across draws are not confidence intervals. Proper prompt-clustered bootstrap intervals come in the main study.
