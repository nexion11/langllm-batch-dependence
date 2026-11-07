# Follow-up plan (fixed before any follow-up result was computed)

Written 2026-11-06, after the pilot, and committed before `src/followup.py` is run. It uses the same data and the prompt-disjoint setup as pilot Experiment B: train on language A's H1 prompts, test on language B's H2 prompts, 42 language pairs, and 50 new random 6/6 prompt splits (seed 20261107).

"Gap" means batch accuracy minus none accuracy in the same split and pair. "Recovery" means (arm − none) / gap, computed on means over pairs and splits.

## F1. Reference-size curve
- **Design:** draw a balanced reference set of k essays per language from B-H1, with k/5 essays per model, for k = 5, 10, 20, 40 and all 60. Ten draws per split and pair.
- **Rule:** if k = 10 recovers at least 50% of the gap, the paper may call the fix "cheap: about two essays per model".

## F2. Lopsided and incomplete reference sets
- **(a) Dominance:** one model contributes d essays and the other four contribute 1 each, for d = 1, 4, 8 or 12 (dominant shares of 20%, 50%, 67% and 75%). The dominant model rotates, with ten draws each.
- **(b) Coverage:** the reference contains only m of the 5 models, 12 essays each, for m = 1 to 5. Every subset of that size is used.
- **Rule:** if a 1-model reference recovers less than 50% of the gap, the paper states that the reference set must cover the candidate models.

## F3. Clue table (descriptive)
For each of the 21 features, starting from the reference arm, normalise that one feature with source-language statistics (as in the none arm) and the rest with the reference set. The resulting accuracy drop measures how much the feature relies on target-language statistics.

Also reported per feature is the language shift: the mean over pairs of |mean_B − mean_A| / sd_A, using H1 essays.

## F4. Length control without whole-language statistics
Within each normalisation source, regress each feature on log(n_tokens) and keep the residuals. The source is A-H1 for training, and B-H1 (reference), B-H2 (batch) or A-H1 (none) for the test. Residuals are then z-scored with statistics from the same source.

**Rule:** if the length-controlled reference arm is within 3 points of the uncontrolled one, the reference result is not driven by length.

## Pre-registration for the main study, not computed here
- **Imbalance analysis:** in the main study, the imbalance analysis will report plain accuracy and the dominant model's recall alongside balanced accuracy, broken down by dominant model.
- **Exploratory status:** the pilot observation (plain accuracy falling 49% → 31% at an 80% mix) stays exploratory until it is tested on the new essays.
