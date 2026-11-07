# Main study plan (pre-registered)

Approved by Aariv and committed 2026-11-06, before any new essay was collected. This commit is the pre-registration. Later changes are logged in a "Deviations" section.

## Questions
1. **Replication:** does the single-essay collapse hold on new prompts, collected two months later?
2. **Later reference set:** does a reference set from the candidate models, collected later on different prompts, still recover the gap?
3. **Outside reference set:** does a reference set from models outside the candidate set recover the gap? If yes, the fix needs no access to the candidates.
4. **Imbalance:** does plain accuracy fall under lopsided test batches? This was exploratory in the pilot.

Dropped from the original plan: collecting essays from unseen models to test rejection. The pilot showed rejection does not work with these features (AUROC 0.49), so it stays a short negative result on existing data.

## New essays (all at LangLLM's settings)
Settings, as in LangLLM:
- Temperature 0.7, 4,000 max tokens, reasoning effort low with the trace excluded.
- No system prompt; one user turn in the target language.
- The same validation: language ID confidence at least 0.6, 150–900 words, and a refusal check.

**Prompts:** 12 new prompt schemas with the same structure as LangLLM's: topic, stance, three sub-claims and reading level, with stance split 6/6 and reading level 4/4/4. Native prompts are written by Llama 4 Maverick and reviewed by Qwen 3.7 Plus, as in LangLLM.

| Set | Models | Essays | Est. cost |
|---|---|---|---|
| N1. Later candidate set | GPT-5.5, Gemini 3.5 Flash, Claude Opus 4.7, Grok 4.3, DeepSeek V4 Pro 0813 | 5 × 7 languages × 12 prompts = 420 | about $6.10 |
| N2. Outside reference set | Llama 4 Scout (Meta), Mistral Small 3.2 24B (Mistral), Qwen 3.8 Flash (Alibaba) | 3 × 7 × 12 = 252 | under $0.50 |
| Prompt writing and review | Llama 4 Maverick, Qwen 3.7 Plus | 84 prompts plus reviews | under $0.50 |

- **Total:** about $7. Set a hard OpenRouter credit limit of $15.
- **Outside-model choice:** none of the outside models comes from a candidate's vendor. Gemma (Google) and gpt-oss (OpenAI) are excluded for that reason.
- **Caveat:** Llama 4 Scout shares a family with the prompt writer.
- **Rates:** prices are OpenRouter list prices on 2026-11-06. The per-essay cost uses LangLLM's logged average of $0.0146.

## Features
The 21 features must be computed exactly as LangLLM computes them, using Stanza parsing.
- **Preferred:** reuse LangLLM's feature code, if Adrian's permission covers code, and cite it.
- **Check:** recompute features for a sample of LangLLM's raw essays and compare them with the published `features.csv`. The study only proceeds if they match.

## Analyses
- **R1, replication with prompts never seen in training:**
  - **Setup:** train on language A's original essays (all 12 prompts), and test on language B's N1 essays (new prompts, new date).
  - **Arms:** none, batch (all N1 essays in B), reference from B's original essays (September), and reference from B's N2 essays (outside models).
- **R2, later reference on the original test set:**
  - **Setup:** the prompt-disjoint splits from the pilot, testing on B's original H2 essays.
  - **Arms:** none, batch, same-date reference (B's H1), later candidate reference (N1 in B), and outside reference (N2 in B), each outside model alone and all three pooled.
  - **Sizes:** each reference arm is run at 5, 10, 20 and all essays.
- **R3, imbalance on the N1 test batches:** report plain accuracy, balanced accuracy and the dominant model's recall at 20%, 40%, 60% and 80% dominant shares, broken down by dominant model.
- **R4, clue table and length control:** repeat the follow-up F3 and F4 analyses with the N1 and N2 reference sets.

## Decision rules
- **Statistics:** prompt-clustered bootstrap, 1,000 resamples. Holm correction across H1–H4.
- **H1:** in R1, none is at least 10 points below batch.
- **H2:** in R2, the later candidate reference recovers at least 50% of the gap.
- **H3:** in R2, the pooled outside reference recovers at least 50% of the gap.
- **H4:** in R3, plain accuracy at an 80% mix is at least 10 points below the 20% mix.
- **Reporting:** every result is reported whatever the outcome. The headline is chosen from what holds.

## Steps
1. Aariv approves this plan and the 12 prompt topics.
2. Aariv adds an OpenRouter API key to `.env` (never shared in chat) and sets the $15 credit limit.
3. Confirm the scope of Adrian's permission (essays, features, code), and paste his reply into `docs/permission.md`.
4. Commit this plan. That commit is the pre-registration.
5. Write the prompts and review them, including a light spot-check if Aariv can read any of the 7 languages.
6. Collect N1 and N2, validate them, and re-collect failures.
7. Compute features and check them against LangLLM's.
8. Run R1–R4, then write the report and figures.

On a MacBook this needs no GPU. Collection should take under an hour at LangLLM's pace. Stanza parsing for 672 essays should take about 15 minutes, plus a one-time model download for each of the 7 languages.

## Addendum 1 (2026-11-07, added before any new essay or human text was collected)

**N3. Human reference set.** For each of the 7 languages, 84 excerpts of human-written Wikipedia text drawn from random articles.
- **Source:** the MediaWiki API, plain-text extracts, CC BY-SA 4.0. Title, page ID, revision ID and URL are logged for each excerpt.
- **Excerpts:** consecutive prose paragraphs from the start of the article. Headings and list lines are skipped, as are paragraphs under 40 word-equivalents. Paragraphs are added until the excerpt reaches about 350 word-equivalents, the same length target as the essays. Articles that cannot reach 150 word-equivalents are discarded.
- **Cost:** $0. No AI model is involved.
- **Genre mismatch:** Wikipedia is expository, not persuasive. That mismatch is part of what the test measures.

**H5.** In R2, a human reference set (N3 in language B) recovers at least 50% of the gap. It is also reported in R1, at the same reference sizes as the other arms.

**Reference-free comparison (descriptive, not a hypothesis).** The classifier is trained and tested on only the 12 features whose language shift was under 1.5 SD in the follow-up clue table, which was computed on LangLLM's essays before any new data existed:
question_rate, first_person_rate, burstiness, para_count, sent_len_sd, para_len_mean, colon_per_1k, dep_depth, comma_per_1k, semicolon_per_1k, digit_rate, sent_len_mean.

It is reported in the none arm, with no target-language statistics, next to the full 21-feature arms. Prior work (Bleaching Text, 2018; DT-grams, 2021) already shows that abstract features transfer across languages, so this is a comparison, not a claim.

Holm correction now covers H1–H5.

## Addendum 2 (2026-11-07, added before any N1 or N2 essay was collected)

**N2 now runs locally in Ollama, at no cost.** The reason is cost: the outside set only has to be text from models outside the candidate set, so it does not need the exact models listed above. N1 is unchanged. It still uses the five LangLLM models through OpenRouter, because the classifier's labels are those five models.

| Key | Ollama model | Vendor | Replaces |
|---|---|---|---|
| llama | `llama3.1:8b` | Meta | Llama 4 Scout (too large for 24 GB of memory) |
| mistral | `mistral-small3.2:24b` | Mistral | Mistral Small 3.2 24B (the same model, run locally) |
| qwen | `qwen3:14b` | Alibaba | Qwen 3.8 Flash (not available locally) |

- **Settings:** the same as LangLLM where Ollama supports them. Temperature 0.7, a 4,000-token output cap, seed 1000, and the native prompt as a single user turn with no system prompt. Qwen's thinking mode is turned off (`think: false`), which plays the role of "low reasoning effort, excluded".
- **Differences from the API:** Ollama runs 4-bit quantised weights on the MacBook (Apple M4 Pro, 24 GB). Exact digests are logged with every essay. The smaller models are also weaker writers in Hindi and Turkish. Both points are reported as limitations, because they make the outside set less like the candidates, which is the very thing H3 tests.
- **Vendor rule:** still met. None of the three comes from a candidate's vendor.
- **Hypotheses and analyses:** unchanged.

## Addendum 3 (2026-11-07, added after the main results; exploratory)
These two analyses were added after R1-R4 were computed. They are exploratory, not confirmatory, and are reported as such. Nothing in R1-R4 or H1-H5 changes.

**E1. Shorter texts.**
- **Texts:** every original and N1 essay is cut to its first 50, 100 and 200 word-equivalents. Word-equivalents follow the same rule as validation: CJK characters ÷ 1.6 for zh and ÷ 2.2 for ja, whitespace tokens otherwise. Cuts fall at token or character boundaries.
- **Features:** recomputed with LangLLM's extractor.
- **Analysis:** the R1 setup run at each length. Train on language A's original essays cut to length N; test on language B's N1 essays cut to length N.
- **Arms:** none, batch, and reference (B's original essays cut to length N).
- **Question:** how does single-text accuracy, with and without a reference, change with text length?

**E2. Neural-embedding baseline.**
- **Embeddings:** each text is embedded with `intfloat/multilingual-e5-base` (prefix "passage: ", mean pooling, 512-token limit).
- **Analysis:** R1 and R2 are repeated with the embedding instead of the 21 features, using the same classifier, arms and seeds. Reference arms use the original, N1, pooled N2 and N3 sets at full size.
- **Question:** does the dependence on batch statistics, and the reference-set recovery, also hold for a neural multilingual representation?

## Addendum 4 (2026-11-07, after an outside review of the draft; exploratory robustness checks)
None of these change H1-H5 or R1-R4.

1. **E2 correction.** E5's model card says to use the `query: ` prefix when embeddings serve as features for classification, and its example L2-normalises them. E2 used `passage: ` with no L2 normalisation.
   - E2 is re-run as E2b: `query: ` prefix, mean pooling, L2-normalised, 512-token limit.
   - Two pipelines are compared: (a) the paper's pipeline (z-scoring arms, then StandardScaler and logistic regression); (b) the usual embedding pipeline (L2-normalised vectors, then logistic regression with no per-dimension scaling and no target-language statistics).
   - The original E2 numbers are kept and reported as superseded.
2. **Imbalance isolation (R3b).** The evaluation essays are fixed: all 60 N1 essays of language B, which are balanced. Only the set used to estimate the normalisation statistics varies: 15 essays drawn from B's N1 at a 20/40/60/80% dominant share, 50 draws, averaged over the five dominant models. This separates "bad statistics" from "a different test mix".
3. **Reference-sampling spread.** R1 with candidate references of 5, 10 and 20 essays, drawn 200 times. Each draw uses one reference set per test language, and accuracy is averaged over the 42 pairs. We report the 2.5-97.5 percentile range across draws.
4. **Per-language breakdown** of R1 (none, reference, batch), by test language.
