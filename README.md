# Batch Dependence in Multilingual Attribution of Frontier LLMs

A follow-up to *Interpretable Multilingual Attribution of Frontier LLMs* (Erlikhman, Tarekegn and Juang, IEEE BigData 2026 High School Symposium; [LangLLM repo](https://github.com/adrian-erlikhman/LangLLM)).

LangLLM shows that 21 interpretable style features identify which of five frontier LLMs wrote an essay, and that a classifier trained in one language transfers to six others at 48.6% (chance 20%). That cross-language protocol z-scores each target language using all of its test essays at once. This project asks what happens when you don't have that batch.

**This project uses LangLLM's 21 features and classifier unchanged.** The only thing that varies is how the target language is normalised:

| Arm | Target-language statistics come from |
|---|---|
| none | nowhere: source-language statistics only, so a single essay can be scored |
| batch | all target-language test essays (LangLLM's setup) |
| reference | a separate, small, unlabeled set of target-language texts on other prompts |

## Main results (pre-registered)

| | Result |
|---|---|
| Single essays vs. batch, on 420 new essays collected about 2 months later | −20.5 points (49.1% → 28.7%) |
| A candidate-model reference set from 2 months later recovers | 89% of the gap (10 essays: over 80%) |
| An outside-model reference (Llama, Mistral, Qwen) recovers | 54% (below the 50%-with-confidence bar) |
| A human (Wikipedia) reference recovers | 48% |
| Plain accuracy when one model writes 80% of the batch | −18 points, mostly that model's own essays |

On LangLLM's own essays (prompt-disjoint splits): none 27.7%, reference 43.4%, batch 46.0%. Our code reproduces LangLLM's published transfer matrix exactly (48.61%), and LangLLM's extractor reproduces its `features.csv` exactly at Stanza 1.14.0.

Exploratory analyses: multilingual-e5 embeddings show the same dependence at about half the size, and essays truncated to 50–200 words lose most of the style features' accuracy. See `results/main_report.md` for everything, including negative results.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Pre-registered plans for the pilot, follow-ups and main study, with addenda 1–4 |
| `prompts/` | 12 new prompt schemas, 84 native prompts, the review log |
| `data/new/` | N1: 420 essays from the 5 LangLLM models. N2: 251 essays from Llama 3.1 8B, Mistral Small 3.2 24B and Qwen3 14B (raw API/Ollama records, including retries) |
| `data/human/` | N3: 588 Wikipedia excerpts (CC BY-SA 4.0, with title, page and revision IDs) |
| `data/features/` | Features for N1, N2, N3 and the truncated N1 essays |
| `src/` | Collection, features, analyses and figures |
| `results/` | Reports, summaries (JSON), tables and figures |

## Reproduce

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
sh scripts/fetch_langllm.sh                       # LangLLM data and extractor (not redistributed here)
cd src
../.venv/bin/python reproduce.py                  # LangLLM's transfer matrix, exactly
../.venv/bin/python pilot.py && ../.venv/bin/python followup.py
../.venv/bin/python features_new.py check         # extractor vs. LangLLM's features.csv
../.venv/bin/python main_study.py                 # R1-R4, H1-H5
../.venv/bin/python short_texts.py && ../.venv/bin/python embed.py && ../.venv/bin/python extra.py && ../.venv/bin/python robustness.py
```

Stanza models for `features_new.py` are downloaded once into `vendor/langllm/stanza_resources_1.14/` (see `stanza.download`). Re-collecting essays needs an OpenRouter key in `.env` (N1) and Ollama (N2). The collected essays are already included.

## Credit

The essays, features, extractor and prompt-writing templates come from LangLLM and are used with Adrian Erlikhman's permission. LangLLM's essays and code are not redistributed in this repository.
