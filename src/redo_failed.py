"""Re-collect essays that fail LangLLM's validation (main study plan, step 6).

Validation: language-ID confidence at least 0.6, 150-900 word-equivalents, a normal stop, and no refusal.
A failed cell is requested again with the next seed (1001, 1002, ... up to 3 retries). The retry is appended
to the same data/new/{model}.jsonl with "attempt" > 1. The analysis keeps, per cell, the last attempt that passes.

    python redo_failed.py --dry-run
    python redo_failed.py
"""
import argparse
import json
import re

from lingua import Language, LanguageDetectorBuilder

from collect_new import ALL, run
from new_prompts import load
from study import GENERATION, NEW, ollama_digests

LANG = {"en": Language.ENGLISH, "es": Language.SPANISH, "zh": Language.CHINESE, "ru": Language.RUSSIAN,
        "ja": Language.JAPANESE, "tr": Language.TURKISH, "hi": Language.HINDI}
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
REFUSAL = re.compile(r"I can't|I cannot|I'm sorry|as an AI|no puedo|我不能|не могу|できません|yapamam|मैं नहीं कर सकता", re.I)
DET = LanguageDetectorBuilder.from_all_languages().build()


def word_equiv(text, lang):
    if lang == "zh":
        return round(len(CJK.findall(text)) / 1.6)
    if lang == "ja":
        return round(len(CJK.findall(text)) / 2.2)
    return len(text.split())


def problems(r):
    out = []
    if r["error"]:
        return ["error"]
    if r["finish_reason"] != "stop":
        out.append(f"finish={r['finish_reason']}")
    n = word_equiv(r["text"], r["lang"])
    if not 150 <= n <= 900:
        out.append(f"length={n}")
    if DET.compute_language_confidence(r["text"], LANG[r["lang"]]) < 0.6:
        out.append("language")
    if REFUSAL.search(r["text"][:300]):
        out.append("refusal")
    return out


def latest(key):
    """Last record per cell (later attempts replace earlier ones)."""
    p = NEW / f"{key}.jsonl"
    cells = {}
    for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if p.exists() else []:
        cells[r["cell_id"]] = r
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(ALL))
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    digests = None
    for key in a.models:
        for cid, r in latest(key).items():
            bad = problems(r)
            if not bad:
                continue
            attempt = r.get("attempt", 1)
            print(f"{cid}: {', '.join(bad)} (attempt {attempt})", flush=True)
            if a.dry_run or attempt >= a.max_attempts:
                continue
            if ALL[key][1] == "N2" and digests is None:
                digests = ollama_digests()
            job = {k: r[k] for k in ("cell_id", "model_key", "model", "set", "prompt_id", "lang", "gen", "fk_tier", "stance")}
            job["prompt"] = load(r["lang"])[r["prompt_id"]]["prompt"]
            while attempt < a.max_attempts:
                attempt += 1
                seed = GENERATION["seed"] + attempt - 1
                GENERATION_SEED = {**GENERATION, "seed": seed}
                new = run(job, digests, params=GENERATION_SEED)
                new.update(attempt=attempt, retry_reason=bad)
                with open(NEW / f"{key}.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(new, ensure_ascii=False) + "\n")
                bad = problems(new)
                print(f"   attempt {attempt} (seed {seed}): {'pass' if not bad else ', '.join(bad)}", flush=True)
                if not bad:
                    break


if __name__ == "__main__":
    main()
