"""E1 (addendum 3): features for the original and N1 essays cut to 50, 100 and 200 word-equivalents.

    python short_texts.py      # data/features/short.csv (resumable)
"""
import json
import re

import pandas as pd

from features_new import OUT, ROOT, WINDOW, extract, tqdm

LENGTHS = [50, 100, 200]
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
RATIO = {"zh": 1.6, "ja": 2.2}


def cut(text, lang, n):
    """Leading part of text with n word-equivalents (same counting rule as validation)."""
    if lang in RATIO:
        need = round(n * RATIO[lang])
        for i, m in enumerate(CJK.finditer(text), 1):
            if i == need:
                return text[:m.end()]
        return text
    toks = list(re.finditer(r"\S+", text))
    return text[:toks[n - 1].end()] if len(toks) >= n else text


def records():
    for f in sorted((ROOT / "data" / "original" / "raw").glob("*.jsonl")):
        for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            yield {**r, "set": "original"}
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from redo_failed import latest, problems
    for key in ("gpt", "gemini", "claude", "grok", "deepseek"):
        for r in latest(key).values():
            if not problems(r):
                yield r


def main():
    keep = set(pd.read_csv(ROOT / "data" / "original" / "features.csv").cell_id)  # LangLLM's 839 kept essays
    path = OUT / "short.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["cell_id", "length"])
    done = set(zip(have.cell_id, have.length))
    todo = [(r, n) for r in records() if r["set"] != "original" or r["cell_id"] in keep
            for n in LENGTHS if (r["cell_id"], n) not in done]
    rows = []
    try:
        for r, n in tqdm(todo, unit="doc"):
            rows.append({**{k: r.get(k) for k in ("cell_id", "model_key", "set", "prompt_id", "lang", "gen")},
                         "length": n, **extract(cut(r["text"], r["lang"], n), r["lang"], WINDOW)})
    finally:
        df = pd.concat([d for d in (have, pd.DataFrame(rows)) if len(d)], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"{len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
