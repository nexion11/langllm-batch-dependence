"""N3: human reference excerpts from random Wikipedia articles (docs/main_study_plan.md, addendum 1).

Uses the MediaWiki API's plain-text extracts. Text is CC BY-SA 4.0; each excerpt keeps its
title, page ID, revision ID and URL for attribution. Resumable.

    python human_reference.py            # 84 excerpts per language
    python human_reference.py --per 10   # quick test
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request

from study import LANGUAGES, ROOT

OUT = ROOT / "data" / "human"
UA = "llm-fingerprint-paper/0.1 (student research script; Wikipedia random-article sampler)"
WIKI = {"en": "en", "es": "es", "zh": "zh", "ru": "ru", "ja": "ja", "tr": "tr", "hi": "hi"}
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
END = re.compile(r"[.!?。！？।]\s*$")


def word_equiv(text, lang):
    """Same word-equivalent rule as LangLLM's validator: CJK characters / 1.6 (zh) or / 2.2 (ja), else whitespace tokens."""
    if lang == "zh":
        return round(len(CJK.findall(text)) / 1.6)
    if lang == "ja":
        return round(len(CJK.findall(text)) / 2.2)
    return len(text.split())


def get(lang, params):
    url = f"https://{WIKI[lang]}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            err = e
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"Wikipedia API failed for {lang}: {err}")


def excerpt(text, lang):
    """Leading prose paragraphs until about 350 word-equivalents; None if under 150."""
    paras = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("=") or re.match(r"^([-*•]|\d+[.)])\s", line):
            continue
        if word_equiv(line, lang) < 40 or not END.search(line):
            continue
        paras.append(line)
        if word_equiv("\n\n".join(paras), lang) >= 350:
            break
    body = "\n\n".join(paras)
    n = word_equiv(body, lang)
    return (body, n) if 150 <= n <= 900 else (None, n)


def sample(lang, per):
    path = OUT / f"{lang}.jsonl"
    have = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    seen = {r["pageid"] for r in have}
    tries = 0
    while len(have) < per and tries < per * 15:
        tries += 1
        d = get(lang, {"action": "query", "generator": "random", "grnnamespace": 0, "grnlimit": 1,
                       "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "wiki",
                       "rvprop": "ids", "inprop": "url"})
        page = d["query"]["pages"][0]
        if page["pageid"] in seen or not page.get("revisions"):
            continue
        seen.add(page["pageid"])
        body, n = excerpt(page.get("extract", ""), lang)
        if body:
            rec = {"cell_id": f"wiki|{page['pageid']}|{lang}|0", "lang": lang, "title": page["title"],
                   "pageid": page["pageid"], "revid": page["revisions"][0]["revid"], "url": page["fullurl"],
                   "license": "CC BY-SA 4.0", "word_equiv": n, "text": body}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            have.append(rec)
        time.sleep(0.3)
    print(f"[{lang}] {len(have)}/{per} excerpts after {tries} random pages", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=84)
    ap.add_argument("--langs", nargs="*", default=list(LANGUAGES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in a.langs:
        sample(lang, a.per)


if __name__ == "__main__":
    main()
