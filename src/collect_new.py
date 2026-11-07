"""Collect the new essays: N1 (5 candidates) and N2 (3 outside models), one essay per model, prompt and language.

N1 goes through OpenRouter; N2 runs locally in Ollama, one model at a time (addendum 2).
Same request as LangLLM's collector: the native prompt as one user turn, no system prompt,
temperature 0.7, 4,000 max tokens, low reasoning effort with the trace excluded. Resumable:
cells already in data/new/{model}.jsonl without an error are skipped.

    python collect_new.py --dry-run
    python collect_new.py                 # everything missing
    python collect_new.py --models llama  # a subset
"""
import argparse
import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from study import (CANDIDATES, GENERATION, LANGUAGES, NEW, OUTSIDE, chat, ollama_chat, ollama_digests, schemas, spent_usd,
                   text_of, usage_of)
from new_prompts import load

ALL = {**{k: (v, "N1") for k, v in CANDIDATES.items()}, **{k: (v, "N2") for k, v in OUTSIDE.items()}}


def done(key):
    p = NEW / f"{key}.jsonl"
    if not p.exists():
        return set()
    return {r["cell_id"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if not r["error"]}


def jobs(keys, langs):
    out = []
    for k in keys:
        have = done(k)
        for lang in langs:
            native = load(lang)
            for s in schemas():
                cid = f"{k}|{s['id']}|{lang}|0"
                if cid in have:
                    continue
                if s["id"] not in native:
                    raise SystemExit(f"missing prompt {s['id']} for {lang}: run new_prompts.py write")
                out.append({"cell_id": cid, "model_key": k, "model": ALL[k][0], "set": ALL[k][1], "prompt_id": s["id"],
                            "lang": lang, "gen": 0, "fk_tier": s["fk_tier"], "stance": s["stance"],
                            "prompt": native[s["id"]]["prompt"]})
    return out


def run(job, digests=None, params=GENERATION):
    rec = {k: v for k, v in job.items() if k != "prompt"}
    rec["requested_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    rec["params"] = params
    msgs = [{"role": "user", "content": job["prompt"]}]
    try:
        if job["set"] == "N2":
            rec["digest"] = (digests or {}).get(job["model"])
            resp = ollama_chat(job["model"], msgs, temperature=params["temperature"],
                               max_tokens=params["max_tokens"], seed=params["seed"])
        else:
            resp = chat(job["model"], msgs, temperature=params["temperature"], max_tokens=params["max_tokens"],
                        seed=params["seed"], reasoning=params["reasoning"])
        ch = resp["choices"][0]
        rec.update(text=text_of(resp), model_served=resp.get("model"), provider=resp.get("provider"),
                   finish_reason=ch.get("finish_reason"), usage=usage_of(resp), error=None)
    except Exception as e:
        rec.update(text="", model_served=None, provider=None, finish_reason="error", usage={}, error=str(e)[:500])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(ALL))
    ap.add_argument("--langs", nargs="*", default=list(LANGUAGES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    todo = jobs(a.models, a.langs)
    print(f"{len(todo)} essays to collect; logged spend so far ${spent_usd():.2f}")
    if a.dry_run or not todo:
        return
    NEW.mkdir(parents=True, exist_ok=True)
    api = [j for j in todo if j["set"] == "N1"]
    local = sorted((j for j in todo if j["set"] == "N2"), key=lambda j: j["model_key"])  # one model loaded at a time
    i = 0

    def save(r):
        nonlocal i
        i += 1
        with open(NEW / f"{r['model_key']}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if i % 25 == 0 or r["error"]:
            print(f"{i}/{len(todo)} {r['cell_id']} {'ERROR ' + r['error'][:120] if r['error'] else ''}", flush=True)

    if api:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(run, j) for j in api]):
                save(fut.result())
    if local:
        digests = ollama_digests()
        missing = {j["model"] for j in local} - set(digests)
        if missing:
            raise SystemExit(f"not pulled in Ollama: {sorted(missing)} (run: ollama pull <model>)")
        for j in local:
            save(run(j, digests))
    print(f"done; logged spend ${spent_usd():.2f}")


if __name__ == "__main__":
    main()
