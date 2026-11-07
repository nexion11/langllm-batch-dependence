"""Compute LangLLM's 21 features with Adrian Erlikhman's own extractor (vendor/langllm, commit 06109dc, used with
the author's permission; fetch it with scripts/fetch_langllm.sh). Nothing in the extractor is changed.

    python features_new.py check   # recompute 5 of his essays per language, compare with his features.csv
    python features_new.py new     # data/features/new.csv: N1 + N2 essays (last passing attempt per cell)
    python features_new.py human   # data/features/human.csv: N3 Wikipedia excerpts
"""
import json
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "langllm"))
sys.modules["langllm.collect"] = types.SimpleNamespace(iter_raw=lambda: iter(()))  # only needed by his CLI, not extract()
from langllm.features import FEATURE_NAMES, extract  # noqa: E402
from tqdm import tqdm  # noqa: E402

OUT = ROOT / "data" / "features"
WINDOW = 50  # his config.yaml features.mattr_window


def rows_for(records, out_path):
    have = pd.read_csv(out_path) if out_path.exists() else None
    done = set(have["cell_id"]) if have is not None else set()
    rows = []
    try:
        for r in tqdm([r for r in records if r["cell_id"] not in done], unit="doc"):
            rows.append({**{k: r.get(k) for k in ("cell_id", "model_key", "model_served", "set", "prompt_id", "lang",
                                                 "gen", "fk_tier", "stance", "attempt")},
                         **extract(r["text"], r["lang"], WINDOW)})
    finally:  # keep partial progress
        df = pd.concat([d for d in (have, pd.DataFrame(rows)) if d is not None and len(d)], ignore_index=True)
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"{len(df)} rows -> {out_path}")
    return df


def check(n=5):
    ref = pd.read_csv(ROOT / "data" / "original" / "features.csv").set_index("cell_id")
    raw = {r["cell_id"]: r for f in (ROOT / "data" / "original" / "raw").glob("*.jsonl")
           for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
    pick = ref.groupby("lang").head(n).index
    diffs = []
    for cid in tqdm(pick, unit="doc"):
        got = extract(raw[cid]["text"], raw[cid]["lang"], WINDOW)
        for f in FEATURE_NAMES:
            a, b = got[f], ref.loc[cid, f]
            diffs.append({"cell_id": cid, "lang": ref.loc[cid, "lang"], "feature": f, "ours": a, "his": b,
                          "abs_diff": abs(a - b) if pd.notna(a) and pd.notna(b) else (0.0 if pd.isna(a) and pd.isna(b) else float("inf"))})
    d = pd.DataFrame(diffs)
    exact = (d["abs_diff"] < 1e-6).mean()
    print(f"{len(pick)} essays x 21 features: {exact:.1%} match to 1e-6")
    bad = d[d["abs_diff"] >= 1e-6]
    if len(bad):
        print(bad.groupby(["lang", "feature"])["abs_diff"].agg(["count", "max"]).to_string())
    return d


def new():
    recs = {}
    for f in sorted((ROOT / "data" / "new").glob("*.jsonl")):
        for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if not r["error"]:
                recs[r["cell_id"]] = r  # later attempts replace earlier ones
    sys.path.insert(0, str(ROOT / "src"))
    from redo_failed import problems
    keep = [r for r in recs.values() if not problems(r)]
    print(f"{len(keep)} valid essays of {len(recs)} cells")
    return rows_for(keep, OUT / "new.csv")


def human():
    recs = [{**r, "model_key": "human", "set": "N3", "prompt_id": None, "gen": 0}
            for f in sorted((ROOT / "data" / "human").glob("*.jsonl"))
            for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())]
    return rows_for(recs, OUT / "human.csv")


if __name__ == "__main__":
    {"check": check, "new": new, "human": human}[sys.argv[1]]()
