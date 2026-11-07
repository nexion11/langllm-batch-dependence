"""Exploratory analyses E1 (shorter texts) and E2 (embedding baseline), docs/main_study_plan.md addendum 3.
Writes results/extra_summary.json.
"""
import json

import numpy as np
import pandas as pd

from common import LANGS, RESULTS, Linear, norm, stats
from main_study import FEATURES, SPLITS, Tally, impute, load_all

ROOT = RESULTS.parent
SEED = 20261109


def rec(none, batch, x):
    return round((x - none) / (batch - none), 3)


def r1_simple(train, test, refs, cols):
    """Train on A's train rows (all prompts); test on B's test rows. refs: name -> frame of reference rows."""
    t = Tally(sorted(test.prompt_id.unique()))
    for a in LANGS:
        da = train[train.lang == a]
        Xa = da[cols].to_numpy(float)
        st_a = stats(Xa)
        clf = Linear(norm(Xa, st_a), da.model_key.to_numpy())
        for b in LANGS:
            if b == a:
                continue
            dt = test[test.lang == b]
            Xt, yt, pt = dt[cols].to_numpy(float), dt.model_key.to_numpy(), dt.prompt_id.to_numpy()
            hit = lambda st: clf.predict(norm(Xt, st)) == yt
            t.add("none", pt, hit(st_a))
            t.add("batch", pt, hit(stats(Xt)))
            for name, r in refs.items():
                t.add(name, pt, hit(stats(r[r.lang == b][cols].to_numpy(float))))
    return {k: round(t.acc(k), 4) for k in t.c}


def r2_simple(o, refs, cols, rng):
    prompts = sorted(o.prompt_id.unique())
    t = Tally(prompts)
    for _ in range(SPLITS):
        p = rng.permutation(prompts)
        h1, h2 = set(p[:6]), set(p[6:])
        for a in LANGS:
            da = o[(o.lang == a) & o.prompt_id.isin(h1)]
            Xa = da[cols].to_numpy(float)
            st_a = stats(Xa)
            clf = Linear(norm(Xa, st_a), da.model_key.to_numpy())
            for b in LANGS:
                if b == a:
                    continue
                dt = o[(o.lang == b) & o.prompt_id.isin(h2)]
                Xt, yt, pt = dt[cols].to_numpy(float), dt.model_key.to_numpy(), dt.prompt_id.to_numpy()
                hit = lambda st: clf.predict(norm(Xt, st)) == yt
                t.add("none", pt, hit(st_a))
                t.add("batch", pt, hit(stats(Xt)))
                t.add("ref_original", pt, hit(stats(o[(o.lang == b) & o.prompt_id.isin(h1)][cols].to_numpy(float))))
                for name, r in refs.items():
                    t.add(name, pt, hit(stats(r[r.lang == b][cols].to_numpy(float))))
    return {k: round(t.acc(k), 4) for k in t.c}


def with_recovery(d):
    out = dict(d)
    for k, v in d.items():
        if k.startswith("ref_"):
            out[f"recovery_{k}"] = rec(d["none"], d["batch"], v)
    return out


def e1():
    s = pd.read_csv(ROOT / "data" / "features" / "short.csv")
    out = {}
    for n in sorted(s.length.unique()):
        d = impute(s[s.length == n])
        o, n1 = d[d.set == "original"], d[d.set == "N1"]
        out[int(n)] = with_recovery(r1_simple(o, n1, {"ref_original": o}, FEATURES))
    return out


def e2():
    z = np.load(ROOT / "data" / "features" / "embeddings.npz")
    cols = [f"e{i}" for i in range(z["vec"].shape[1])]
    emb = pd.DataFrame(z["vec"], columns=cols)
    emb["cell_id"] = z["cell_id"]
    o, n1, n2, n3 = load_all()
    meta = lambda d: d[["cell_id", "model_key", "prompt_id", "lang"]].merge(emb, on="cell_id")
    o, n1, n2, n3 = meta(o), meta(n1), meta(n2), meta(n3)
    r1 = with_recovery(r1_simple(o, n1, {"ref_original": o, "ref_n2": n2, "ref_human": n3}, cols))
    r2 = with_recovery(r2_simple(o, {"ref_n1": n1, "ref_n2": n2, "ref_human": n3}, cols, np.random.default_rng(SEED)))
    return {"R1": r1, "R2": r2, "dims": len(cols), "model": str(z["model"])}


def main():
    out = {"E1_shorter_texts_R1": e1(), "E2_embedding": e2()}
    (RESULTS / "extra_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
