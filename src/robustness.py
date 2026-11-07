"""Addendum 4 robustness checks. Writes results/robustness_summary.json.

1. E2b: E5 embeddings with the 'query: ' prefix, L2-normalised. (a) The paper's pipeline (z-scoring arms);
   (b) the usual embedding pipeline (logistic regression on the L2 vectors, no scaling, no target statistics).
2. R3b: imbalance isolation. Fixed balanced evaluation essays; only the normalisation sample is lopsided.
3. Reference-sampling spread for R1 at 5, 10 and 20 candidate reference essays (200 draws).
4. Per-language R1 breakdown.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import LANGS, MODELS, RESULTS, Linear, norm, stats
from extra import r1_simple, r2_simple, with_recovery
from main_study import FEATURES, load_all

ROOT = RESULTS.parent
SEED = 20261110


def emb_frames(path):
    z = np.load(ROOT / "data" / "features" / path)
    cols = [f"e{i}" for i in range(z["vec"].shape[1])]
    emb = pd.DataFrame(z["vec"], columns=cols)
    emb["cell_id"] = z["cell_id"]
    meta = lambda d: d[["cell_id", "model_key", "prompt_id", "lang"]].merge(emb, on="cell_id")
    return [meta(d) for d in load_all()], cols


def plain_lr(o, test, cols, splits=None, rng=None):
    """Usual embedding pipeline: logistic regression on L2-normalised vectors; no per-dimension scaling."""
    accs = []
    if splits is None:  # R1: train on all original essays of A, test on N1 of B
        for a in LANGS:
            da = o[o.lang == a]
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(da[cols].to_numpy(), da.model_key)
            for b in LANGS:
                if b != a:
                    dt = test[test.lang == b]
                    accs.append(np.mean(clf.predict(dt[cols].to_numpy()) == dt.model_key.to_numpy()))
        return round(float(np.mean(accs)), 4)
    prompts = sorted(o.prompt_id.unique())
    for _ in range(splits):
        p = rng.permutation(prompts)
        h1, h2 = set(p[:6]), set(p[6:])
        for a in LANGS:
            da = o[(o.lang == a) & o.prompt_id.isin(h1)]
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(da[cols].to_numpy(), da.model_key)
            for b in LANGS:
                if b != a:
                    dt = o[(o.lang == b) & o.prompt_id.isin(h2)]
                    accs.append(np.mean(clf.predict(dt[cols].to_numpy()) == dt.model_key.to_numpy()))
    return round(float(np.mean(accs)), 4)


def e2b():
    (o, n1, n2, n3), cols = emb_frames("embeddings_query.npz")
    r1 = with_recovery(r1_simple(o, n1, {"ref_original": o, "ref_n2": n2, "ref_human": n3}, cols))
    r2 = with_recovery(r2_simple(o, {"ref_n1": n1, "ref_n2": n2, "ref_human": n3}, cols, np.random.default_rng(SEED)))
    return {"R1_zscore_pipeline": r1, "R2_zscore_pipeline": r2,
            "R1_plain_lr_no_target_stats": plain_lr(o, n1, cols),
            "R2_plain_lr_no_target_stats": plain_lr(o, None, cols, splits=50, rng=np.random.default_rng(SEED))}


def r3b(o, n1, rng, draws=50):
    """Evaluation set fixed (all of B's N1, balanced); normalisation sample of 15 at a given dominant share."""
    rows = []
    for a in LANGS:
        da = o[o.lang == a]
        st_a = stats(da[FEATURES].to_numpy(float))
        clf = Linear(norm(da[FEATURES].to_numpy(float), st_a), da.model_key.to_numpy())
        for b in LANGS:
            if b == a:
                continue
            db = n1[n1.lang == b]
            X, y = db[FEATURES].to_numpy(float), db.model_key.to_numpy()
            by = {m: np.flatnonzero(y == m) for m in MODELS}
            for dom in MODELS:
                others = [m for m in MODELS if m != dom]
                for d in (3, 6, 9, 12):
                    acc, rec_dom = [], []
                    for _ in range(draws):
                        rest = 15 - d
                        cnt = {m: rest // 4 for m in others}
                        for m in rng.choice(others, rest % 4, replace=False):
                            cnt[m] += 1
                        cnt[dom] = d
                        idx = np.concatenate([rng.choice(by[m], c, replace=False) for m, c in cnt.items()])
                        p = clf.predict(norm(X, stats(X[idx])))
                        acc.append(np.mean(p == y))
                        rec_dom.append(np.mean(p[y == dom] == dom))
                    rows.append({"test": b, "dominant": dom, "share": round(100 * d / 15),
                                 "accuracy": float(np.mean(acc)), "dominant_recall": float(np.mean(rec_dom))})
    df = pd.DataFrame(rows)
    return {"by_share": df.groupby("share")[["accuracy", "dominant_recall"]].mean().round(4).to_dict(orient="index"),
            "by_dominant_at_80": df[df.share == 80].groupby("dominant").accuracy.mean().round(4).to_dict()}


def ref_spread(o, n1, rng, draws=200):
    """Spread of R1 accuracy across random balanced reference sets of k original essays."""
    clfs = {}
    for a in LANGS:
        da = o[o.lang == a]
        st_a = stats(da[FEATURES].to_numpy(float))
        clfs[a] = Linear(norm(da[FEATURES].to_numpy(float), st_a), da.model_key.to_numpy())
    out = {}
    for k in (5, 10, 20):
        per_draw = []
        for _ in range(draws):
            accs = []
            for b in LANGS:
                ob = o[o.lang == b]
                idx = np.concatenate([rng.choice(np.flatnonzero(ob.model_key.to_numpy() == m), k // 5, replace=False)
                                      for m in MODELS])
                st_r = stats(ob.iloc[idx][FEATURES].to_numpy(float))
                dt = n1[n1.lang == b]
                Xt, yt = dt[FEATURES].to_numpy(float), dt.model_key.to_numpy()
                accs += [np.mean(clfs[a].predict(norm(Xt, st_r)) == yt) for a in LANGS if a != b]
            per_draw.append(np.mean(accs))
        per_draw = np.array(per_draw)
        out[k] = {"mean": round(float(per_draw.mean()), 4), "p2.5": round(float(np.percentile(per_draw, 2.5)), 4),
                  "p97.5": round(float(np.percentile(per_draw, 97.5)), 4), "min": round(float(per_draw.min()), 4)}
    return out


def per_language(o, n1):
    rows = []
    for a in LANGS:
        da = o[o.lang == a]
        st_a = stats(da[FEATURES].to_numpy(float))
        clf = Linear(norm(da[FEATURES].to_numpy(float), st_a), da.model_key.to_numpy())
        for b in LANGS:
            if b == a:
                continue
            dt = n1[n1.lang == b]
            Xt, yt = dt[FEATURES].to_numpy(float), dt.model_key.to_numpy()
            for arm, st in (("none", st_a), ("reference", stats(o[o.lang == b][FEATURES].to_numpy(float))),
                            ("batch", stats(Xt))):
                rows.append({"test": b, "arm": arm, "acc": np.mean(clf.predict(norm(Xt, st)) == yt)})
    t = pd.DataFrame(rows).groupby(["test", "arm"]).acc.mean().unstack()
    t["recovery"] = (t.reference - t.none) / (t.batch - t.none)
    return t.loc[LANGS, ["none", "reference", "batch", "recovery"]].round(3).to_dict(orient="index")


def main():
    rng = np.random.default_rng(SEED)
    o, n1, n2, n3 = load_all()
    out = {"E2b_embeddings_query_l2": e2b(), "R3b_imbalance_isolation": r3b(o, n1, rng),
           "R1_reference_sampling_spread": ref_spread(o, n1, rng), "R1_per_test_language": per_language(o, n1)}
    (RESULTS / "robustness_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
