"""Main study analyses R1-R4 and hypotheses H1-H5 (docs/main_study_plan.md, addenda 1-2).

Inputs: LangLLM's features (data/original/features.csv), our N1 and N2 essays (data/features/new.csv) and the
N3 Wikipedia excerpts (data/features/human.csv), all computed with LangLLM's extractor at Stanza 1.14.0.
Writes results/main_*.csv and results/main_summary.json.

Analysis details the plan left open, fixed here before the results were seen (see results/main_report.md):
- **Missing values:** filled with the within-language median of the same set (original, N1, N2 or N3).
- **Reference subsets:** candidate sets (original and N1) are drawn balanced, k/5 per model. Outside and
  human sets are drawn at random. Each size has 10 draws in R2 and 50 in R1.
- **R3 batch size:** 15, not the pilot's 20, because N1 has only 12 essays per model and language. Dominant
  counts are 3, 6, 9 and 12 (20-80%); the rest are split as evenly as possible across the other four models.
- **Bootstrap:** resample the test prompts with replacement (1,000 times). Normalisation statistics and
  draws are held fixed, and essays are reweighted by how often their prompt was drawn. Each hypothesis's
  p-value is the share of resamples on the wrong side of its threshold. Holm correction across H1-H5.
"""
import json
import time

import numpy as np
import pandas as pd

from common import DATA, FEATURES, LANGS, MODELS, RESULTS, Linear, balanced_accuracy, norm, stats

ROOT = DATA.parents[1]
SEED = 20261108
SPLITS = 50
SIZES = [5, 10, 20]
B = 1000
OUTSIDE = ["llama", "mistral", "qwen"]
FREE12 = ["question_rate", "first_person_rate", "burstiness", "para_count", "sent_len_sd", "para_len_mean",
          "colon_per_1k", "dep_depth", "comma_per_1k", "semicolon_per_1k", "digit_rate", "sent_len_mean"]


def impute(df):
    df = df.copy()
    for f in FEATURES:
        df[f] = df.groupby("lang")[f].transform(lambda s: s.fillna(s.median()))
    df["logn"] = np.log(df.n_tokens.clip(lower=1))
    return df


def load_all():
    o = impute(pd.read_csv(DATA / "features.csv"))
    new = pd.read_csv(ROOT / "data" / "features" / "new.csv")
    n1 = impute(new[new.set == "N1"])
    n2 = impute(new[new.set == "N2"])
    n3 = impute(pd.read_csv(ROOT / "data" / "features" / "human.csv"))
    return o, n1, n2, n3


class Tally:
    """Correct counts per (key, prompt), for point estimates and prompt-clustered bootstrap."""

    def __init__(self, prompts):
        self.prompts = list(prompts)
        self.ix = {p: i for i, p in enumerate(self.prompts)}
        self.c, self.n = {}, {}

    def add(self, key, prompt_ids, correct, weight=1.0):
        c = self.c.setdefault(key, np.zeros(len(self.prompts)))
        n = self.n.setdefault(key, np.zeros(len(self.prompts)))
        idx = np.fromiter((self.ix[p] for p in prompt_ids), int, len(prompt_ids))
        np.add.at(c, idx, weight * correct.astype(float))
        np.add.at(n, idx, weight)

    def acc(self, key, w=None):
        w = np.ones(len(self.prompts)) if w is None else w
        return float(self.c[key] @ w / (self.n[key] @ w))

    def boot(self, rng, fn):
        k = len(self.prompts)
        out = np.empty(B)
        for i in range(B):
            w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
            out[i] = fn(lambda key: self.acc(key, w))
        return out


def arr(d, cols=FEATURES):
    return d[cols].to_numpy(float)


def draw_balanced(rng, d, k):
    idx = np.concatenate([rng.choice(np.flatnonzero(d.model_key.to_numpy() == m),
                                     min(k // 5, int((d.model_key == m).sum())), replace=False) for m in MODELS])
    return d.iloc[idx]


def draw_random(rng, d, k):
    return d.iloc[rng.choice(len(d), min(k, len(d)), replace=False)]


def ref_sets(o_ref, n1, n2, n3, b):
    """Reference sets in test language b: name -> (frame, drawer)."""
    sets = {"ref_original": (o_ref, draw_balanced), "ref_n1": (n1[n1.lang == b], draw_balanced),
            "ref_n2": (n2[n2.lang == b], draw_random), "ref_human": (n3[n3.lang == b], draw_random)}
    for m in OUTSIDE:
        sets[f"ref_{m}"] = (n2[(n2.lang == b) & (n2.model_key == m)], None)
    return sets


def score_arms(tally, tag, clf, Xt, yt, pt, st_none, st_batch, sets, rng, draws, cols=FEATURES):
    hit = lambda st: clf.predict(norm(Xt, st)) == yt
    tally.add((tag, "none", "all"), pt, hit(st_none))
    tally.add((tag, "batch", "all"), pt, hit(st_batch))
    for name, (d, drawer) in sets.items():
        tally.add((tag, name, "all"), pt, hit(stats(arr(d, cols))))
        if drawer is None:
            continue
        for k in SIZES:
            for _ in range(draws):
                tally.add((tag, name, k), pt, hit(stats(arr(drawer(rng, d, k), cols))), 1.0 / draws)


def r1(o, n1, n2, n3, rng):
    """Train on A's original essays (all prompts); test on B's N1 essays."""
    t = Tally(sorted(n1.prompt_id.unique()))
    for cols, tag in ((FEATURES, "R1"), (FREE12, "R1_free12")):
        for a in LANGS:
            da = o[o.lang == a]
            Xa = arr(da, cols)
            st_a = stats(Xa)
            clf = Linear(norm(Xa, st_a), da.model_key.to_numpy())
            for b in LANGS:
                if b == a:
                    continue
                dt = n1[n1.lang == b]
                Xt, yt, pt = arr(dt, cols), dt.model_key.to_numpy(), dt.prompt_id.to_numpy()
                if tag == "R1_free12":
                    hit = clf.predict(norm(Xt, st_a)) == yt
                    t.add((tag, "none", "all"), pt, hit)
                    continue
                sets = ref_sets(o[o.lang == b], n1, n2, n3, b)
                del sets["ref_n1"]  # the test set itself
                score_arms(t, tag, clf, Xt, yt, pt, st_a, stats(Xt), sets, rng, 50, cols)
    return t


def r2(o, n1, n2, n3, rng):
    """Prompt-disjoint splits on the original essays; later, outside and human reference sets."""
    prompts = sorted(o.prompt_id.unique())
    t = Tally(prompts)
    clue_rows, length_rows = [], []
    for s in range(SPLITS):
        p = rng.permutation(prompts)
        h1, h2 = set(p[:6]), set(p[6:])
        for a in LANGS:
            da = o[(o.lang == a) & o.prompt_id.isin(h1)]
            Xa, ya = arr(da), da.model_key.to_numpy()
            st_a = stats(Xa)
            clf = Linear(norm(Xa, st_a), ya)
            st_a12 = stats(arr(da, FREE12))
            clf12 = Linear(norm(arr(da, FREE12), st_a12), ya)
            fit_a = length_fit(Xa, da.logn.to_numpy())
            ra = Xa - pred_len(da.logn.to_numpy(), fit_a)
            st_ra = stats(ra)
            clf_len = Linear(norm(ra, st_ra), ya)
            for b in LANGS:
                if b == a:
                    continue
                dt = o[(o.lang == b) & o.prompt_id.isin(h2)]
                Xt, yt, pt = arr(dt), dt.model_key.to_numpy(), dt.prompt_id.to_numpy()
                sets = ref_sets(o[(o.lang == b) & o.prompt_id.isin(h1)], n1, n2, n3, b)
                score_arms(t, "R2", clf, Xt, yt, pt, st_a, stats(Xt), sets, rng, 10)
                t.add(("R2_free12", "none", "all"), pt, clf12.predict(norm(arr(dt, FREE12), st_a12)) == yt)
                # R4: clue table (F3) and length control (F4) with the N1 and N2 reference sets
                for name in ("ref_original", "ref_n1", "ref_n2"):
                    dr = sets[name][0]
                    st_r = stats(arr(dr))
                    for j, f in enumerate(FEATURES):
                        mu, sd = st_r[0].copy(), st_r[1].copy()
                        mu[j], sd[j] = st_a[0][j], st_a[1][j]
                        clue_rows.append({"split": s, "train": a, "test": b, "ref": name, "feature": f,
                                          "acc": float(np.mean(clf.predict(norm(Xt, (mu, sd))) == yt))})
                    nr = dr.logn.to_numpy()
                    fit_r = length_fit(arr(dr), nr)
                    rr = arr(dr) - pred_len(nr, fit_r)
                    rt = Xt - pred_len(dt.logn.to_numpy(), fit_r)
                    length_rows.append({"split": s, "train": a, "test": b, "ref": name,
                                        "len_ref_acc": float(np.mean(clf_len.predict(norm(rt, stats(rr))) == yt))})
                rt_none = Xt - pred_len(dt.logn.to_numpy(), fit_a)
                fit_t = length_fit(Xt, dt.logn.to_numpy())
                rt_b = Xt - pred_len(dt.logn.to_numpy(), fit_t)
                length_rows.append({"split": s, "train": a, "test": b, "ref": "none",
                                    "len_ref_acc": float(np.mean(clf_len.predict(norm(rt_none, st_ra)) == yt))})
                length_rows.append({"split": s, "train": a, "test": b, "ref": "batch",
                                    "len_ref_acc": float(np.mean(clf_len.predict(norm(rt_b, stats(rt_b))) == yt))})
    return t, pd.DataFrame(clue_rows), pd.DataFrame(length_rows)


def length_fit(X, logn):
    lc = logn - logn.mean()
    slope = (lc @ (X - X.mean(axis=0))) / (lc @ lc)
    return slope, X.mean(axis=0) - slope * logn.mean()


def pred_len(logn, fit):
    return np.outer(logn, fit[0]) + fit[1]


def r3(o, n1, rng, draws=50):
    """Lopsided test batches of 15 from B's N1 essays; train on A's original essays."""
    t = Tally(sorted(n1.prompt_id.unique()))
    rows = []
    for a in LANGS:
        da = o[o.lang == a]
        st_a = stats(arr(da))
        clf = Linear(norm(arr(da), st_a), da.model_key.to_numpy())
        for b in LANGS:
            if b == a:
                continue
            db = n1[n1.lang == b]
            st_ref = stats(arr(o[o.lang == b]))
            by = {m: np.flatnonzero(db.model_key.to_numpy() == m) for m in MODELS}
            for dom in MODELS:
                for d in (3, 6, 9, 12):
                    others = [m for m in MODELS if m != dom]
                    for _ in range(draws):
                        rest = 15 - d
                        cnt = {m: rest // 4 for m in others}
                        for m in rng.choice(others, rest % 4, replace=False):
                            cnt[m] += 1
                        cnt[dom] = d
                        idx = np.concatenate([rng.choice(by[m], min(c, len(by[m])), replace=False) for m, c in cnt.items()])
                        dt = db.iloc[idx]
                        Xt, yt, pt = arr(dt), dt.model_key.to_numpy(), dt.prompt_id.to_numpy()
                        share = round(100 * d / 15)
                        for arm, st in (("batch", stats(Xt)), ("ref_original", st_ref), ("none", st_a)):
                            p = clf.predict(norm(Xt, st))
                            t.add(("R3", arm, share), pt, p == yt, 1.0 / draws)
                            rows.append({"train": a, "test": b, "dominant": dom, "share": share, "arm": arm,
                                         "accuracy": float(np.mean(p == yt)), "balanced_accuracy": balanced_accuracy(yt, p),
                                         "dominant_recall": float(np.mean(p[yt == dom] == dom))})
    return t, pd.DataFrame(rows)


def holm(p):
    order = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, h in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[h]))
        out[h] = run
    return out


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    o, n1, n2, n3 = load_all()
    print(f"loaded: original {len(o)}, N1 {len(n1)}, N2 {len(n2)}, N3 {len(n3)}", flush=True)
    t1 = r1(o, n1, n2, n3, rng)
    print(f"R1 done {time.time() - t0:.0f}s", flush=True)
    t2, clue, length = r2(o, n1, n2, n3, rng)
    print(f"R2/R4 done {time.time() - t0:.0f}s", flush=True)
    t3, r3rows = r3(o, n1, rng)
    print(f"R3 done {time.time() - t0:.0f}s", flush=True)

    table = []
    for t in (t1, t2, t3):
        for key in t.c:
            table.append({"analysis": key[0], "arm": key[1], "size": str(key[2]), "accuracy": round(t.acc(key), 4)})
    table = pd.DataFrame(table)
    table.to_csv(RESULTS / "main_accuracy.csv", index=False)
    clue.to_csv(RESULTS / "main_clues.csv", index=False)
    length.to_csv(RESULTS / "main_length.csv", index=False)
    r3rows.to_csv(RESULTS / "main_imbalance.csv", index=False)

    brng = np.random.default_rng(SEED + 1)
    rec = lambda tag, arm: (lambda A: (A((tag, arm, "all")) - A((tag, "none", "all")))
                            / (A((tag, "batch", "all")) - A((tag, "none", "all"))))
    stat = {
        "H1": (t1, lambda A: A(("R1", "batch", "all")) - A(("R1", "none", "all")), 0.10),
        "H2": (t2, rec("R2", "ref_n1"), 0.5),
        "H3": (t2, rec("R2", "ref_n2"), 0.5),
        "H4": (t3, lambda A: A(("R3", "batch", 20)) - A(("R3", "batch", 80)), 0.10),
        "H5": (t2, rec("R2", "ref_human"), 0.5),
    }
    hyp, pvals = {}, {}
    for h, (t, fn, thr) in stat.items():
        est = fn(t.acc)
        bs = t.boot(brng, fn)
        pvals[h] = float(np.mean(bs < thr))
        hyp[h] = {"estimate": round(est, 4), "threshold": thr, "ci95": ci(bs), "p": round(pvals[h], 4)}
    for h, adj in holm(pvals).items():
        hyp[h].update(p_holm=round(adj, 4), holds=bool(adj < 0.05))

    recovery = {}
    for tag, t in (("R1", t1), ("R2", t2)):
        for key in t.c:
            if key[0] == tag and key[1].startswith("ref_"):
                none, batch = t.acc((tag, "none", "all")), t.acc((tag, "batch", "all"))
                recovery[f"{tag}|{key[1]}|{key[2]}"] = round((t.acc(key) - none) / (batch - none), 3)
    base = {f"{tag}|{arm}": round(t.acc((tag, arm, "all")), 4) for tag, t in (("R1", t1), ("R2", t2))
            for arm in ("none", "batch")}
    base.update({"R1_free12|none": round(t1.acc(("R1_free12", "none", "all")), 4),
                 "R2_free12|none": round(t2.acc(("R2_free12", "none", "all")), 4)})
    imb = r3rows.groupby(["share", "arm"])[["accuracy", "balanced_accuracy", "dominant_recall"]].mean().round(4)
    imb_dom = r3rows[(r3rows.arm == "batch")].groupby(["dominant", "share"]).accuracy.mean().unstack().round(4)
    cl = clue.groupby(["ref", "feature"]).acc.mean().unstack(0)
    ref_acc = {r: t2.acc(("R2", r, "all")) for r in ("ref_original", "ref_n1", "ref_n2")}
    clue_drop = {r: (100 * (ref_acc[r] - cl[r])).sort_values(ascending=False).round(2).head(8).to_dict() for r in ref_acc}
    out = {
        "counts": {"original": len(o), "N1": len(n1), "N2": len(n2), "N3": len(n3)},
        "hypotheses": hyp,
        "baselines": base,
        "recovery": recovery,
        "R3_imbalance": {f"{s}|{a}": v for (s, a), v in imb.to_dict(orient="index").items()},
        "R3_batch_accuracy_by_dominant": imb_dom.to_dict(orient="index"),
        "R4_clue_drop_points": clue_drop,
        "R4_length_controlled": length.groupby("ref").len_ref_acc.mean().round(4).to_dict(),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "main_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
