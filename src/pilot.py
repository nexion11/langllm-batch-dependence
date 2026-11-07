"""Pilot experiments A, B and C from docs/pilot_plan.md. Writes results/pilot_*.csv and results/pilot_summary.json."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import FEATURES, LANGS, MODELS, RESULTS, Linear, balanced_accuracy, load, norm, stats

SEED = 20261106
DRAWS = 200
SPLITS = 50
MIX = {0.2: 4, 0.4: 8, 0.6: 12, 0.8: 16}  # dominant share -> dominant count in a batch of 20
SIZES = [5, 10, 20, 40]


def pairs():
    return [(a, b) for a in LANGS for b in LANGS if a != b]


def experiment_a(df, rng):
    """Original protocol (same prompts); vary where the test language's statistics come from."""
    X = {l: df.loc[df.lang == l, FEATURES].to_numpy(float) for l in LANGS}
    y = {l: df.loc[df.lang == l, "model_key"].to_numpy() for l in LANGS}
    st = {l: stats(X[l]) for l in LANGS}
    clf = {l: Linear(norm(X[l], st[l]), y[l]) for l in LANGS}
    rows = []
    for a, b in pairs():
        pred_full = clf[a].predict(norm(X[b], st[b]))
        pred_none = clf[a].predict(norm(X[b], st[a]))
        by_model = {m: np.flatnonzero(y[b] == m) for m in MODELS}

        def run(kind, level, dom, counts):
            truth, p_batch, p_full, p_none = [], [], [], []
            for _ in range(DRAWS):
                idx = np.concatenate([rng.choice(by_model[m], counts[m], replace=False) for m in MODELS])
                truth.append(y[b][idx])
                p_batch.append(clf[a].predict(norm(X[b][idx], stats(X[b][idx]))))
                p_full.append(pred_full[idx])
                p_none.append(pred_none[idx])
            t = np.concatenate(truth)
            for arm, p in (("batch", p_batch), ("full", p_full), ("none", p_none)):
                p = np.concatenate(p)
                rows.append({"sweep": kind, "level": level, "dominant": dom, "train": a, "test": b, "arm": arm,
                             "balanced_accuracy": balanced_accuracy(t, p), "accuracy": float(np.mean(t == p))})

        for share, k in MIX.items():
            for d in MODELS:
                run("mix", share, d, {m: k if m == d else (20 - k) // 4 for m in MODELS})
        for n in SIZES:
            run("size", n, "", {m: n // 5 for m in MODELS})
    return pd.DataFrame(rows)


def split_prompts(rng):
    prompts = sorted(pd.unique(DF.prompt_id))
    out = []
    for _ in range(SPLITS):
        p = rng.permutation(prompts)
        out.append((set(p[:6]), set(p[6:])))
    return out


def subset(df, lang, prompts, exclude=None):
    d = df[(df.lang == lang) & df.prompt_id.isin(prompts)]
    if exclude is not None:
        d = d[d.model_key != exclude]
    return d[FEATURES].to_numpy(float), d.model_key.to_numpy()


def experiment_b(df, splits):
    """Prompt-disjoint: train on A's half-1 prompts, test on B's half-2 prompts."""
    rows = []
    for s, (h1, h2) in enumerate(splits):
        for a in LANGS:
            Xa, ya = subset(df, a, h1)
            st_a = stats(Xa)
            clf = Linear(norm(Xa, st_a), ya)
            for b in LANGS:
                if b == a:
                    continue
                Xt, yt = subset(df, b, h2)
                Xr, _ = subset(df, b, h1)
                for arm, st in (("batch", stats(Xt)), ("reference", stats(Xr)), ("none", st_a)):
                    p = clf.predict(norm(Xt, st))
                    rows.append({"split": s, "train": a, "test": b, "arm": arm,
                                 "accuracy": float(np.mean(p == yt)), "balanced_accuracy": balanced_accuracy(yt, p)})
    return pd.DataFrame(rows)


def experiment_c(df, splits):
    """Leave one model out of training; score how well 1 - max probability flags its essays."""
    pooled, rows = {}, []
    for u in MODELS:
        for s, (h1, h2) in enumerate(splits):
            for a in LANGS:
                Xa, ya = subset(df, a, h1, exclude=u)
                st_a = stats(Xa)
                clf = Linear(norm(Xa, st_a), ya)
                for b in LANGS:
                    if b == a:
                        continue
                    Xt, yt = subset(df, b, h2)
                    Xr, _ = subset(df, b, h1, exclude=u)
                    unknown = yt == u
                    for arm, st in (("reference", stats(Xr)), ("none", st_a), ("batch", stats(Xt))):
                        pr = clf.proba(norm(Xt, st))
                        score = 1 - pr.max(axis=1)
                        known_acc = float(np.mean(clf.classes[pr.argmax(axis=1)][~unknown] == yt[~unknown]))
                        key = (u, arm)
                        pooled.setdefault(key, ([], []))
                        pooled[key][0].append(unknown)
                        pooled[key][1].append(score)
                        rows.append({"held_out": u, "split": s, "train": a, "test": b, "arm": arm,
                                     "auroc": roc_auc_score(unknown, score), "known_accuracy": known_acc})
    auroc = []
    for (u, arm), (t, sc) in pooled.items():
        auroc.append({"held_out": u, "arm": arm, "pooled_auroc": roc_auc_score(np.concatenate(t), np.concatenate(sc))})
    return pd.DataFrame(rows), pd.DataFrame(auroc)


def summarise(a, b, c_rows, c_auroc):
    mix = a[a.sweep == "mix"].groupby(["level", "arm"]).balanced_accuracy.mean().unstack()
    size = a[a.sweep == "size"].groupby(["level", "arm"]).balanced_accuracy.mean().unstack()
    per_split = b.groupby(["split", "arm"]).accuracy.mean().unstack()
    bm = per_split.mean()
    gap = bm["batch"] - bm["none"]
    pooled_all = {}
    for arm in ("reference", "none", "batch"):
        pooled_all[arm] = float(c_auroc[c_auroc.arm == arm].pooled_auroc.mean())
    drop_80 = float(mix.loc[0.2, "batch"] - mix.loc[0.8, "batch"])
    out = {
        "A_mix_balanced_accuracy": {str(k): v for k, v in mix.round(4).to_dict(orient="index").items()},
        "A_size_balanced_accuracy": {str(k): v for k, v in size.round(4).to_dict(orient="index").items()},
        "A_batch_drop_20_to_80pct_mix_points": round(100 * drop_80, 2),
        "B_mean_accuracy": bm.round(4).to_dict(),
        "B_split_sd": per_split.std().round(4).to_dict(),
        "B_none_minus_batch_points": round(100 * float(bm["none"] - bm["batch"]), 2),
        "B_recovery_fraction": round(float((bm["reference"] - bm["none"]) / gap), 3) if gap >= 0.03 else None,
        "C_pooled_auroc_by_model": c_auroc.pivot(index="held_out", columns="arm", values="pooled_auroc").round(4).to_dict(orient="index"),
        "C_pooled_auroc_mean_over_models": {k: round(v, 4) for k, v in pooled_all.items()},
        "C_known_accuracy": c_rows.groupby("arm").known_accuracy.mean().round(4).to_dict(),
    }
    out["gate_G3a_matters"] = bool(drop_80 >= 0.03 or gap >= 0.03)
    out["gate_G3b_recovers_most"] = None if out["B_recovery_fraction"] is None else bool(out["B_recovery_fraction"] >= 0.5)
    out["gate_G3c_rejection_weak"] = bool(pooled_all["reference"] <= 0.55)
    return out


DF = None


def main():
    global DF
    t0 = time.time()
    DF = load()
    rng = np.random.default_rng(SEED)
    a = experiment_a(DF, rng)
    splits = split_prompts(rng)
    b = experiment_b(DF, splits)
    c_rows, c_auroc = experiment_c(DF, splits)
    a.to_csv(RESULTS / "pilot_A.csv", index=False)
    b.to_csv(RESULTS / "pilot_B.csv", index=False)
    c_rows.to_csv(RESULTS / "pilot_C.csv", index=False)
    c_auroc.to_csv(RESULTS / "pilot_C_auroc.csv", index=False)
    out = summarise(a, b, c_rows, c_auroc)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    (RESULTS / "pilot_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
