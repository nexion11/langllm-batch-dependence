"""Follow-up analyses F1-F4 from docs/followup_plan.md. Writes results/followup_*.csv and results/followup_summary.json."""
import itertools
import json
import time

import numpy as np
import pandas as pd

from common import FEATURES, LANGS, MODELS, RESULTS, Linear, load, norm, stats

SEED = 20261107
SPLITS = 50
DRAWS = 10
SIZES = [5, 10, 20, 40, 60]
DOMINANT = [1, 4, 8, 12]


def length_fit(X, logn):
    """Per-feature OLS of X on log length: returns (slope, intercept) arrays."""
    lc = logn - logn.mean()
    slope = (lc @ (X - X.mean(axis=0))) / (lc @ lc)
    return slope, X.mean(axis=0) - slope * logn.mean()


def residual(X, logn, fit):
    return X - (np.outer(logn, fit[0]) + fit[1])


def main():
    t0 = time.time()
    df = load()
    df["logn"] = np.log(df.n_tokens.clip(lower=1))
    rng = np.random.default_rng(SEED)
    prompts = sorted(df.prompt_id.unique())
    sizes, dom, cover, clue, length = [], [], [], [], []
    shift = {f: [] for f in FEATURES}

    for s in range(SPLITS):
        p = rng.permutation(prompts)
        h1, h2 = set(p[:6]), set(p[6:])
        part = {}
        for l in LANGS:
            for h, ps in (("h1", h1), ("h2", h2)):
                d = df[(df.lang == l) & df.prompt_id.isin(ps)]
                part[l, h] = (d[FEATURES].to_numpy(float), d.model_key.to_numpy(), d.logn.to_numpy())
        for a in LANGS:
            Xa, ya, na = part[a, "h1"]
            st_a = stats(Xa)
            clf = Linear(norm(Xa, st_a), ya)
            fit_a = length_fit(Xa, na)
            ra = residual(Xa, na, fit_a)
            st_ra = stats(ra)
            clf_len = Linear(norm(ra, st_ra), ya)
            for b in LANGS:
                if b == a:
                    continue
                Xt, yt, nt = part[b, "h2"]
                Xr, yr, nr = part[b, "h1"]
                acc = lambda st: float(np.mean(clf.predict(norm(Xt, st)) == yt))
                base = {"split": s, "train": a, "test": b, "none": acc(st_a), "batch": acc(stats(Xt)),
                        "reference": acc(stats(Xr))}
                by_model = {m: np.flatnonzero(yr == m) for m in MODELS}

                # F1: balanced reference sets of k essays
                for k in SIZES:
                    vals = [acc(stats(Xr[np.concatenate([rng.choice(by_model[m], min(k // 5, len(by_model[m])), replace=False)
                                                          for m in MODELS])])) for _ in range(DRAWS)]
                    sizes.append({**base, "k": k, "arm_acc": float(np.mean(vals))})

                # F2a: one dominant model with d essays, the other four with 1 each
                for d in DOMINANT:
                    for dm in MODELS:
                        vals = [acc(stats(Xr[np.concatenate([rng.choice(by_model[m], min(d if m == dm else 1, len(by_model[m])), replace=False)
                                                              for m in MODELS])])) for _ in range(DRAWS)]
                        dom.append({**base, "d": d, "dominant": dm, "arm_acc": float(np.mean(vals))})

                # F2b: reference from only m of the 5 models
                for m in range(1, 6):
                    for subset in itertools.combinations(MODELS, m):
                        idx = np.concatenate([by_model[x] for x in subset])
                        cover.append({**base, "m": m, "models": "+".join(subset), "arm_acc": acc(stats(Xr[idx]))})

                # F3: one feature at a time normalised with source statistics, the rest with the reference
                st_r = stats(Xr)
                for j, f in enumerate(FEATURES):
                    mu, sd = st_r[0].copy(), st_r[1].copy()
                    mu[j], sd[j] = st_a[0][j], st_a[1][j]
                    clue.append({**base, "feature": f, "arm_acc": acc((mu, sd))})
                    if s == 0:
                        shift[f].append(abs(Xr[:, j].mean() - Xa[:, j].mean()) / st_a[1][j])

                # F4: length-residualised features, each source fitting its own length regression
                fit_r, fit_t = length_fit(Xr, nr), length_fit(Xt, nt)
                lacc = lambda resid, st: float(np.mean(clf_len.predict(norm(resid, st)) == yt))
                length.append({**base,
                               "len_none": lacc(residual(Xt, nt, fit_a), st_ra),
                               "len_reference": lacc(residual(Xt, nt, fit_r), stats(residual(Xr, nr, fit_r))),
                               "len_batch": lacc(residual(Xt, nt, fit_t), stats(residual(Xt, nt, fit_t)))})

    frames = {n: pd.DataFrame(r) for n, r in
              (("sizes", sizes), ("dominance", dom), ("coverage", cover), ("clues", clue), ("length", length))}
    for n, f in frames.items():
        f.to_csv(RESULTS / f"followup_{n}.csv", index=False)

    base = frames["length"][["none", "batch", "reference"]].mean()
    gap = base["batch"] - base["none"]
    rec = lambda v: round(float((v - base["none"]) / gap), 3)
    f1 = frames["sizes"].groupby("k").arm_acc.mean()
    f2a = frames["dominance"].groupby("d").arm_acc.mean()
    f2a_by = frames["dominance"][frames["dominance"].d == 12].groupby("dominant").arm_acc.mean()
    f2b = frames["coverage"].groupby("m").arm_acc.mean()
    f2b_one = frames["coverage"][frames["coverage"].m == 1].groupby("models").arm_acc.mean()
    clues = frames["clues"].groupby("feature").arm_acc.mean()
    clue_tbl = pd.DataFrame({"accuracy_drop_points": (100 * (base["reference"] - clues)).round(2),
                             "language_shift_sd": pd.Series({f: float(np.mean(v)) for f, v in shift.items()}).round(2)}
                            ).sort_values("accuracy_drop_points", ascending=False)
    clue_tbl.to_csv(RESULTS / "followup_clue_table.csv")
    ln = frames["length"][["len_none", "len_reference", "len_batch"]].mean()

    out = {
        "baseline_accuracy": base.round(4).to_dict(),
        "F1_accuracy_by_k": f1.round(4).to_dict(), "F1_recovery_by_k": {int(k): rec(v) for k, v in f1.items()},
        "F1_rule_k10_recovers_half": bool(rec(f1[10]) >= 0.5),
        "F2a_accuracy_by_dominant_count": f2a.round(4).to_dict(),
        "F2a_recovery_by_dominant_count": {int(k): rec(v) for k, v in f2a.items()},
        "F2a_accuracy_at_d12_by_dominant_model": f2a_by.round(4).to_dict(),
        "F2b_accuracy_by_models_covered": f2b.round(4).to_dict(),
        "F2b_recovery_by_models_covered": {int(k): rec(v) for k, v in f2b.items()},
        "F2b_one_model_reference_accuracy": f2b_one.round(4).to_dict(),
        "F2b_rule_one_model_below_half": bool(rec(f2b[1]) < 0.5),
        "F3_top_features": clue_tbl.head(8).to_dict(orient="index"),
        "F4_length_controlled_accuracy": ln.round(4).to_dict(),
        "F4_rule_reference_within_3pts": bool(abs(ln["len_reference"] - base["reference"]) <= 0.03),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "followup_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
