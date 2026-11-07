"""Step 0: reproduce LangLLM's cross-language transfer matrix (mean off-diagonal 0.4861)."""
import json

import numpy as np
import pandas as pd

from common import DATA, FEATURES, LANGS, RESULTS, Linear, load, norm, stats


def main():
    df = load()
    X = {l: df.loc[df.lang == l, FEATURES].to_numpy(float) for l in LANGS}
    y = {l: df.loc[df.lang == l, "model_key"].to_numpy() for l in LANGS}
    z = {l: norm(X[l], stats(X[l])) for l in LANGS}

    M = pd.DataFrame(index=LANGS, columns=LANGS, dtype=float)
    for a in LANGS:
        clf = Linear(z[a], y[a])
        for b in LANGS:
            if a != b:
                M.loc[a, b] = np.mean(clf.predict(z[b]) == y[b])

    published = pd.read_csv(DATA / "rq3_transfer_matrix.csv", index_col=0)
    off = ~np.eye(len(LANGS), dtype=bool)
    ours = M.to_numpy()[off]
    theirs = published.loc[LANGS, LANGS].to_numpy()[off]
    out = {
        "n_essays": len(df),
        "mean_offdiag_ours": float(ours.mean()),
        "mean_offdiag_published": float(theirs.mean()),
        "max_abs_cell_difference": float(np.abs(ours - theirs).max()),
        "gate_G2_pass": bool(abs(ours.mean() - 0.4861) <= 0.02),
    }
    RESULTS.mkdir(exist_ok=True)
    M.to_csv(RESULTS / "reproduced_transfer_matrix.csv")
    (RESULTS / "reproduction.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
