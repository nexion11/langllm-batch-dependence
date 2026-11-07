"""E2 (addendum 3): multilingual-e5-base embeddings for every text (original, N1, N2, N3).

    python embed.py      # data/features/embeddings.npz (cell_id array + 768-d vectors)
"""
import json
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from features_new import OUT, ROOT

MODEL = "intfloat/multilingual-e5-base"


def texts():
    keep = set(pd.read_csv(ROOT / "data" / "original" / "features.csv").cell_id)
    for f in sorted((ROOT / "data" / "original" / "raw").glob("*.jsonl")):
        for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if r["cell_id"] in keep:
                yield r["cell_id"], r["text"]
    sys.path.insert(0, str(ROOT / "src"))
    from redo_failed import latest, problems
    for key in ("gpt", "gemini", "claude", "grok", "deepseek", "llama", "mistral", "qwen"):
        for r in latest(key).values():
            if not problems(r):
                yield r["cell_id"], r["text"]
    for f in sorted((ROOT / "data" / "human").glob("*.jsonl")):
        for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            yield r["cell_id"], r["text"]


def main():
    """Default (addendum 4, E2b): 'query: ' prefix, L2-normalised, as E5's model card says for classification
    features. `python embed.py passage` reproduces the superseded E2 run ('passage: ', not normalised)."""
    variant = sys.argv[1] if len(sys.argv) > 1 else "query"
    ids, docs = zip(*texts())
    model = SentenceTransformer(MODEL, device="mps")
    model.max_seq_length = 512
    vec = model.encode([f"{variant}: " + t for t in docs], batch_size=16, show_progress_bar=True,
                       normalize_embeddings=(variant == "query"))
    out = OUT / ("embeddings.npz" if variant == "passage" else "embeddings_query.npz")
    np.savez_compressed(out, cell_id=np.array(ids), vec=vec.astype(np.float32), model=MODEL, prefix=variant)
    print(f"{len(ids)} texts x {vec.shape[1]} dims -> {out}")


if __name__ == "__main__":
    main()
