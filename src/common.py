"""Shared pieces for the pilot: loading, normalisation statistics and the classifier.

The classifier matches LangLLM's transfer setup (StandardScaler then logistic regression,
C=1), rewritten here rather than imported.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "original"
RESULTS = ROOT / "results"

FEATURES = [
    "mattr", "hapax_rate", "mean_token_len", "zipf_slope",
    "sent_len_mean", "sent_len_sd", "burstiness", "dep_depth", "subord_rate", "func_word_ratio", "first_person_rate",
    "para_count", "para_len_mean", "question_rate", "connective_rate",
    "comma_per_1k", "colon_per_1k", "dash_per_1k", "semicolon_per_1k",
    "bigram_entropy", "digit_rate",
]
LANGS = ["en", "es", "zh", "ru", "ja", "tr", "hi"]  # LangLLM's resource-rank order
MODELS = ["claude", "deepseek", "gemini", "gpt", "grok"]


def load() -> pd.DataFrame:
    """Load the feature table and impute missing values with the within-language median, as LangLLM does."""
    df = pd.read_csv(DATA / "features.csv")
    for f in FEATURES:
        df[f] = df.groupby("lang")[f].transform(lambda s: s.fillna(s.median()))
    return df


def stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample SD per column. A zero or undefined SD becomes 1, matching LangLLM's guard."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1) if len(X) > 1 else np.full(X.shape[1], np.nan)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return mu, sd


def norm(X: np.ndarray, st: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return (X - st[0]) / st[1]


class Linear:
    """A fitted StandardScaler + logistic regression, collapsed to one linear map for fast scoring."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=0)).fit(X, y)
        self.clf = clf
        sc, lr = clf[0], clf[1]
        self.classes = lr.classes_
        self.W = lr.coef_ / sc.scale_
        self.b = lr.intercept_ - (sc.mean_ / sc.scale_) @ lr.coef_.T

    def scores(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W.T + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[self.scores(X).argmax(axis=1)]

    def proba(self, X: np.ndarray) -> np.ndarray:
        s = self.scores(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the classes present in y_true."""
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]))
