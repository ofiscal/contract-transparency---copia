"""Text price model: description -> expected log-price (stacked into XGBoost).

Why: contract descriptions carry price information that the categorical
columns do not (professional level, scope, quantities, object details).
Compressing the text to a few SVD components loses most of it, and trees
learn word-level effects poorly from dense components. The established,
cost-efficient approach for "price from text" (e.g. the winning solutions of
the Mercari Price Suggestion Challenge) is a linear model on sparse TF-IDF:

* word 1-2 grams      -> "abogado especializado", "mantenimiento rutinario"
* char_wb 2-5 grams   -> robust to misspellings, abbreviations and inflections
                         ("computador/computadores/comput.")
* Ridge regression    -> one weight per n-gram, trained in seconds on CPU

Its prediction becomes ONE feature of the XGBoost model. Training rows get
**out-of-fold** predictions (K-fold), so the booster never sees a text
prediction fitted on its own target (no leakage).

Storage is pickle-free (vocabularies + idf + coefficients in JSON/npz), like
:mod:`features.portable`, so it survives library upgrades.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

from contract_transparency.features.text import SPANISH_STOP_WORDS

JSON_FILE = "text_model.json"
NPZ_FILE = "text_model.npz"
DECIMALS = 6  # same quantisation rule as features.portable


def _texts(values) -> list[str]:
    return ["" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in values]


class TextPriceModel:
    def __init__(
        self,
        word_max_features: int = 50_000,
        char_max_features: int = 50_000,
        char_ngram: tuple[int, int] = (2, 5),
        min_df: int = 3,
        alpha: float = 1.0,
        n_folds: int = 5,
        seed: int = 0,
    ):
        self.params = {
            "word_max_features": word_max_features,
            "char_max_features": char_max_features,
            "char_ngram": list(char_ngram),
            "min_df": min_df,
            "alpha": alpha,
            "n_folds": n_folds,
            "seed": seed,
        }
        self.blocks: list[dict] = []
        self.arrays: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- fitting
    def _vectorizers(self, n_docs: int) -> list[tuple[str, TfidfVectorizer]]:
        p = self.params
        min_df = min(p["min_df"], max(1, n_docs // 10))
        common = {
            "lowercase": True,
            "strip_accents": "unicode",
            "sublinear_tf": True,
            "min_df": min_df,
            "dtype": np.float64,
        }
        return [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    stop_words=SPANISH_STOP_WORDS,
                    max_features=p["word_max_features"],
                    **common,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(p["char_ngram"]),
                    max_features=p["char_max_features"],
                    **common,
                ),
            ),
        ]

    def fit_oof(self, texts, y: np.ndarray) -> np.ndarray:
        """Fit on all rows; return out-of-fold predictions for those rows."""
        docs = _texts(texts)
        y = np.asarray(y, dtype=np.float64)
        mats, self.blocks, self.arrays = [], [], {}
        for name, vec in self._vectorizers(len(docs)):
            mats.append(vec.fit_transform(docs))
            self.arrays[f"{name}_idf"] = np.asarray(vec.idf_, dtype=np.float64)
            self.blocks.append(
                {
                    "name": name,
                    "analyzer": vec.analyzer,
                    "ngram_range": list(vec.ngram_range),
                    "stop_words": sorted(vec.stop_words) if vec.stop_words else None,
                    "vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()},
                }
            )
        X = sp.hstack(mats, format="csr")
        alpha = self.params["alpha"]
        oof = np.zeros(len(docs))
        folds = KFold(self.params["n_folds"], shuffle=True, random_state=self.params["seed"])
        for tr, te in folds.split(X):
            oof[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
        full = Ridge(alpha=alpha).fit(X, y)
        self.arrays["coef"] = np.asarray(full.coef_, dtype=np.float64)
        self.arrays["intercept"] = np.array([float(full.intercept_)])
        return np.round(oof, DECIMALS)

    # -------------------------------------------------------------- inference
    def _matrix(self, docs: list[str]) -> sp.csr_matrix:
        mats = []
        for b in self.blocks:
            counter = CountVectorizer(
                analyzer=b["analyzer"],
                ngram_range=tuple(b["ngram_range"]),
                vocabulary=b["vocabulary"],
                stop_words=b["stop_words"],
                lowercase=True,
                strip_accents="unicode",
                dtype=np.float64,
            )
            tf = counter.transform(docs).tocsr()
            tf.data = np.log(tf.data) + 1.0  # sublinear_tf
            mats.append(normalize(tf @ sp.diags(self.arrays[f"{b['name']}_idf"]), norm="l2"))
        return sp.hstack(mats, format="csr")

    def predict(self, texts) -> np.ndarray:
        X = self._matrix(_texts(texts))
        pred = X @ self.arrays["coef"] + self.arrays["intercept"][0]
        return np.round(np.asarray(pred).ravel(), DECIMALS)

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> list[Path]:
        directory = Path(directory)
        spec = {"format_version": 1, "params": self.params, "blocks": self.blocks}
        (directory / JSON_FILE).write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        np.savez(directory / NPZ_FILE, **self.arrays)
        return [directory / JSON_FILE, directory / NPZ_FILE]

    @classmethod
    def load(cls, directory: Path) -> TextPriceModel:
        directory = Path(directory)
        spec = json.loads((directory / JSON_FILE).read_text(encoding="utf-8"))
        model = cls()
        model.params, model.blocks = spec["params"], spec["blocks"]
        with np.load(directory / NPZ_FILE, allow_pickle=False) as npz:
            model.arrays = {k: npz[k] for k in npz.files}
        return model

    @staticmethod
    def exists(directory: Path) -> bool:
        return (Path(directory) / JSON_FILE).exists()
