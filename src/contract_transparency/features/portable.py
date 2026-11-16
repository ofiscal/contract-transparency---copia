"""Version-independent (pickle-free) form of the fitted feature preprocessor.

Why: a fitted scikit-learn object saved with joblib/pickle is only guaranteed
to work with the *same* scikit-learn version. A model trained with
scikit-learn 1.4 crashes under 1.9 (``'SimpleImputer' object has no attribute
'_fill_dtype'``). Upgrading a library must not break every saved model.

How: scikit-learn is still used to *fit* (see ``features/builder.py``). The
learned parameters are then exported to plain data, and :class:`PortablePreprocessor`
applies them using only stable public building blocks (numpy/scipy and
``CountVectorizer`` with a fixed vocabulary):

=======================  ==============================================
fitted sklearn step       stored as
=======================  ==============================================
OneHotEncoder             category -> output column maps (JSON)
TfidfVectorizer           vocabulary (JSON) + idf weights (npz)
TruncatedSVD              components matrix (npz)
SimpleImputer             medians + missing-indicator columns
numeric derivations       pure function (``add_numeric_features``)
=======================  ==============================================

Files: ``preprocessor.json`` + ``preprocessor.npz`` (loaded with
``allow_pickle=False``: no code execution on load). ``tests/test_portable.py``
checks the output equals scikit-learn's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from contract_transparency.features.builder import _text_1d, _to_str_frame, add_numeric_features
from contract_transparency.features.text import SentenceEmbedder

FORMAT_VERSION = 1
JSON_FILE = "preprocessor.json"
NPZ_FILE = "preprocessor.npz"


def _steps(transformer) -> dict[str, Any]:
    return dict(transformer.steps) if hasattr(transformer, "steps") else {"only": transformer}


# Text features are rounded at train AND score time. XGBoost split thresholds
# often coincide with training values (many contracts share the same text), so a
# last-digit floating-point difference -- float32 vs float64, or a different
# summation order after a numpy/scipy upgrade -- could otherwise move a whole
# group of contracts to the other side of a split.
TEXT_DECIMALS = 6


def _quantize(m: sp.csr_matrix) -> sp.csr_matrix:
    m.data = np.round(m.data, TEXT_DECIMALS)
    m.eliminate_zeros()
    return m


class PortablePreprocessor:
    def __init__(self, blocks: list[dict[str, Any]], arrays: dict[str, np.ndarray]):
        self.blocks = blocks
        self.arrays = arrays

    # ------------------------------------------------------------------ export
    @classmethod
    def from_sklearn(cls, ct: ColumnTransformer) -> PortablePreprocessor:
        blocks: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        for name, fitted, columns in ct.transformers_:
            if fitted == "drop" or name == "remainder":
                continue
            steps = _steps(fitted)
            if "onehot" in steps:
                blocks.append(cls._export_onehot(steps["onehot"], list(columns)))
            elif "tfidf" in steps:
                key = f"{name}_idf"
                vec = steps["tfidf"]
                arrays[key] = np.asarray(vec.idf_, dtype=np.float64)
                block = {
                    "kind": "tfidf",
                    "column": columns,
                    "vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()},
                    "analyzer": {
                        "lowercase": vec.lowercase,
                        "strip_accents": vec.strip_accents,
                        "stop_words": sorted(vec.stop_words) if vec.stop_words else None,
                        "ngram_range": list(vec.ngram_range),
                        "token_pattern": vec.token_pattern,
                    },
                    "sublinear_tf": bool(vec.sublinear_tf),
                    "norm": vec.norm,
                    "idf": key,
                    "svd": None,
                }
                if "svd" in steps:
                    block["svd"] = f"{name}_svd"
                    arrays[block["svd"]] = np.asarray(steps["svd"].components_, dtype=np.float64)
                blocks.append(block)
            elif "st" in steps:
                blocks.append(
                    {"kind": "embedder", "column": columns, "model_name": steps["st"].model_name}
                )
            elif "impute" in steps:
                imp = steps["impute"]
                key = f"{name}_medians"
                arrays[key] = np.nan_to_num(np.asarray(imp.statistics_, dtype=np.float64), nan=0.0)
                indicator = getattr(imp, "indicator_", None)
                blocks.append(
                    {
                        "kind": "numeric",
                        "columns": list(columns),
                        "medians": key,
                        "indicator": [int(i) for i in indicator.features_]
                        if indicator is not None
                        else [],
                    }
                )
            else:  # pragma: no cover - guard for future feature blocks
                raise TypeError(f"Cannot export feature block '{name}' ({list(steps)})")
        return cls(blocks, arrays)

    @staticmethod
    def _export_onehot(enc, columns: list[str]) -> dict[str, Any]:
        """Map every category to its output column, replicating
        handle_unknown='infrequent_if_exist' (rare and unseen -> 'infrequent')."""
        infrequent = getattr(enc, "infrequent_categories_", None) or [None] * len(columns)
        maps, unknown, offset = [], [], 0
        for cats, infreq in zip(enc.categories_, infrequent, strict=True):
            infreq_set = set(infreq.tolist()) if infreq is not None else set()
            frequent = [c for c in cats.tolist() if c not in infreq_set]
            mapping = {str(c): offset + i for i, c in enumerate(frequent)}
            n_out = len(frequent)
            infreq_idx = None
            if infreq_set:
                infreq_idx = offset + n_out
                mapping.update({str(c): infreq_idx for c in infreq_set})
                n_out += 1
            maps.append(mapping)
            unknown.append(infreq_idx)
            offset += n_out
        return {
            "kind": "onehot",
            "columns": columns,
            "maps": maps,
            "unknown": unknown,
            "n_out": offset,
        }

    # --------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        parts = [self._apply(b, df) for b in self.blocks]
        return sp.hstack(parts, format="csr", dtype=np.float64)

    def _apply(self, b: dict[str, Any], df: pd.DataFrame) -> sp.csr_matrix:
        n = len(df)
        if b["kind"] == "onehot":
            values = _to_str_frame(df[b["columns"]])
            rows, cols = [], []
            for j, (mapping, unk) in enumerate(zip(b["maps"], b["unknown"], strict=True)):
                idx = [mapping.get(v, unk) for v in values[:, j]]
                for r, c in enumerate(idx):
                    if c is not None:
                        rows.append(r)
                        cols.append(c)
            data = np.ones(len(rows))
            return sp.csr_matrix((data, (rows, cols)), shape=(n, b["n_out"]))
        if b["kind"] == "tfidf":
            a = b["analyzer"]
            counter = CountVectorizer(
                vocabulary=b["vocabulary"],
                lowercase=a["lowercase"],
                strip_accents=a["strip_accents"],
                stop_words=a["stop_words"],
                ngram_range=tuple(a["ngram_range"]),
                token_pattern=a["token_pattern"],
                dtype=np.float64,
            )
            tf = counter.transform(_text_1d(df[b["column"]])).tocsr()
            if b["sublinear_tf"]:
                tf.data = np.log(tf.data) + 1.0
            tfidf = tf @ sp.diags(self.arrays[b["idf"]])
            if b["norm"]:
                tfidf = normalize(tfidf, norm=b["norm"], copy=False)
            out = tfidf if b["svd"] is None else sp.csr_matrix(tfidf @ self.arrays[b["svd"]].T)
            return _quantize(sp.csr_matrix(out))
        if b["kind"] == "embedder":
            emb = SentenceEmbedder(b["model_name"])
            return sp.csr_matrix(emb.transform(_text_1d(df[b["column"]])))
        if b["kind"] == "numeric":
            x = add_numeric_features(df[b["columns"]]).to_numpy(dtype=np.float64)
            missing = np.isnan(x)
            filled = np.where(missing, self.arrays[b["medians"]][None, :], x)
            if b["indicator"]:
                filled = np.hstack([filled, missing[:, b["indicator"]].astype(np.float64)])
            return sp.csr_matrix(filled)
        raise ValueError(f"Unknown block kind {b['kind']}")  # pragma: no cover

    # ------------------------------------------------------------- persistence
    def save(self, directory: Path) -> list[Path]:
        directory = Path(directory)
        spec = {"format_version": FORMAT_VERSION, "blocks": self.blocks}
        (directory / JSON_FILE).write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        np.savez(directory / NPZ_FILE, **self.arrays)
        return [directory / JSON_FILE, directory / NPZ_FILE]

    @classmethod
    def load(cls, directory: Path) -> PortablePreprocessor:
        directory = Path(directory)
        spec = json.loads((directory / JSON_FILE).read_text(encoding="utf-8"))
        if spec.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(
                f"Preprocessor format {spec['format_version']} is newer than this code "
                f"supports ({FORMAT_VERSION}); upgrade contract-transparency."
            )
        with np.load(directory / NPZ_FILE, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
        return cls(spec["blocks"], arrays)

    @staticmethod
    def exists(directory: Path) -> bool:
        return (Path(directory) / JSON_FILE).exists()
