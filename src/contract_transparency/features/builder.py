"""Feature matrix construction (a single sklearn ``ColumnTransformer``).

Replaces the legacy per-column Keras ``Tokenizer`` files
(``trainedmodels\\tokenizers\\<column>tokenizer.json``) and the hard-coded
``argumentos`` object. The fitted transformer is saved together with the model
so training and scoring can never drift apart.

Model inputs deliberately **exclude** personal data (supplier / legal
representative ids, names) and post-signature information (days added,
payments) that would leak the outcome.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder

from contract_transparency.config import Settings
from contract_transparency.data import columns as C
from contract_transparency.features.text import (
    SPANISH_STOP_WORDS,
    SafeTruncatedSVD,
    SentenceEmbedder,
)

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "f_log_duration_days",
    "f_signed_month",
    "f_signed_year",
    "f_signed_dayofweek",
    "f_log_text_quantity",
]

# A quantity written in the description next to a unit / object word:
# "9 equipos", "3,5 km", "2.500 galones", "250 m2". Not followed by "de" (dates,
# "No. 045 de 2023") nor by duration words (duration has its own feature).
_QTY = re.compile(
    r"(?<![\w.,/<-])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?\s+"
    r"(?!de\b|del\b|y\b|a\b|al\b|meses?\b|d[ií]as?\b|a[ñn]os?\b|horas?\b|semanas?\b)"
    r"(m2|m3|[a-záéíóúñ]{2,})",
    re.IGNORECASE,
)
_QTY_PREFIX = re.compile(
    r"(?i)(?:\bno\.?|n[°º]|n[uú]mero|contrato|proceso|resoluci[oó]n|decreto)\s*$"
)


def text_quantity(text: object) -> float:
    """Largest quantity written in the text (heuristic), NaN if none."""
    if not isinstance(text, str) or not text:
        return np.nan
    best = np.nan
    for m in _QTY.finditer(text):
        if _QTY_PREFIX.search(text[: m.start()]):
            continue
        integer = m.group(1).replace(".", "")
        value = float(integer + ("." + m.group(2) if m.group(2) else ""))
        if m.group(2) is None and len(integer) == 4 and 1990 <= value <= 2039:
            continue  # a year
        best = value if np.isnan(best) else max(best, value)
    return best


_UNSPSC = [C.UNSPSC_SEGMENT, C.UNSPSC_FAMILY, C.UNSPSC_CLASS]


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived numeric columns used by the model (pure function, picklable)."""
    out = pd.DataFrame(index=df.index)
    out["f_log_duration_days"] = np.log1p(pd.to_numeric(df.get(C.DURATION_DAYS), errors="coerce"))
    signed = pd.to_datetime(df.get(C.SIGNED_AT), errors="coerce")
    out["f_signed_month"] = signed.dt.month.astype(float)
    out["f_signed_year"] = signed.dt.year.astype(float)
    out["f_signed_dayofweek"] = signed.dt.day_of_week.astype(float)
    text = df.get(C.TEXT)
    if text is None:
        out["f_log_text_quantity"] = np.nan
    else:
        out["f_log_text_quantity"] = np.log1p(text.map(text_quantity).astype(float))
    return out


def _numeric_names(_transformer, _input_features) -> np.ndarray:
    return np.array(NUMERIC_FEATURES, dtype=object)


def _to_str_frame(x: pd.DataFrame) -> np.ndarray:
    return pd.DataFrame(x).astype("object").where(pd.notna(x), "NA").astype(str).to_numpy()


def _text_1d(x) -> np.ndarray:
    arr = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    return np.asarray(arr, dtype=object).ravel()


def categorical_columns(cfg: Settings, available: pd.Index) -> list[str]:
    cols = [c for c in cfg.schema_.categorical if c in available]
    if cfg.features.add_unspsc_levels:
        cols += [c for c in _UNSPSC if c in available]
    if cfg.features.add_person_type and C.PERSON_TYPE in available:
        cols.append(C.PERSON_TYPE)
    missing = sorted(set(cfg.schema_.categorical) - set(available))
    if missing:
        logger.warning("Categorical columns not found and skipped: %s", missing)
    return cols


def build_preprocessor(cfg: Settings, df: pd.DataFrame) -> ColumnTransformer:
    f = cfg.features
    transformers: list[tuple] = []

    cat_cols = categorical_columns(cfg, df.columns)
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "to_str",
                            FunctionTransformer(_to_str_frame, feature_names_out="one-to-one"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=f.min_category_frequency,
                                max_categories=f.max_categories,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )

    if f.text_embedder != "none" and C.TEXT in df.columns:
        to_1d = FunctionTransformer(_text_1d, feature_names_out="one-to-one")
        if f.text_embedder in {"tfidf_svd", "tfidf"}:
            steps = [
                ("to_1d", to_1d),
                (
                    "tfidf",
                    TfidfVectorizer(
                        strip_accents="unicode",
                        lowercase=True,
                        stop_words=SPANISH_STOP_WORDS,
                        ngram_range=(1, f.text_ngram_max),
                        max_features=f.text_max_features,
                        # A term must appear in several contracts: rare tokens (e.g. a
                        # person's name) never become a feature the model can memorise.
                        min_df=f.text_min_df,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
            ]
            if f.text_embedder == "tfidf_svd":
                steps.append(
                    ("svd", SafeTruncatedSVD(n_components=f.text_svd_components, random_state=0))
                )
            transformers.append(("text", Pipeline(steps), C.TEXT))
        else:
            transformers.append(
                (
                    "text",
                    Pipeline([("to_1d", to_1d), ("st", SentenceEmbedder(f.sentence_model))]),
                    C.TEXT,
                )
            )

    transformers.append(
        (
            "num",
            Pipeline(
                [
                    (
                        "derive",
                        FunctionTransformer(add_numeric_features, feature_names_out=_numeric_names),
                    ),
                    (
                        "impute",
                        SimpleImputer(
                            strategy="median", add_indicator=True, keep_empty_features=True
                        ),
                    ),
                ]
            ),
            [
                c
                for c in (C.DURATION_DAYS, C.SIGNED_AT, C.TEXT if f.add_text_quantity else None)
                if c and c in df.columns
            ],
        )
    )
    return ColumnTransformer(transformers, sparse_threshold=1.0, remainder="drop", n_jobs=None)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add missing input columns (as NaN) so a fitted preprocessor can run."""
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return df
    return df.assign(**{c: np.nan for c in missing})


def input_columns(pre: ColumnTransformer) -> list[str]:
    cols: list[str] = []
    for _, _, sel in pre.transformers:
        if isinstance(sel, str):
            cols.append(sel)
        elif isinstance(sel, list):
            cols.extend(sel)
    return cols
