"""Text featurisers for contract descriptions.

* ``tfidf_svd`` (default) -- TF-IDF on word n-grams + TruncatedSVD (LSA). Fast,
  CPU-only, no downloads; a good replacement for the legacy Keras tokenizer +
  transformer that needed GPUs and could not be reloaded.
* ``sentence_transformer`` -- multilingual sentence embeddings (768 dims), the
  same size as the embeddings used by the legacy ``modelxgboostnrow`` model.
  Requires ``pip install -e .[nlp]``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

# Compact Spanish stop-word list (sklearn only ships English).
SPANISH_STOP_WORDS = sorted(
    {
        "a",
        "al",
        "ante",
        "con",
        "contra",
        "de",
        "del",
        "desde",
        "durante",
        "e",
        "el",
        "en",
        "entre",
        "esta",
        "este",
        "esto",
        "estos",
        "estas",
        "hacia",
        "hasta",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "mediante",
        "o",
        "para",
        "por",
        "que",
        "se",
        "segun",
        "sin",
        "sobre",
        "su",
        "sus",
        "tras",
        "u",
        "un",
        "una",
        "unas",
        "unos",
        "y",
        "como",
        "mas",
        "muy",
        "ser",
        "son",
        "fue",
        "ha",
        "han",
        "cual",
        "cuales",
        "dicho",
        "dicha",
        "dichos",
        "dichas",
        "otro",
        "otra",
        "otros",
        "otras",
        "cada",
        "todo",
        "toda",
        "todos",
        "todas",
    }
)


class SafeTruncatedSVD(TruncatedSVD):
    """TruncatedSVD that shrinks ``n_components`` when the vocabulary is small."""

    def fit_transform(self, X, y=None):
        max_k = max(1, min(X.shape) - 1)
        if self.n_components > max_k:
            self.n_components = max_k
        return super().fit_transform(X, y)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Wraps a ``sentence-transformers`` model as a sklearn transformer.

    The underlying network is loaded lazily and is *not* pickled with the
    pipeline (only its name is); it is re-downloaded/re-loaded from the local
    Hugging Face cache on first use after loading a model from the registry.
    """

    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

    def _model(self):
        if getattr(self, "_st_model", None) is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "text_embedder='sentence_transformer' requires `pip install -e .[nlp]`"
                ) from exc
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
        return self._st_model

    def fit(self, X, y=None):
        # Fitted state (sklearn >= 1.8 refuses to run "unfitted" steps): the
        # embedding dimension, used for feature names.
        self.dim_ = int(np.asarray(self._encode(["dimension probe"])).shape[1])
        return self

    def transform(self, X):
        return self._encode([str(t) for t in np.asarray(X).ravel()])

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model().encode(texts, batch_size=self.batch_size, show_progress_bar=False)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "dim_")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"st_{i}" for i in range(self.dim_)], dtype=object)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_st_model", None)
        return state
