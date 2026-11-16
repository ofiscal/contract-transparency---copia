"""Zero-shot sustainability tagging of contract objects (legacy ``sustainability.py``).

Uses a multilingual NLI model to assign each contract to sustainability
themes without labelled data. Requires ``pip install -e .[nlp]`` and a model
download (~1 GB) the first time.

Changes from the legacy script: the model is loaded lazily (importing the
module no longer downloads it nor requires a GPU), texts are processed in
batches, failures are reported per row instead of silently returning
"nada", and the hard-coded Windows paths are gone.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_LABELS = [
    "agua",
    "energia",
    "biodiversidad",
    "cambio climático",
    "residuos sólidos",
    "calidad atmosférica",
    "calidad del aire",
    "generación de empleo",
    "costo total de propiedad",
    "otros",
]


def load_classifier(model: str = DEFAULT_MODEL, device: int | str | None = None):
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Sustainability tagging requires `pip install -e .[nlp]`") from exc
    if device is None:
        try:
            import torch

            device = 0 if torch.cuda.is_available() else -1
        except ImportError:  # pragma: no cover
            device = -1
    return pipeline("zero-shot-classification", model=model, device=device)


def classify_sustainability(
    texts: Iterable[str],
    classifier=None,
    labels: list[str] | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Return the top label, its score and the full ranking for every text."""
    labels = labels or DEFAULT_LABELS
    classifier = classifier or load_classifier()
    texts = [str(t) if t is not None else "" for t in texts]
    rows: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        non_empty = [t for t in batch if t.strip()]
        raw = classifier(non_empty, candidate_labels=labels) if non_empty else []
        if isinstance(raw, dict):  # some pipeline versions un-list single results
            raw = [raw]
        results = iter(raw)
        for t in batch:
            if not t.strip():
                rows.append(
                    {
                        "sustainability_label": None,
                        "sustainability_score": None,
                        "sustainability_ranking": None,
                    }
                )
                continue
            r = next(results)
            rows.append(
                {
                    "sustainability_label": r["labels"][0],
                    "sustainability_score": float(r["scores"][0]),
                    "sustainability_ranking": "; ".join(r["labels"]),
                }
            )
    return pd.DataFrame(rows)
