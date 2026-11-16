"""One price model per supplier segment (natural persons vs legal entities).

Why: natural persons and companies use the same information very differently.
A lawyer hired as an individual and a law firm, or a small merchant and a
distributor, have different price levels, different vocabularies in their
descriptions and different price variability. A single model can split on
person type, but its linear parts (text price model, blend) cannot: they apply
the same word weights to everyone. Separate models learn each population's own
text weights, trees, blend weight and early stopping.

Layout on disk (all pickle-free, see models/price_model.py)::

    segments.json            {"segment_column", "segments": {...}, "fallback": "_global"}
    segment_natural/         a complete PriceModel
    segment_juridica/        a complete PriceModel
    segment__global/         PriceModel on all rows: unclassified suppliers and
                             segments too small to train on their own

Routing at prediction time: a contract goes to its segment's model when one
exists, otherwise to the global model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from contract_transparency.config import Settings
from contract_transparency.data import columns as C
from contract_transparency.data.person_type import JURIDICA, NATURAL
from contract_transparency.models.price_model import PriceModel

logger = logging.getLogger(__name__)

SPEC_FILE = "segments.json"
GLOBAL = "_global"
SEGMENT_COLUMNS = {"person_type": C.PERSON_TYPE}
SEGMENT_VALUES = {"person_type": (NATURAL, JURIDICA)}


class SegmentedPriceModel:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.segment_by = cfg.model.segment_by
        self.column = SEGMENT_COLUMNS[self.segment_by]
        self.models: dict[str, PriceModel] = {}
        self.sizes: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> SegmentedPriceModel:
        m = self.cfg.model
        logger.info("Training the global model (fallback)")
        self.models[GLOBAL] = PriceModel(self.cfg).fit(train, valid)
        self.sizes[GLOBAL] = {"train": len(train), "valid": 0 if valid is None else len(valid)}
        for value in SEGMENT_VALUES[self.segment_by]:
            tr = train[train[self.column] == value]
            va = valid[valid[self.column] == value] if valid is not None else None
            n_va = 0 if va is None else len(va)
            self.sizes[value] = {"train": len(tr), "valid": n_va}
            if len(tr) < m.min_segment_train_rows or n_va < m.min_segment_valid_rows:
                logger.info(
                    "Segment %s too small (%s train, %s valid): uses the global model",
                    value,
                    len(tr),
                    n_va,
                )
                continue
            logger.info("Training segment model: %s (%s rows)", value, len(tr))
            self.models[value] = PriceModel(self.cfg).fit(tr, va)
        return self

    # -------------------------------------------------------------- predict
    def _route(self, df: pd.DataFrame) -> pd.Series:
        seg = df[self.column] if self.column in df.columns else pd.Series(None, index=df.index)
        return seg.where(seg.isin([k for k in self.models if k != GLOBAL]), GLOBAL)

    def _combine(self, df: pd.DataFrame, method: str) -> np.ndarray:
        route = self._route(df).to_numpy()
        out = np.full(len(df), np.nan)
        for key in np.unique(route):
            idx = np.flatnonzero(route == key)
            out[idx] = getattr(self.models[key], method)(df.iloc[idx])
        return out

    def predict_transformed(self, df: pd.DataFrame) -> np.ndarray:
        return self._combine(df, "predict_transformed")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._combine(df, "predict")

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return np.log1p(self.predict(df))

    def segment_of(self, df: pd.DataFrame) -> pd.Series:
        """Which model scores each row (natural / juridica / _global)."""
        return self._route(df)

    # ------------------------------------------------------ introspection
    @property
    def global_model(self) -> PriceModel:
        return self.models[GLOBAL]

    @property
    def feature_names_(self) -> list[str]:
        return self.global_model.feature_names_

    @property
    def best_iteration_(self) -> int | None:
        return self.global_model.best_iteration_

    def top_features(self, n: int = 20) -> pd.DataFrame:
        frames = []
        for key, model in self.models.items():
            top = model.top_features(n)
            top.insert(0, "segment", key)
            frames.append(top)
        return pd.concat(frames, ignore_index=True)

    # ---------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for key, model in self.models.items():
            files += model.save(directory / f"segment_{key}")
        spec = {
            "format_version": 1,
            "segment_by": self.segment_by,
            "segment_column": self.column,
            "segments": sorted(self.models),
            "sizes": self.sizes,
            "fallback": GLOBAL,
        }
        (directory / SPEC_FILE).write_text(json.dumps(spec, indent=2), encoding="utf-8")
        files.append(directory / SPEC_FILE)
        return files

    @classmethod
    def load(cls, directory: str | Path) -> SegmentedPriceModel:
        directory = Path(directory)
        spec = json.loads((directory / SPEC_FILE).read_text(encoding="utf-8"))
        models = {key: PriceModel.load(directory / f"segment_{key}") for key in spec["segments"]}
        obj = cls.__new__(cls)
        obj.cfg = models[GLOBAL].cfg
        obj.segment_by = spec["segment_by"]
        obj.column = spec["segment_column"]
        obj.models = models
        obj.sizes = spec.get("sizes", {})
        return obj


# --------------------------------------------------------------------------- factory
def build_model(cfg: Settings) -> PriceModel | SegmentedPriceModel:
    """The model configured in ``model.segment_by`` (single or segmented)."""
    if cfg.model.segment_by and cfg.model.type == "xgboost":
        return SegmentedPriceModel(cfg)
    return PriceModel(cfg)


def load_model(directory: str | Path) -> PriceModel | SegmentedPriceModel:
    """Load either kind of model from a registry version directory."""
    if (Path(directory) / SPEC_FILE).exists():
        return SegmentedPriceModel.load(directory)
    return PriceModel.load(directory)
