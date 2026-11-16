"""Expected-price model: predicts the value a contract *should* have.

The gap between the real and the predicted value is the main anomaly signal
(see :mod:`contract_transparency.scoring.anomaly`).

Consolidates the experiments in ``legacy/python3_10/modules/models``
(``xgboost.py``, ``gradientboosting.py``, ``xgboost_weights.py``,
``wiki40b_regresion_adapt.py`` ...) into one class with a stable API::

    model = PriceModel(cfg).fit(train_df, valid_df)
    model.predict(df)          # COP
    model.save(directory); PriceModel.load(directory)

Artifacts contain **no pickles**, so a saved model keeps working after
upgrading scikit-learn / numpy / xgboost:

* preprocessing -> ``preprocessor.json`` + ``.npz`` (:mod:`features.portable`)
* XGBoost      -> native ``booster.ubj`` (readable by newer xgboost versions)
* Ridge        -> coefficients in ``ridge.npz``

Models saved by v1.0 (``preprocessor.joblib``) can still be loaded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import ValidationError
from sklearn.linear_model import Ridge

from contract_transparency.config import Settings, with_overrides
from contract_transparency.data import columns as C
from contract_transparency.features.builder import (
    build_preprocessor,
    ensure_columns,
    input_columns,
)
from contract_transparency.features.portable import PortablePreprocessor
from contract_transparency.features.text_model import TextPriceModel
from contract_transparency.models.target import TargetTransform

logger = logging.getLogger(__name__)

LINEAR_DIR = "linear"  # blended sparse linear sub-model
BLEND_GRID = (0.0, 0.25, 0.5, 0.75)
LEGACY_PREPROCESSOR_FILE = "preprocessor.joblib"  # v1.0 models
LEGACY_RIDGE_FILE = "ridge.joblib"  # v1.0 models
BOOSTER_FILE = "booster.ubj"
RIDGE_FILE = "ridge.npz"
META_FILE = "model.json"


class LinearModel:
    """Pickle-free linear predictor (coefficients exported from sklearn Ridge)."""

    def __init__(self, coef: np.ndarray, intercept: float):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)

    def predict(self, X) -> np.ndarray:
        return np.asarray(X @ self.coef_).ravel() + self.intercept_


_DENSE_MAX_CELLS = 50_000_000  # ~200 MB of float32


class PriceModel:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.model_type = cfg.model.type
        self.target = TargetTransform(
            cfg.model.target_transform, cfg.model.zscore_mean, cfg.model.zscore_std
        )
        self.preprocessor: PortablePreprocessor | None = None
        self.text_model: TextPriceModel | None = None
        self.linear: PriceModel | None = None  # blended sparse linear model
        self.blend_weight_: float = 0.0
        self.estimator = None
        self.input_columns_: list[str] = []
        self.feature_names_: list[str] = []
        self.best_iteration_: int | None = None
        self.evals_result_: dict = {}

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _as_matrix(X) -> sp.csr_matrix | np.ndarray:
        """Dense float32 when it fits comfortably in memory (much faster for
        XGBoost ``hist``), CSR otherwise."""
        X = sp.csr_matrix(X, dtype=np.float32)
        if X.shape[0] * X.shape[1] <= _DENSE_MAX_CELLS:
            return X.toarray()
        return X

    def _matrix(
        self, df: pd.DataFrame, text_pred: np.ndarray | None = None
    ) -> sp.csr_matrix | np.ndarray:
        X = sp.csr_matrix(self.preprocessor.transform(ensure_columns(df, self.input_columns_)))
        if self.text_model is not None:
            if text_pred is None:  # inference: full text model
                texts = df[C.TEXT] if C.TEXT in df.columns else [""] * len(df)
                text_pred = self.text_model.predict(texts)
            X = sp.hstack([X, sp.csr_matrix(np.asarray(text_pred).reshape(-1, 1))], format="csr")
        return self._as_matrix(X)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> PriceModel:
        y_train = self.target.fit(train[C.VALUE].to_numpy()).transform(train[C.VALUE].to_numpy())
        # Fit with scikit-learn, then keep only the portable (pickle-free) form.
        # Training and scoring both use the portable transform: identical features.
        fitted = build_preprocessor(self.cfg, train).fit(train)
        self.input_columns_ = input_columns(fitted)
        self.preprocessor = PortablePreprocessor.from_sklearn(fitted)
        oof = None
        f = self.cfg.features
        if f.text_price_model and C.TEXT in train.columns:
            self.text_model = TextPriceModel(
                word_max_features=f.text_price_max_features // 2,
                char_max_features=f.text_price_max_features // 2,
                char_ngram=(2, f.text_price_char_ngram_max),
                min_df=f.text_min_df,
                alpha=f.text_price_alpha,
                n_folds=f.text_price_folds,
            )
            oof = self.text_model.fit_oof(train[C.TEXT], y_train)  # out-of-fold: no leakage
        X_train = self._matrix(train, text_pred=oof)
        try:
            self.feature_names_ = [str(n) for n in fitted.get_feature_names_out()]
        except Exception as exc:  # pragma: no cover - names are informative only
            logger.warning("Feature names unavailable (%s); using positional names", exc)
            self.feature_names_ = [f"f{i}" for i in range(X_train.shape[1])]
        if self.text_model is not None and len(self.feature_names_) < X_train.shape[1]:
            self.feature_names_.append("text__price_model")
        logger.info("Training %s on %s rows x %s features", self.model_type, *X_train.shape)

        if self.model_type == "ridge":
            ridge = Ridge(alpha=1.0).fit(X_train, y_train)
            self.estimator = LinearModel(ridge.coef_, ridge.intercept_)
            return self

        import xgboost as xgb

        params = dict(self.cfg.model.params)
        params.setdefault("base_score", float(np.median(y_train)))
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self._xgb_names())
        # Only the validation set is evaluated each round (evaluating the training
        # set too doubled the training time without adding information).
        evals = []
        if valid is not None and len(valid):
            y_valid = self.target.transform(valid[C.VALUE].to_numpy())
            dvalid = xgb.DMatrix(
                self._matrix(valid), label=y_valid, feature_names=self._xgb_names()
            )
            evals.append((dvalid, "valid"))
        self.evals_result_ = {}
        self.estimator = xgb.train(
            params,
            dtrain,
            num_boost_round=self.cfg.model.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.cfg.model.early_stopping_rounds if evals else None,
            evals_result=self.evals_result_,
            verbose_eval=False,
        )
        self.best_iteration_ = int(
            getattr(self.estimator, "best_iteration", self.cfg.model.num_boost_round - 1)
        )
        logger.info("XGBoost stopped at iteration %s", self.best_iteration_)
        self._fit_blend(train, valid)
        return self

    def _linear_config(self) -> Settings:
        return with_overrides(
            self.cfg,
            {
                "model.type": "ridge",
                "features.text_embedder": "tfidf",
                "features.text_price_model": False,
                "model.blend_linear_weight": 0.0,
            },
        )

    def _fit_blend(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> None:
        """XGBoost + sparse Ridge, averaged in the transformed-target space.
        Different model families make different errors; averaging them is a
        cheap, well-established gain (weight chosen on validation data)."""
        weight = self.cfg.model.blend_linear_weight
        if weight == 0 or (weight == "auto" and (valid is None or not len(valid))):
            return
        self.linear = PriceModel(self._linear_config()).fit(train)
        if weight != "auto":
            self.blend_weight_ = float(weight)
            return
        y_valid = self.target.transform(valid[C.VALUE].to_numpy())
        xgb_pred = self._predict_booster(valid)
        lin_pred = self.linear.predict_transformed(valid)
        errors = {
            w: float(np.mean(np.abs((1 - w) * xgb_pred + w * lin_pred - y_valid)))
            for w in BLEND_GRID
        }
        self.blend_weight_ = min(errors, key=errors.get)
        logger.info(
            "Blend weight of the linear model: %s (validation MAE %s)", self.blend_weight_, errors
        )
        if self.blend_weight_ == 0:
            self.linear = None

    def _xgb_names(self) -> list[str]:
        # XGBoost forbids [, ] and < in feature names.
        return [
            n.replace("[", "(").replace("]", ")").replace("<", "lt") for n in self.feature_names_
        ]

    # -------------------------------------------------------------- predict
    def predict_transformed(self, df: pd.DataFrame) -> np.ndarray:
        pred = self._predict_booster(df)
        if self.linear is not None and self.blend_weight_ > 0:
            w = self.blend_weight_
            pred = (1 - w) * pred + w * self.linear.predict_transformed(df)
        return pred

    def _predict_booster(self, df: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        X = self._matrix(df)
        if self.model_type == "ridge":
            return self.estimator.predict(X)
        import xgboost as xgb

        dm = xgb.DMatrix(X, feature_names=self._xgb_names())
        rng = (0, self.best_iteration_ + 1) if self.best_iteration_ is not None else (0, 0)
        return self.estimator.predict(dm, iteration_range=rng)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Expected contract value in COP."""
        return np.clip(self.target.inverse(self.predict_transformed(df)), 0, None)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        """Expected value on the log1p scale (used for residuals)."""
        return np.log1p(self.predict(df))

    # ------------------------------------------------------ explainability
    def top_features(self, n: int = 20) -> pd.DataFrame:
        if self.model_type == "ridge":
            coef = np.abs(self.estimator.coef_)
            imp = pd.Series(coef, index=self.feature_names_)
        else:
            gain = self.estimator.get_score(importance_type="total_gain")
            names = dict(zip(self._xgb_names(), self.feature_names_, strict=True))
            imp = pd.Series({names.get(k, k): v for k, v in gain.items()}, dtype=float)
        imp = imp.sort_values(ascending=False).head(n)
        total = imp.sum() or 1.0
        return pd.DataFrame({"feature": imp.index, "importance": imp.values / total})

    # ---------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = self.preprocessor.save(directory)
        if self.text_model is not None:
            files += self.text_model.save(directory)
        if self.model_type == "ridge":
            files.append(directory / RIDGE_FILE)
            np.savez(
                files[-1],
                coef=self.estimator.coef_,
                intercept=np.array([self.estimator.intercept_]),
            )
        else:
            files.append(directory / BOOSTER_FILE)
            self.estimator.save_model(files[-1])
        if self.linear is not None:
            files += self.linear.save(directory / LINEAR_DIR)
        meta = {
            "format_version": 2,
            "blend_weight": self.blend_weight_,
            "model_type": self.model_type,
            "target": self.target.to_dict(),
            "input_columns": self.input_columns_,
            "feature_names": self.feature_names_,
            "best_iteration": self.best_iteration_,
            "config": self.cfg.model_dump(mode="json", by_alias=True),
        }
        files.append(directory / META_FILE)
        files[-1].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return files

    @staticmethod
    def _restore_config(raw: dict) -> Settings:
        """Config stored with the model. Tolerant: a model must stay loadable even
        if later versions of this package rename/remove settings."""
        try:
            return Settings.model_validate(raw)
        except ValidationError as exc:
            logger.warning(
                "Stored model config is not valid for this version (%s); using defaults",
                exc.error_count(),
            )
            known = {k: raw[k] for k in ("model", "features") if k in raw}
            try:
                return Settings.model_validate(known)
            except ValidationError:
                return Settings()

    @classmethod
    def load(cls, directory: str | Path) -> PriceModel:
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        model = cls(cls._restore_config(meta.get("config", {})))
        model.model_type = meta["model_type"]
        model.target = TargetTransform.from_dict(meta["target"])
        model.input_columns_ = meta["input_columns"]
        model.feature_names_ = meta["feature_names"]
        model.best_iteration_ = meta["best_iteration"]
        if PortablePreprocessor.exists(directory):
            model.preprocessor = PortablePreprocessor.load(directory)
        else:  # v1.0 model: pickled sklearn preprocessor (same-version only)
            import joblib

            logger.warning(
                "Loading a v1.0 model with a pickled preprocessor; re-train with `ct train` to make it portable"
            )
            model.preprocessor = joblib.load(directory / LEGACY_PREPROCESSOR_FILE)
        if TextPriceModel.exists(directory):
            model.text_model = TextPriceModel.load(directory)
        model.blend_weight_ = float(meta.get("blend_weight", 0.0))
        if model.blend_weight_ > 0 and (directory / LINEAR_DIR).exists():
            model.linear = cls.load(directory / LINEAR_DIR)
        if meta["model_type"] == "ridge":
            if (directory / RIDGE_FILE).exists():
                with np.load(directory / RIDGE_FILE, allow_pickle=False) as npz:
                    model.estimator = LinearModel(npz["coef"], float(npz["intercept"][0]))
            else:
                import joblib

                model.estimator = joblib.load(directory / LEGACY_RIDGE_FILE)
        else:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(directory / BOOSTER_FILE)
            model.estimator = booster
        return model
