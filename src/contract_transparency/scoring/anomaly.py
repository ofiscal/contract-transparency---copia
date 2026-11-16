"""Price-anomaly scoring: compare real vs. expected contract value.

Consolidates the legacy post-processing:

* ``distance_functions.distance_percentage``  -> :func:`percentage_error`
* ``desviación.distance_proportion``          -> :func:`within_tolerance`
* ``xgboost.flagging`` (duplicated in 4 files) -> :func:`legacy_flag`
  (its output ``"probability"`` was *not* a probability; it is now called
  ``excess_ratio``; ``"value in danger"`` is ``weighted_excess``)
* ``desviación`` Student-t fit on errors       -> :class:`ResidualDistribution`

Improvements over the legacy approach:

1. Residuals are computed on the log scale, where they are roughly symmetric
   and comparable between a 5 M and a 5 000 M COP contract.
2. The t distribution is fitted on **out-of-sample** (validation) residuals,
   not on training residuals (which are shrunk by over-fitting and would
   produce far too many alerts).
3. The alert uses the tail probability ``P(residual >= r)`` instead of the
   density value ``pdf(r) > 0.7`` (a density is not a probability and its
   scale depends on the units).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from contract_transparency.config import ScoringConfig
from contract_transparency.data import columns as C


def percentage_error(real: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> np.ndarray:
    """(real - predicted) / predicted  (legacy ``distance_percentage``)."""
    real = np.asarray(real, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(predicted > 0, (real - predicted) / predicted, np.nan)


def within_tolerance(
    real: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray, tolerance: float
) -> np.ndarray:
    """True when ``predicted`` lies inside ``real * (1 +- tolerance)``
    (legacy ``distance_proportion``; bounds are exclusive as in the original)."""
    real = np.asarray(real, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return (predicted > real * (1 - tolerance)) & (predicted < real * (1 + tolerance))


def legacy_flag(
    predicted: pd.Series | np.ndarray, real: pd.Series | np.ndarray, floor: float = 1_000_000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised version of the legacy ``flagging(predict, real)``.

    * non-positive predictions are replaced by ``floor`` (1 000 000 COP);
    * if the contract costs more than predicted:
      ``excess_ratio = (real - pred) / (real + 1)`` (in [0, 1)) and
      ``weighted_excess  = excess_ratio * (real - pred)``;
    * otherwise both are 0.
    """
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(real, dtype=float)
    pred = np.where(pred <= 0, floor, pred)
    excess = real - pred
    ratio = np.where(excess > 0, excess / (real + 1), 0.0)
    value = np.where(excess > 0, ratio * excess, 0.0)
    return ratio, value


def group_keys(df: pd.DataFrame, spec: str | None) -> pd.Series | None:
    """Calibration group of each row. ``spec`` is one column or several joined by
    "|" (e.g. "ct_person_type|Tipo de Contrato"). None if a column is missing."""
    if not spec:
        return None
    cols = spec.split("|")
    if any(c not in df.columns for c in cols):
        return None
    keys = df[cols[0]].astype("object").fillna("NA").astype(str)
    for c in cols[1:]:
        keys = keys + "|" + df[c].astype("object").fillna("NA").astype(str)
    return keys


@dataclass
class ResidualDistribution:
    """Student-t fitted on out-of-sample log residuals (real - predicted).

    With ``groups`` (e.g. contract type), residuals are first divided by the
    robust spread of their group (normalised / "Mondrian" conformal
    calibration): public works vary much more than service fees, and one global
    threshold would over-alert the former and miss deviations in the latter.
    Group spreads are shrunk towards the global spread when a group is small.
    """

    df: float = 5.0
    loc: float = 0.0
    scale: float = 1.0
    n: int = 0
    group_column: str | None = None
    group_scales: dict[str, float] = field(default_factory=dict)  # relative to global

    @staticmethod
    def _robust_scale(r: np.ndarray) -> float:
        return float(1.4826 * np.median(np.abs(r - np.median(r)))) or float(np.std(r)) or 1.0

    @classmethod
    def fit(
        cls,
        residuals: np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
        group_column: str | None = None,
        shrinkage: float = 50.0,
    ) -> ResidualDistribution:
        r = np.asarray(residuals, dtype=float)
        ok = np.isfinite(r)
        if ok.sum() < 30:
            raise ValueError("Need at least 30 residuals to fit the residual distribution")
        scales: dict[str, float] = {}
        z = r[ok]
        if groups is not None:
            g = pd.Series(np.asarray(groups, dtype=object)[ok]).fillna("NA").astype(str).to_numpy()
            global_scale = cls._robust_scale(z)
            for key in np.unique(g):
                rg = z[g == key]
                sg = cls._robust_scale(rg) if len(rg) >= 5 else global_scale
                shrunk = (len(rg) * sg + shrinkage * global_scale) / (len(rg) + shrinkage)
                scales[str(key)] = float(shrunk / global_scale)
            z = z / np.array([scales[k] for k in g])
        df, loc, scale = stats.t.fit(z)
        # Guard against degenerate fits on tiny / synthetic samples.
        return cls(
            df=float(np.clip(df, 1.0, 1e3)),
            loc=float(loc),
            scale=float(max(scale, 1e-6)),
            n=int(ok.sum()),
            group_column=group_column if scales else None,
            group_scales=scales,
        )

    def relative_scale(self, groups: pd.Series | np.ndarray | None, n: int) -> np.ndarray:
        if not self.group_scales or groups is None:
            return np.ones(n)
        keys = pd.Series(np.asarray(groups, dtype=object)).fillna("NA").astype(str)
        return keys.map(self.group_scales).fillna(1.0).to_numpy(dtype=float)

    def tail_probability(
        self, residuals: np.ndarray, groups: pd.Series | np.ndarray | None = None
    ) -> np.ndarray:
        """P(R >= r): small values = unusually expensive for comparable contracts."""
        r = np.asarray(residuals, dtype=float)
        z = r / self.relative_scale(groups, len(r))
        return stats.t.sf(z, self.df, self.loc, self.scale)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ResidualDistribution:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def score_prices(
    df: pd.DataFrame,
    predicted: np.ndarray,
    dist: ResidualDistribution,
    cfg: ScoringConfig,
) -> pd.DataFrame:
    """Price-anomaly columns for each contract (index aligned with ``df``)."""
    real = df[C.VALUE].to_numpy(dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    log_res = np.log1p(np.clip(real, 0, None)) - np.log1p(np.clip(predicted, 0, None))
    ratio, var = legacy_flag(predicted, real, cfg.min_prediction_floor)
    tail = dist.tail_probability(log_res, group_keys(df, dist.group_column))
    valid_value = np.isfinite(real)
    out = pd.DataFrame(
        {
            C.PREDICTED: np.round(predicted, 0),
            C.LOG_RESIDUAL: log_res,
            "pct_error": percentage_error(real, predicted),
            "excess_value": np.where(real > predicted, real - predicted, 0.0),
            "excess_ratio": ratio,
            "weighted_excess": var,
            "tail_probability": np.where(valid_value, tail, np.nan),
            "within_tolerance": within_tolerance(real, predicted, cfg.tolerance),
        },
        index=df.index,
    )
    out["price_deviation"] = valid_value & (tail < cfg.tail_probability_alert) & (log_res > 0)
    return out
