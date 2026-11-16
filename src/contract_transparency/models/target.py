"""Target transformations for contract values.

Contract values span ~8 orders of magnitude (from 10^3 to 10^12 COP). The
legacy code z-scored them with hard-coded constants (``mean=163740447``,
``ssd=1716771980``) and -- in ``main.py`` -- de-normalised with the wrong sign
(``x*ssd - mean``). With such a heavy tail the z-score squeezes ordinary
contracts into a narrow band (10 M COP -> -0.090, 100 M COP -> -0.037) while a
few huge contracts dominate the squared loss, so models learn little about
typical contracts. ``log1p`` is the default now; ``zscore`` is kept for
reproducing legacy experiments (its constants are *learned* unless given).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TargetTransform:
    kind: str = "log1p"  # log1p | zscore | none
    mean: float | None = None
    std: float | None = None

    def fit(self, y: np.ndarray) -> TargetTransform:
        y = np.asarray(y, dtype=float)
        if self.kind == "zscore":
            if self.mean is None:
                self.mean = float(np.mean(y))
            if self.std is None:
                self.std = float(np.std(y)) or 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.kind == "log1p":
            return np.log1p(np.clip(y, 0, None))
        if self.kind == "zscore":
            return (y - self.mean) / self.std
        return y

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.kind == "log1p":
            return np.expm1(z)
        if self.kind == "zscore":
            return z * self.std + self.mean  # legacy main.py had `- mean` (bug)
        return z

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TargetTransform:
        return cls(**d)
