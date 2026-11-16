"""Train / validation / test splits.

The legacy scripts used ``train_test_split(..., random_state=153)`` on
contracts from several years, and ``main.py`` even *predicted on rows that
were also used for training* (``skiprows=[1, subsetsize]`` skips just two
rows). Both leak information. The default here is a **temporal split**: the
model is trained on older contracts and evaluated on the most recent ones,
which is exactly how it is used in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from contract_transparency.config import EvaluationConfig
from contract_transparency.data import columns as C


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "valid": len(self.valid), "test": len(self.test)}


def split_data(df: pd.DataFrame, ev: EvaluationConfig) -> Split:
    n = len(df)
    n_test = max(1, int(round(n * ev.test_size)))
    n_valid = max(1, int(round(n * ev.valid_size)))
    if n - n_test - n_valid < 1:
        raise ValueError(f"Not enough rows ({n}) for the requested split sizes")

    if ev.split == "temporal":
        # Undated contracts go to the training set (never to the test set).
        order = df[C.SIGNED_AT].fillna(pd.Timestamp.min).argsort(kind="stable").to_numpy()
    else:
        order = np.random.default_rng(ev.random_state).permutation(n)

    test_idx = order[n - n_test :]
    valid_idx = order[n - n_test - n_valid : n - n_test]
    train_idx = order[: n - n_test - n_valid]
    take = df.iloc
    return Split(
        train=take[train_idx].reset_index(drop=True),
        valid=take[valid_idx].reset_index(drop=True),
        test=take[test_idx].reset_index(drop=True),
    )
