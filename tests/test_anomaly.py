import numpy as np
import pandas as pd
import pytest
from scipy import stats

from contract_transparency.config import ScoringConfig
from contract_transparency.data import columns as C
from contract_transparency.scoring.anomaly import (
    ResidualDistribution,
    legacy_flag,
    percentage_error,
    score_prices,
    within_tolerance,
)


# --- verbatim legacy implementations used as test oracles --------------------
def legacy_flagging(predict, real):  # legacy/.../xgboost.py
    if predict <= 0:
        predict = 1000000
    if (real - predict) > 0:
        size = (real - predict) / (real + 1)
        value = size * (real - predict)
    else:
        size = 0
        value = 0
    return float(size), float(value)


def legacy_distance_proportion(size, value, predicted):  # legacy/.../desviación.py
    superior = value * (1 + size)
    inferior = value * (1 - size)
    return predicted > inferior and predicted < superior


def test_legacy_flag_parity():
    rng = np.random.default_rng(0)
    pred = np.concatenate([rng.lognormal(17, 2, 500), [0.0, -5.0, 1e6]])
    real = np.concatenate([rng.lognormal(17, 2, 500), [5e6, 5e5, 1e6]])
    ratio, value = legacy_flag(pred, real)
    for p, r, a, b in zip(pred, real, ratio, value, strict=True):
        ea, eb = legacy_flagging(p, r)
        assert a == pytest.approx(ea) and b == pytest.approx(eb)


def test_within_tolerance_parity():
    rng = np.random.default_rng(1)
    real, pred = rng.lognormal(15, 1, 300), rng.lognormal(15, 1, 300)
    ours = within_tolerance(real, pred, 0.8)
    assert ours.tolist() == [
        legacy_distance_proportion(0.8, r, p) for r, p in zip(real, pred, strict=True)
    ]


def test_percentage_error_matches_legacy_definition():
    assert percentage_error([150.0], [100.0])[0] == pytest.approx(0.5)  # (X1-X2)/X2
    assert np.isnan(percentage_error([1.0], [0.0])[0])


def test_residual_distribution_fit_and_tail():
    r = stats.t.rvs(4, loc=0.0, scale=0.3, size=5000, random_state=2)
    dist = ResidualDistribution.fit(r)
    assert dist.scale == pytest.approx(0.3, rel=0.15)
    tails = dist.tail_probability(np.array([-1.0, 0.0, 1.0, 3.0]))
    assert np.all(np.diff(tails) < 0)
    assert ResidualDistribution.from_dict(dist.to_dict()) == dist
    with pytest.raises(ValueError):
        ResidualDistribution.fit(np.array([0.1, 0.2]))


def test_score_prices_only_alerts_overpricing():
    dist = ResidualDistribution(df=5, loc=0, scale=0.3, n=1000)
    df = pd.DataFrame({C.VALUE: [100e6, 10e6, 1e6, np.nan]})
    pred = np.array([10e6, 10e6, 100e6, 10e6])  # 10x over, exact, 100x under, missing
    out = score_prices(df, pred, dist, ScoringConfig())
    assert out["price_deviation"].tolist() == [True, False, False, False]
    assert out.loc[0, "excess_value"] == pytest.approx(90e6)
    assert out.loc[2, "excess_value"] == 0
    assert out.loc[0, "tail_probability"] < 0.01 < out.loc[1, "tail_probability"]
