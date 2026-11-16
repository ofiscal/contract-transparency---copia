import numpy as np
import pandas as pd
import pytest

from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import (
    clean_categorical,
    clean_contracts,
    combine_text,
    normalize_text,
    parse_dates,
    parse_duration_days,
    parse_money,
    training_mask,
    unspsc_levels,
)
from tests.conftest import TEST_SALT


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234,567", 1_234_567.0),  # legacy "VALOR NORM" format
        ("1.234.567,89", 1_234_567.89),  # es-CO
        ("1,234,567.89", 1_234_567.89),  # en-US
        ("1234567", 1_234_567.0),  # datos.gov.co export
        ("$ 12.000.000", 12_000_000.0),
        ("12,5", 12.5),  # decimal comma
        ("COP 5,000", 5000.0),
        ("", np.nan),
        ("no definido", np.nan),
    ],
)
def test_parse_money(raw, expected):
    out = parse_money(pd.Series([raw]))[0]
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == pytest.approx(expected)


def test_parse_money_numeric_passthrough():
    s = pd.Series([1, 2.5, None])
    out = parse_money(s)
    assert out.iloc[0] == 1 and out.iloc[1] == 2.5 and np.isnan(out.iloc[2])


@pytest.mark.parametrize(
    ("raw", "days"),
    [
        ("30 Dia(s)", 30),
        (" 3 Mes(es)", 3 * 30.4375),
        ("1 Año(s)", 365.25),
        ("2 Semana(s)", 14),
        ("12 Hora(s)", 0.5),
        ("No definido", np.nan),
        (None, np.nan),
    ],
)
def test_parse_duration(raw, days):
    out = parse_duration_days(pd.Series([raw], dtype=object))[0]
    assert (np.isnan(out) and np.isnan(days)) or out == pytest.approx(days)


def test_parse_dates_mixed_and_invalid():
    out = parse_dates(pd.Series(["05/31/2022", "2022-06-01T00:00:00.000", "not a date", None]))
    assert out[0] == pd.Timestamp("2022-05-31")
    assert out[1] == pd.Timestamp("2022-06-01")
    assert pd.isna(out[2]) and pd.isna(out[3])


def test_normalize_text_removes_line_breaks():
    s = pd.Series(["a\r\nb", "c\\r\\nd", "  e \t f  ", None])
    assert normalize_text(s).tolist() == ["a b", "c d", "e f", ""]


def test_combine_text_skips_duplicates_and_missing():
    df = pd.DataFrame({"a": ["Compra de equipos", "x"], "b": ["COMPRA DE EQUIPOS", None]})
    assert combine_text(df, ["a", "b", "missing"]).tolist() == ["Compra de equipos", "x"]


def test_unspsc_levels():
    out = unspsc_levels(pd.Series(["V1.80111600", "80111600", "bad", None]))
    assert out.iloc[0].tolist() == ["80", "8011", "801116"]
    assert out.iloc[1].tolist() == ["80", "8011", "801116"]
    assert out.iloc[2].isna().all() and out.iloc[3].isna().all()


def test_clean_categorical_missing_tokens():
    out = clean_categorical(pd.Series([" Obra ", "No Definido", "", None, "N/A"]))
    assert out.iloc[0] == "Obra"
    assert out.iloc[1:].isna().all()


def test_clean_contracts_creates_canonical_columns(clean_df):
    for col in (
        C.ID,
        C.VALUE,
        C.SIGNED_AT,
        C.DURATION_DAYS,
        C.TEXT,
        C.UNSPSC_CLASS,
        C.SUPPLIER,
        C.ENTITY,
    ):
        assert col in clean_df.columns
    assert clean_df[C.ID].is_unique
    assert clean_df[C.VALUE].notna().all()
    assert clean_df[C.SIGNED_AT].notna().mean() > 0.99
    assert clean_df[C.DURATION_DAYS].between(1, 400).mean() > 0.99


def test_clean_contracts_dedup_and_range(raw_df, session_cfg):
    raw = pd.concat([raw_df.head(10), raw_df.head(3)], ignore_index=True)
    raw.loc[5, "Valor del Contrato"] = "0"  # zero-value contract
    raw.loc[10, "Valor del Contrato"] = "777000"  # newer version of contract 0
    out = clean_contracts(raw, session_cfg, TEST_SALT)
    assert len(out) == 10
    # duplicates: the last occurrence (most recent extract) wins
    assert out.loc[out[C.ID] == raw.loc[0, "ID Contrato"], C.VALUE].item() == 777000
    # out-of-range values are kept (and scored) but excluded from training
    assert out.loc[out[C.ID] == raw.loc[5, "ID Contrato"], C.VALUE_OUT_OF_RANGE].item()
    assert training_mask(out).sum() == 9


def test_duration_falls_back_to_dates(raw_df, session_cfg):
    raw = raw_df.head(5).copy()
    raw["Duración del contrato"] = None
    out = clean_contracts(raw, session_cfg, TEST_SALT)
    assert out[C.DURATION_DAYS].notna().all()


def test_missing_required_columns(raw_df, session_cfg):
    with pytest.raises(KeyError, match="ID column"):
        clean_contracts(raw_df.drop(columns=["ID Contrato"]), session_cfg, TEST_SALT)
    with pytest.raises(KeyError, match="Target column"):
        clean_contracts(raw_df.drop(columns=["Valor del Contrato"]), session_cfg, TEST_SALT)
