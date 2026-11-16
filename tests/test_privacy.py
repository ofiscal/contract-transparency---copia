import hashlib
import os
import stat

import pandas as pd
import pytest

from contract_transparency.data import columns as C
from contract_transparency.data.io import read_contracts
from contract_transparency.data.privacy import (
    apply_privacy,
    get_salt,
    mask_person_names,
    normalize_identifier,
    pseudonymize,
    redact_text,
)


def test_pseudonyms_are_deterministic_keyed_and_normalised():
    s = pd.Series(["1.234.567", "1234567", 1234567.0, " 1234567 ", None, "7654321"])
    p = pseudonymize(s, b"key-a")
    assert p[0] == p[1] == p[2] == p[3]
    assert p[0].startswith("p_") and len(p[0]) == 18
    assert p[4] is None or pd.isna(p[4])
    assert p[5] != p[0]
    # keyed: another key gives another pseudonym, and it is not a plain hash
    assert pseudonymize(s, b"key-b")[0] != p[0]
    assert hashlib.sha256(b"1234567").hexdigest()[:16] not in p[0]


def test_normalize_identifier():
    assert normalize_identifier("900.123.456-7") == "9001234567"
    assert normalize_identifier("No Definido") is None
    assert normalize_identifier(float("nan")) is None


@pytest.mark.parametrize(
    ("text", "must_not_contain", "must_contain"),
    [
        ("Contratista identificado con C.C. 1.234.567.890 para apoyo", "1.234.567", "<ID>"),
        ("cédula de ciudadanía No. 52123456", "52123456", "<ID>"),
        ("NIT 900.123.456-7", "900.123", "<ID>"),
        ("escribir a juan.perez@example.com", "juan.perez", "<EMAIL>"),
        ("celular 310 123 4567", "4567", "<TEL>"),
        ("por valor de $12.000.000 pesos", "12.000.000", "<NUM>"),
        ("cuenta 1234567890", "1234567890", "<NUM>"),
    ],
)
def test_redact_text(text, must_not_contain, must_contain):
    out = redact_text(pd.Series([text]))[0]
    assert must_not_contain not in out
    assert must_contain in out


def test_redact_text_keeps_ordinary_numbers():
    out = redact_text(pd.Series(["Suministro de 3 computadores para la vigencia 2023"]))[0]
    assert out == "Suministro de 3 computadores para la vigencia 2023"


def test_salt_from_env_and_file(cfg, monkeypatch):
    assert get_salt(cfg) == b"test-salt-not-secret"
    monkeypatch.delenv("CT_PII_SALT")
    first = get_salt(cfg)
    path = cfg.paths.secrets_dir / "pii_salt"
    assert path.exists()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert get_salt(cfg) == first  # stable between runs
    assert len(first) == 64


def test_apply_privacy_drops_and_pseudonymizes(raw_df, session_cfg):
    out = apply_privacy(raw_df.head(50), session_cfg, b"k")
    for col in session_cfg.privacy.drop_columns:
        assert col not in out.columns
    assert out["Documento Proveedor"].str.startswith("p_").all()
    assert not out["Documento Proveedor"].isin(raw_df["Documento Proveedor"]).any()


def test_pii_columns_never_loaded(cfg, raw_df, write_raw):
    path = write_raw(raw_df.head(20))
    df = read_contracts(path, cfg)
    assert "Número de cuenta" in raw_df.columns
    assert not set(cfg.privacy.drop_columns) & set(df.columns)


def test_mask_person_names():
    names = pd.Series(["Juan Ficticio", "Empresa Demo S.A.S."])
    out = mask_person_names(names, pd.Series([True, False]), pd.Series(["p_abc", "p_def"]))
    assert out.tolist() == ["Persona natural p_abc", "Empresa Demo S.A.S."]


def test_cleaned_table_contains_no_personal_data(raw_df, clean_df):
    """End-to-end guarantee: no raw ID, name, bank data or supervisor in the clean table."""
    flat = clean_df.astype(str).to_numpy().ravel()
    text = "\n".join(flat)
    natural = raw_df["TipoDocProveedor"] == "Cédula de Ciudadanía"
    for raw_id in raw_df.loc[natural, "Documento Proveedor"].unique()[:200]:
        assert raw_id not in text
    assert "Contratista Ficticio" not in text
    assert "Representante Ficticio" not in text
    assert "Supervisor Ficticio" not in text
    assert "000-DEMO" not in text
    # companies keep their (public) names
    assert clean_df[C.SUPPLIER_DISPLAY].str.contains("Empresa Demo").any()
