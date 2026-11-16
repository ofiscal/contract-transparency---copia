"""Rule tests on small hand-made frames with known answers."""

import numpy as np
import pandas as pd
import pytest

from contract_transparency.data import columns as C
from contract_transparency.scoring import rules as R


def frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        C.ID: None,
        C.VALUE: 10e6,
        C.VALUE_OUT_OF_RANGE: False,
        C.SIGNED_AT: "2022-08-01",
        C.DURATION_DAYS: 90.0,
        C.TEXT: "Prestación de servicios",
        C.UNSPSC_CLASS: "801116",
        C.NATURAL_PERSON: False,
        C.SUPPLIER: "p_s1",
        C.ENTITY: "E1",
        C.LEGAL_REP: None,
        "Modalidad de Contratacion": "Mínima cuantía",
        "Tipo de Contrato": "Prestación de servicios",
        "Justificacion Modalidad de Contratacion": "No Especificado",
        "Orden": "Territorial",
        "Dias adicionados": "0",
    }
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    df[C.ID] = [f"c{i}" for i in range(len(df))]
    df[C.SIGNED_AT] = pd.to_datetime(df[C.SIGNED_AT])
    return df


DIRECT = "Contratación directa"


def test_ley_garantias_windows(session_cfg):
    # congress 2022-03-13 -> restriction from 2021-11-13; presidential 2022-05-29
    # -> restriction 2022-01-29 .. 2022-06-19 (second round)
    df = frame(
        [
            {"Modalidad de Contratacion": DIRECT, C.SIGNED_AT: "2022-03-01"},  # during presidential
            {"Modalidad de Contratacion": DIRECT, C.SIGNED_AT: "2022-06-15"},  # before 2nd round
            {"Modalidad de Contratacion": DIRECT, C.SIGNED_AT: "2021-12-01"},  # congress only
            {
                "Modalidad de Contratacion": DIRECT,
                C.SIGNED_AT: "2021-10-25",
            },  # rush before 2021-11-13
            {"Modalidad de Contratacion": "Licitación pública", C.SIGNED_AT: "2022-03-01"},
            {"Modalidad de Contratacion": DIRECT, C.SIGNED_AT: "2022-08-01"},
            {  # territorial convenio interadministrativo during a congress restriction
                "Modalidad de Contratacion": DIRECT,
                "Justificacion Modalidad de Contratacion": "ContratosConveniosInteradministrativosConValor",
                C.SIGNED_AT: "2021-12-01",
            },
            {  # same, but national entity -> art. 38 does not apply
                "Modalidad de Contratacion": DIRECT,
                "Justificacion Modalidad de Contratacion": "ContratosConveniosInteradministrativosConValor",
                "Orden": "Nacional",
                C.SIGNED_AT: "2021-12-01",
            },
        ]
    )
    f = R.ley_garantias_flags(df, session_cfg)
    assert f["direct_during_ley_garantias"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert f["pre_ley_garantias_rush"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert f["interadministrative_during_ley_garantias"].tolist() == [False] * 6 + [True, False]


def test_days_to_next_restriction(session_cfg):
    s = pd.Series(pd.to_datetime(["2021-11-03", "2030-01-01", None]))
    out = R.days_to_next_restriction(s, session_cfg.rules)
    assert out[0] == 10  # restriction for congress 2022-03-13 starts 2021-11-13
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_concurrent_direct_services(session_cfg):
    person = {C.NATURAL_PERSON: True, C.SUPPLIER: "p_person", "Modalidad de Contratacion": DIRECT}
    df = frame(
        [person] * 4
        + [{**person, C.SUPPLIER: "p_other"}] * 3
        + [{**person, C.SIGNED_AT: "2023-01-01"}]
    )
    out = R.concurrent_direct_services(df, session_cfg)
    assert out.tolist() == [True] * 4 + [False] * 3 + [False]


def test_possible_contract_splitting(session_cfg):
    close = [
        {C.SUPPLIER: "p_a", C.SIGNED_AT: d} for d in ("2022-08-01", "2022-08-10", "2022-08-20")
    ]
    far = [{C.SUPPLIER: "p_b", C.SIGNED_AT: d} for d in ("2022-01-01", "2022-03-01", "2022-05-01")]
    other_class = [{C.SUPPLIER: "p_a", C.SIGNED_AT: "2022-08-05", C.UNSPSC_CLASS: "999999"}]
    out = R.possible_contract_splitting(frame(close + far + other_class), session_cfg)
    assert out.tolist() == [True, True, True, False, False, False, False]


def test_shared_legal_representative(session_cfg):
    df = frame(
        [
            {C.SUPPLIER: "p_c1", C.LEGAL_REP: "p_rep"},
            {C.SUPPLIER: "p_c2", C.LEGAL_REP: "p_rep"},
            {C.SUPPLIER: "p_c3", C.LEGAL_REP: "p_rep3"},
            {C.SUPPLIER: "p_n1", C.LEGAL_REP: "p_n1", C.NATURAL_PERSON: True},
        ]
    )
    assert R.shared_legal_representative(df, session_cfg).tolist() == [True, True, False, False]


def test_spending_shares(session_cfg):
    small = [{C.SUPPLIER: f"p_{i}", C.VALUE: 1e6} for i in range(6)]
    big = [{C.SUPPLIER: "p_big", C.VALUE: 50e6}]
    out = R.spending_shares(frame(small + big), session_cfg)
    assert out["large_share_of_entity_spending"].tolist() == [False] * 6 + [True]
    assert out["dominant_supplier"].tolist() == [False] * 6 + [True]


def test_spending_shares_needs_enough_contracts(session_cfg):
    out = R.spending_shares(frame([{C.VALUE: 1e6}, {C.VALUE: 9e6}]), session_cfg)
    assert not out["large_share_of_entity_spending"].any()


def test_additions(session_cfg):
    df = frame(
        [
            {C.TEXT: "Adición No. 1 al contrato de obra"},
            {C.TEXT: "OTROSÍ modificatorio"},
            {C.TEXT: "Suministro de papelería", "Dias adicionados": "60"},
            {C.TEXT: "Suministro de papelería", "Dias adicionados": "10"},
        ]
    )
    out = R.additions(df, session_cfg)
    assert out["mentions_addition"].tolist() == [True, True, False, False]
    assert out["large_days_added"].tolist() == [False, False, True, False]


def test_atypical_duration(session_cfg):
    rows = [{C.DURATION_DAYS: d} for d in [88, 90, 92, 91, 89, 90, 87, 93, 90, 1800]]
    out = R.atypical_duration(frame(rows), session_cfg)
    assert out.tolist() == [False] * 9 + [True]


def test_single_bidder_requires_configured_column(session_cfg):
    df = frame([{"Numero Oferentes": "1"}, {"Numero Oferentes": "4"}])
    assert R.single_bidder(df, session_cfg) is None
    cfg = session_cfg.model_copy(
        update={"schema_": session_cfg.schema_.model_copy(update={"n_bidders": "Numero Oferentes"})}
    )
    assert R.single_bidder(df, cfg).tolist() == [True, False]


def test_evaluate_rules_skips_missing_columns(session_cfg):
    df = frame([{}] * 3).drop(
        columns=["Modalidad de Contratacion", "Tipo de Contrato", "Dias adicionados"]
    )
    out = R.evaluate_rules(df, session_cfg)
    assert "sig_direct_during_ley_garantias" not in out  # cannot be evaluated
    assert "sig_large_days_added" not in out
    assert "sig_mentions_addition" in out and "sig_data_quality_suspect" in out


def test_alert_index_weights_and_explanations():
    flags = pd.DataFrame(
        {
            "sig_price_deviation": [True, False, False],
            "sig_dominant_supplier": [True, True, False],
            "sig_data_quality_suspect": [True, True, True],
        }
    )
    out = R.alert_index(
        flags, {"price_deviation": 3.0, "dominant_supplier": 1.0, "data_quality_suspect": 0.0}
    )
    assert out["alert_index"].tolist() == pytest.approx([1.0, 0.25, 0.0])
    assert out["n_signals"].tolist() == [2, 1, 0]
    assert out["signals"].tolist() == [
        "price_deviation; dominant_supplier",
        "dominant_supplier",
        "",
    ]


def test_planted_patterns_detected_on_synthetic(clean_df, session_cfg):
    flags = R.evaluate_rules(clean_df, session_cfg)
    split_rows = clean_df["Nombre Entidad"] == "Alcaldía Demo Fraccionamiento"
    assert flags.loc[split_rows, "sig_possible_contract_splitting"].all()
    concurrent = clean_df[C.ENTITY] == "899900001"
    assert flags.loc[concurrent, "sig_concurrent_direct_services"].sum() == 5
    assert flags["sig_shared_legal_representative"].any()
