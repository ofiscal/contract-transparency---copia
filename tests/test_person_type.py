"""Natural vs legal-entity classification and the per-person-type (segmented) model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contract_transparency.config import with_overrides
from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import clean_contracts, training_mask
from contract_transparency.data.person_type import (
    JURIDICA,
    NATURAL,
    UNKNOWN,
    agreement_with_doc_type,
    classify_person_type,
    nit_check_digit,
    number_rule,
)
from contract_transparency.data.splits import split_data
from contract_transparency.data.synthetic import generate_contracts
from contract_transparency.mlops.registry import ModelRegistry
from contract_transparency.models.price_model import PriceModel
from contract_transparency.models.segmented import (
    GLOBAL,
    SegmentedPriceModel,
    build_model,
    load_model,
)
from tests.conftest import TEST_SALT


# --------------------------------------------------------------------------- check digit
@pytest.mark.parametrize(
    ("nit", "dv"),
    [
        (800197268, 4),
        (899999068, 1),
        (890903938, 8),
        (899999063, 3),
    ],  # DIAN, Ecopetrol, Bancolombia, U. Nacional
)
def test_dian_check_digit_matches_published_nits(nit, dv):
    assert nit_check_digit(nit) == dv


# --------------------------------------------------------------------------- number rule
@pytest.mark.parametrize(
    ("number", "expected", "rule"),
    [
        ("900123456", JURIDICA, "nit_range"),
        ("800197268-4", JURIDICA, "nit_range_with_dv"),
        ("8001972684", JURIDICA, "nit_range_with_dv"),  # DV appended without separator
        ("1012345678", NATURAL, "cedula_nuip"),
        ("52123456", NATURAL, "cedula_old_series"),
        ("52.123.456", NATURAL, "cedula_old_series"),
        ("52123456-" + str(nit_check_digit(52123456)), NATURAL, "cedula_with_dv"),
        ("1012345678" + str(nit_check_digit(1012345678)), NATURAL, "cedula_with_dv"),
        ("AB1234567", UNKNOWN, "not_numeric"),  # passport
        ("12345678901234", UNKNOWN, "unusual_length"),
        (None, UNKNOWN, "no_number"),
        (52123456.0, NATURAL, "cedula_old_series"),  # float-coerced
    ],
)
def test_number_rule(number, expected, rule):
    assert number_rule(number) == (expected, rule)


def test_rule_precedence_and_ambiguity():
    # 9 digits starting with 8 whose last digit is the check digit of the first 8:
    # a company NIT or an old cédula (80 million range) + DV -> cannot tell without a name
    ambiguous = f"80123456{nit_check_digit(80123456)}"
    df = pd.DataFrame(
        [
            ("Cédula de Ciudadanía", "900123456", "Juan Ficticio", "No"),  # doc type wins
            ("NIT", "1012345678", "Consorcio Vial 2023", "No"),  # legal form in name wins
            ("NIT", "52123456", "Pedro Ficticio", "Si"),  # group flag wins
            ("NIT", ambiguous, "Pedro Ficticio", "No"),  # could be cédula + DV: undetermined
            ("NIT", ambiguous, None, "No"),  # no name: NIT range
            ("NIT", "ABC", "Nombre sin forma", "No"),  # unusable number + NIT doc type
            ("No Definido", None, None, None),
        ],
        columns=["doc", "num", "name", "group"],
    )
    out = classify_person_type(df["doc"], df["num"], df["name"], df["group"])
    assert out["person_type"].tolist() == [
        NATURAL,
        JURIDICA,
        JURIDICA,
        UNKNOWN,
        JURIDICA,
        JURIDICA,
        UNKNOWN,
    ]
    assert out["rule"].tolist() == [
        "doc_type",
        "legal_form_in_name",
        "group_flag",
        "ambiguous_nit_or_cedula",
        "nit_range",
        "doc_type_nit",
        "no_number",
    ]


def test_configured_natural_doc_types_are_honoured():
    out = classify_person_type(
        pd.Series(["Documento Especial X"]),
        pd.Series(["900123456"]),
        natural_doc_types=["Documento Especial X"],
    )
    assert out["person_type"].iloc[0] == NATURAL


def test_agreement_with_doc_type():
    doc = pd.Series(["Cédula de Ciudadanía"] * 3 + ["NIT"])
    num = pd.Series(["52123456", "1012345678", "900123456", "900123456"])
    assert agreement_with_doc_type(doc, num) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- cleaning
def test_classification_on_synthetic_data_is_never_wrong(raw_df, clean_df):
    truth = raw_df.set_index("ID Contrato").loc[clean_df[C.ID], "synthetic_person_type"].to_numpy()
    got = clean_df[C.PERSON_TYPE].to_numpy()
    decided = got != UNKNOWN
    assert (got[decided] == truth[decided]).all()  # no natural<->juridica confusion
    assert (~decided).mean() < 0.03  # only genuinely ambiguous numbers left undetermined
    assert set(clean_df[C.PERSON_TYPE_RULE]) >= {
        "doc_type",
        "legal_form_in_name",
        "nit_range",
        "cedula_nuip",
    }
    assert (clean_df[C.NATURAL_PERSON] == (clean_df[C.PERSON_TYPE] == NATURAL)).all()


def test_names_are_masked_unless_positively_a_legal_entity(clean_df):
    display = clean_df[C.SUPPLIER_DISPLAY]
    kind = clean_df[C.PERSON_TYPE]
    assert display[kind == NATURAL].str.startswith("Persona natural p_").all()
    assert display[kind == UNKNOWN].str.startswith("Proveedor sin clasificar p_").all()
    assert not display[kind == JURIDICA].str.contains("p_").any()


def test_person_type_stats_in_the_prepare_report(cfg, raw_df, write_raw):
    from contract_transparency.mlops import pipeline

    write_raw(raw_df)
    _, report = pipeline.prepare(cfg)
    stats = report["clean"]["stats"]
    assert stats["share_person_natural"] > 0.3 and stats["share_person_juridica"] > 0.3
    assert stats["person_type_number_rule_agreement"] > 0.99


# --------------------------------------------------------------------------- segmented model
@pytest.fixture(scope="module")
def split(session_cfg):
    df = clean_contracts(generate_contracts(6000, seed=8), session_cfg, TEST_SALT)
    return split_data(df[training_mask(df)].reset_index(drop=True), session_cfg.evaluation)


@pytest.fixture(scope="module")
def seg_cfg(session_cfg):
    return with_overrides(
        session_cfg, {"model.min_segment_train_rows": 300, "model.min_segment_valid_rows": 30}
    )


@pytest.fixture(scope="module")
def segmented(seg_cfg, split):
    return build_model(seg_cfg).fit(split.train, split.valid)


def test_one_model_per_person_type_plus_global(segmented, split):
    assert isinstance(segmented, SegmentedPriceModel)
    assert set(segmented.models) == {GLOBAL, NATURAL, JURIDICA}
    route = segmented.segment_of(split.test)
    kind = split.test[C.PERSON_TYPE]
    assert (route[kind == NATURAL] == NATURAL).all() and (route[kind == JURIDICA] == JURIDICA).all()
    assert (route[kind == UNKNOWN] == GLOBAL).all()
    # each segment's predictions come from its own model
    nat = split.test[kind == NATURAL]
    np.testing.assert_allclose(segmented.predict(nat), segmented.models[NATURAL].predict(nat))


def test_small_segments_fall_back_to_the_global_model(session_cfg, split):
    cfg = with_overrides(session_cfg, {"model.min_segment_train_rows": 10**9})
    model = build_model(cfg).fit(split.train, split.valid)
    assert set(model.models) == {GLOBAL}
    np.testing.assert_allclose(model.predict(split.test), model.global_model.predict(split.test))


def test_segmented_model_persists_and_is_portable(segmented, split, tmp_path):
    reg = ModelRegistry(tmp_path, "m")
    version = reg.register(segmented, {})
    assert reg.is_portable(version)
    artifacts = reg.metadata(version)["artifacts"]
    assert "segments.json" in artifacts and any(k.startswith("segment_natural/") for k in artifacts)
    loaded, _ = reg.load(version)
    assert isinstance(loaded, SegmentedPriceModel)
    np.testing.assert_allclose(loaded.predict(split.test), segmented.predict(split.test))
    assert isinstance(load_model(reg.versions_dir / version), SegmentedPriceModel)
    top = loaded.top_features(5)
    assert set(top["segment"]) == {GLOBAL, NATURAL, JURIDICA}


def test_single_model_when_segmentation_is_off(session_cfg):
    assert isinstance(
        build_model(with_overrides(session_cfg, {"model.segment_by": None})), PriceModel
    )
    assert isinstance(build_model(with_overrides(session_cfg, {"model.type": "ridge"})), PriceModel)


def test_calibration_groups_combine_person_and_contract_type(seg_cfg, segmented, split):
    from contract_transparency.mlops.pipeline import calibration_column, fit_residual_distribution

    spec = calibration_column(seg_cfg)
    assert spec == f"{C.PERSON_TYPE}|Tipo de Contrato"
    dist = fit_residual_distribution(
        seg_cfg, split.valid, split.valid[C.VALUE].to_numpy(), segmented.predict(split.valid)
    )
    assert dist.group_column == spec
    assert any(k.startswith("natural|Prestación de servicios") for k in dist.group_scales)
    assert any(k.startswith("juridica|") for k in dist.group_scales)


def test_person_type_is_a_feature_but_no_identifier_is(session_cfg, split):
    model = PriceModel(session_cfg).fit(split.train, split.valid)
    names = " ".join(model.feature_names_)
    assert C.PERSON_TYPE in names
    assert C.SUPPLIER not in names and "Documento Proveedor" not in names
