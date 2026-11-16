import pandas as pd

from contract_transparency.analysis.contractors import entity_concentration, top_suppliers
from contract_transparency.analysis.sustainability import classify_sustainability
from contract_transparency.data import columns as C
from contract_transparency.data.validation import validate_clean, validate_raw


def test_validate_raw(raw_df, session_cfg):
    safe = raw_df.drop(columns=[c for c in session_cfg.privacy.drop_columns if c in raw_df])
    assert validate_raw(safe, session_cfg).ok
    rep = validate_raw(raw_df, session_cfg)  # PII not removed -> error
    assert not rep.ok and "PII" in rep.errors[0]
    rep = validate_raw(safe.drop(columns=["Valor del Contrato", "Ciudad"]), session_cfg)
    assert any("Valor del Contrato" in e for e in rep.errors)
    assert any("Ciudad" in w for w in rep.warnings)
    assert not validate_raw(pd.DataFrame(), session_cfg).ok


def test_validate_clean(clean_df, session_cfg):
    rep = validate_clean(clean_df, session_cfg)
    assert rep.ok and rep.stats["n_trainable"] > 2500
    tiny = validate_clean(clean_df.head(50), session_cfg)
    assert not tiny.ok and "need >= 200" in tiny.errors[0]
    undated = clean_df.assign(**{C.SIGNED_AT: pd.NaT})
    assert any("temporal split" in e for e in validate_clean(undated, session_cfg).errors)


def test_top_suppliers_masks_people_and_excludes_universities(clean_df):
    df = clean_df.copy()
    df.loc[df.index[:50], C.SUPPLIER_DISPLAY] = "Universidad Demo Nacional"
    df.loc[df.index[:50], C.SUPPLIER] = "p_uni"
    df.loc[df.index[:50], C.VALUE] = 1e12
    top = top_suppliers(df, n=10)
    assert "p_uni" not in top["supplier_pseudonym"].tolist()
    assert top_suppliers(df, n=10, exclude_regex=None)["supplier_pseudonym"].iloc[0] == "p_uni"
    people = top_suppliers(clean_df, n=2000)
    assert people.loc[people["natural_person"], "supplier"].str.startswith("Persona natural").all()
    assert top_suppliers(clean_df, since="2023-06-01")["last_signed"].min() >= pd.Timestamp(
        "2023-06-01"
    )


def test_entity_concentration(clean_df):
    hhi = entity_concentration(clean_df)
    assert hhi["hhi"].between(0, 10_000).all()


def test_sustainability_with_fake_classifier():
    calls = []

    def fake(texts, candidate_labels):
        calls.append(len(texts))
        if len(texts) == 1:  # mimic pipelines that un-list single results
            return {
                "labels": candidate_labels,
                "scores": [0.9] + [0.01] * (len(candidate_labels) - 1),
            }
        return [
            {"labels": candidate_labels[::-1], "scores": [0.8] * len(candidate_labels)}
            for _ in texts
        ]

    texts = ["agua potable", "", "energía solar"]
    # batch_size=2 -> each batch has ONE non-empty text -> dict (un-listed) path
    out = classify_sustainability(texts, classifier=fake, batch_size=2)
    labels = out["sustainability_label"]
    assert labels.isna().tolist() == [False, True, False]  # None (pandas 2) / NaN (pandas 3)
    assert labels.dropna().tolist() == ["agua", "agua"]
    assert calls == [1, 1]  # empty texts are never sent to the model
    # batch_size=3 -> one batch with two texts -> list path
    out = classify_sustainability(texts, classifier=fake, batch_size=3)
    assert out["sustainability_label"].dropna().tolist() == ["otros", "otros"]
    assert out["sustainability_score"].iloc[0] == 0.8
