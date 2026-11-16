import numpy as np
import pandas as pd

from contract_transparency.data import columns as C
from contract_transparency.data.incremental import find_modified_contracts, find_new_contracts
from contract_transparency.mlops.drift import build_profile, compare_to_profile, psi


def test_find_new_contracts_legacy_scenario():
    """Same data as legacy cleaning.new_data_test()."""
    X = pd.DataFrame(
        {
            "col1": ["a", "a", "b", "b", "a"],
            "col2": [1.0, 2.0, 3.0, np.nan, 5.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = X.copy()
    y.loc[0, "col1"] = "c"
    y.loc[2, "col3"] = 4.0
    new = find_new_contracts(X, y, key="col1")
    assert new["col1"].tolist() == ["c"]


def test_find_modified_contracts():
    prev = pd.DataFrame({"id": ["1", "2", "3"], "value": [10, 20, 30], "end": ["a", "b", "c"]})
    cur = pd.DataFrame({"id": ["1", "2", "4"], "value": [10, 25, 40], "end": ["a", "b", "d"]})
    mod = find_modified_contracts(prev, cur, "id", ["value", "end"])
    assert mod["id"].tolist() == ["2"]
    assert mod[["value", "value_previous"]].iloc[0].tolist() == [25, 20]
    assert find_modified_contracts(prev, cur, "id", ["missing"]).empty


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25


def test_drift_profile_stable_vs_shifted(clean_df, session_cfg):
    half = len(clean_df) // 2
    prof = build_profile(clean_df.iloc[:half], session_cfg)
    assert "log_value" in prof["numeric"] and "Departamento" in prof["categorical"]
    same = compare_to_profile(clean_df.iloc[half:], prof, session_cfg)
    assert (same["status"] == "ok").all(), same

    shifted = clean_df.iloc[half:].copy()
    shifted["Departamento"] = "Nuevo Departamento"
    shifted[C.VALUE] = shifted[C.VALUE] * 50
    rep = compare_to_profile(shifted, prof, session_cfg).set_index("feature")
    assert rep.loc["Departamento", "status"] == "retrain"
    assert rep.loc["log_value", "status"] == "retrain"


def test_drift_profile_holds_no_row_level_data(clean_df, session_cfg):
    prof = build_profile(clean_df, session_cfg)
    text = str(prof)
    assert "p_" not in text  # no pseudonyms
    assert clean_df[C.ID].iloc[0] not in text
