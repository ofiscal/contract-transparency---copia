"""Incremental updates: detect new and modified contracts between two extracts.

Evolves ``cleaning.new_data`` from the legacy code ("identify the new values ...
then return the new rows to load the new contracts"). Modified contracts are
interesting on their own: a change in value or dates after signature is how
additions (*otrosíes*) show up in the open-data extracts.
"""

from __future__ import annotations

import pandas as pd


def find_new_contracts(previous: pd.DataFrame, current: pd.DataFrame, key: str) -> pd.DataFrame:
    """Rows of ``current`` whose ``key`` does not exist in ``previous``."""
    known = set(previous[key].dropna().astype(str))
    return current[~current[key].astype(str).isin(known)].copy()


def _row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[cols].astype(str), index=False)


def find_modified_contracts(
    previous: pd.DataFrame, current: pd.DataFrame, key: str, compare: list[str]
) -> pd.DataFrame:
    """Contracts present in both extracts whose ``compare`` columns changed.

    Returns the *current* rows plus ``<col>_previous`` columns for every
    compared column, so the change can be audited.
    """
    compare = [c for c in compare if c in previous.columns and c in current.columns]
    if not compare:
        return current.iloc[0:0].copy()
    prev = previous[[key, *compare]].drop_duplicates(subset=[key], keep="last").copy()
    prev["_hash"] = _row_hash(prev, compare).values
    cur = current.drop_duplicates(subset=[key], keep="last").copy()
    cur["_hash"] = _row_hash(cur, compare).values
    merged = cur.merge(prev, on=key, how="inner", suffixes=("", "_previous"))
    changed = merged[merged["_hash"] != merged["_hash_previous"]]
    return changed.drop(columns=["_hash", "_hash_previous"]).reset_index(drop=True)
