"""Supplier rankings (legacy ``trials_of_higher_contractors.py``).

The legacy script summed "Precio Base" by ``CodigoProveedor`` for contracts
published after 07/07/2022 and exported names to Excel, with a pending note
"excluir universidades públicas". Here:

* suppliers are grouped by their *pseudonymised* id;
* names of natural persons are masked (companies keep their names);
* public universities (and any other regex) can be excluded;
* a Herfindahl-Hirschman index per entity measures supplier concentration.
"""

from __future__ import annotations

import pandas as pd

from contract_transparency.data import columns as C

DEFAULT_EXCLUDE = r"universidad|instituci[oó]n universitaria|escuela superior"


def top_suppliers(
    df: pd.DataFrame,
    n: int = 50,
    since: str | pd.Timestamp | None = None,
    exclude_regex: str | None = DEFAULT_EXCLUDE,
) -> pd.DataFrame:
    data = df[df[C.SUPPLIER].notna() & df[C.VALUE].notna() & ~df[C.VALUE_OUT_OF_RANGE]]
    if since is not None:
        data = data[data[C.SIGNED_AT] >= pd.Timestamp(since)]
    if exclude_regex:
        names = data[C.SUPPLIER_DISPLAY].fillna("").astype(str)
        data = data[~names.str.contains(exclude_regex, case=False, regex=True)]
    grouped = (
        data.groupby(C.SUPPLIER)
        .agg(
            supplier=(C.SUPPLIER_DISPLAY, "first"),
            natural_person=(C.NATURAL_PERSON, "max"),
            total_value=(C.VALUE, "sum"),
            n_contracts=(C.ID, "count"),
            n_entities=(C.ENTITY, "nunique"),
            first_signed=(C.SIGNED_AT, "min"),
            last_signed=(C.SIGNED_AT, "max"),
        )
        .sort_values("total_value", ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={C.SUPPLIER: "supplier_pseudonym"})
    )
    grouped["share_of_total"] = grouped["total_value"] / data[C.VALUE].sum()
    return grouped


def entity_concentration(df: pd.DataFrame, min_contracts: int = 10) -> pd.DataFrame:
    """Herfindahl-Hirschman index (0-10 000) of supplier shares per entity."""
    data = df[df[C.SUPPLIER].notna() & df[C.VALUE].notna() & ~df[C.VALUE_OUT_OF_RANGE]]
    by_pair = data.groupby([C.ENTITY, C.SUPPLIER])[C.VALUE].sum()
    shares = by_pair / by_pair.groupby(level=0).transform("sum")
    hhi = (shares.pow(2) * 10_000).groupby(level=0).sum()
    counts = data.groupby(C.ENTITY)[C.ID].count()
    out = pd.DataFrame(
        {"hhi": hhi, "n_contracts": counts, "n_suppliers": by_pair.groupby(level=0).size()}
    )
    return (
        out[out["n_contracts"] >= min_contracts].sort_values("hhi", ascending=False).reset_index()
    )
