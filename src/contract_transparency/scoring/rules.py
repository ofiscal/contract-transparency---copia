"""Rule-based alert signals for the review of public contracts.

Implements the ideas sketched (but never finished -- the file did not even
parse) in ``legacy/python3_10/modules/models/rules_based.py``:

=====================================  =======================================
Legacy idea (comment / stub)           Rule here
=====================================  =======================================
date_near_change / date_near_control   ``pre_ley_garantias_rush``,
("ley de garantías 4 meses antes ...   ``direct_during_ley_garantias``,
justo antes de la vigencia disparan")  ``days_to_next_restriction``
contratación_directa_prestación        ``concurrent_direct_services``
("persona natural ... 0 if x<4")
otrosies ("adición, otrosí, otro si")  ``mentions_addition``, ``large_days_added``
big_contracts ("% del gasto del        ``large_share_of_entity_spending``
municipio en un solo contrato")
proveedor_monopolistico ("empresas     ``dominant_supplier``,
con el mismo representante")           ``shared_legal_representative``
licitación_un_oferente                 ``single_bidder`` (needs ``schema.n_bidders``)
fechas_atípicos_de_duración            ``atypical_duration``
ponderador ("un solo indicador")       :func:`alert_index`
cne_cuentas (campaign donors)          not implemented -> docs/ROADMAP.md
=====================================  =======================================

New: ``possible_contract_splitting`` (fraccionamiento de contratos).

IMPORTANT: a signal is a reason to *look*, never evidence of wrongdoing.
Ley 996 de 2005 has many exceptions (defensa, salud, emergencias, crédito
público...), and entities legitimately sign many direct contracts. Rules
whose input columns are missing are skipped (reported as ``None``), never
guessed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from contract_transparency.config import RulesConfig, Settings
from contract_transparency.data import columns as C

logger = logging.getLogger(__name__)

SIGNAL_PREFIX = "sig_"

# Neutral Spanish labels for the dashboard (output files keep the stable codes).
SIGNAL_LABELS: dict[str, str] = {
    "price_deviation": "Valor por encima de contratos comparables",
    "pre_ley_garantias_rush": "Firmado justo antes de Ley de Garantías",
    "direct_during_ley_garantias": "Contratación directa en periodo de Ley de Garantías",
    "interadministrative_during_ley_garantias": "Convenio interadministrativo en periodo de Ley de Garantías",
    "concurrent_direct_services": "Varios contratos directos simultáneos con la misma persona",
    "large_share_of_entity_spending": "Alta proporción del gasto anual de la entidad",
    "dominant_supplier": "Proveedor con alta proporción del gasto de la entidad",
    "shared_legal_representative": "Representante legal compartido entre proveedores",
    "possible_contract_splitting": "Contratos similares cercanos en el tiempo (posible fraccionamiento)",
    "mentions_addition": "Menciona adición, otrosí o prórroga",
    "large_days_added": "Días adicionados altos frente a la duración",
    "atypical_duration": "Duración poco común para el tipo de contrato",
    "single_bidder": "Proceso competitivo con un solo oferente",
    "supplier_in_fiscal_bulletin": "Proveedor o representante en boletín de responsables fiscales",
    "supplier_previous_sanction": "Proveedor con sanciones previas en SECOP",
    "data_quality_suspect": "Valor con posible error de registro",
}


def signal_label(code: str) -> str:
    code = code[len(SIGNAL_PREFIX) :] if code.startswith(SIGNAL_PREFIX) else code
    return SIGNAL_LABELS.get(code, code.replace("_", " "))


def _matches_any(values: pd.Series, patterns: list[str]) -> pd.Series:
    s = values.fillna("").astype(str).str.casefold()
    out = pd.Series(False, index=values.index)
    for p in patterns:
        out |= s.str.contains(p.casefold(), regex=False)
    return out


def _col(df: pd.DataFrame, name: str | None) -> pd.Series | None:
    return df[name] if name and name in df.columns else None


# --------------------------------------------------------------------------- #
# Ley de garantías (Ley 996 de 2005)
# --------------------------------------------------------------------------- #
def restriction_periods(
    cfg: RulesConfig, types: set[str] | None = None
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of each restricted period: ``restriction_months`` before
    election day until election day (or the second round, if any)."""
    periods = []
    for e in sorted(cfg.elections, key=lambda x: x.date):
        if types is not None and e.type not in types:
            continue
        end = pd.Timestamp(e.second_round or e.date)
        periods.append((pd.Timestamp(e.date) - pd.DateOffset(months=cfg.restriction_months), end))
    return periods


def days_to_next_restriction(signed: pd.Series, cfg: RulesConfig) -> pd.Series:
    """Days between signature and the start of the next restricted period
    (legacy ``date_near_change`` returned the same idea in years)."""
    starts = np.array(sorted(s for s, _ in restriction_periods(cfg)), dtype="datetime64[ns]")
    vals = signed.to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(starts, vals, side="left")
    out = np.full(len(vals), np.nan)
    ok = (idx < len(starts)) & ~np.isnat(vals)
    out[ok] = (starts[idx[ok]] - vals[ok]).astype("timedelta64[D]").astype(float)
    return pd.Series(out, index=signed.index)


def _in_periods(signed: pd.Series, periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    out = pd.Series(False, index=signed.index)
    for start, end in periods:
        out |= signed.between(start, end)
    return out


def ley_garantias_flags(df: pd.DataFrame, settings: Settings) -> dict[str, pd.Series | None]:
    cfg, s = settings.rules, settings.schema_
    signed = df[C.SIGNED_AT]
    names = (
        "pre_ley_garantias_rush",
        "direct_during_ley_garantias",
        "interadministrative_during_ley_garantias",
    )
    modality = _col(df, s.modality)
    if signed.isna().all() or modality is None:
        return dict.fromkeys(names)

    # Where agreements (convenios) can be recognised: justification, type, text.
    kind_text = pd.Series("", index=df.index)
    for col in (s.modality_justification, s.contract_type):
        c = _col(df, col)
        if c is not None:
            kind_text = kind_text + " " + c.fillna("").astype(str)
    kind_text = kind_text + " " + df[C.TEXT].fillna("")

    direct = _matches_any(modality, cfg.direct_modality_patterns)
    interadmin = _matches_any(kind_text, cfg.interadministrative_patterns)
    association = _matches_any(kind_text, cfg.association_patterns)
    order = _col(df, s.entity_order)
    territorial = (
        _matches_any(order, cfg.territorial_order_values)
        if order is not None
        else pd.Series(True, index=df.index)
    )

    presidential = restriction_periods(cfg, {"presidential"})
    any_election = restriction_periods(cfg)
    during_presidential = _in_periods(signed, presidential)
    during_any = _in_periods(signed, any_election)

    # Legacy note: "justo antes de la vigencia disparan" -- contracting spikes
    # right before the restriction starts (direct contracts and convenios).
    window = pd.Timedelta(days=cfg.pre_restriction_window_days)
    before = pd.Series(False, index=df.index)
    for start, _ in any_election:
        before |= (signed >= start - window) & (signed < start)
    rush = (direct | interadmin | association) & before & ~during_any

    return {
        "pre_ley_garantias_rush": rush,
        "direct_during_ley_garantias": direct & during_presidential,
        "interadministrative_during_ley_garantias": interadmin & territorial & during_any,
    }


# --------------------------------------------------------------------------- #
# Parties & concentration
# --------------------------------------------------------------------------- #
def concurrent_direct_services(df: pd.DataFrame, settings: Settings) -> pd.Series | None:
    """Natural person with >= N direct service contracts in the same entity & year."""
    cfg, s = settings.rules, settings.schema_
    modality, ctype = _col(df, s.modality), _col(df, s.contract_type)
    if modality is None or ctype is None or df[C.SUPPLIER].isna().all():
        return None
    mask = (
        df[C.NATURAL_PERSON].astype(bool)
        & df[C.SUPPLIER].notna()
        & _matches_any(modality, cfg.direct_modality_patterns)
        & _matches_any(ctype, cfg.services_type_patterns)
    )
    out = pd.Series(False, index=df.index)
    if mask.any():
        sub = df[mask]
        keys = [sub[C.ENTITY], sub[C.SUPPLIER], sub[C.SIGNED_AT].dt.year]
        n = sub.groupby(keys, dropna=False)[C.ID].transform("size")
        out.loc[n.index] = n >= cfg.concurrent_contracts_threshold
    return out


def spending_shares(df: pd.DataFrame, settings: Settings) -> dict[str, pd.Series | None]:
    cfg = settings.rules
    if df[C.ENTITY].isna().all():
        return {"large_share_of_entity_spending": None, "dominant_supplier": None}
    value = df[C.VALUE].where(~df[C.VALUE_OUT_OF_RANGE])
    year = df[C.SIGNED_AT].dt.year.fillna(-1)
    keys = [df[C.ENTITY], year]
    entity_total = value.groupby(keys).transform("sum")
    entity_n = value.groupby(keys).transform("count")
    enough = entity_n >= 5
    large = enough & (value / entity_total > cfg.entity_spending_share_threshold)

    dominant: pd.Series | None = None
    if not df[C.SUPPLIER].isna().all():
        supplier_total = value.groupby([df[C.ENTITY], year, df[C.SUPPLIER]]).transform("sum")
        n_suppliers = df[C.SUPPLIER].groupby(keys).transform("nunique")
        dominant = (
            enough
            & (n_suppliers >= 2)
            & (supplier_total / entity_total > cfg.supplier_share_threshold)
        )
    return {"large_share_of_entity_spending": large.fillna(False), "dominant_supplier": dominant}


def shared_legal_representative(df: pd.DataFrame, settings: Settings) -> pd.Series | None:
    """Legal representative linked to >= N distinct *companies* (pseudonymised ids)."""
    if df[C.LEGAL_REP].isna().all() or df[C.SUPPLIER].isna().all():
        return None
    companies = df[~df[C.NATURAL_PERSON].astype(bool) & df[C.LEGAL_REP].notna()]
    n_companies = companies.groupby(C.LEGAL_REP)[C.SUPPLIER].nunique()
    shared = n_companies[n_companies >= settings.rules.shared_legal_rep_threshold].index
    return df[C.LEGAL_REP].isin(shared) & ~df[C.NATURAL_PERSON].astype(bool)


def possible_contract_splitting(df: pd.DataFrame, settings: Settings) -> pd.Series | None:
    """>= N contracts, same entity + supplier + UNSPSC class, within W days."""
    cfg = settings.rules
    if df[C.SUPPLIER].isna().all() or df[C.UNSPSC_CLASS].isna().all():
        return None
    out = np.zeros(len(df), dtype=bool)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    window = np.timedelta64(cfg.splitting_window_days, "D")
    keys = [C.ENTITY, C.SUPPLIER, C.UNSPSC_CLASS]
    data = df[[*keys, C.SIGNED_AT]].dropna()
    sizes = data.groupby(keys)[C.SIGNED_AT].transform("size")
    data = data[sizes >= cfg.splitting_min_contracts]  # only groups that can qualify
    for _, g in data.groupby(keys):
        g = g.sort_values(C.SIGNED_AT)
        dates = g[C.SIGNED_AT].to_numpy(dtype="datetime64[ns]")
        rows = pos.loc[g.index].to_numpy()
        ends = np.searchsorted(dates, dates + window, side="right")
        for i, j in enumerate(ends):
            if j - i >= cfg.splitting_min_contracts:
                out[rows[i:j]] = True
    return pd.Series(out, index=df.index)


# --------------------------------------------------------------------------- #
# Contract content
# --------------------------------------------------------------------------- #
def additions(df: pd.DataFrame, settings: Settings) -> dict[str, pd.Series | None]:
    cfg, s = settings.rules, settings.schema_
    mentions = df[C.TEXT].fillna("").str.contains(cfg.addition_regex, case=False, regex=True)
    days_added = _col(df, s.days_added)
    large = None
    if days_added is not None:
        added = pd.to_numeric(days_added, errors="coerce").fillna(0)
        ratio = added / df[C.DURATION_DAYS].where(df[C.DURATION_DAYS] > 0)
        large = (ratio > cfg.days_added_ratio_threshold).fillna(False)
    return {"mentions_addition": mentions, "large_days_added": large}


def atypical_duration(df: pd.DataFrame, settings: Settings) -> pd.Series | None:
    """Robust z-score of log(duration) within the contract type."""
    ctype = _col(df, settings.schema_.contract_type)
    if ctype is None or df[C.DURATION_DAYS].isna().all():
        return None
    logd = np.log1p(df[C.DURATION_DAYS])
    med = logd.groupby(ctype).transform("median")
    mad = (logd - med).abs().groupby(ctype).transform("median") * 1.4826
    z = (logd - med) / mad.replace(0, np.nan)
    return (z.abs() > settings.rules.duration_robust_z_threshold).fillna(False)


def single_bidder(df: pd.DataFrame, settings: Settings) -> pd.Series | None:
    s = settings.schema_
    bidders, modality = _col(df, s.n_bidders), _col(df, s.modality)
    if bidders is None:
        return None
    competitive = (
        ~_matches_any(modality, settings.rules.direct_modality_patterns)
        if modality is not None
        else True
    )
    return (pd.to_numeric(bidders, errors="coerce") == 1) & competitive


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_RULES: list[Callable[[pd.DataFrame, Settings], pd.Series | dict | None]] = [
    ley_garantias_flags,
    concurrent_direct_services,
    spending_shares,
    shared_legal_representative,
    possible_contract_splitting,
    additions,
    atypical_duration,
    single_bidder,
]


def evaluate_rules(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Evaluate every rule; returns ``sig_<rule>`` boolean columns.

    Rules that cannot be evaluated (missing columns) are omitted and logged.
    """
    out = pd.DataFrame(index=df.index)
    out["days_to_next_restriction"] = days_to_next_restriction(df[C.SIGNED_AT], settings.rules)
    out[SIGNAL_PREFIX + "data_quality_suspect"] = df[C.VALUE].isna() | df[C.VALUE_OUT_OF_RANGE]
    if not settings.rules.enabled:
        return out
    for rule in _RULES:
        res = rule(df, settings)
        items = res.items() if isinstance(res, dict) else [(rule.__name__, res)]
        for name, flags in items:
            if flags is None:
                logger.info("Rule '%s' skipped (required columns missing)", name)
                continue
            out[SIGNAL_PREFIX + name] = flags.reindex(df.index).fillna(False).astype(bool)
    return out


def alert_index(flags: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Weighted combination of alert signals into one index in [0, 1]
    (the legacy ``ponderador``), plus a human-readable explanation."""
    names = [c[len(SIGNAL_PREFIX) :] for c in flags.columns if c.startswith(SIGNAL_PREFIX)]
    w = pd.Series({n: float(weights.get(n, 1.0)) for n in names})
    if w.sum() <= 0:
        return pd.DataFrame({"alert_index": 0.0, "n_signals": 0, "signals": ""}, index=flags.index)
    mat = flags[[SIGNAL_PREFIX + n for n in names]].astype(float).to_numpy()
    score = mat @ w.to_numpy() / w.sum()
    active = [
        "; ".join(n for n, on in zip(names, row, strict=True) if on and w[n] > 0)
        for row in mat.astype(bool)
    ]
    counted = mat[:, (w > 0).to_numpy()].sum(axis=1).astype(int)
    return pd.DataFrame(
        {"alert_index": score, "n_signals": counted, "signals": active}, index=flags.index
    )
