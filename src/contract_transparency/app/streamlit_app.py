"""Streamlit dashboard for scored contracts.

Run with ``ct app`` or ``streamlit run src/contract_transparency/app/streamlit_app.py``.
The config is taken from ``$CT_CONFIG`` (default ``configs/secop_ii.yaml``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from contract_transparency.analysis.contractors import DEFAULT_EXCLUDE, top_suppliers
from contract_transparency.config import DEFAULT_CONFIG_PATH, Settings, load_config
from contract_transparency.data.person_type import LABELS as PERSON_LABELS
from contract_transparency.mlops import pipeline
from contract_transparency.scoring.rules import signal_label

PRIVACY_NOTE = (
    "Solo se usa información pública y cada alerta enlaza a su fuente oficial en SECOP. "
    "Los nombres de personas naturales se muestran seudonimizados (Ley 1581 de 2012)."
)
SIGNAL_PREFIX = "sig_"
# Most useful columns first; the rest is available in the download.
TABLE_COLUMNS = [
    "review_priority",
    "ID Contrato",
    "contract_value",
    "predicted_value",
    "alert_index",
    "signals",
    "tail_probability",
    "secop_url",
    "open_data_url",
    "Nombre Entidad",
    "Departamento",
    "Tipo de Contrato",
    "Modalidad de Contratacion",
    "supplier",
    "person_type",
]
FILTER_COLUMNS = ["Departamento", "Modalidad de Contratacion", "Tipo de Contrato"]


# --------------------------------------------------------------------------- helpers
@st.cache_resource
def get_config(path: str) -> Settings:
    return load_config(path)


@st.cache_data(show_spinner="Scoring contracts...")
def score_snapshot(config_path: str, version: str) -> tuple[pd.DataFrame, dict]:
    cfg = get_config(config_path)
    scored, _ = pipeline.score(
        cfg,
        data=pipeline.load_processed(cfg),
        version=version,
        write=False,
        source=pipeline.snapshot_source(cfg),
    )
    return scored, scored.attrs["manifest"]


@st.cache_data(show_spinner="Scoring uploaded file...")
def score_upload(
    config_path: str, version: str, content: bytes, name: str
) -> tuple[pd.DataFrame, dict]:
    cfg = get_config(config_path)
    with tempfile.TemporaryDirectory() as tmp:  # deleted right after scoring
        path = Path(tmp) / name
        path.write_bytes(content)
        scored, _ = pipeline.score(cfg, input_path=path, version=version, write=False)
    return scored, scored.attrs["manifest"]


def fmt_cop(x: float) -> str:
    return "-" if pd.isna(x) else f"$ {x:,.0f}".replace(",", ".")


def fmt_cop_compact(x: float) -> str:
    """Short Colombian notation: billones (10^12) or MM (millones)."""
    if pd.isna(x):
        return "-"
    if abs(x) >= 1e12:
        number = f"{x / 1e12:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"$ {number} billones"
    if abs(x) >= 1e6:
        return f"$ {x / 1e6:,.0f} MM".replace(",", ".")
    return fmt_cop(x)


def signals_es(codes: str) -> str:
    return "; ".join(signal_label(c) for c in str(codes).split("; ") if c) if codes else ""


def hbar(data: pd.Series, label: str, value: str, fmt: str = ",.0f") -> alt.Chart:
    """Sorted horizontal bar chart (keeps long Spanish labels readable)."""
    df = data.rename(value).rename_axis(label).reset_index()
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=value, axis=alt.Axis(format=fmt)),
            y=alt.Y(
                f"{label}:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=320, labelOverlap=False),
            ),
            tooltip=[label, alt.Tooltip(f"{value}:Q", format=fmt)],
        )
        .properties(height=max(120, 28 * len(df)))
    )


def real_vs_expected(scored: pd.DataFrame) -> alt.Chart:
    sample = scored.dropna(subset=["contract_value", "predicted_value"]).head(5000)
    plot = pd.DataFrame(
        {
            "expected (log10 COP)": np.log10(sample["predicted_value"].clip(lower=1)),
            "real (log10 COP)": np.log10(sample["contract_value"].clip(lower=1)),
            "status": np.where(sample["sig_price_deviation"], "desviación de precio", "normal"),
        }
    )
    lo = float(plot.iloc[:, :2].min().min())
    hi = float(plot.iloc[:, :2].max().max())
    scale = alt.Scale(domain=[lo, hi], zero=False)
    points = (
        alt.Chart(plot)
        .mark_circle(size=18, opacity=0.6)
        .encode(
            x=alt.X("expected (log10 COP):Q", scale=scale),
            y=alt.Y("real (log10 COP):Q", scale=scale),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["normal", "desviación de precio"], range=["#9ecae1", "#d62728"]
                ),
            ),
        )
    )
    diagonal = (  # same field names as the points -> shared axis titles
        alt.Chart(pd.DataFrame({"expected (log10 COP)": [lo, hi], "real (log10 COP)": [lo, hi]}))
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="expected (log10 COP):Q", y="real (log10 COP):Q")
    )
    return (points + diagonal).properties(height=380)


# --------------------------------------------------------------------------- page
st.set_page_config(page_title="Contract transparency - SECOP II", page_icon="🔎", layout="wide")

config_path = st.sidebar.text_input("Config", os.environ.get("CT_CONFIG", str(DEFAULT_CONFIG_PATH)))
if not Path(config_path).exists():
    st.error(f"Config not found: {config_path}")
    st.stop()
cfg = get_config(config_path)
registry = pipeline.get_registry(cfg)
versions = registry.versions()
if not versions:
    st.warning(
        "No trained model yet. Run `ct run -c configs/demo.yaml` (demo) or `ct prepare && ct train`."
    )
    st.stop()
champion = registry.champion()
ordered = versions[::-1]
version = st.sidebar.selectbox(
    "Model version",
    ordered,
    index=ordered.index(champion) if champion in ordered else 0,
    format_func=lambda v: f"{v} (champion)" if v == champion else v,
)
source = st.sidebar.radio("Data", ["Processed snapshot", "Upload CSV / Parquet"], key="source")

st.title("🔎 Transparencia en contratación pública - SECOP II")
if cfg.public_data.synthetic:
    st.error(
        "**DATOS SINTÉTICOS - NO REALES.** Entidades, contratistas y enlaces son ficticios (demostración)."
    )
st.info(f"{cfg.public_data.notice_es}\n\n{PRIVACY_NOTE}")

if source == "Upload CSV / Parquet":
    upload = st.sidebar.file_uploader("SECOP II extract", type=["csv", "parquet", "xlsx"])
    if upload is None:
        st.write("Upload a raw SECOP II extract with the columns configured in the YAML file.")
        st.stop()
    scored, manifest = score_upload(config_path, version, upload.getvalue(), Path(upload.name).name)
else:
    if not pipeline.processed_path(cfg).exists():
        st.warning("No processed snapshot found: run `ct prepare` first or upload a file.")
        st.stop()
    scored, manifest = score_snapshot(config_path, version)

# --------------------------------------------------------------------------- filters
flag_cols = [c for c in scored.columns if c.startswith("sig_")]
with st.sidebar.expander("Filters", expanded=True):
    filt = scored
    for col in FILTER_COLUMNS:
        if col in scored:
            chosen = st.multiselect(col, sorted(scored[col].dropna().unique()), key=f"f_{col}")
            if chosen:
                filt = filt[filt[col].isin(chosen)]
    if "person_type" in scored:
        kinds = st.multiselect(
            "Tipo de proveedor",
            sorted(scored["person_type"].dropna().unique()),
            format_func=lambda k: PERSON_LABELS.get(k, k),
            key="f_person_type",
        )
        if kinds:
            filt = filt[filt["person_type"].isin(kinds)]
    min_index = st.slider("Índice de alerta mínimo", 0.0, 1.0, 0.0, 0.05, key="min_index")
    only_alerts = st.checkbox("Solo desviaciones de precio", value=False, key="only_alerts")
    flags_chosen = st.multiselect(
        "Con la señal",
        [c[len(SIGNAL_PREFIX) :] for c in flag_cols],
        format_func=signal_label,
        key="flags",
    )
filt = filt[filt["alert_index"] >= min_index]
if only_alerts:
    filt = filt[filt["sig_price_deviation"]]
for f in flags_chosen:
    filt = filt[filt["sig_" + f]]

tab_alerts, tab_detail, tab_dist, tab_suppliers, tab_models, tab_sources = st.tabs(
    ["Alertas", "Detalle de contrato", "Distribuciones", "Proveedores", "Modelos", "Fuentes"]
)

# --------------------------------------------------------------------------- alerts
with tab_alerts:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contratos", f"{len(filt):,}")
    c2.metric("Desviaciones de precio", f"{int(filt['sig_price_deviation'].sum()):,}")
    excess = filt.loc[filt["sig_price_deviation"], "excess_value"].sum()
    c3.metric("Valor sobre lo esperado (desviaciones)", fmt_cop_compact(excess))
    c4.metric("Modelo", version)
    table = filt[[c for c in TABLE_COLUMNS if c in filt]].head(500).copy()
    table["signals"] = table["signals"].map(signals_es)
    if "person_type" in table:
        table["person_type"] = table["person_type"].map(lambda k: PERSON_LABELS.get(k, k))
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "contract_value": st.column_config.NumberColumn("Valor real (COP)", format="localized"),
            "predicted_value": st.column_config.NumberColumn(
                "Valor esperado (COP)", format="localized"
            ),
            "alert_index": st.column_config.ProgressColumn(
                "Índice de alerta", min_value=0.0, max_value=1.0
            ),
            "review_priority": st.column_config.NumberColumn("Prioridad de revisión"),
            "signals": st.column_config.TextColumn("Señales"),
            "person_type": st.column_config.TextColumn("Tipo de proveedor"),
            "tail_probability": st.column_config.NumberColumn("P(tan caro)", format="%.4f"),
            "secop_url": st.column_config.LinkColumn("SECOP", display_text="Ver en SECOP"),
            "open_data_url": st.column_config.LinkColumn("Datos abiertos", display_text="Registro"),
        },
    )
    d1, d2 = st.columns(2)
    d1.download_button(
        "Descargar resultados filtrados (CSV)",
        filt.to_csv(index=False, sep=cfg.output.csv_sep).encode("utf-8-sig"),
        file_name=f"scored_{version}.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Descargar manifiesto de procedencia (JSON)",
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
        file_name=f"scored_{version}.csv.manifest.json",
        mime="application/json",
    )
    st.caption(manifest["how_to_verify"])

# --------------------------------------------------------------------------- detail
with tab_detail:
    if filt.empty:
        st.write("No contracts with the current filters.")
    else:
        has_id = "ID Contrato" in filt
        ids = filt["ID Contrato"].head(200).tolist() if has_id else filt.index[:200].tolist()
        chosen = st.selectbox("Contrato (primeros 200 por prioridad de revisión)", ids)
        row = filt[filt["ID Contrato"] == chosen].iloc[0] if has_id else filt.loc[chosen]
        a, b, c = st.columns(3)
        a.metric("Valor real", fmt_cop(row["contract_value"]))
        b.metric(
            "Valor esperado",
            fmt_cop(row["predicted_value"]),
            delta=f"{row['pct_error']:+.0%} real vs esperado"
            if pd.notna(row["pct_error"])
            else None,
            delta_color="inverse",
        )
        c.metric("Índice de alerta", f"{row['alert_index']:.2f}")
        st.markdown("**Señales:** " + (signals_es(row["signals"]) or "ninguna"))
        if "person_type" in row.index:
            model_note = ""
            if isinstance(row.get("model_segment"), str):
                seg = row["model_segment"]
                model_note = f" · valor esperado estimado con el modelo de {PERSON_LABELS.get(seg, 'todos los proveedores').lower()}"
            st.markdown(
                f"**Tipo de proveedor:** {PERSON_LABELS.get(row['person_type'], row['person_type'])}{model_note}"
            )
        links = []
        if isinstance(row.get("secop_url"), str):
            links.append(f"[Ver el proceso en SECOP]({row['secop_url']})")
        if isinstance(row.get("open_data_url"), str):
            links.append(f"[Registro en datos abiertos]({row['open_data_url']})")
        st.markdown("**Verificar en la fuente oficial:** " + " · ".join(links))
        st.caption(cfg.public_data.row_notice)
        desc_col = next((col for col in cfg.schema_.text if col in row.index), None)
        if desc_col:
            st.markdown(f"**Descripción:** {row[desc_col]}")
        details = {k: (None if pd.isna(v) else v) for k, v in row.drop(labels=flag_cols).items()}
        st.json(details, expanded=False)

# --------------------------------------------------------------------------- distributions
with tab_dist:
    left, right = st.columns(2)
    with left:
        st.subheader("Real vs. expected value")
        st.altair_chart(real_vs_expected(filt))
        st.caption(
            "Línea punteada: valor real = valor esperado. Los puntos muy por encima tienen un "
            "valor mayor que contratos comparables; puede haber explicaciones legítimas."
        )
    with right:
        st.subheader("Señales (proporción de contratos)")
        freq = filt[flag_cols].mean().rename(signal_label)
        st.altair_chart(hbar(freq[freq > 0], "señal", "proporción", ".1%"))
    if "Departamento" in filt:
        st.subheader("Desviaciones de precio por departamento")
        by_dep = filt.groupby("Departamento")["sig_price_deviation"].sum()
        st.altair_chart(hbar(by_dep[by_dep > 0], "Departamento", "desviaciones"))

# --------------------------------------------------------------------------- suppliers
with tab_suppliers:
    if source == "Processed snapshot":
        keep_uni = st.checkbox("Include public universities", value=False, key="keep_uni")
        ranking = top_suppliers(
            pipeline.load_processed(cfg), n=50, exclude_regex=None if keep_uni else DEFAULT_EXCLUDE
        )
        ranking["share_of_total"] = ranking["share_of_total"] * 100  # fraction -> percentage points
        st.dataframe(
            ranking,
            hide_index=True,
            column_config={
                "total_value": st.column_config.NumberColumn(
                    "total value (COP)", format="localized"
                ),
                "share_of_total": st.column_config.NumberColumn("share", format="%.2f%%"),
            },
        )
        st.caption("Natural persons appear as 'Persona natural p_…' (pseudonymised).")
    else:
        st.write("Available for the processed snapshot.")

# --------------------------------------------------------------------------- models
with tab_models:
    st.subheader("Model registry")
    st.dataframe(registry.summary(), hide_index=True)
    meta = registry.metadata(version)
    m = meta.get("metrics", {})
    st.subheader(f"{version} - metrics")
    regression = {k: v for k, v in m.items() if isinstance(v, dict) and "mae_log" in v}
    st.dataframe(pd.DataFrame(regression).T)
    st.caption(
        "`test` = most recent contracts (temporal split) · `baseline_test` = median price of similar "
        "contracts · `champion_test` = previous champion on the same test set."
    )
    if "detection_test" in m:
        st.markdown("**Detection of known anomalies** (`evaluation.label_column`)")
        st.dataframe(pd.DataFrame([m["detection_test"]]), hide_index=True)
    st.caption(
        f"Train dates {meta['data']['train_dates']} · test dates {meta['data']['test_dates']} · "
        f"git {meta.get('git_commit')} · config {meta.get('config_fingerprint')}"
    )
    st.subheader("Most important features")
    feats = pd.DataFrame(meta.get("top_features", [])).head(15)
    if not feats.empty:
        st.dataframe(
            feats,
            hide_index=True,
            column_config={
                "importance": st.column_config.ProgressColumn(
                    "share of gain",
                    min_value=0.0,
                    max_value=float(feats["importance"].max()),
                    format="%.3f",
                )
            },
        )
    with st.expander("Promotion decision & history"):
        st.json(meta.get("promotion", {}))
        st.json(registry.history()[-20:])

# --------------------------------------------------------------------------- sources
with tab_sources:
    st.subheader("Fuentes públicas utilizadas")
    for src in manifest["sources"]:
        title = src.get("name") or src.get("title")
        url = src.get("portal_url") or src.get("official_url")
        st.markdown(f"**{title}** - {src.get('publisher')}  \n[{url}]({url})")
        details = {
            k: src.get(k)
            for k in (
                "file",
                "file_sha256",
                "rows_read",
                "rows_with_valid_id",
                "imported_at",
                "legal_basis",
            )
            if src.get(k) is not None
        }
        if "retrieval" in src:
            details["retrieval"] = src["retrieval"]
        st.json(details, expanded=False)
    r = manifest["rows"]
    st.markdown(
        f"Contratos publicados: **{r['published']:,}** de {r['input']:,}. "
        f"Excluidos por no tener enlace oficial verificable: **{r['excluded_not_auditable']:,}**."
    )
    st.subheader("Diccionario de columnas")
    st.caption(
        "'source' = copiado de la fuente pública · 'derived' = estimación estadística de este proyecto."
    )
    st.dataframe(
        pd.DataFrame.from_dict(manifest["columns"], orient="index")
        .rename_axis("columna")
        .reset_index(),
        hide_index=True,
    )
