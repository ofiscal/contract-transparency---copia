"""Synthetic SECOP II-like data for tests, demos and CI.

Real SECOP extracts contain personal data and must not be committed or used
in CI, so the test-suite runs on this generator. The data mimics the *shape*
of "SECOP II - Contratos Electrónicos" (same column names, category values
recovered from the legacy models, realistic price structure) and plants known
anomalies so that detection quality can be measured:

* ``synthetic_is_anomaly``   -- contract value inflated x6 - x30 (ground truth);
* contract splitting, concurrent direct contracts and shared legal
  representatives patterns for the rule-based alert signals.

All names and identifiers are fictitious ("Contratista Ficticio 0001",
"Empresa Demo 001 S.A.S.", IDs starting with 999...).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from contract_transparency.data.person_type import JURIDICA, nit_check_digit, number_rule

_PLACES = [
    ("Distrito Capital de Bogotá", "Bogotá"),
    ("Antioquia", "Medellín"),
    ("Valle del Cauca", "Cali"),
    ("Atlántico", "Barranquilla"),
    ("Bolívar", "Cartagena"),
    ("Santander", "Bucaramanga"),
    ("Nariño", "Pasto"),
    ("Cauca", "Popayán"),
    ("Chocó", "Quibdó"),
    ("La Guajira", "Riohacha"),
    ("Boyacá", "Tunja"),
    ("Meta", "Villavicencio"),
]
# Departments with higher prices (remote logistics) -- a learnable effect.
_PLACE_EFFECT = {"Chocó": 0.25, "La Guajira": 0.2, "Distrito Capital de Bogotá": 0.15}


@dataclass(frozen=True)
class _Product:
    contract_type: str
    unspsc: str
    # (description template, log-price multiplier): the text carries price
    # information that is NOT in the categorical columns (level, scope).
    variants: tuple[tuple[str, float], ...]
    log_base: float  # log(COP) for a 1-month contract at the reference quantity
    months: tuple[int, int]
    modalities: tuple[tuple[str, float], ...]
    noise: float = 0.3  # heteroscedastic: public works vary more than service fees
    # (min, max, reference, elasticity) of a quantity written in the text as {q}
    quantity: tuple[float, float, float, float] | None = None
    duration_elasticity: float = 1.0


_PRODUCTS: list[tuple[_Product, float]] = [
    (
        _Product(
            "Prestación de servicios",
            "V1.80111600",
            (
                (
                    "Prestación de servicios de apoyo a la gestión como auxiliar administrativo en la {area}",
                    0.0,
                ),
                (
                    "Prestar servicios de apoyo a la gestión como técnico administrativo en la {area}",
                    0.35,
                ),
                (
                    "Prestación de servicios como tecnólogo en sistemas de información para la {area}",
                    0.55,
                ),
            ),
            np.log(2_400_000),
            (2, 11),
            (("Contratación directa", 0.9), ("Mínima cuantía", 0.1)),
            noise=0.15,
        ),
        0.40,
    ),
    (
        _Product(
            "Prestación de servicios",
            "V1.80121700",
            (
                ("Prestación de servicios profesionales como abogado para la {area}", 0.0),
                (
                    "Prestación de servicios profesionales como abogado especializado en contratación estatal para la {area}",
                    0.45,
                ),
                ("Prestar servicios profesionales como asesor jurídico experto en la {area}", 0.8),
            ),
            np.log(6_000_000),
            (3, 11),
            (("Contratación directa", 0.95), ("Mínima cuantía", 0.05)),
            noise=0.25,
        ),
        0.18,
    ),
    (
        _Product(
            "Suministros",
            "V1.15101506",
            (
                ("Suministro de {q} galones de combustible para los vehículos de la {area}", 0.0),
                ("Suministro de {q} galones de gasolina y ACPM para el parque automotor", 0.0),
            ),
            np.log(30_000_000),
            (3, 12),
            (("Mínima cuantía", 0.5), ("Selección abreviada subasta inversa", 0.5)),
            noise=0.3,
            quantity=(300, 20_000, 2_500, 0.95),
            duration_elasticity=0.1,
        ),
        0.08,
    ),
    (
        _Product(
            "Compraventa",
            "V1.43211500",
            (
                ("Adquisición de {q} computadores portátiles para la {area}", 0.0),
                ("Compra de {q} equipos de cómputo de escritorio para la {area}", -0.15),
            ),
            np.log(60_000_000),
            (1, 2),
            (("Selección abreviada subasta inversa", 0.6), ("Mínima cuantía", 0.4)),
            noise=0.25,
            quantity=(3, 300, 17, 0.95),
            duration_elasticity=0.0,
        ),
        0.08,
    ),
    (
        _Product(
            "Obra",
            "V1.72141100",
            (
                ("Construcción de {q} km de placa huella en la vereda {vereda} del municipio", 0.0),
                (
                    "Mantenimiento rutinario de {q} km de vía terciaria hacia la vereda {vereda}",
                    -1.2,
                ),
            ),
            np.log(400_000_000),
            (2, 10),
            (
                ("Licitación pública Obra Publica", 0.6),
                ("Selección Abreviada de Menor Cuantía", 0.4),
            ),
            noise=0.45,
            quantity=(0.3, 8.0, 2.0, 0.9),
            duration_elasticity=0.2,
        ),
        0.08,
    ),
    (
        _Product(
            "Consultoría",
            "V1.81101500",
            (
                ("Consultoría para los estudios y diseños de la infraestructura de la {area}", 0.0),
                ("Consultoría para la actualización del diagnóstico técnico de la {area}", -0.7),
            ),
            np.log(45_000_000),
            (2, 8),
            (("Concurso de méritos abierto", 1.0),),
            noise=0.4,
        ),
        0.06,
    ),
    (
        _Product(
            "Interventoría",
            "V1.81102200",
            (
                (
                    "Interventoría técnica, administrativa y financiera a la construcción de {q} km de placa huella en la vereda {vereda}",
                    0.0,
                ),
            ),
            np.log(40_000_000),
            (3, 10),
            (("Concurso de méritos abierto", 1.0),),
            noise=0.35,
            quantity=(0.3, 8.0, 2.0, 0.8),
            duration_elasticity=0.3,
        ),
        0.04,
    ),
    (
        _Product(
            "Arrendamiento de inmuebles",
            "V1.80131500",
            (("Arrendamiento de inmueble de {q} m2 para el funcionamiento de la {area}", 0.0),),
            np.log(4_000_000),
            (6, 12),
            (("Contratación directa", 1.0),),
            noise=0.25,
            quantity=(60, 1_500, 250, 0.8),
        ),
        0.08,
    ),
]

ANNUAL_INFLATION = 0.09  # Colombia's CPI was ~5-13 % a year in 2021-2023

# Share of contracts awarded to natural persons, by contract type.
_NATURAL_SHARE = {
    "Prestación de servicios": 0.85,
    "Suministros": 0.15,
    "Compraventa": 0.15,
    "Obra": 0.05,
    "Consultoría": 0.2,
    "Interventoría": 0.1,
    "Arrendamiento de inmuebles": 0.6,
}
# Log-price gap natural minus legal entity for the SAME object (split +-half):
# firms selling services charge overhead; individual merchants/builders work at
# a smaller scale; rental prices are similar.
_PERSON_EFFECT = {
    "Prestación de servicios": -0.4,
    "Suministros": -0.5,
    "Compraventa": -0.4,
    "Obra": -0.6,
    "Consultoría": -0.5,
    "Interventoría": -0.4,
    "Arrendamiento de inmuebles": -0.1,
}
_LEGAL_FORMS = ["S.A.S.", "S.A.S.", "S.A.S.", "Ltda.", "S.A."]


def _person_registry(rng: np.random.Generator, n: int) -> list[str]:
    """Random cédula numbers in the two real formats: old series (8 digits) and
    NUIP (10 digits starting with 1). Random, so they could coincide with real
    IDs: they are pseudonymised on load and never displayed."""
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            out.append(str(int(rng.integers(10_000_000, 99_999_999))))
        else:
            out.append(str(int(rng.integers(1_000_000_000, 1_130_000_000))))
    return out


def _company_registry(rng: np.random.Generator, n: int) -> list[tuple[str, str]]:
    """(NIT without check digit, fictitious name) for legal entities."""
    out = []
    for i in range(n):
        nit = str(
            int(
                rng.choice(
                    [rng.integers(800_000_000, 900_000_000), rng.integers(900_000_000, 902_000_000)]
                )
            )
        )
        r = rng.random()
        if r < 0.04:
            name = f"Consorcio Demo {i:03d}"
        elif r < 0.06:
            name = f"Unión Temporal Demo {i:03d}"
        elif r < 0.16:
            name = f"Comercializadora Demo {i:03d}"  # no legal form: decided by the number
        elif r < 0.20:
            name = f"Fundación Demo {i:03d}"
        else:
            name = f"Empresa Demo {i:03d} {_LEGAL_FORMS[int(rng.integers(0, len(_LEGAL_FORMS)))]}"
        out.append((nit, name))
    return out


_AREAS = [
    "Secretaría de Gobierno",
    "Secretaría de Salud",
    "Secretaría de Educación",
    "Oficina Jurídica",
    "Secretaría de Hacienda",
    "Secretaría de Infraestructura",
    "Oficina de Planeación",
]
_VEREDAS = ["El Carmen", "La Esperanza", "San José", "Buenos Aires", "El Progreso", "La Palma"]
_ESTADOS = [
    ("En ejecución", 0.45),
    ("terminado", 0.3),
    ("Modificado", 0.1),
    ("Cerrado", 0.1),
    ("Activo", 0.05),
]
_JUSTIFICACION = {
    "Contratación directa": "ServiciosProfesionales",
    "Mínima cuantía": "Presupuesto menor al 10% de la Menor Cuantía",
}


def _choice(
    rng: np.random.Generator, options: list[tuple[str, float]] | tuple, size: int
) -> np.ndarray:
    labels = [o[0] for o in options]
    p = np.array([o[1] for o in options], dtype=float)
    return rng.choice(labels, size=size, p=p / p.sum())


def generate_contracts(
    n: int = 20_000,
    seed: int = 24111996,
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    anomaly_rate: float = 0.02,
    anomaly_factor: tuple[float, float] = (1.8, 8.0),
) -> pd.DataFrame:
    """Generate ``n`` fictitious contracts with the SECOP II column names.

    Planted anomalies multiply the value by a log-uniform factor in
    ``anomaly_factor`` (default x1.8 - x8, realistic and not trivial to detect).
    """
    rng = np.random.default_rng(seed)

    # --- entities ---------------------------------------------------------------
    n_entities = 40
    ent_place = rng.integers(0, len(_PLACES), n_entities)
    ent_nacional = rng.random(n_entities) < 0.3
    entities = pd.DataFrame(
        {
            "Nombre Entidad": [
                (f"Ministerio Demo {i:02d}" if nac else f"Alcaldía Demo de {_PLACES[p][1]} {i:02d}")
                for i, (p, nac) in enumerate(zip(ent_place, ent_nacional, strict=True))
            ],
            "Nit Entidad": [f"8999{i:05d}" for i in range(n_entities)],
            "Departamento": [_PLACES[p][0] for p in ent_place],
            "Ciudad": [_PLACES[p][1] for p in ent_place],
            "Orden": np.where(ent_nacional, "Nacional", "Territorial"),
            "Sector": rng.choice(
                [
                    "Salud y Protección Social",
                    "Educación Nacional",
                    "Transporte",
                    "Hacienda y Crédito Público",
                    "interior",
                ],
                n_entities,
            ),
            "Rama": "Ejecutivo",
            "Entidad Centralizada": rng.choice(
                ["Centralizada", "Descentralizada"], n_entities, p=[0.7, 0.3]
            ),
        }
    )
    ent_idx = rng.integers(0, n_entities, n)
    df = entities.iloc[ent_idx].reset_index(drop=True)

    # --- products / modality ----------------------------------------------------
    prod_idx = rng.choice(len(_PRODUCTS), size=n, p=[w for _, w in _PRODUCTS])
    products = [_PRODUCTS[i][0] for i in prod_idx]
    df["Tipo de Contrato"] = [p.contract_type for p in products]
    df["Codigo de Categoria Principal"] = [p.unspsc for p in products]
    df["Modalidad de Contratacion"] = [str(_choice(rng, p.modalities, 1)[0]) for p in products]
    df["Justificacion Modalidad de Contratacion"] = [
        _JUSTIFICACION.get(m, "No Especificado") for m in df["Modalidad de Contratacion"]
    ]
    df["Estado Contrato"] = _choice(rng, _ESTADOS, n)

    # --- dates & duration --------------------------------------------------------
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    span = (end_ts - start_ts).days
    signed = start_ts + pd.to_timedelta(rng.integers(0, span, n), unit="D")
    months = np.array([rng.integers(p.months[0], p.months[1] + 1) for p in products])
    use_days = rng.random(n) < 0.2
    duration_txt = np.where(
        use_days, [f"{m * 30} Dia(s)" for m in months], [f"{m} Mes(es)" for m in months]
    )
    begin = signed + pd.to_timedelta(rng.integers(0, 10, n), unit="D")
    finish = begin + pd.to_timedelta(months * 30, unit="D")
    df["Fecha de Firma"] = signed.strftime("%m/%d/%Y")
    df["Fecha de Inicio del Contrato"] = begin.strftime("%m/%d/%Y")
    df["Fecha de Fin del Contrato"] = finish.strftime("%m/%d/%Y")
    df["Duración del contrato"] = duration_txt
    df["Dias adicionados"] = np.where(rng.random(n) < 0.05, rng.integers(15, 200, n), 0).astype(str)

    # --- suppliers -------------------------------------------------------------
    # Both person types appear in every contract type, with different pricing.
    natural = rng.random(n) < np.array([_NATURAL_SHARE[p.contract_type] for p in products])
    person_ids = rng.integers(0, 1500, n)
    company_ids = rng.integers(0, 200, n)
    people = _person_registry(rng, 1500)
    companies = _company_registry(rng, 200)
    cedula = np.array([people[i] for i in person_ids], dtype=object)
    nit = np.array([companies[i][0] for i in company_ids], dtype=object)
    company_name = np.array([companies[i][1] for i in company_ids], dtype=object)
    # How the number is written: company NITs sometimes with the check digit appended.
    with_dv = rng.random(n) < 0.2
    nit_written = np.where(with_dv, [f"{x}{nit_check_digit(x)}" for x in nit], nit)
    # Document type as published: mostly consistent, sometimes "NIT" for merchants
    # (natural persons invoicing with their cédula) or simply missing.
    u = rng.random(n)
    doc_type_nat = np.where(
        u < 0.90, "Cédula de Ciudadanía", np.where(u < 0.97, "NIT", "No Definido")
    )
    doc_type_jur = np.where(u < 0.95, "NIT", "No Definido")
    df["TipoDocProveedor"] = np.where(natural, doc_type_nat, doc_type_jur)
    df["Documento Proveedor"] = np.where(natural, cedula, nit_written)
    df["Proveedor Adjudicado"] = np.where(
        natural, [f"Contratista Ficticio {i:04d}" for i in person_ids], company_name
    )
    df["Es Grupo"] = np.where(
        ~natural
        & pd.Series(company_name).str.startswith(("Consorcio", "Unión Temporal")).to_numpy(),
        "Si",
        "No",
    )
    df["synthetic_person_type"] = np.where(natural, "natural", "juridica")  # ground truth
    # Legal representative: companies 0-9 share the same person (planted pattern).
    rep = np.where(company_ids < 10, 5_000, company_ids)
    rep_ids = {i: people[(i * 7919) % len(people)] for i in np.unique(rep)}
    df["Identificación Representante Legal"] = np.where(
        natural, df["Documento Proveedor"], [rep_ids[i] for i in rep]
    )
    df["Nombre Representante Legal"] = np.where(
        natural, df["Proveedor Adjudicado"], [f"Representante Ficticio {i:04d}" for i in rep]
    )
    # Personal data that must be dropped by the privacy layer.
    df["Género Representante Legal"] = rng.choice(["Femenino", "Masculino"], n)
    df["Nombre del banco"] = "Banco Demo"
    df["Número de cuenta"] = [f"000-DEMO-{i:06d}" for i in range(n)]
    df["Nombre supervisor"] = [f"Supervisor Ficticio {i:03d}" for i in rng.integers(0, 80, n)]
    df["Número de documento supervisor"] = [f"9996{i:06d}" for i in rng.integers(0, 80, n)]

    # --- descriptions (carry level / scope / quantity information) -----------------
    descr, text_mult, qty_effect = [], np.zeros(n), np.zeros(n)
    for i, p in enumerate(products):
        template, mult = p.variants[rng.integers(0, len(p.variants))]
        text_mult[i] = mult
        q_txt = ""
        if p.quantity:
            qmin, qmax, qref, elasticity = p.quantity
            q = float(np.exp(rng.uniform(np.log(qmin), np.log(qmax))))
            q = round(q, 1) if qmax < 20 else int(round(q))
            qty_effect[i] = elasticity * np.log(q / qref)
            q_txt = (
                f"{q:.1f}".replace(".", ",") if isinstance(q, float) else f"{q:,}".replace(",", ".")
            )
        descr.append(
            template.format(
                area=_AREAS[rng.integers(0, len(_AREAS))],
                vereda=_VEREDAS[rng.integers(0, len(_VEREDAS))],
                q=q_txt,
            )
        )
    descr = np.array(descr, dtype=object)
    # Personal data and amounts typed inside free text (must be redacted).
    with_id = natural & (rng.random(n) < 0.05)
    descr[with_id] = [
        f"{d} contratista identificado con C.C. 1.0{rng.integers(10, 99)}.{rng.integers(100, 999)}.{rng.integers(100, 999)}"
        for d in descr[with_id]
    ]
    df["Descripcion del Proceso"] = descr
    df["Objeto del Contrato"] = np.where(
        rng.random(n) < 0.7, descr, df["Tipo de Contrato"].str.upper()
    )

    # --- value -------------------------------------------------------------------
    years = (signed - start_ts).days.to_numpy() / 365.25
    log_value = (
        np.array([p.log_base for p in products])
        + text_mult
        + qty_effect
        + np.log(months) * np.array([p.duration_elasticity for p in products])
        + df["Departamento"].map(_PLACE_EFFECT).fillna(0.0).to_numpy()
        + np.where(df["Orden"] == "Nacional", 0.2, 0.0)
        + years * np.log1p(ANNUAL_INFLATION)
        + np.array([_PERSON_EFFECT[p.contract_type] for p in products])
        * np.where(natural, 1.0, -1.0)
        / 2
        + rng.normal(0, 1, n) * np.array([p.noise for p in products])
    )
    value = np.exp(log_value)
    anomaly = rng.random(n) < anomaly_rate
    lo, hi = np.log(anomaly_factor[0]), np.log(anomaly_factor[1])
    factor = np.ones(n)
    factor[anomaly] = np.exp(rng.uniform(lo, hi, anomaly.sum()))
    value *= factor
    df["Valor del Contrato"] = np.round(value, -3).astype(np.int64).astype(str)
    df["synthetic_is_anomaly"] = anomaly
    df["synthetic_anomaly_factor"] = np.round(factor, 3)  # analysis only, never a feature

    df["ID Contrato"] = [f"CO1.PCCNTR.DEMO{i:07d}" for i in range(n)]
    df["URLProceso"] = [
        f"https://community.secop.gov.co/Public/Tendering/OpportunityDetail/Index?noticeUID=CO1.NTC.DEMO{i:07d}"
        for i in range(n)
    ]

    df = _plant_patterns(df, rng)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _plant_patterns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Plant rule-based patterns on a few rows (splitting / concurrent contracts)."""
    df = df.copy()
    # Contract splitting: 4 small purchases, same entity + supplier + class, 10 days.
    rows = df.index[df["Tipo de Contrato"] == "Compraventa"][:4]
    if len(rows) == 4:
        base = pd.Timestamp("2022-08-01")
        df.loc[rows, "Nit Entidad"] = "899900000"
        df.loc[rows, "Nombre Entidad"] = "Alcaldía Demo Fraccionamiento"
        df.loc[rows, "Documento Proveedor"] = "901999999"
        df.loc[rows, "TipoDocProveedor"] = "NIT"
        df.loc[rows, "synthetic_person_type"] = "juridica"
        df.loc[rows, "Proveedor Adjudicado"] = "Empresa Demo Fraccionada S.A.S."
        df.loc[rows, "Fecha de Firma"] = [
            (base + pd.Timedelta(days=3 * i)).strftime("%m/%d/%Y") for i in range(4)
        ]
    # Concurrent direct contracts for one natural person in one entity/year.
    rows = df.index[df["Modalidad de Contratacion"] == "Contratación directa"][:5]
    if len(rows) == 5:
        df.loc[rows, "Nit Entidad"] = "899900001"
        df.loc[rows, "Documento Proveedor"] = "1099999999"
        df.loc[rows, "Proveedor Adjudicado"] = "Contratista Ficticio Concurrente"
        df.loc[rows, "synthetic_person_type"] = "natural"
        df.loc[rows, "Identificación Representante Legal"] = "1099999999"  # natural person: self
        df.loc[rows, "TipoDocProveedor"] = "Cédula de Ciudadanía"
        df.loc[rows, "Tipo de Contrato"] = "Prestación de servicios"
        df.loc[rows, "Fecha de Firma"] = [f"0{m}/15/2023" for m in range(1, 6)]
    return df


def generate_fiscal_bulletin(contracts: pd.DataFrame, n: int = 40, seed: int = 7) -> pd.DataFrame:
    """Fictitious 'Boletín de Responsables Fiscales'-like table for demos/tests.

    Mimics an official Excel publication: two title rows above the header,
    NITs written with verification digit, dates as dd/mm/yyyy. About 60 % of
    the listed suppliers come from contracts with a planted price anomaly, so
    the weak-label evaluation has a (synthetic) signal to find.
    The real bulletin's layout may differ; the importer auto-detects columns.
    """
    rng = np.random.default_rng(seed)
    anomalous = contracts.loc[
        contracts["synthetic_is_anomaly"].astype(bool), "Documento Proveedor"
    ].unique()
    others = contracts["Documento Proveedor"].unique()
    n_anom = min(len(anomalous), int(n * 0.6))
    chosen = list(rng.choice(anomalous, n_anom, replace=False)) if n_anom else []
    pool = np.setdiff1d(others, chosen)
    chosen += list(rng.choice(pool, n - len(chosen), replace=False))

    docs, types, names, dates = [], [], [], []
    for doc in chosen:
        digits = "".join(ch for ch in doc if ch.isdigit())
        is_nit = number_rule(doc)[0] == JURIDICA
        if is_nit and len(digits) == 10:  # written with DV: keep the 9-digit base
            digits = digits[:9]
        docs.append(
            f"{int(digits):,}".replace(",", ".") + f"-{nit_check_digit(digits)}" if is_nit else doc
        )
        types.append("NIT" if is_nit else "CC")
        names.append("Empresa Sancionada Ficticia S.A.S." if is_nit else "Persona Ficticia Listada")
        dates.append(
            (pd.Timestamp("2020-06-01") + pd.Timedelta(days=int(rng.integers(0, 900)))).strftime(
                "%d/%m/%Y"
            )
        )
    body = pd.DataFrame(
        {
            "No.": range(1, n + 1),
            "TIPO DE DOCUMENTO": types,
            "NÚMERO DE DOCUMENTO": docs,
            "NOMBRE O RAZÓN SOCIAL": names,
            "ENTIDAD AFECTADA": "Entidad Demo",
            "FECHA DE INCLUSIÓN": dates,
        }
    )
    title = pd.DataFrame(
        [
            ["BOLETÍN DE RESPONSABLES FISCALES - DATOS SINTÉTICOS (NO REALES)"] + [None] * 5,
            ["Edición de demostración generada por contract-transparency"] + [None] * 5,
            list(body.columns),
        ],
        columns=body.columns,
    )
    return pd.concat([title, body.astype(object)], ignore_index=True)
