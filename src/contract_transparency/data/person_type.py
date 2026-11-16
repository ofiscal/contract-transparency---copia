"""Classify suppliers as natural persons or legal entities (persona natural / jurídica).

Uses only information published in SECOP: the document type, the document
number and the supplier name. Rules are applied in order of reliability, and the
rule that decided is recorded (``ct_person_type_rule``) so the classification
can be audited and its agreement with the published document type measured.

Colombian numbering conventions used:

* **NIT of legal entities**: 9 digits, starting with 8 or 9 (800.000.000 -
  999.999.999), optionally followed by the DIAN check digit (DV).
* **Cédula de ciudadanía**: up to 8 digits (older series) or 10 digits starting
  with 1 (NUIP, issued since the 2000s).
* **NIT of a natural person** (e.g. a merchant): their cédula + check digit.
* **DIAN check digit**: modulo 11 with weights 3, 7, 13, 17, 19, 23, 29, 37, 41,
  43, 47, 53, 59, 67, 71 applied from the rightmost digit (verified against
  published NITs in ``tests/test_person_type.py``).

These are conventions, not guarantees. Ambiguous cases (e.g. a 9-digit number
starting with 8 could be an old cédula in the 80-million range + DV) are
resolved with the document type and name, or left as ``desconocido``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NATURAL = "natural"
JURIDICA = "juridica"
UNKNOWN = "desconocido"
LABELS = {NATURAL: "Persona natural", JURIDICA: "Persona jurídica", UNKNOWN: "Sin determinar"}

_DV_WEIGHTS = (3, 7, 13, 17, 19, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71)

NATURAL_DOC_TYPES = {
    "cedula de ciudadania",
    "cc",
    "cedula de extranjeria",
    "ce",
    "pasaporte",
    "tarjeta de identidad",
    "ti",
    "permiso por proteccion temporal",
    "ppt",
    "permiso especial de permanencia",
    "pep",
}
NIT_DOC_TYPES = {"nit", "numero de identificacion tributaria"}

# Legal forms and collective suppliers in the supplier name.
_LEGAL_FORM = re.compile(
    r"(?i)(\bs\.?\s?a\.?\s?s\b\.?|\bsas\b|\bs\.?\s?a\b\.?|\bltda\b|\blimitada\b|\be\.?\s?u\b\.?|"
    r"\bs\.?\s?c\.?\s?a\b|\be\.?\s?s\.?\s?p\b|\be\.?\s?s\.?\s?e\b|\bfundaci[oó]n\b|\basociaci[oó]n\b|"
    r"\bcorporaci[oó]n\b|\bcooperativa\b|\buniversidad\b|\binstituto\b|\bsociedad\b|\bcompa[nñ][ií]a\b|"
    r"\bempresa\b|\bconsorcio\b|\buni[oó]n\s+temporal\b|\bcaja de compensaci[oó]n\b|\bbanco\b|"
    r"\bfederaci[oó]n\b|\bcolegio\b|\bhospital\b|\bcl[ií]nica\b)"
)
_GROUP_VALUES = {"si", "sí", "s", "yes", "true", "1"}


def nit_check_digit(base: str | int) -> int:
    """DIAN check digit (dígito de verificación) of a NIT / cédula number."""
    digits = [int(c) for c in str(base)][::-1]
    if len(digits) > len(_DV_WEIGHTS):
        raise ValueError("number too long for a NIT")
    r = sum(d * w for d, w in zip(digits, _DV_WEIGHTS, strict=False)) % 11
    return r if r in (0, 1) else 11 - r


def _norm(text: object) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def number_rule(raw: object) -> tuple[str, str]:
    """Classification from the document number alone -> (type, rule)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return UNKNOWN, "no_number"
    text = str(raw).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    explicit_dv = re.search(r"\d\s*-\s*\d\s*$", text) is not None
    digits = re.sub(r"\D", "", text)
    if not digits or re.search(r"[A-Za-z]", text):
        return UNKNOWN, "not_numeric"  # passports and other alphanumeric documents
    n, first = len(digits), digits[0]

    def dv_ok(d: str) -> bool:
        return len(d) >= 2 and nit_check_digit(d[:-1]) == int(d[-1])

    if explicit_dv:
        base = digits[:-1]
        if len(base) == 9 and base[0] in "89":
            return JURIDICA, "nit_range_with_dv"
        if len(base) <= 8 or (len(base) == 10 and base[0] == "1"):
            return NATURAL, "cedula_with_dv"
        return UNKNOWN, "unusual_length"
    if n == 9 and first in "89":
        return JURIDICA, "nit_range"
    if n == 10 and first in "89" and dv_ok(digits):
        return JURIDICA, "nit_range_with_dv"
    if n == 10 and first == "1":
        return NATURAL, "cedula_nuip"
    if n == 11 and first == "1" and dv_ok(digits):
        return NATURAL, "cedula_with_dv"
    if 3 <= n <= 8:
        return NATURAL, "cedula_old_series"
    if n == 9 and dv_ok(digits):
        return NATURAL, "cedula_with_dv"
    return UNKNOWN, "unusual_length"


def _looks_like_cedula_with_dv(raw: object) -> bool:
    digits = re.sub(r"\D", "", str(raw))
    return len(digits) == 9 and nit_check_digit(digits[:-1]) == int(digits[-1])


def classify_person_type(
    doc_type: pd.Series | None,
    number: pd.Series | None,
    name: pd.Series | None = None,
    is_group: pd.Series | None = None,
    natural_doc_types: list[str] | None = None,
) -> pd.DataFrame:
    """Return ``person_type`` (natural | juridica | desconocido) and the deciding ``rule``.

    Order (most reliable first):
    1. a natural-person document type (CC, CE, passport, ...)          -> natural
    2. consortium / temporary union ("Es Grupo" = Sí)                  -> juridica
    3. a legal form in the name (S.A.S., LTDA, fundación, consorcio...) -> juridica
    4. the document number (NIT range, cédula formats, check digit)
    5. document type "NIT" without other evidence                      -> juridica
    """
    index = next(s.index for s in (doc_type, number, name, is_group) if s is not None)
    n = len(index)

    def col(s: pd.Series | None) -> list:
        return [None] * n if s is None else s.tolist()

    natural_docs = NATURAL_DOC_TYPES | {_norm(t) for t in (natural_doc_types or [])}
    out_type, out_rule = [], []
    for dt, num, nm, grp in zip(col(doc_type), col(number), col(name), col(is_group), strict=True):
        dt_n = _norm(dt) if dt is not None and dt == dt else ""
        if dt_n in natural_docs:
            out_type.append(NATURAL), out_rule.append("doc_type")
            continue
        if grp is not None and grp == grp and _norm(grp) in _GROUP_VALUES:
            out_type.append(JURIDICA), out_rule.append("group_flag")
            continue
        if isinstance(nm, str) and _LEGAL_FORM.search(nm):
            out_type.append(JURIDICA), out_rule.append("legal_form_in_name")
            continue
        ptype, rule = number_rule(num)
        if (
            rule == "nit_range"
            and isinstance(nm, str)
            and nm.strip()
            and _looks_like_cedula_with_dv(num)
        ):
            # 9 digits starting with 8/9 could also be an old cédula (70-99 million)
            # + check digit, and the name shows no legal form: do not guess.
            ptype, rule = UNKNOWN, "ambiguous_nit_or_cedula"
        if ptype == UNKNOWN and dt_n in NIT_DOC_TYPES and rule != "ambiguous_nit_or_cedula":
            ptype, rule = JURIDICA, "doc_type_nit"
        out_type.append(ptype)
        out_rule.append(rule)
    return pd.DataFrame({"person_type": out_type, "rule": out_rule}, index=index)


def agreement_with_doc_type(doc_type: pd.Series, number: pd.Series) -> float | None:
    """Share of rows with a natural-person document type whose NUMBER alone is
    also classified as natural (quality check of the number rule on real data)."""
    natural_doc = doc_type.map(
        lambda d: _norm(d) in NATURAL_DOC_TYPES if d == d and d is not None else False
    )
    if not natural_doc.any():
        return None
    by_number = number[natural_doc].map(lambda v: number_rule(v)[0])
    decided = by_number[by_number != UNKNOWN]
    return float((decided == NATURAL).mean()) if len(decided) else None
