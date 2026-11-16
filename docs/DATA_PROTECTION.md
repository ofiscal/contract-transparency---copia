# Data protection & responsible use

> Not legal advice. Have the organisation's data-protection officer review
> any deployment that publishes results.

## 0. Public-data policy (what the project may use and publish)

1. **Only public, official sources.** The inputs are SECOP II *Contratos
   Electrónicos* from datos.gov.co (API via `ct download`, or a portal export)
   and official public records imported with their official URL (e.g. the
   Contraloría's *Boletín de Responsables Fiscales*, published under Ley 610
   de 2000, art. 60). No private databases, leaked data or scraped personal
   profiles. The legacy code's non-public sources (observatory warehouse
   tables, pre-normalised "VALOR NORM" files) are not used.
2. **Link, don't copy.** Every published result links to its official source:
   `secop_url` (the SECOP process page, accepted only for official domains)
   and `open_data_url` (the exact record in the datos.gov.co API). Rows
   without a verifiable official link are left out of the outputs and
   counted in the manifest.
3. **Statistics, not accusations.** Every output carries the notice below:
   in the CLI, the dashboard, each row (`aviso`), the manifest, and the first
   sheet of Excel files.

   > *Una alerta solo indica que un contrato tiene características poco
   > comunes frente a contratos comparables y que podría ser útil revisarlo. No
   > constituye prueba, acusación, juicio ni señalamiento de irregularidad o
   > responsabilidad de ninguna entidad, contratista o persona, y no reemplaza
   > la verificación en la fuente oficial ni la labor de las autoridades
   > competentes. Los autores no emiten conclusiones sobre los contratos ni
   > asumen responsabilidad por las decisiones que terceros tomen con base en
   > esta información.* (full text: `config.py::NOTICE_ES` / `NOTICE_EN`)

6. **Language policy.** Outputs speak of *alerts for review*, never of
   risk, suspicion, overpricing or guilt:

   | Use | Avoid |
   |---|---|
   | alerta, señal, prioridad de revisión, desviación de precio, valor sobre lo esperado | riesgo, sospechoso, sobrecosto, irregular, culpable, bandera roja |

   Column names follow the same rule (`alert_index`, `review_priority`,
   `signals`, `sig_*`, `price_deviation`, `excess_value`).
   `tests/test_modeling.py` fails if forbidden terms appear in notices,
   output columns or dashboard text.

4. **Traceability.** `<output>.manifest.json` lists each source (publisher,
   URL, retrieval query/time, SHA-256 of the exact file used), the model
   version and code version, and a dictionary that marks every column as
   `source` (copied from the public data), `source (redacted)` or `derived`
   (a statistical estimate made by this project).
5. **Synthetic data is watermarked** ("DATOS SINTÉTICOS – NO REALES") so demo
   outputs can never be mistaken for findings.

Public does not mean free of personal data, so the privacy controls below
still apply to public information.

## 1. Why this matters

SECOP is public (Ley 1712 de 2014 — transparency and access to public
information). But public is not the same as *free of personal data*:

* Most SECOP II contracts are *prestación de servicios* with **natural
  persons**. Their name and cédula appear as the supplier.
* Depending on the extract, the SECOP II contracts dataset also exposes
  **legal representatives** (name, ID, gender, nationality, domicile),
  **supervisors** and **ordenadores del gasto** (name, ID), and even
  **bank name, account type and account number**.
* Free-text descriptions often repeat the contractor's name and ID number
  ("…identificado con C.C. 1.234.567…").

Processing this data falls under **Ley 1581 de 2012** (Habeas Data) and its
regulation (Decreto 1377 de 2013, compiled in Decreto 1074 de 2015).
Ley 1712 de 2014 (art. 18) itself restricts access to information whose
disclosure harms privacy.

There is a second concern, specific to this project: **an automated label
attached to a named person can cause reputational harm**, even though a
statistical anomaly is not evidence of wrongdoing (*presunción de inocencia*,
Constitución art. 29).

## 2. What the code does (privacy by default)

| Control | Where | Details |
|---|---|---|
| Drop at load | `privacy.drop_columns`, `data/io.py` | Bank data, representative/supervisor/ordenador names and IDs, gender, domicile. For CSV files they are excluded by `usecols` and **never enter memory**. |
| Keyed pseudonyms | `privacy.pseudonymize_columns`, `data/privacy.py` | Supplier and legal-rep IDs become `p_<HMAC-SHA256>`. They stay joinable for concentration analyses, but are not reversible without the key. A *plain* hash of a cédula is **not** anonymous, since the whole ID space can be enumerated in minutes. |
| Secret key handling | `CT_PII_SALT` env var or `.secrets/pii_salt` (0600, git-ignored) | Rotate it and you get new pseudonyms. Share it only through a secret manager. |
| Free-text redaction | `redact_text` | Masks ID numbers (`C.C.`, `NIT`, `cédula No.`), e-mails, phones and numbers ≥ 6 digits. This also removes **target leakage** ("por valor de $12.000.000"). |
| Name masking | `data/person_type.py`, `mask_person_names` | Suppliers are classified from the document type, number and name. Natural persons are shown as `Persona natural p_…`, undetermined suppliers as `Proveedor sin clasificar p_…`. Only suppliers positively identified as legal entities keep their names. The classification runs on the raw number *before* pseudonymisation, and only its result (`natural`/`juridica`/`desconocido`) is stored. |
| No PII as features | `config.py` validator | Config loading fails if a dropped or pseudonymised column is configured as a model feature or output column. |
| Rare-term filter | `features.text_min_df` | A word must appear in ≥ N contracts to become a feature, so the model cannot memorise individual names. |
| Official records | `data/records.py` | Contraloría / sanctions files: names and raw IDs are discarded on import. Only HMAC pseudonyms of IDs and record dates are stored, plus a manifest (official URL, publisher, file SHA-256). |
| Minimal download | `data/download.py` | `ct download` requests only the columns the pipeline needs. Columns in `privacy.drop_columns` are never requested from the API. |
| JSON from the API URL | `data/download.py::frame_from_api_records` | A raw URL such as `…/resource/jbjy-vk9h.json?$limit=10000` returns **every** column, including bank accounts and supervisors' IDs. When such a file is read, only an **allowlist** of needed columns is kept. Anything else, known or unknown, is discarded immediately. Prefer `ct download`, which never requests those columns. |
| No raw data in models | `mlops/drift.py` | Drift profiles store only aggregate proportions. |
| Tests | `tests/test_privacy.py`, `tests/test_pipeline.py` | Assert that no raw ID, person name, bank or supervisor data appears in cleaned tables or scored outputs. |
| Repository | `.gitignore`, `data/.gitignore`, pre-commit | Data, models and secrets are never committed. Large-file and private-key hooks are enabled. |

### Known limits

* Redaction is regex-based. **Names written in free text are not removed.**
  Only the rare-term filter stops them from becoming features, and they are
  still visible in the description column of outputs. Adding Spanish NER is
  on the [roadmap](ROADMAP.md).
* Pseudonymised data is still **personal data** under Ley 1581: pseudonymisation
  is not anonymisation. Keep `data/processed` and `data/outputs` access-controlled.
* The `Proveedor Adjudicado` column keeps company names. Sole proprietors
  registered with a NIT may still be natural persons.

## 3. Rules for using the results

1. Results are **statistical alerts to prioritise reviews**, never conclusions. Every
   output and the dashboard carry this disclaimer.
2. **Do not publish** rankings that identify natural persons. Aggregate by
   entity, municipality or contract type when results are made public.
3. Before acting on an alert, a person must check the contract in SECOP
   (`URLProceso` is included for that).
4. Keep the scored files for the shortest time needed, and log who accessed them.
5. Document false positives: they are the best data for improving the model
   (see `evaluation.label_column`).
6. When in doubt about a column, add it to `privacy.drop_columns`. The
   pipeline keeps working without it.
