# Changelog

Changes that matter to **users of the outputs** (renamed columns, new
signals, retraining needed) are marked **⚠**.

## 1.3.1 — clean-up

* Removed dead code: the `features.add_date_features` option (it was never
  read; date features are always computed), `Settings.resolve_paths`,
  `io.read_processed`, and unused helpers of the text model.
  **⚠** Configs that still set `add_date_features` fail validation: delete
  that line.
* Tests: shared helpers (`write_cli_config`, `fast_download`) moved to
  `tests/conftest.py`; test modules no longer import each other.
* `configs/secop_ii.yaml`: the `features` block is grouped by component, with
  comments saying which option affects what. Values are unchanged.
* Documentation reviewed:
  * README (installation, live check, dashboard, model summary, project structure);
  * MLOPS (on-disk layout of a model version);
  * MODELING (benchmark variant names);
  * ROADMAP (completed items, wrong cross-references, internal identifiers removed);
  * MIGRATION (v1.3 columns).

## 1.3.0 — natural persons vs legal entities

* New `data/person_type.py`: each supplier is classified as `natural`,
  `juridica` or `desconocido` from the published document type, the document
  number (NIT range, cédula series, DIAN check digit) and the name. The
  deciding rule is kept in `ct_person_type_rule`.
* New `models/segmented.py`: one model per person type plus a global fallback
  (`model.segment_by: person_type`, default). Calibration groups are now
  person type × contract type.
* Person type is also a model feature (`features.add_person_type`).
* **⚠** Outputs gain `person_type` and `model_segment`. `natural_person` now
  comes from the classification (not only the document type). Unclassified
  suppliers are masked as `Proveedor sin clasificar p_…`.
* `ct prepare` reports the share of each person type and the agreement of the
  number rule with the published document type.
* New config keys: `schema.supplier_group` ("Es Grupo"),
  `features.add_person_type`, `model.segment_by`,
  `model.min_segment_train_rows`, `model.min_segment_valid_rows`.
* **⚠** Retrain (`ct train`) to get the segmented model. Older models keep working.

## 1.2.0 — neutral language, better language model, calibrated alerts

* **⚠** Outputs renamed to review-oriented, non-accusatory language:
  `risk_index` → `alert_index`, `risk_rank` → `review_priority`,
  `red_flags` → `signals`, `rf_*` → `sig_*`, `price_alert` → `price_deviation`,
  `overprice_ratio` → `excess_ratio`, `value_at_risk` → `weighted_excess`.
  The notice states that an alert is not evidence and that the authors accept
  no responsibility for third parties' decisions. Neutral Spanish signal
  labels appear in the dashboard. A test guards the vocabulary.
* Stacked text price model (word + character TF-IDF → Ridge, out-of-fold),
  quantity extraction from descriptions, masking of amounts written in words,
  and a blend with a sparse Ridge.
  * `features.text_embedder` now defaults to `none`;
  * the default sentence model is now MiniLM.
* Calibration of the price-deviation signal per contract type
  (`scoring.calibration: by_group`).
* `ct benchmark`: accuracy, detection and cost of each architecture.
* `ct live-check` hardened:
  * OS certificate store (`truststore` dependency);
  * TLS hints;
  * metadata and text-date fallbacks;
  * duplicated ids tolerated;
  * errors and hints printed in the terminal.
* More realistic synthetic data: text signal, heteroscedastic noise, inflation,
  ×1.8–×8 anomalies.
* **⚠** Retrain to use the new model. Scored files from 1.1 use the old column names.

## 1.1.0 — uv, flexible dependencies, portable models

* uv is the standard runner (`make install`, `make demo`, `make app`). Docker is optional.
* Dependencies declare minimum versions only. `uv.lock` replaces the
  hash-pinned `requirements/*.txt`. New targets: `make upgrade`,
  `make rollback-deps`, `make test-lowest`, `make test-latest` and
  `make portability-check`. Two false minimums were fixed
  (xgboost ≥ 2.0.3, typer ≥ 0.15.4).
* Pickle-free models (`features/portable.py`): a model trained with the oldest
  supported libraries scores identically with the newest.
* `ct doctor`: environment and model compatibility check. The CLI shows short
  messages for expected errors (`CT_DEBUG=1` for tracebacks).
* JSON saved from the datos.gov.co API URL is accepted, read through an
  allowlist that drops personal-data columns.
* **⚠** Retrain once so the champion is saved in the portable format.

## 1.0.0 — refactor of the legacy code

* Package `contract_transparency` with a validated YAML config, privacy by
  default (PII dropped at read time, keyed pseudonyms, text redaction,
  name masking), an XGBoost price model with a median baseline, rule-based
  signals, a file registry with promotion gates, drift monitoring, optional
  MLflow, the `ct` CLI and a Streamlit dashboard.
* Public-data update:
  * audit links to SECOP and to the datos.gov.co record on every row;
  * provenance manifests;
  * a statistical-information notice;
  * `ct download`, `ct records import` (Contraloría bulletin, SECOP sanctions)
    and `ct live-check`;
  * Docker packaging.
* Legacy code moved to `legacy/` (reference only). See `docs/MIGRATION.md`
  for the mapping and the 17 defects found.
