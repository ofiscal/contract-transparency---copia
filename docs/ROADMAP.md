# Insights & roadmap

## Insights from the refactor

1. **The project has no ground truth.** Every model so far predicts *price*,
   and "anomaly" means "far from the prediction". Nobody has measured whether
   the alerts match real irregularities. This is the biggest limitation, more
   than the choice of model (see N1).
2. **The last legacy models already used the right ideas**: UNSPSC codes,
   duration, location, modality, text embeddings and a robust loss (recovered
   from the pickles, see `legacy/README.md`). They were lost because encoders
   were not saved with the models. The registry now saves preprocessing and
   model together.
3. **Evaluation was optimistic.** Random splits, in-sample predictions
   (`skiprows` bug) and raw-scale R² hid how the models generalise to *new*
   contracts. The temporal split and the baseline gate make that visible.
   Expect real-data numbers to be lower than the synthetic demo's.
4. **A raw-scale R² near zero is not a failure**, and a high one is not
   success: it is dominated by a handful of mega-contracts. Use MAE in log
   space and MdAPE ("half of contracts predicted within X %").
5. **Residuals are not homogeneous.** The error spread differs between a
   *prestación de servicios* contract and a public work. One global
   t-distribution over-alerts high-variance types and under-alerts
   low-variance ones. Addressed in v1.2/v1.3: calibration per person type ×
   contract type.
6. **Descriptions leak the target.** Texts such as "por valor de
   $12.000.000" or "12 millones" let a text model "predict" the price by
   reading it. Redaction masks long numbers (for privacy) and amounts written
   in words (v1.2), which also removes the leak.
7. **Underpricing is a signal too.** Contracts just below a modality
   threshold (mínima cuantía, menor cuantía) and clusters of small contracts
   suggest *fraccionamiento*. The `possible_contract_splitting` rule is a
   first step; unit prices (X3) would go further.
8. **Natural persons and companies are different populations.** The same
   object is priced differently by an individual and by a firm. Knowing the
   person type cut the error by 12 % on synthetic data (v1.3), and each type
   now has its own model.

## Done in the second iteration (v1.0: public data, ingestion, packaging)

* **Public-data policy.** Only public, official sources are used. Every row
  links to SECOP (`secop_url`) and to its open-data record (`open_data_url`).
  Rows without an official link are not published. Provenance manifests carry
  source hashes and a column dictionary, and the statistical-information
  notice appears everywhere.
* **Automated ingestion** (former X1): `ct download` from the datos.gov.co API,
  resolving column names from live metadata and never requesting personal-data
  columns.
* **Official records** (part of N1): Contraloría *Boletín de Responsables
  Fiscales* and SECOP sanctions importer. It adds a legally grounded alert signal
  (Ley 610/2000 art. 60) and a weak-label metric per model version.
* **`ct live-check`**: end-to-end verification against the live sources.
* **Packaging**: Docker image + compose, hash-pinned lock files, PEP 668-safe
  setup scripts, CI building and health-checking the real image.

## Done in the third iteration (v1.1)

* **uv as the standard runner** (`make install/demo/app/test`), with no Docker
  needed. Docker remains optional and also installs from `uv.lock`.
* **Flexible dependencies:** minimums only, no upper caps. The minimums are
  verified (`make test-lowest`; two false ones fixed: xgboost ≥ 2.0.3,
  typer ≥ 0.15.4), the newest releases are checked weekly (`make test-latest`),
  and upgrades and rollbacks are one command each.
* **Pickle-free, portable models:** a model trained with the oldest
  libraries scores identically with the newest (`make portability-check`).
  `ct doctor` checks the environment after any upgrade.
* **JSON saved from the API URL** (`…/jbjy-vk9h.json?$limit=10000`) is
  accepted, with an allowlist that discards personal-data columns.

## Done in the fourth iteration (v1.2)

* **Neutral, review-oriented language everywhere:** alerts and signals for
  review, no "risk". Renamed outputs; a notice stating that the authors draw
  no conclusions and accept no responsibility; a test guards the vocabulary.
* **Language model rebuilt:** stacked word + character TF-IDF → Ridge text price
  model (out-of-fold), quantity extraction, masking of amounts written in
  words, and a blend with a sparse linear model. Error −37 % vs v1.1 on
  realistic synthetic data (docs/MODELING.md).
* **Per-contract-type calibration** of the price-deviation signal (former N3).
  Detection of ×1.8–×8 overpricing rose from 9 % to about 77 %, with false
  alerts below 0.1 % per type on synthetic data.
* **`ct benchmark`:** accuracy, detection and cost of each architecture on any
  dataset.
* **`ct live-check` hardened:** OS certificate store (truststore),
  certificate-error hints, metadata fallback, text-date fallback, tolerance
  for duplicated ids, per-step errors and hints printed in the terminal.

## Done in the fifth iteration (v1.3)

* **Natural person vs legal entity classification** from the published
  document type, the number (NIT / cédula conventions, DIAN check digit) and
  the name, with the deciding rule recorded. Undetermined suppliers are masked
  like natural persons.
* **One model per person type** (plus a global fallback), calibrated by
  person type × contract type. On synthetic data, knowing the person type cut
  the error by 12 %; separate models tie with a single model using the
  feature (docs/MODELING.md §3.4).

## Now: requires internet access (could not be done from the build environment)

| # | Task |
|---|---|
| V1 | Run `ct live-check` against the real API. If it reports a different `public_data.id_api_field`, or missing columns, update `configs/secop_ii.yaml`. |
| V2 | Download the current *Boletín de Responsables Fiscales* from contraloria.gov.co and run `ct live-check --fiscal-bulletin <file>`. Pin `id_column` / `date_column` if auto-detection picks the wrong ones. Replace `official_url` with the bulletin's exact page. |
| V3 | Find the datos.gov.co dataset for *SECOP – Multas y sanciones* (catalog search), document its URL in `records.sources.secop_multas_sanciones.official_url`, and import it. |
| V4 | (Optional, Docker only) Let the CI `docker` job build the image. The Dockerfile was validated here by instruction-level emulation. |
| V5 | Run `scripts/setup.ps1` (or `uv sync` + `uv run ct demo-data`) once on Windows. |
| V6 | Open `https://www.datos.gov.co/resource/jbjy-vk9h.json?$limit=10000`, save it, run `uv run ct prepare --input <file>.json`, and check the log for "Fields not found". Any listed field needs an entry in `public_data.api_field_overrides`; `ct live-check` reports the exact names. |
| V7 | Run `uv run ct benchmark` on a real extract (≥ 50k contracts). Confirm that `segmented` and `v1_2` beat `xgb_tfidf_svd` on real descriptions; try `xgb_sentence_embeddings` with `uv sync --extra nlp`. |
| V8 | Check how often `f_log_text_quantity` is filled on real descriptions, and extend the unit list in `features/builder.py::_QTY` if needed. |
| V9 | On the real extract, read `share_person_*` and `person_type_number_rule_agreement` in `data/processed/prepare_report.json`. An agreement below ~0.95, or more than 10 % `desconocido`, means the numbering conventions need adjusting (`data/person_type.py`). Then run `ct benchmark --variants v1_2,v1_2_person_type,segmented`. |

## Now (next 2–4 weeks)

| # | Task | Why |
|---|---|---|
| N1 | **Build a labelled set.** Contraloría bulletin and SECOP sanctions are now importable as weak labels (done). Still missing: Procuraduría (SIRI) / PACO sanctions, and auditor feedback ("confirm / dismiss" buttons in the dashboard stored as `evaluation.label_column`). | Turns "far from the prediction" into measured precision/recall. Lets the `alert_index` weights be *learned* (logistic regression over the `sig_*` flags) instead of hand-set. |
| N2 | Run `ct validate` + `ct run` on the **real** SECOP II extract and review the first 100 alerts with a domain expert. | Check the schema mapping and set thresholds (`tail_probability_alert`, rule thresholds) on real data. |
| N3 | ~~Group-wise calibration~~ **done** (v1.2: per contract type; v1.3: per person type × contract type). Next: per UNSPSC segment when data allows. | |
| N4 | **Deflate values** to constant pesos (DANE IPC, or SMMLV for *prestación de servicios* fees) before training. | Prices from 2019 and 2025 are not comparable; the temporal split exposes this. |
| N5 | **Per-contract explanations**: `booster.predict(..., pred_contribs=True)` (built-in SHAP values) shown in the dashboard "Detalle" tab. | Auditors need to know *why* the expected price is X. |
| N6 | Verify the **election calendar** and the Ley 996 rule simplifications with a lawyer. Review all rule thresholds with procurement experts (Ley 80/1993, Ley 1150/2007, Decreto 1082/2015). | Rules must be defensible. |

## Next (1–3 months)

| # | Task |
|---|---|
| X2 | **Join more open data**: SECOP II *Procesos* (number of bidders → enables `single_bidder`), *Adiciones* (real otrosí values), RUES (company creation date → "company created < 1 year before a large contract"), CNE *Cuentas Claras* campaign donors (the legacy `cne_cuentas` idea), SIGEP (public servants, conflicts of interest). All joins on *pseudonymised* IDs, using the same key. Each must be a public, official source registered with its URL (the `records` mechanism already supports ID lists). |
| X3 | **Unit-price benchmarking**: compare item prices against Tienda Virtual del Estado Colombiano / Acuerdos Marco reference prices by UNSPSC class. Total value mixes price and quantity; unit prices are far more precise. |
| X4 | **Supplier–entity network analysis** (bipartite graph, shared representatives, communities) to detect collusion rings. This extends `shared_legal_representative`. |
| X5 | **Spanish NER** (spaCy `es_core_news_md`) to remove person names from free text before storage/output. This closes the known privacy gap. |
| X6 | **Fairness monitoring**: alert rates by department and municipality category, tracked per model version, so the model does not systematically target poorer or remote regions (where logistics legitimately raise prices). |
| X7 | **Output to BigQuery/Tableau** (`output.format: bigquery`), which completes the legacy plan "send the updated file to Tableau" (the legacy code wrote to a BigQuery table; its name belongs in the config, not in the code). |
| X8 | **Model cards** generated from `metadata.json` (intended use, data window, metrics, limitations) for every promoted version. |

## Later

* Fine-tune a Spanish language model (e.g. RoBERTa-BNE / BETO) for price
  regression, continuing the legacy Hugging Face experiments. It must beat the
  TF-IDF model through the same promotion gates.
* Unsupervised complement (Isolation Forest on features + residuals,
  continuing the legacy GMM idea) for contract types with little history.
* Scale-out for the full history (> 5 M rows): year-partitioned parquet,
  DuckDB/Polars in `prepare`, out-of-core XGBoost.
* Deployment: the Docker image exists. Add internal hosting behind SSO.
  Because outputs contain personal data, **do not deploy the dashboard
  publicly**. For public releases, publish aggregated indicators (by entity /
  municipality) that link to SECOP.
* Shrink the repository: remove `legacy/artifacts` (113 MB) from git history
  (`git filter-repo`). This rewrites history, so coordinate with everyone who
  has a clone.
