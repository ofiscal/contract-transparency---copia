# Contract Transparency — SECOP II

**Statistical alerts to support the review of Colombian public contracts**,
built **only from public, auditable information**: SECOP II *Contratos
Electrónicos* (datos.gov.co) and, optionally, official public records such as
the Contraloría's *Boletín de Responsables Fiscales*.

1. **Expected value.** A model estimates the value a contract *would
   typically have*, given its type, modality, UNSPSC code, location, duration
   and description (including quantities and level of service written in the
   text). Suppliers are classified as natural persons or legal entities, and
   each group has its own model. When the published value is far above the
   estimate for comparable contracts, the contract gets a **price-deviation signal**.
2. **Rule signals.** Contracts signed in Ley de Garantías windows, suppliers
   listed in the Contraloría bulletin (Ley 610 de 2000, art. 60), concurrent
   direct contracts, possible contract splitting, dominant suppliers, shared
   legal representatives, additions, atypical durations.
3. **Alert index and review priority.** A weighted combination of the
   signals, with the list of active signals **and a direct link to the contract
   in SECOP**, shown in a Streamlit dashboard and in exported files.

> ⚠️ **AVISO / NOTICE.** An alert only says that a contract is *unusual
> compared with similar contracts* and that reviewing it could be useful. It
> is **not** evidence, accusation or judgement of wrongdoing or liability of
> any entity, contractor or person. It does not replace verification at the
> official source (every row links to SECOP) or the work of the competent
> authorities. The authors draw no conclusions about any contract and accept
> no responsibility for decisions third parties make based on this
> information. See [§4](#4-public-data--auditability) and
> [docs/DATA_PROTECTION.md](docs/DATA_PROTECTION.md).

---

## 1. Installation (uv)

Everything runs through **[uv](https://docs.astral.sh/uv/)**:
- it creates a local `.venv`;
- it downloads a suitable Python if needed;
- it **never touches the system Python**, so there are no
  `externally-managed-environment` errors;
- Docker is **not** required.

```bash
# 1. Install uv once (no admin rights needed)
curl -LsSf https://astral.sh/uv/install.sh | sh                  # Linux / macOS
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"       # Windows
#    (or: pipx install uv | brew install uv | winget install astral-sh.uv)

# 2. Install the project (tested versions from uv.lock)
make install          # = uv sync
```

Without `make` (e.g. Windows), use the same commands directly: `uv sync`,
then `uv run ct <command>`. Or run `scripts/setup.sh` / `scripts\setup.ps1`,
which install uv if missing and run `uv sync`.

| Need | Command |
|---|---|
| MLflow tracking | `uv sync --extra mlflow` |
| Transformer text models (large) | `uv sync --extra nlp` |
| Plain pip in your own venv | `python -m venv .venv && .venv/bin/pip install -e .` |
| Docker (optional) | `make docker-demo` / `make docker-app` (see `Dockerfile`) |

After installing or upgrading anything, run **`uv run ct doctor`**. It checks
package versions, folders, the pseudonymisation key, and that the current
model still loads and predicts.

## 2. Quick start (synthetic data, no internet needed)

```bash
make demo     # = uv run ct demo-data --with-records && uv run ct run -c configs/demo.yaml
make app      # = uv run ct app -c configs/demo.yaml   -> http://localhost:8501
```

Demo outputs are watermarked **"DATOS SINTÉTICOS – NO REALES"**
(`public_data.synthetic: true` in `configs/demo.yaml`).

## 3. Running on real public data

Commands below are shown as `ct …`: run them as `uv run ct …`, or activate
`.venv` first (`source .venv/bin/activate`).

```bash
export CT_PII_SALT=$(python -c "import secrets; print(secrets.token_hex(32))")  # or put it in .env

# 1. Download from the datos.gov.co API: only needed, non-personal columns,
#    with provenance written to data/raw/secop_ii_contratos.csv.source.json
ct download --since 2024-01-01 --until 2025-01-01          # or: make download SINCE=2024-01-01
#    Alternatives:
#    * Export -> CSV on the portal, saved as data/raw/secop_ii_contratos.csv
#    * JSON saved from the API URL in a browser, e.g.
#      https://www.datos.gov.co/resource/jbjy-vk9h.json?$limit=10000
#      -> ct prepare --input jbjy-vk9h.json
#      (that URL returns ALL columns, including bank accounts; only the columns
#       the pipeline needs are kept, everything else is discarded on read)

# 2. Optional: official public records
ct records list
ct records import contraloria_responsables_fiscales path/to/boletin_responsables_fiscales.xlsx

# 3. Pipeline
ct prepare && ct train && ct score
ct app

# 4. Monthly update with a new extract
ct download --since 2025-01-01 --output data/raw/secop_2025.csv
ct update data/raw/secop_2025.csv
```

**First time on a new machine, run `ct live-check`.** It verifies the
complete pipeline against the live sources in an isolated folder
(`data/live_check/`; your registry is not touched) and writes a
PASS/WARN/FAIL report. It checks that:
- the API is reachable and the configured columns exist in the live dataset;
- the API field for contract ids is correct;
- a real sample downloads, cleans and trains, and the promotion gates are evaluated;
- ≥ 95 % of contracts have an official SECOP link;
- sampled audit links return that contract with the same value.

Every WARN/FAIL is printed in the terminal with its error and a hint. The
report also records the environment (Python, platform, proxy and certificate
settings). Certificate errors behind corporate proxies are the most common
cause: HTTPS uses the operating-system certificate store (`truststore`), and
`SSL_CERT_FILE` can point to your organisation's CA bundle.

Add `--fiscal-bulletin <file>` to also check a real Contraloría file.

```bash
ct live-check --limit 20000 --fiscal-bulletin boletin.xlsx
cat data/live_check/report.md
```

**About official records.** The Contraloría publishes the *Boletín de
Responsables Fiscales* ([contraloria.gov.co](https://www.contraloria.gov.co)).
Download the current edition yourself and import it. The importer:
- auto-detects the header row and the ID and date columns (override them in `records.sources` if the layout changes);
- matches NITs with or without the check digit;
- keeps **only keyed pseudonyms and dates**;
- records the file's SHA-256 and official URL in every output manifest.

Each imported source is also used as a **weak label**: every model version
reports how much more often it alerts on contracts of listed suppliers
(`weak_labels_test.lift`).

## 4. Public data & auditability

| Rule | How it is enforced |
|---|---|
| Only public sources | Inputs are SECOP II from datos.gov.co (API or portal export) and official records imported with their official URL. Each is listed with publisher, retrieval details and SHA-256 in every output manifest. |
| Every result is verifiable | Each row has `secop_url` (the official SECOP process page from the dataset's `URLProceso`, accepted only for `secop.gov.co` / `colombiacompra.gov.co` domains) and `open_data_url` (the exact record in the datos.gov.co API). Rows without a valid official link are **not published**; they are counted in the manifest (`excluded_not_auditable`). |
| Facts vs statistics are separated | The manifest's column dictionary marks each column `source` (copied from the public data), `source (redacted)` or `derived` (a statistical estimate by this project). |
| "Statistical information, not proof" travels with the data | The notice (es/en) appears in the CLI, the dashboard, the manifest (`<file>.manifest.json`), an `aviso` column on every row, and as the **first sheet** of Excel outputs. |
| Reproducible | Manifests include the model version, config fingerprint, package version and git commit (baked into Docker images). |
| Personal data minimised | Personal-data columns are never downloaded. IDs become keyed pseudonyms, and names of natural persons are masked ([DATA_PROTECTION.md](docs/DATA_PROTECTION.md)). |

## 5. Dashboard

`ct app [-c config]` (or the Docker `app` service).

| Tab | Content |
|---|---|
| Alertas | KPIs, contracts in review-priority order with **SECOP** and **open-data** links, filters (department, modality, contract type, supplier type, alert index, signal), CSV + provenance-manifest downloads |
| Detalle de contrato | Real vs expected value, signals (in Spanish), supplier type and the model that estimated the value, links to verify at the official source |
| Distribuciones | Real vs expected scatter, signal frequencies, alerts by department |
| Proveedores | Supplier ranking (natural persons pseudonymised) |
| Modelos | Registry, metrics vs baseline/champion, weak-label lift, feature importance, promotion history |
| Fuentes | Public sources used (publisher, URL, file hash, retrieval query), rows excluded, column dictionary |

## 6. CLI reference

| Command | What it does |
|---|---|
| `ct demo-data [--rows N] [--with-records]` | Synthetic SECOP II-like data (+ synthetic bulletin) |
| `ct download [--since --until --limit --output]` | datos.gov.co API → CSV + `.source.json` provenance |
| `ct records list` / `ct records import SOURCE FILE` | Official public records (Contraloría, SECOP sanctions) |
| `ct validate [--input f]` | Schema report, no files written |
| `ct prepare [--input f.csv\|f.json\|f.parquet --nrows N]` | Privacy → cleaning → validation → snapshot |
| `ct train` | Temporal split, model + baseline, metrics, weak labels, registry, auto-promotion |
| `ct models` / `ct promote vNNNN` | Registry listing / promotion and rollback |
| `ct score [--input f --version v --output f.csv\|f.xlsx]` | Ranked results + `<file>.manifest.json` |
| `ct drift [--input f]` / `ct update FILE` | Drift report / incremental MLOps loop |
| `ct run` | `prepare` + `train` + `score` |
| `ct live-check [--since --limit --fiscal-bulletin f]` | End-to-end verification against the live sources |
| `ct benchmark [--variants a,b --nrows N]` | Compare model architectures on accuracy, alert precision/recall and cost (run it on the real extract) |
| `ct doctor [--network]` | Environment check after installing/upgrading (versions, folders, key, champion model) |
| `ct suppliers`, `ct sustainability` | Supplier ranking; zero-shot sustainability tags (`.[nlp]`) |
| `ct app [--address 0.0.0.0]` | Dashboard |

All commands accept `-c/--config` and `-v/--verbose`; see `ct <command> --help`.

## 7. Output columns

| Column | Meaning |
|---|---|
| `review_priority`, `alert_index`, `signals` | Suggested review order (1 = first), alert index in [0, 1], active signals |
| `secop_url`, `open_data_url` | **Where to verify**: official SECOP page and open-data record |
| `contract_value`, `predicted_value`, `pct_error`, `excess_value` | Real vs statistically expected value (COP) |
| `tail_probability` | Estimated probability that a comparable contract is at least this expensive relative to its prediction (small = unusual) |
| `sig_*` | One boolean per signal (`sig_price_deviation`, `sig_supplier_in_fiscal_bulletin`, `sig_direct_during_ley_garantias`, …) |
| `excess_ratio`, `weighted_excess` | Legacy `flagging()` outputs |
| `supplier`, `supplier_pseudonym` | Company name, or `Persona natural p_…` / `Proveedor sin clasificar p_…` |
| `person_type`, `model_segment` | `natural` / `juridica` / `desconocido`, and the model that estimated the value |
| `model_version`, `aviso` | Model that produced the row; notice ("alerta estadística para revisión…") |

## 8. How the model works (summary)

```
supplier document + name ──► natural person | legal entity ──► one model per type (+ global fallback)
inside each model:
description ──► word + char TF-IDF ──► Ridge (out-of-fold) ──┐  text price estimate
description ──► quantity extraction ("92 equipos", "3,5 km") ┤
categorical, UNSPSC hierarchy, duration, dates ──────────────┴─► XGBoost (pseudo-Huber, log value)
one-hot + TF-IDF ──► sparse Ridge ───────────────────────────────► blend (weight chosen on validation)
                                                                   │
       real − expected (log) ─► scaled by the spread of its person type × contract type ─► Student-t tail probability
                                                                   │
                             price-deviation signal if P < 0.01 ◄──┘  (+ rule signals → alert index)
```

Suppliers are classified as natural persons or legal entities from the
published document type, the number (Colombian NIT / cédula conventions,
including the DIAN check digit) and the name. Each type gets its own model.
Knowing the person type cut the error by a further 12 % on synthetic data.

On realistic synthetic data (20,000 contracts) v1.2 reduced the error by 37 %
versus v1.1 and raised the share of planted overpricing that is detected
(×1.8–×8) from 9 % to about 77 %, at 87 % precision. Costs stay CPU-only:
roughly 30–45 s of training per 13,000 contracts for the three models
(natural, legal entity, global), 0.25–0.5 ms per contract to score, and about
5 MB on disk (half of that with `model.segment_by: null`). Architecture choices, comparisons with LightGBM,
CatBoost and transformer text models, detection limits and the threshold
trade-off are in **[docs/MODELING.md](docs/MODELING.md)**. Run
`ct benchmark` on your real extract before changing defaults.

## 9. MLOps (how models are upgraded)

```
download ─► prepare ─► train ─► candidate vN ─► gates ─► champion ─► score / dashboard
                                               │ fail
                                               ▼
                                         rejected (audit log)
```

Summary:
- **Registry:** immutable versions with checksums.
- **Promotion gates:** beat the baseline, and beat the champion on the most recent contracts.
- **Retraining:** PSI drift monitoring and `ct update`.
- **Tracking:** optional MLflow (`sqlite:///models/mlflow.db`).
- **CI:**
  - locked versions on Python 3.10, 3.12 and 3.14;
  - the **oldest** allowed versions;
  - the **newest** releases (weekly early warning);
  - model portability (train old, score new);
  - an optional Docker build.

Details: [docs/MLOPS.md](docs/MLOPS.md). **Upgrading packages:**
[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) (`make upgrade`, `make rollback-deps`).

## 10. Configuration

YAML validated at load time (unknown keys fail). `configs/demo.yaml` inherits
from `configs/secop_ii.yaml`. Main blocks:

| Block | Controls |
|---|---|
| `schema` | Which SECOP column plays which role |
| `public_data` | Dataset, publisher, official domains, `require_secop_url`, notices, `synthetic` watermark |
| `records` | Official record sources (title, publisher, official URL, legal basis, flag, columns) |
| `privacy` | Dropped / pseudonymised columns, text redaction |
| `features`, `model`, `evaluation`, `promotion` | Features, model (incl. `segment_by`, blend), split and quality gates |
| `scoring`, `rules` | Alert threshold, election calendar, rule thresholds and weights |
| `download`, `drift`, `tracking`, `output` | API paging/retries, PSI, MLflow, output format |

## 11. Testing

```bash
make test                 # full suite (uv run pytest --cov), ~2 min
make test-fast            # skip end-to-end tests
make test-lowest          # oldest dependency versions allowed by pyproject.toml (Python 3.10)
make test-latest          # newest releases of everything (LATEST_PYTHON=3.14)
make portability-check    # model trained with oldest libs == scored with newest libs
make lint
```

The tests need no internet and use no real personal data:
- synthetic contracts;
- a **mock datos.gov.co API** (including the raw `…/jbjy-vk9h.json?$limit=…` response with every column);
- a synthetic Contraloría-like bulletin.

## 12. What has been verified, and where

| Item | Verified in the refactoring environment | Still to verify with internet |
|---|---|---|
| Pipeline, rules, privacy, provenance | ✅ 226 tests on the locked versions (Python 3.10, 3.12, 3.14), on the **oldest** allowed versions (Python 3.10, pandas 2.1, scikit-learn 1.4, xgboost 2.0.3) and on the **newest** releases (Python 3.14, pandas 3.1, scikit-learn 1.9, xgboost 3.4) | — |
| Saved models across upgrades | ✅ trained on the oldest stack, scored on the newest: identical predictions, alerts and ranking | — |
| `uv` workflow (`make install/demo/app/test*`) | ✅ on a fresh copy, on a PEP 668-managed system Python, without Docker | — |
| datos.gov.co downloader, JSON from the API URL, live check | ✅ against a mock API implementing the endpoints used | `ct live-check` / the real URL: datos.gov.co was **not reachable** from the refactoring environment (DNS blocked; only PyPI was available) |
| Contraloría importer | ✅ with a synthetic bulletin in an assumed layout | Import a real edition (`ct live-check --fiscal-bulletin …`) |
| Docker (optional) | ✅ Dockerfile executed instruction by instruction in an emulated root | A real `docker build` (the CI `docker` job) |

## 13. Project structure

```
pyproject.toml, uv.lock                           minimum versions (no caps) + known-good lock
Makefile                                          uv-based commands (make help)
scripts/                                          setup.sh / setup.ps1 (install uv + uv sync),
                                                  check_model_portability.sh
Dockerfile, docker-compose.yml, .dockerignore     optional container packaging
configs/                                          secop_ii.yaml (default), demo.yaml
src/contract_transparency/
  config.py, provenance.py                        settings; audit links, manifests, notices
  environment.py                                  `ct doctor`, version compatibility checks
  data/        io, cleaning, privacy, person_type (natural / jurídica), validation,
               splits, incremental, synthetic, download (datos.gov.co API),
               records (Contraloría / sanctions)
  features/    builder (sklearn fit), portable (pickle-free saved preprocessing),
               text_model (stacked text price model), text (embeddings option)
  models/      price_model (XGBoost + blend), segmented (one model per person type),
               baseline, target
  scoring/     anomaly (price deviation, calibration), rules (signals, alert index)
  analysis/    supplier rankings, sustainability tags
  mlops/       pipeline, registry, drift, metrics, tracking, live_check, benchmark
  app/streamlit_app.py, cli.py
tests/                                            pytest suite (+ mock_socrata.py)
docs/                                             MODELING, MLOPS, DEPENDENCIES, DATA_PROTECTION,
                                                  MIGRATION, ROADMAP
CHANGELOG.md                                      what changed in each version
legacy/                                           original code, reference only
```

## 14. Further reading

* [docs/DATA_PROTECTION.md](docs/DATA_PROTECTION.md): public-data policy, privacy controls, rules of use.
* [docs/MODELING.md](docs/MODELING.md): model architecture, evidence, state of the art, costs, limits.
* [docs/MLOPS.md](docs/MLOPS.md): lifecycle, registry, gates, drift, scheduling, live check.
* [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md): upgrade policy, `make upgrade`, rollback, verified version span.
* [docs/MIGRATION.md](docs/MIGRATION.md): legacy → new mapping and the 17 defects found.
* [docs/ROADMAP.md](docs/ROADMAP.md): insights and next steps.
* [CHANGELOG.md](CHANGELOG.md): what changed in each version, including renamed outputs.

## License

MIT, see [LICENSE](LICENSE).
