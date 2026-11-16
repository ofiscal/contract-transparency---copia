# MLOps lifecycle

```
            ┌──────────── configs/*.yaml (validated, fingerprinted) ────────────┐
            ▼                                                                   │
 raw extract ─► prepare ─► processed snapshot ─► train ─► candidate vN ─► gates ─┤
 (datos.gov.co)  │ privacy    (parquet, no PII)    │ temporal split        │     │
                 │ cleaning                        │ model + baseline      │ pass│fail
                 │ validation                      │ residual t-dist       ▼     ▼
                 ▼                                 │ drift profile     champion  rejected
           prepare_report.json                     ▼                     │   (audit log)
                                           registry + MLflow             ▼
                                                                  score ─► outputs/*.csv
                                                                         ─► dashboard
 new extract ─► update ─► new / modified contracts ─► drift (PSI) ─► retrain? ─► score changes
```

## Stages

| Stage | Command | Output |
|---|---|---|
| Acquire | `ct download --since … --until …` | Raw CSV (only needed, non-personal columns) + `.source.json` (query, time, rows, SHA-256). Also accepted: a portal CSV export, or JSON saved from the API URL (`ct prepare --input file.json`) |
| Official records | `ct records import <source> <file>` | Pseudonymised record keys + manifest (official URL, publisher, SHA-256) |
| Environment | `ct doctor` | Versions vs minimums, folders, key, champion loads & predicts (run after every install/upgrade) |
| Validate | `ct validate` | Schema report (exit code 1 on errors), no files written |
| Prepare | `ct prepare` | `data/processed/contracts.parquet` + `prepare_report.json` |
| Train | `ct train` | New immutable version in `models/registry/<name>/versions/vNNNN/` |
| Promote | automatic, or `ct promote vNNNN` | `champion.json`, `history.jsonl` |
| Score | `ct score [--input file] [--version v]` | `data/outputs/scored_<version>_<ts>.csv` + `.manifest.json` (notice, sources, column dictionary); only rows with an official SECOP link |
| Benchmark | `ct benchmark` | Accuracy, alert precision/recall and cost of each model architecture on the current snapshot (`<outputs>/benchmark.md`) |
| Live check | `ct live-check` | `data/live_check/report.md`: the whole chain verified against the live API in an isolated folder |
| Drift | `ct drift --input new.csv` | PSI per feature; `retrain` status above `drift.psi_retrain` |
| Update | `ct update new.csv` | Merges the snapshot, retrains if needed, scores new and modified contracts |

## What is stored with every model version

`metadata.json` contains:
- **provenance:** config fingerprint, git commit, library versions, user and timestamp;
- **data:** split sizes and date ranges, plus the validation report;
- **metrics:** on the validation and test sets, the baseline on the test set, the *previous champion on the same test set*, and detection precision/recall when labels exist;
- **model state:** residual distribution parameters, top features and the drift profile;
- **integrity:** SHA-256 checksums of every artifact, verified before loading;
- **promotion:** the decision and the reasons for it.

### Files of a model version (no pickles)

```
models/registry/<name>/versions/vNNNN/
  metadata.json                  see above
  segments.json                  routing: natural / juridica / _global (segmented models)
  segment_natural/               one complete model per supplier type ...
  segment_juridica/
  segment__global/               ... plus a global model (unclassified suppliers, small segments)
    model.json                   target transform, feature names, blend weight, config
    preprocessor.json + .npz     one-hot maps, quantity/date features, medians
    text_model.json + .npz       text price model (vocabularies, idf, Ridge weights)
    booster.ubj                  XGBoost (native format)
    linear/                      blended sparse Ridge (same file types)
```

With `model.segment_by: null` the files of a single model sit directly in
`vNNNN/`. Every file is listed with its SHA-256 in `metadata.json` and checked
before loading. Models saved by v1.0 (`preprocessor.joblib`) still load; run
`ct doctor` after upgrades.

## Weak labels from official records

There is no ground truth of contracts confirmed by the authorities. As a first, measurable
proxy, every training run computes the alert rate on test contracts whose
supplier (or its legal representative) appears in an imported official record,
compared with all other contracts (`metrics.weak_labels_test.<source>.lift`).
A lift well above 1 means the model's statistical alerts concentrate where
official findings exist. Being listed is **not** proof that a given contract
is irregular, which is why this is reported, not used for training. Configure
it with `records.weak_label_sources`.

## Verifying against the live sources (`ct live-check`)

Run it on any new machine, after schema changes announced by Colombia Compra
Eficiente, and before the first production scoring. It checks:
- API reachability;
- that the configured columns exist in the live metadata;
- the API field of the contract id (used by `open_data_url`);
- download, cleaning, training, promotion gates and scoring on a fresh sample;
- SECOP-link coverage ≥ 95 %;
- that sampled `open_data_url`s return exactly that contract with the same value.

`--fiscal-bulletin <file>` adds a real Contraloría import.

## Promotion gates (`promotion:` in the config)

A candidate becomes champion only if **all** of these hold:

1. `r2_log >= min_r2_log`.
2. It beats the hierarchical-median baseline by `min_improvement_over_baseline`.
3. It beats the current champion by at least `min_relative_improvement`, with
   both evaluated **on the candidate's test set**, i.e. the most recent contracts.

Set `auto_promote: false` to require a human `ct promote`. Rollback means
promoting an older version; every change goes into `history.jsonl`.

## Retraining triggers (`update`)

* No champion yet.
* New contracts ≥ `drift.min_new_rows_for_retrain`.
* PSI ≥ `drift.psi_retrain` on a monitored feature or on the target.
* `--force-retrain`.

## Scheduling in production

The pipeline is a set of idempotent CLI commands, so any scheduler works.
The simplest example is a monthly cron job:

```cron
# 1st of every month, 03:00: download last month, update, score (Docker)
0 3 1 * *  cd /srv/contract-transparency && \
           uv run ct download --since $(date -d "-1 month" +\%Y-\%m-01) --until $(date +\%Y-\%m-01) --output data/raw/secop_$(date +\%Y\%m).csv && \
           uv run ct update data/raw/secop_$(date +\%Y\%m).csv >> logs/update.log 2>&1
```

For Airflow, Prefect or GitHub Actions, use one task per stage: `validate → prepare → train → score`.
`ct validate` exits non-zero on schema errors, which stops the DAG before a
broken extract reaches the model.

## Experiment tracking

Set `tracking.enabled: true` and install `.[mlflow]`. Every training run then logs
params, metrics (valid, test, baseline, champion, detection), tags (version,
promoted, git commit) and the model files to MLflow (`sqlite:///models/mlflow.db` by
default; run `mlflow ui --backend-store-uri sqlite:///models/mlflow.db`).
The file-based registry stays the source of truth for which model is the champion,
so tracking can be switched off without affecting scoring.

## Packaging & reproducibility

* **uv** is the standard runner: `make install` (= `uv sync`), then
  `uv run ct …` / `make demo`, `make app`. No Docker and no system Python needed.
* `pyproject.toml` declares minimum versions only, and `uv.lock` records a
  known-good set. Upgrades are one command (`make upgrade`) and reversible
  (`make rollback-deps`). See [DEPENDENCIES.md](DEPENDENCIES.md).
* Saved models are **pickle-free** (portable preprocessing + native XGBoost
  booster), so they keep working after library upgrades. `ct doctor` verifies
  it; `make portability-check` proves it.
* Docker is optional: the `Dockerfile` installs `uv.lock` with uv and runs as
  non-root, with data/models/secrets as volumes and a `HEALTHCHECK`.

## CI (`.github/workflows/ci.yml`)

| Job | Purpose |
|---|---|
| `lint` | ruff + `uv lock --check` |
| `test` | Locked versions on Python 3.10 / 3.12 / 3.14 (synthetic data and a mock API; never real SECOP data in CI) |
| `lowest` | Oldest versions allowed by `pyproject.toml` |
| `latest` | Newest releases, with deprecations as errors. Non-blocking, also runs weekly |
| `portability` | Model trained with the oldest libraries == scored with the newest |
| `docker` | Optional real image build, containerised demo, health check |
