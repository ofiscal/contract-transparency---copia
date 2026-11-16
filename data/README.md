# data/

Local working area, **never committed** (see `.gitignore`): SECOP extracts and
official records contain personal data.

| Folder | Content |
|---|---|
| `raw/` | Raw extracts: `ct download` output (+ `<file>.source.json` provenance) or portal exports; official record files (e.g. Contraloría bulletin) |
| `processed/` | Cleaned, pseudonymised snapshot (`contracts.parquet`, `prepare_report.json`); `records/` holds pseudonymised official records + manifests |
| `outputs/` | Scored files (`scored_<version>_<timestamp>.csv`) + `<file>.manifest.json` (notice, sources, column dictionary) |
| `live_check/` | Isolated workspace and report of `ct live-check` |

Get the data:
* `ct download --since 2024-01-01` (datos.gov.co API), or
* <https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h>
  (Export → CSV), saved as `raw/secop_ii_contratos.csv`;
* `ct demo-data --with-records` for synthetic data.
