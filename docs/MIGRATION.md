# Migration guide: legacy code → `contract_transparency`

This document maps every legacy file to its replacement. It also lists the
defects found during the refactor, so nothing important is lost silently.

## 1. File mapping

| Legacy file | Status | New location |
|---|---|---|
| `main.py`, `python3_10/maintry.py` | Replaced | `mlops/pipeline.py` + `cli.py` (`ct run`, `ct train`, `ct score`, `ct update`). The `maintry.py` plan (*load → train or update → run → export*) is exactly the new pipeline. |
| `modules/cleaning.py` · `secop2_general/valor/categoric/date/numeric` | Replaced | `data/io.py` (read once), `data/cleaning.py` (`parse_money`, `parse_dates`, `parse_duration_days`, `clean_contracts`) |
| `cleaning.secop_for_prediction` | Replaced | `ct score --input <file>` / `pipeline.score` |
| `cleaning.new_data` + `new_data_test` | Replaced & extended | `data/incremental.py` (`find_new_contracts`, new `find_modified_contracts`); the legacy test is kept in `tests/test_incremental_drift.py` |
| `models/models_main.py` (`argumentos`) | Replaced | `configs/*.yaml` validated by `config.py` |
| `models/xgboost.py`, `gradientboosting.py`, `xgboost_weights.py`, `wiki40b_regresion_adapt.py`, `trials_enph_geih.py` (same script ×5) | Consolidated | `models/price_model.py` (`PriceModel`) + `features/builder.py` |
| Per-column Keras tokenizers (`trainedmodels\tokenizers\*.json`) | Replaced | `OneHotEncoder` in the saved preprocessor (no separate files to lose) |
| `flagging()` (copied in 4 files) | Kept, vectorised | `scoring/anomaly.py::legacy_flag`; outputs renamed `probability` → `excess_ratio`, `value in danger` → `weighted_excess`. Parity tested against the original. |
| `post_processing/distance_functions.py` | Kept | `scoring/anomaly.py::percentage_error` |
| `post_processing/desviación.py` | Kept & fixed | `within_tolerance` (parity-tested) and `ResidualDistribution` (Student-t on out-of-sample log residuals, tail probability instead of density) |
| `post_processing/post_processing.py` (concat Excel files, strip `\r\n`) | Replaced | `normalize_text`; consolidation isn't needed because scoring writes one file per run |
| `models/rules_based.py` (stubs, did not parse) | Implemented | `scoring/rules.py`: 13 alert signals + `alert_index` (the "ponderador") |
| `models/trials_of_higher_contractors.py` | Replaced | `analysis/contractors.py::top_suppliers` (pseudonymised, universities excluded per the legacy TODO) + `entity_concentration` (HHI) |
| `models/sustainability.py` | Replaced | `analysis/sustainability.py` (lazy model load, batching) + `ct sustainability` |
| `models/baseline.py` (OLS, MLP) | Replaced | `models/baseline.py::MedianBaseline`, now a mandatory promotion gate |
| `models/Transformers.py`, `full.py`, `NN_categorical_embedding.py`, `recurrents.py` | Not ported | Keras text models. The saved weights cannot be reloaded (model-building functions missing). Transformer text features are available through `features.text_embedder: sentence_transformer`. See ROADMAP for fine-tuning. |
| `hugginface_learn.py`, `paraguay_hugging*.py`, `gpt-bert.py`, `wiki40b_example.py`, `language_embedding.py` | Not ported | Exploratory notebooks-as-scripts. The idea (pre-trained language-model embeddings) is covered by `sentence_transformer`. |
| `models/gmm.py` | Not ported | Unsupervised clustering of tokenised categories. See ROADMAP (unsupervised anomaly detection). |
| `models/algo.py` | Removed | Absenteeism-at-work example unrelated to contracts |
| `pipeline_execution.py` (PySpark), `abrir tensorboard.py`, `calling.py`, `testing.py`, `untitled0.py`, `NN_continues_input.py` | Removed | Empty, broken or one-off utilities |

## 2. Defects found in the legacy code

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | `main.py` | Calls functions that don't exist (`transformer_train`, `create_model_tr`, `keep_train`), so the entry point could never run. | New CLI, covered by end-to-end tests. |
| 2 | `main.py` | De-normalisation `x*ssd - mean` has the wrong sign (should be `+ mean`), so every prediction was off by about 327 M COP. | `TargetTransform.inverse` + regression test. |
| 3 | `secop_for_prediction` | `skiprows=[1, subsetsize]` skips only **two rows**, so "prediction" data overlapped the training data. Reported metrics were in-sample. | Temporal train/valid/test split; test asserts no overlap. |
| 4 | all model scripts | Random `train_test_split` on multi-year data, with metrics on the training set or discarded (`r2_score(...)` not assigned). | Temporal split, metrics stored with every model version. |
| 5 | all model scripts | Hard-coded `mean=163740447, ssd=1716771980` z-score. With this heavy-tailed target it squeezes ordinary contracts into a narrow band. | `log1p` target by default; z-score still available and learned from data. |
| 6 | tokenizer loops | `try: ... except: ...` silently **dropped any column** whose tokenizer failed. | Missing columns are logged; unknown categories are handled by the encoder. |
| 7 | `cleaning.secop2_date` | `pd.to_datetime(DataFrame)` raises for these columns. | `parse_dates` (per column, invalid → NaT). |
| 8 | `cleaning` | Values ≥ 1e100 replaced by 1e10; unparseable values crashed. | `parse_money` handles `$1,234`, `1.234,56`, `1234`; out-of-range values are flagged, kept for scoring, excluded from training. |
| 9 | `rules_based.py` | Syntax errors, so the module could not be imported. | Implemented and tested. |
| 10 | `NN_categorical_embedding.py` | Imports `python_3_10` (typo), so it could not be imported. | n/a (not ported). |
| 11 | `xgboost_weights.py` | Uses undefined `ssd`, `mean`, `setsize`, `path_to_result`; reads `.xlsx` with `read_csv`. | n/a (consolidated). |
| 12 | `desviación.py` | Uses a density value `pdf(x) > 0.7` as "in range" (not a probability; depends on units). `distance_distribution` ignores its `ditribution` argument and reads a global. | Tail probability `P(R ≥ r)` from a t fitted on validation residuals. |
| 13 | many scripts | Windows-only paths (`"\model1_tr.h5"`, `r"C:\Users\usuario\..."`) and a personal user path committed. | Config paths with `pathlib`; nothing user-specific in code. |
| 14 | many scripts | Every script saved its model to `modelxgboost.pkl` in the working directory, overwriting the others. | Versioned registry with checksums and an audit log. |
| 15 | repo | 113 MB of binaries and TensorBoard logs in git; `__pycache__` committed. | `.gitignore`, registry outside git, `check-added-large-files` pre-commit hook. |
| 16 | `sustainability.py` | Loading the module downloads a ~1 GB model and requires a GPU (`device=0`); errors replaced by the label `"nada"`. | Lazy load, CPU fallback, per-row `None` for empty text. |
| 17 | privacy | Outputs exported names and supplier IDs of natural persons (e.g. supplier rankings to Excel). | See [DATA_PROTECTION.md](DATA_PROTECTION.md). |

## 3. Behaviour you will notice

* Output column names changed (`probability` → `excess_ratio`,
  `value in danger` → `weighted_excess`, `predicted_valu` → `predicted_value`,
  `real_valu` → `contract_value`). The legacy formulas are unchanged.
* v1.2 renamed the outputs to neutral, review-oriented language:
  `risk_index` → `alert_index`, `risk_rank` → `review_priority`,
  `red_flags` → `signals`, `rf_*` → `sig_*`, `price_alert` → `price_deviation`,
  `overprice_ratio` → `excess_ratio`, `value_at_risk` → `weighted_excess`.
* v1.3 added `person_type` (`natural` / `juridica` / `desconocido`) and
  `model_segment` (the model that estimated the value). `natural_person` is now
  derived from the natural/legal-entity classification (document type, number
  and name), not from the document type alone. Unclassified suppliers are
  shown as `Proveedor sin clasificar p_…`.
* Output CSVs use `;` and UTF-8 with BOM, so Excel in Spanish opens them correctly.
* Supplier IDs are pseudonyms (`p_3fa2…`), and names of natural persons are masked.
* A model is only used for scoring once it has been **promoted**
  (`ct models` shows the champion).
