# Modelling: expected value, language model and alert calibration

This document explains how the price-deviation signal is produced, why this
architecture was chosen (accuracy **and** cost), what it can and cannot
detect, and what must be verified on real data.

> All numbers below come from the **synthetic** SECOP-like generator
> (`ct demo-data`). Its descriptions carry information on service level, scope
> and quantity; noise depends on the contract type; prices include 9 %/year
> inflation; and 2 % of contracts are overpriced by a log-uniform factor
> between ×1.8 and ×8. Synthetic results show that the method *works as
> designed*. They do not measure real-world performance: run `ct benchmark`
> on a real extract for that (see the last section).

## 1. What the model does

1. Estimate the value a contract **would typically have** given its
   characteristics (expected value, `predicted_value`).
2. Compare it with the published value on the log scale (`log_residual`).
3. Decide how unusual that difference is **for comparable contracts**
   (`tail_probability`), and raise a `sig_price_deviation` signal when it is
   very unusual and the published value is *above* the estimate.

A signal means "unusual compared with similar contracts, worth reviewing".
Legitimate explanations are common: special specifications, remote locations,
urgency, or errors in the source data.

## 2. Architecture

```
supplier document type + number + name ──► natural | juridica | desconocido ──► one model per type
                                                                                 (global model for the rest)
inside each model:
description ──► word 1-2-gram + char 2-5-gram TF-IDF ──► Ridge ──(out-of-fold)──┐
description ──► quantity written in the text ("92 equipos", "3,5 km", "2.500 galones")
categorical (type, modality, department, city, sector, …), UNSPSC segment/family/class,
duration, signing date ───────────────────────────────────────────────────────┴─► XGBoost
one-hot + sparse TF-IDF ──► Ridge (linear model) ─────────────────────► blend (weight on validation)
residuals on validation ─► divided by the robust spread of each contract type ─► Student-t
```

| Component | Module | Why |
|---|---|---|
| Target = log(1 + value), pseudo-Huber loss | `models/target.py`, `models/price_model.py` | Values span 8 orders of magnitude; a robust loss limits the influence of the very anomalies we want to find |
| **Text price model** (stacked) | `features/text_model.py` | Descriptions carry price information not present in the categorical columns. A linear model on sparse word and character n-grams learns one weight per n-gram ("especializado", "mantenimiento rutinario"), is robust to misspellings and inflections, and trains in seconds. Training rows receive **out-of-fold** predictions (K = 5), so the booster never sees a text prediction fitted on its own target |
| **Quantity feature** | `features/builder.py::text_quantity` | TF-IDF sees "9" and "92" as unrelated tokens; price scales with quantity |
| **Amounts written in words are masked** | `data/privacy.py` | "por valor de 12 millones" would let a text model echo the price (target leakage) and hide exactly the contracts we want to find |
| XGBoost | `models/price_model.py` | Standard for tabular data at this scale; native, forward-compatible model files |
| **Blend with a sparse Ridge** | `models/price_model.py::_fit_blend` | Two model families make different errors, so averaging improves accuracy and alert precision. The weight is chosen on validation data from {0, .25, .5, .75}, so it turns itself off when it does not help |
| **Per-type calibration** | `scoring/anomaly.py::ResidualDistribution` | Normalised ("Mondrian") conformal-style calibration: each residual is divided by the robust spread (MAD) of its contract type, shrunk toward the global spread for small groups. Public works vary far more than service fees; one global threshold would over-alert the former and miss deviations in the latter |
| Student-t tail probability, alert if P < 0.01 and value above expected | `scoring/anomaly.py` | Heavy-tailed residuals; the threshold is the precision/recall knob (§5) |

All artifacts are pickle-free, so they survive library upgrades (see [DEPENDENCIES.md](DEPENDENCIES.md)).

## 3. Evidence

### 3.1 Architecture comparison (`ct benchmark`, 20,000 contracts, 13,000 for training)

Measured with v1.2, before the person-type split (see §3.4 for v1.3). The
"+ SVD on top" variant was dropped from the benchmark afterwards because it
added cost without gain.

| Variant | MAE log | MdAPE | R² log | Alert precision | Alert recall | Training | Scoring / 1k | Model |
|---|---|---|---|---|---|---|---|---|
| Median of similar contracts (baseline) | 0.677 | 0.492 | 0.48 | | | | | |
| XGBoost, no text | 0.541 | 0.336 | 0.59 | 0.82 | 0.58 | 1 s | 12 ms | 1.1 MB |
| **v1.1** (TF-IDF → SVD, global calibration) | 0.370 | 0.204 | 0.78 | 0.88 | **0.09** | 7 s | 56 ms | 2.4 MB |
| v1.1 features + per-type calibration | 0.370 | 0.204 | 0.78 | 0.74 | 0.68 | 7 s | 56 ms | 2.4 MB |
| + stacked text price model | 0.271 | 0.180 | 0.89 | 0.81 | 0.76 | 8 s | 195 ms | 1.7 MB |
| + quantity feature | 0.246 | 0.168 | 0.91 | 0.83 | 0.78 | 10 s | 251 ms | 1.7 MB |
| + SVD on top (redundant) | 0.252 | 0.175 | 0.90 | 0.78 | 0.76 | 14 s | 200 ms | 3.2 MB |
| Sparse Ridge alone | 0.266 | 0.190 | 0.90 | 0.91 | 0.76 | 2 s | 55 ms | 0.1 MB |
| **v1.2 default** (text model + quantity + blend, per-type calibration) | **0.234** | **0.158** | **0.92** | 0.86 | 0.78 | 11 s | 248 ms | 1.8 MB |

Timings are from one run on a shared 4-CPU machine and vary by ±50 % between
runs. Compare them within a column, not across machines.

Reading the table:
* The **language model was v1.1's weak point**: the stacked text model
  plus quantity extraction removes a third of the error.
* **v1.1's alerting was far less sensitive than it looked.** With the
  original, trivial ×6–×30 anomalies it seemed perfect. With realistic ×1.8–×8
  overpricing, its single global threshold found only 9 %: the mixture of
  low- and high-variance contract types produced very heavy fitted tails.
  Per-type calibration is the biggest single improvement in detection.
* SVD becomes redundant once the text model exists (more cost, no gain), so
  it is off by default (`features.text_embedder: none`).

### 3.2 Is XGBoost still the right learner? (same features, 3 random datasets)

| Learner | MAE log (mean ± sd) | Note |
|---|---|---|
| XGBoost | 0.252 ± 0.003 | current |
| LightGBM | 0.249 ± 0.003 | |
| CatBoost (RMSE) | 0.247 ± 0.002 | needs `boost_from_average`; a Huber loss without it diverged |
| CatBoost (MAE) | 0.254 ± 0.003 | |
| Sparse Ridge | 0.275 ± 0.003 | |
| **XGBoost + Ridge blend** | **0.246 ± 0.002** | alert precision 0.88 vs 0.84 for XGBoost alone, same recall |

The three gradient-boosting libraries are equivalent within noise, which is
consistent with the literature (§4). Switching libraries would add
dependencies and change the model format for no measurable gain. Text
representation, calibration and model blending matter far more.

### 3.3 What can be detected (5 random datasets, 4 % overpriced contracts)

| Overpricing factor | Share detected |
|---|---|
| ×1.8 – ×2.5 | 25 % |
| ×2.5 – ×3.5 | 63 % |
| ×3.5 – ×5 | 80 % |
| ×5 – ×8 | 88 % |

| Contract type (noise of normal prices) | Share detected |
|---|---|
| Prestación de servicios (low) | 85 % |
| Arrendamiento, Compraventa, Suministros, Interventoría | 46 – 66 % |
| Consultoría, Obra (high) | 15 – 21 % |

False alerts on clean contracts stayed below 0.1 % in every contract type.
The limit is statistical: if normal public-works prices vary ±57 % (one
standard deviation) for the same characteristics, a ×2 price is not unusual
*from the characteristics alone*. Detecting it requires more information, such
as quantities and unit prices (roadmap X3: reference prices per UNSPSC class).

### 3.4 Natural persons vs legal entities (v1.3)

**Classification** (`data/person_type.py`). Each supplier becomes `natural`,
`juridica` or `desconocido` using only published information. Rules, in
order of reliability (the deciding rule is kept in `ct_person_type_rule`):

| # | Rule | Example |
|---|---|---|
| 1 | Natural-person document type (CC, CE, passport, PPT…) | `Cédula de Ciudadanía` → natural |
| 2 | Consortium / temporary union (`Es Grupo` = Sí) | → jurídica |
| 3 | Legal form in the name (S.A.S., LTDA, S.A., fundación, consorcio, E.S.P.…) | `Constructora Andina S.A.S.` → jurídica |
| 4 | Document number, Colombian conventions | `900123456`, `800197268-4`, `8001972684` (valid DIAN check digit) → jurídica; `52123456` (old series), `1012345678` (NUIP), cédula + check digit → natural |
| 5 | Document type `NIT` with no other evidence | → jurídica |

A 9-digit number starting with 8 whose last digit is a valid check digit of
its first 8 digits could be a company NIT *or* an old cédula (80 million
range) + check digit. With a name that shows no legal form, it is left as
`desconocido` rather than guessed. Names are masked unless the supplier is
positively a legal entity, so undetermined suppliers are treated like people.
The check-digit algorithm is verified against published NITs (DIAN,
Ecopetrol, Bancolombia, Universidad Nacional). On the synthetic data
(20,000 contracts): **0 natural/legal-entity confusions**, 0.8 %
undetermined. On real data, `ct prepare` reports the share of each type and
`person_type_number_rule_agreement`: the share of suppliers with a cédula
document type whose *number alone* is also classified as natural.

**Segmented model** (`models/segmented.py`, `model.segment_by: person_type`).
One complete model per type: its own text price model (word weights),
trees, blend weight and early stopping. A global model handles
`desconocido` and any segment with fewer than `min_segment_train_rows`.
Calibration groups become *person type × contract type*. The dashboard shows
which model estimated each contract (`model_segment`).

Evidence (4 random datasets, 20,000 contracts each, 4 % overpriced; MAE on
normal contracts):

| Variant | MAE all | MAE natural | MAE legal entities | Precision | Recall | Training |
|---|---|---|---|---|---|---|
| v1.2 (no person type) | 0.241 | 0.185 | 0.316 | 0.99 | 0.56 | 26 s |
| One model + person type as a feature | 0.213 | 0.174 | 0.267 | 0.99 | 0.60 | 17 s |
| **Segmented (default)** | **0.212** | **0.171** | 0.269 | 0.99 | **0.60** | 46 s |

Std over datasets: ±0.005 MAE, ±0.03 recall.

* Knowing the person type helps clearly: **−12 % error** (−15 % for legal
  entities) and **+4 points** of recall.
* On this synthetic data, separate models and a single model with the
  person-type feature **tie** (differences below one standard deviation), and
  segmentation costs ~2.7× the training time. The generator gives both person
  types the *same* description templates. In real SECOP, individuals
  ("prestación de servicios profesionales como…") and companies ("suministro,
  instalación y puesta en funcionamiento…") write very differently, which is
  where separate text models should pay off. Segmentation is therefore the
  default. Confirm on real data with
  `ct benchmark --variants v1_2,v1_2_person_type,segmented`, and switch to
  `model.segment_by: null` (half the cost) if they still tie.

## 4. State of the art, and why this design (accuracy vs. cost)

| Option | Size / cost | Assessment for this project |
|---|---|---|
| **Gradient-boosted trees** (XGBoost / LightGBM / CatBoost) | MBs; CPU; minutes | Still the reference for medium-to-large tabular data (Grinsztajn et al., NeurIPS 2022; Shwartz-Ziv & Armon, 2022). **Used.** |
| Tabular foundation models (TabPFN v2, Hollmann et al., *Nature* 2025) | Pre-trained transformer; best on small tables (≈ ≤ 10k rows); inference cost grows with the training-set size | Attractive for small subgroups (a single entity or UNSPSC class), not for the full SECOP history. Possible future experiment |
| **Sparse linear text models** (TF-IDF word + char → Ridge) | ~100k weights; CPU; seconds | Very strong for *price from text*. Linear and sparse models were central to the winning Mercari Price Suggestion Challenge solutions (Kaggle, 2018). **Used (stacked + blended).** |
| Multilingual sentence embeddings (e.g. `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small`, ~118M parameters; `…-mpnet-base-v2`, ~278M) | Hundreds of MB; CPU-feasible but orders of magnitude slower than TF-IDF; GPU recommended for millions of texts | Captures synonyms and paraphrases. Available as an option (`features.text_embedder: sentence_transformer`, `uv sync --extra nlp`); the default model was changed to the smaller MiniLM. **Not benchmarked here** (model hub unreachable from the build environment): run `ct benchmark --variants xgb_sentence_embeddings,segmented` |
| Fine-tuned Spanish transformers (BETO, Cañete et al. 2020; RoBERTa-BNE / MarIA, Gutiérrez-Fandiño et al. 2022) | ~110–125M parameters; GPU training | Best for nuanced Spanish semantics. Worth it only if embeddings beat TF-IDF on real data *and* labelled outcomes exist |
| Large language models (API or local) | Per-call cost or large GPUs; slow for millions of rows | Not suited to bulk numeric regression. Good future use: extracting quantities and units from messy descriptions for the top alerts, or drafting the "why" explanation for a reviewer |
| Calibration: split / normalised / Mondrian conformal prediction (Vovk et al., 2005; Lei et al., 2018) | Negligible | **Used** in its per-group normalised form |

Design principle: spend complexity where the measured gain is (text and
calibration), keep everything CPU-only and pickle-free, and let
`ct benchmark` decide on real data whether heavier text models pay off.

## 5. Choosing the alert threshold (`scoring.tail_probability_alert`)

Measured over 3 random datasets (2 % overpriced):

| Threshold | Alerts per 1,000 contracts | Precision | Recall | False-alert rate |
|---|---|---|---|---|
| 0.001 | 8.9 | 1.00 | 0.41 | 0.00 % |
| 0.005 | 15.5 | 0.96 | 0.68 | 0.07 % |
| **0.01 (default)** | 19.2 | 0.87 | 0.77 | 0.26 % |
| 0.02 | 28.2 | 0.66 | 0.85 | 1.0 % |
| 0.05 | 59.3 | 0.35 | 0.96 | 3.9 % |
| 0.10 | 115.8 | 0.18 | 0.97 | 9.7 % |

For a tool whose alerts trigger human review of named entities, the default
favours precision. Raise it only if reviewers have capacity and accept more
alerts without a finding.

## 6. Verify on real data (before trusting any number above)

1. `uv run ct download --since <one year ago>` then `uv run ct prepare`.
2. `uv run ct benchmark`: compare `segmented` (default) and `v1_2` against `xgb_tfidf_svd` (v1.1)
   and `ridge_sparse` on **real** descriptions. Add `xgb_sentence_embeddings`
   if you can install the `nlp` extra.
3. Check per-type calibration: the `false_alert_rate_*` columns need labels.
   Without them, look at the alert rate per contract type in the dashboard
   ("Distribuciones"). Large differences deserve a look.
4. Import the Contraloría bulletin and read `weak_labels_test.lift` in
   `ct models` / the registry metadata: alerts should concentrate on listed
   suppliers (lift > 1).
5. Known gaps on real data:
   * **inflation** (roadmap N4): the signing year is a feature, but the most
     recent year lies outside the training range;
   * **quantities** in real descriptions are messier than the heuristic
     assumes; check how often `f_log_text_quantity` is non-empty;
   * **unit prices** would make public-works alerts far more sensitive (X3).

## References

* Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
* Ke et al. (2017). *LightGBM.* NeurIPS. Prokhorenkova et al. (2018). *CatBoost.* NeurIPS.
* Grinsztajn, Oyallon & Varoquaux (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS Datasets & Benchmarks.
* Shwartz-Ziv & Armon (2022). *Tabular data: Deep learning is not all you need.* Information Fusion.
* Hollmann et al. (2025). *Accurate predictions on small data with a tabular foundation model.* Nature (TabPFN v2).
* Kaggle (2018). *Mercari Price Suggestion Challenge*: price prediction from product text.
* Reimers & Gurevych (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.* EMNLP.
* Wang et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv:2402.05672.
* Cañete et al. (2020). *Spanish Pre-Trained BERT Model and Evaluation Data* (BETO). PML4DC @ ICLR.
* Gutiérrez-Fandiño et al. (2022). *MarIA: Spanish Language Models.* Procesamiento del Lenguaje Natural.
* Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World.* Springer.
* Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). *Distribution-Free Predictive Inference for Regression.* JASA.
