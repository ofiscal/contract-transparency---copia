"""Typed, validated project configuration.

Every path, column name and hyper-parameter that used to be hard-coded in the
legacy scripts (``mean=163740447``, ``subsetsize=500000``, ``r"C:\\Users\\..."``)
now lives in a YAML file under ``configs/`` and is validated here with pydantic.

Usage::

    from contract_transparency.config import load_config
    cfg = load_config("configs/secop_ii.yaml")
    cfg.schema_.target  # -> "Valor del Contrato"
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("configs/secop_ii.yaml")


class _Strict(BaseModel):
    """Base model that rejects unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PathsConfig(_Strict):
    raw_data: Path = Path("data/raw/secop_ii_contratos.csv")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("data/outputs")
    registry_dir: Path = Path("models/registry")
    secrets_dir: Path = Path(".secrets")


class SchemaConfig(_Strict):
    """Maps *raw* dataset column names to the roles the pipeline needs.

    Change this block (not the code) to adapt the pipeline to another
    extraction of SECOP (e.g. SECOP I, the COVID consolidated file, or a
    BigQuery export).
    """

    id: str = "ID Contrato"
    target: str = "Valor del Contrato"
    text: list[str] = Field(default_factory=lambda: ["Descripcion del Proceso"])
    categorical: list[str] = Field(default_factory=list)
    unspsc: str | None = "Codigo de Categoria Principal"
    duration: str | None = "Duración del contrato"
    signed_date: str | None = "Fecha de Firma"
    start_date: str | None = "Fecha de Inicio del Contrato"
    end_date: str | None = "Fecha de Fin del Contrato"
    entity_name: str | None = "Nombre Entidad"
    entity_id: str | None = "Nit Entidad"
    supplier_id: str | None = "Documento Proveedor"
    supplier_name: str | None = "Proveedor Adjudicado"
    supplier_doc_type: str | None = "TipoDocProveedor"
    supplier_group: str | None = "Es Grupo"  # consortium / temporary union (Sí/No)
    legal_rep_id: str | None = "Identificación Representante Legal"
    modality: str | None = "Modalidad de Contratacion"
    modality_justification: str | None = "Justificacion Modalidad de Contratacion"
    entity_order: str | None = "Orden"
    contract_type: str | None = "Tipo de Contrato"
    days_added: str | None = "Dias adicionados"
    n_bidders: str | None = None
    url: str | None = "URLProceso"


class PrivacyConfig(_Strict):
    """Personal-data handling (Ley 1581 de 2012 - Habeas Data)."""

    salt_env_var: str = "CT_PII_SALT"
    # Columns removed as soon as data is loaded. Never needed for modelling.
    drop_columns: list[str] = Field(default_factory=list)
    # Columns replaced by a keyed hash (HMAC-SHA256) so they can still be
    # joined/grouped (e.g. supplier concentration) without exposing the ID.
    pseudonymize_columns: list[str] = Field(default_factory=list)
    # Document types that identify a *natural person* (persona natural).
    natural_person_doc_types: list[str] = Field(
        default_factory=lambda: [
            "Cédula de Ciudadanía",
            "Cédula de Extranjería",
            "Pasaporte",
            "Tarjeta de Identidad",
            "Permiso por Protección Temporal",
        ]
    )
    # Mask ID numbers, e-mails, phones and long numbers in free text.
    redact_free_text: bool = True
    # Never show names of natural persons in outputs / UI.
    mask_natural_person_names: bool = True


class DataConfig(_Strict):
    nrows: int | None = None
    csv_sep: str = ","
    encoding: str = "utf-8"
    # Contracts outside this range are excluded from *training* (they are still
    # scored, and flagged as data-quality suspects).
    min_value: float = 1_000.0
    max_value: float = 1e13
    date_format: str | None = None  # None -> let pandas infer
    dayfirst: bool = False


class FeaturesConfig(_Strict):
    min_category_frequency: int = 20
    max_categories: int | None = 300
    # Extra dense text features for XGBoost. "none" by default: the stacked text
    # price model (below) made TF-IDF/SVD redundant (see docs/MODELING.md).
    text_embedder: Literal["tfidf_svd", "tfidf", "sentence_transformer", "none"] = "none"
    text_max_features: int = 20_000
    text_ngram_max: int = 2
    # Minimum number of contracts a term must appear in (privacy + noise control).
    text_min_df: int = 3
    text_svd_components: int = 128
    # Smaller default (118M params, 384 dims): ~3x cheaper than mpnet-base (278M) for
    # an optional feature; measure alternatives with `ct benchmark`.
    sentence_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Stacked text price model (TF-IDF word + char n-grams -> Ridge, out-of-fold
    # predictions as one XGBoost feature). See features/text_model.py.
    text_price_model: bool = True
    text_price_max_features: int = 100_000  # split between word and char n-grams
    text_price_char_ngram_max: int = 5
    text_price_folds: int = 5
    text_price_alpha: float = 1.0
    # Quantity written in the description ("92 equipos", "3,5 km") as a numeric feature.
    add_text_quantity: bool = True
    add_unspsc_levels: bool = True
    # Supplier person type (natural / juridica / desconocido) as a categorical feature.
    add_person_type: bool = True


class ModelConfig(_Strict):
    name: str = "secop_price_model"
    type: Literal["xgboost", "ridge"] = "xgboost"
    target_transform: Literal["log1p", "zscore", "none"] = "log1p"
    # Only used with target_transform == "zscore" (legacy behaviour). The legacy
    # constants were mean=163740447, std=1716771980; they are now *learned*
    # from the training data when left as null.
    zscore_mean: float | None = None
    zscore_std: float | None = None
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "reg:pseudohubererror",  # same robust loss as legacy models
            "huber_slope": 1.0,
            "eta": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "seed": 24111996,
        }
    )
    num_boost_round: int = 2_000
    early_stopping_rounds: int = 50
    # Blend with a sparse linear model (one-hot + TF-IDF -> Ridge), averaged in log
    # space. "auto" picks the weight in {0, .25, .5, .75} on the validation set;
    # 0 disables the blend. Only used with type "xgboost".
    blend_linear_weight: float | Literal["auto"] = "auto"
    # One model per supplier person type (natural / juridica), plus a global model
    # for unclassified suppliers and segments too small to train (models/segmented.py).
    segment_by: Literal["person_type"] | None = "person_type"
    min_segment_train_rows: int = 1_000
    min_segment_valid_rows: int = 100


class EvaluationConfig(_Strict):
    split: Literal["temporal", "random"] = "temporal"
    test_size: float = 0.2
    valid_size: float = 0.15
    random_state: int = 24111996
    # Optional boolean column with *known* anomalies (audited cases, sanctions,
    # or the synthetic ground truth). When present, alert precision/recall are
    # reported for every model version. Never used as a feature.
    label_column: str | None = None

    @field_validator("test_size", "valid_size")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0 < v < 0.5:
            raise ValueError("split fractions must be in (0, 0.5)")
        return v


class PromotionConfig(_Strict):
    """Quality gates a candidate must pass to become the *champion*."""

    metric: Literal["mae_log", "rmse_log", "mdape"] = "mae_log"
    # Candidate must improve the champion's metric (on the *same* test set) by at
    # least this relative amount. 0.0 -> any improvement.
    min_relative_improvement: float = 0.0
    # Candidate must beat the naive baseline by at least this relative amount.
    min_improvement_over_baseline: float = 0.05
    min_r2_log: float = 0.2
    auto_promote: bool = True


class ScoringConfig(_Strict):
    # "by_group": residual spread calibrated per contract type (see scoring/anomaly.py);
    # "global": one distribution for all contracts (v1.0 behaviour).
    calibration: Literal["global", "by_group"] = "by_group"
    # Column defining the calibration groups (None -> schema.contract_type).
    calibration_column: str | None = None
    # Student-t tail probability under which an over-priced contract is alerted.
    tail_probability_alert: float = 0.01
    # Legacy ``flagging`` replaced non-positive predictions with 1,000,000 COP.
    min_prediction_floor: float = 1_000_000.0
    # Legacy ``distance_proportion`` tolerance (+-80 %).
    tolerance: float = 0.8


class Election(_Strict):
    date: dt.date
    type: Literal["presidential", "congress", "territorial"]
    second_round: dt.date | None = None


def _default_elections() -> list[Election]:
    # VERIFY against the official calendar of the Registraduría before each cycle.
    e = Election
    return [
        e(date=dt.date(2018, 3, 11), type="congress"),
        e(date=dt.date(2018, 5, 27), type="presidential", second_round=dt.date(2018, 6, 17)),
        e(date=dt.date(2019, 10, 27), type="territorial"),
        e(date=dt.date(2022, 3, 13), type="congress"),
        e(date=dt.date(2022, 5, 29), type="presidential", second_round=dt.date(2022, 6, 19)),
        e(date=dt.date(2023, 10, 29), type="territorial"),
        e(date=dt.date(2026, 3, 8), type="congress"),
        e(date=dt.date(2026, 5, 31), type="presidential", second_round=dt.date(2026, 6, 21)),
    ]


class RulesConfig(_Strict):
    """Thresholds of the rule-based alert signals (see scoring/rules.py).

    Ley 996 de 2005 (Ley de Garantías), simplified:
      * art. 33 - no *direct contracting* by any entity during the 4 months
        before a **presidential** election (until the second round);
      * art. 38 par. - territorial entities cannot sign *convenios
        interadministrativos* during the 4 months before **any** election.
    Both have legal exceptions, so these are flags, not violations.
    """

    enabled: bool = True
    elections: list[Election] = Field(default_factory=_default_elections)
    restriction_months: int = 4
    pre_restriction_window_days: int = 30
    direct_modality_patterns: list[str] = Field(
        default_factory=lambda: ["contratación directa", "contratacion directa"]
    )
    interadministrative_patterns: list[str] = Field(default_factory=lambda: ["interadministrativ"])
    association_patterns: list[str] = Field(
        default_factory=lambda: ["convenio de asociaci", "convenios de asociaci"]
    )
    territorial_order_values: list[str] = Field(default_factory=lambda: ["Territorial"])
    services_type_patterns: list[str] = Field(
        default_factory=lambda: ["prestación de servicios", "prestacion de servicios"]
    )
    concurrent_contracts_threshold: int = 4
    entity_spending_share_threshold: float = 0.3
    supplier_share_threshold: float = 0.5
    shared_legal_rep_threshold: int = 2
    addition_regex: str = r"\b(?:adici[oó]n|otros[ií]|otro\s+s[ií]|pr[oó]rroga|modificatorio)\b"
    days_added_ratio_threshold: float = 0.5
    splitting_window_days: int = 30
    splitting_min_contracts: int = 3
    duration_robust_z_threshold: float = 3.5
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "price_deviation": 3.0,
            "pre_ley_garantias_rush": 1.0,
            "direct_during_ley_garantias": 2.0,
            "interadministrative_during_ley_garantias": 2.0,
            "concurrent_direct_services": 1.0,
            "mentions_addition": 0.5,
            "large_days_added": 1.0,
            "large_share_of_entity_spending": 1.0,
            "dominant_supplier": 1.0,
            "shared_legal_representative": 1.5,
            "possible_contract_splitting": 1.5,
            "atypical_duration": 0.5,
            "single_bidder": 1.5,
            "data_quality_suspect": 0.0,
            "supplier_in_fiscal_bulletin": 2.0,
            "supplier_previous_sanction": 1.0,
        }
    )


class DriftConfig(_Strict):
    psi_warning: float = 0.1
    psi_retrain: float = 0.25
    min_new_rows_for_retrain: int = 5_000
    monitored_features: list[str] | None = None  # None -> categorical + target


class TrackingConfig(_Strict):
    enabled: bool = False
    tracking_uri: str | None = None  # e.g. "sqlite:///models/mlflow.db" or a server URL
    # Where run artifacts are stored for NEW experiments (default: inside models/).
    artifact_location: str | None = "models/mlruns"
    experiment: str = "contract-transparency"


class OutputConfig(_Strict):
    # Columns (raw names) copied to the scored output in addition to the
    # model/rule columns. Keep this list free of personal data.
    passthrough_columns: list[str] = Field(default_factory=list)
    format: Literal["csv", "parquet", "xlsx"] = "csv"
    csv_sep: str = ";"


NOTICE_ES = (
    "AVISO: Este resultado contiene alertas estadísticas generadas automáticamente a partir de "
    "información pública (SECOP II en datos.gov.co y, cuando se usan, registros públicos "
    "oficiales). Una alerta solo indica que un contrato tiene características poco comunes frente "
    "a contratos comparables y que podría ser útil revisarlo. No constituye prueba, acusación, "
    "juicio ni señalamiento de irregularidad o responsabilidad de ninguna entidad, contratista o "
    "persona, y no reemplaza la verificación en la fuente oficial ni la labor de las autoridades "
    "competentes. Los autores no emiten conclusiones sobre los contratos ni asumen "
    "responsabilidad por las decisiones que terceros tomen con base en esta información, que "
    "puede contener errores propios de las fuentes."
)
NOTICE_EN = (
    "NOTICE: This output contains statistical alerts generated automatically from public "
    "information (SECOP II on datos.gov.co and, when used, official public records). An alert "
    "only indicates that a contract has unusual characteristics compared with similar contracts "
    "and that reviewing it could be useful. It is not evidence, accusation, judgement or "
    "indication of wrongdoing or liability of any entity, contractor or person, and it does not "
    "replace verification at the official source or the work of the competent authorities. The "
    "authors draw no conclusions about any contract and accept no responsibility for decisions "
    "third parties make based on this information, which may contain errors from the sources."
)
ROW_NOTICE = (
    "Alerta estadística para revisión; no implica irregularidad ni responsabilidad. "
    "Verifique en SECOP."
)


class PublicDataConfig(_Strict):
    """Public-data policy: every result must be traceable to a public, official source."""

    dataset_name: str = "SECOP II - Contratos Electrónicos"
    publisher: str = "Agencia Nacional de Contratación Pública - Colombia Compra Eficiente"
    dataset_id: str = "jbjy-vk9h"
    portal_url: str = (
        "https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h"
    )
    api_base: str = "https://www.datos.gov.co"
    # API field name of `schema.id`, used to build a direct link to each record
    # in the open-data API (`ct live-check` verifies it against the live API).
    id_api_field: str = "id_contrato"
    # Display name -> API field name, for fields whose API name is NOT the
    # standard Socrata transformation of the display name (Socrata truncates
    # long names). Used when reading JSON saved straight from the API URL;
    # `ct download` reads the real names from the live metadata instead.
    # `ct live-check` reports any field that differs.
    api_field_overrides: dict[str, str] = Field(
        default_factory=lambda: {
            "Justificacion Modalidad de Contratacion": "justificacion_modalidad_de",
        }
    )
    # Rows without a valid link to an allowed official domain cannot be audited
    # by a third party and are left out of scored outputs (and counted in the manifest).
    require_secop_url: bool = True
    allowed_url_domains: list[str] = Field(
        default_factory=lambda: ["secop.gov.co", "colombiacompra.gov.co"]
    )
    add_row_notice: bool = True
    # True for synthetic/demo data: outputs and dashboard are watermarked.
    synthetic: bool = False
    notice_es: str = NOTICE_ES
    notice_en: str = NOTICE_EN
    row_notice: str = ROW_NOTICE


class RecordSourceConfig(_Strict):
    """An official public record (e.g. Contraloría bulletin) matched against suppliers."""

    title: str
    publisher: str
    official_url: str
    legal_basis: str | None = None
    flag: str  # name of the alert signal (sig_<flag>) raised on matching contracts
    id_column: str | None = None  # None -> auto-detect
    date_column: str | None = None  # None -> auto-detect (optional)
    match_on: list[Literal["supplier", "legal_rep"]] = Field(
        default_factory=lambda: ["supplier", "legal_rep"]
    )
    # If the record has a date, only flag contracts signed on/after it.
    only_after_record_date: bool = True


def _default_record_sources() -> dict[str, RecordSourceConfig]:
    return {
        "contraloria_responsables_fiscales": RecordSourceConfig(
            title="Boletín de Responsables Fiscales",
            publisher="Contraloría General de la República",
            official_url="https://www.contraloria.gov.co",
            legal_basis=(
                "Ley 610 de 2000, art. 60: las entidades deben abstenerse de celebrar contratos "
                "con quienes aparezcan en el boletín de responsables fiscales."
            ),
            flag="supplier_in_fiscal_bulletin",
        ),
        "secop_multas_sanciones": RecordSourceConfig(
            title="SECOP - Multas y sanciones",
            publisher="Agencia Nacional de Contratación Pública - Colombia Compra Eficiente",
            official_url="https://www.datos.gov.co",
            flag="supplier_previous_sanction",
            match_on=["supplier"],
        ),
    }


class RecordsConfig(_Strict):
    enabled: bool = True
    sources: dict[str, RecordSourceConfig] = Field(default_factory=_default_record_sources)
    # Record sources used as *weak labels* when evaluating each model version
    # (alert rate among matched suppliers vs. the rest).
    weak_label_sources: list[str] = Field(
        default_factory=lambda: ["contraloria_responsables_fiscales"]
    )


class DownloadConfig(_Strict):
    app_token_env_var: str = "CT_SOCRATA_APP_TOKEN"
    page_size: int = 50_000
    timeout_s: float = 120.0
    max_retries: int = 5
    backoff_s: float = 2.0  # first retry wait; doubles each attempt


class Settings(_Strict):
    project: str = "contract-transparency"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    public_data: PublicDataConfig = Field(default_factory=PublicDataConfig)
    records: RecordsConfig = Field(default_factory=RecordsConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)

    @model_validator(mode="after")
    def _no_pii_in_model_inputs(self) -> Settings:
        """Personal identifiers must never be used as model features."""
        features = set(self.schema_.categorical) | set(self.schema_.text)
        pii = set(self.privacy.drop_columns) | set(self.privacy.pseudonymize_columns)
        leaked = sorted(features & pii)
        if leaked:
            raise ValueError(f"PII columns configured as model features: {leaked}")
        leaked_out = sorted(set(self.output.passthrough_columns) & set(self.privacy.drop_columns))
        if leaked_out:
            raise ValueError(f"Dropped PII columns configured as output columns: {leaked_out}")
        return self

    def fingerprint(self) -> str:
        """Stable short hash of the configuration (stored with every model)."""
        payload = json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> Settings:
    """Load a YAML config, apply dotted ``overrides`` and validate it.

    Relative paths inside the config are resolved against the current working
    directory (run the CLI from the repository root).
    """
    data: dict[str, Any] = _read_yaml(Path(path)) if path is not None else {}
    for dotted, value in (overrides or {}).items():
        _set_dotted(data, dotted, value)
    return Settings.model_validate(data)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML, resolving ``extends: other.yaml`` (relative to the file)."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _read_yaml((path.parent / parent).resolve())
        data = _deep_merge(base, data)
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def with_overrides(cfg: Settings, overrides: dict[str, Any]) -> Settings:
    """Copy of ``cfg`` with dotted overrides, e.g. {"model.type": "ridge"}."""
    data = cfg.model_dump(mode="json", by_alias=True)
    for dotted, value in overrides.items():
        _set_dotted(data, dotted, value)
    return Settings.model_validate(data)


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
