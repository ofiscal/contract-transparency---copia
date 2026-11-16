"""Canonical (internal) column names created by the cleaning step.

Raw dataset columns keep their original names; everything the pipeline derives
is prefixed with ``ct_`` so it can never collide with a SECOP column.
"""

ID = "ct_id"
VALUE = "ct_value"  # contract value in COP (float)
VALUE_OUT_OF_RANGE = "ct_value_out_of_range"
SIGNED_AT = "ct_signed_at"
START_AT = "ct_start_at"
END_AT = "ct_end_at"
DURATION_DAYS = "ct_duration_days"
TEXT = "ct_text"  # combined + redacted description
UNSPSC_SEGMENT = "ct_unspsc_segment"
UNSPSC_FAMILY = "ct_unspsc_family"
UNSPSC_CLASS = "ct_unspsc_class"
NATURAL_PERSON = "ct_is_natural_person"
PERSON_TYPE = "ct_person_type"  # natural | juridica | desconocido (data/person_type.py)
PERSON_TYPE_RULE = "ct_person_type_rule"  # which rule decided (auditable)
SUPPLIER = "ct_supplier"  # pseudonymised supplier id
SUPPLIER_DISPLAY = "ct_supplier_display"  # company name or "Persona natural p_..."
ENTITY = "ct_entity"  # entity id (NIT) or name
LEGAL_REP = "ct_legal_rep"  # pseudonymised legal representative id

# Model / scoring outputs
PREDICTED = "predicted_value"
LOG_RESIDUAL = "log_residual"
