#!/usr/bin/env bash
# Train a model with the OLDEST supported library versions and score it with
# the NEWEST ones: predictions, alerts and ranking must be identical.
# Proves that upgrading packages does not invalidate saved models.
#   scripts/check_model_portability.sh [OLD_VENV] [NEW_VENV]   (defaults: .venv-lowest .venv-latest)
set -euo pipefail
cd "$(dirname "$0")/.."
OLD="${1:-.venv-lowest}"; NEW="${2:-.venv-latest}"
for v in "$OLD" "$NEW"; do
  [ -x "$v/bin/ct" ] || { echo "$v missing: run 'make test-lowest' and 'make test-latest' first" >&2; exit 2; }
done
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp -r configs "$WORK/"
ROOT="$PWD"
cd "$WORK"
export CT_PII_SALT=portability-check
"$ROOT/$OLD/bin/ct" demo-data --rows 4000 >/dev/null
"$ROOT/$OLD/bin/ct" run -c configs/demo.yaml >/dev/null 2>&1
"$ROOT/$OLD/bin/ct" score -c configs/demo.yaml --output old.csv >/dev/null 2>&1
"$ROOT/$NEW/bin/ct" score -c configs/demo.yaml --output new.csv >/dev/null 2>&1
"$ROOT/$NEW/bin/python" - "$ROOT/$OLD" "$ROOT/$NEW" <<'PY'
import sys, subprocess
import numpy as np, pandas as pd
def versions(venv):
    code = "import sys,sklearn,numpy,xgboost,pandas;print(sys.version.split()[0],'sklearn',sklearn.__version__,'numpy',numpy.__version__,'xgboost',xgboost.__version__,'pandas',pandas.__version__)"
    return subprocess.run([f"{venv}/bin/python", "-c", code], capture_output=True, text=True).stdout.strip()
a = pd.read_csv("old.csv", sep=";").set_index("ID Contrato")
b = pd.read_csv("new.csv", sep=";").set_index("ID Contrato").loc[a.index]
same_pred = np.allclose(a.predicted_value, b.predicted_value, rtol=1e-9)
same_alerts = (a.sig_price_deviation == b.sig_price_deviation).all()
same_rank = (a.review_priority == b.review_priority).all()
print("trained with:", versions(sys.argv[1]))
print("scored with: ", versions(sys.argv[2]))
print(f"identical predictions: {same_pred} | alerts: {same_alerts} | ranking: {same_rank}")
sys.exit(0 if same_pred and same_alerts and same_rank else 1)
PY
