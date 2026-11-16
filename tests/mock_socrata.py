"""A small in-process imitation of the datos.gov.co (Socrata SODA 2.x) API.

Implements what the project uses:
  GET /api/views/<id>.json                 -> {"name", "rowsUpdatedAt", "columns": [{name, fieldName, dataTypeName}]}
  GET /resource/<id>.json?$select&$where&$order&$limit&$offset
  GET /resource/<id>.json?<field>=<value>  (simple equality filter)

Realistic quirks reproduced (each can be switched on to test robustness):
* irregular field names (Socrata truncates long names) -> must come from metadata;
* URL columns returned as objects ({"url": ...}); ISO timestamps;
* typed columns (dates as floating_timestamp, value as number) or dates stored
  as TEXT ("MM/DD/YYYY 12:00:00 AM"), where a date filter silently returns nothing;
* the system field ``:updated_at``;
* a blocked metadata endpoint (``metadata_status``);
* duplicated contract ids (``duplicate_ids``);
* injected 503/429 failures to exercise retries.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from contract_transparency.data.download import socrata_field_name

# Irregular field names (Socrata truncates long names) -> must come from metadata.
SPECIAL_FIELDS = {
    "Justificacion Modalidad de Contratacion": "justificacion_modalidad_de",
    "URLProceso": "urlproceso",
}
DATE_COLUMNS = ["Fecha de Firma", "Fecha de Inicio del Contrato", "Fecha de Fin del Contrato"]


class MockSocrata:
    def __init__(
        self,
        contracts: pd.DataFrame,
        dataset_id: str = "jbjy-vk9h",
        fail_first: int = 0,
        fail_code: int = 503,
        date_type: str = "floating_timestamp",
        metadata_status: int = 200,
        duplicate_ids: int = 0,
    ):
        self.dataset_id = dataset_id
        self.date_type = date_type
        self.metadata_status = metadata_status
        if duplicate_ids:  # same contract id published twice (first rows repeated)
            contracts = pd.concat([contracts, contracts.head(duplicate_ids)], ignore_index=True)
        self.fields = {c: SPECIAL_FIELDS.get(c, socrata_field_name(c)) for c in contracts.columns}
        api = contracts.rename(columns=self.fields).astype(object)
        fmt = "%m/%d/%Y 12:00:00 AM" if date_type == "text" else "%Y-%m-%dT00:00:00.000"
        for c in DATE_COLUMNS:
            if c in contracts:
                api[self.fields[c]] = pd.to_datetime(contracts[c]).dt.strftime(fmt)
        signed = pd.to_datetime(contracts["Fecha de Firma"])
        api[":updated_at"] = (signed + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%dT00:00:00.000")
        api[":id"] = [f"row-{i:07d}" for i in range(len(api))]
        self.rows = api
        self.fail_remaining = fail_first
        self.fail_code = fail_code
        self.requests: list[dict] = []
        self._server: ThreadingHTTPServer | None = None

    # --------------------------------------------------------------- server
    def __enter__(self) -> MockSocrata:
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def do_GET(self):
                mock._handle(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    # -------------------------------------------------------------- handling
    def _send(self, h: BaseHTTPRequestHandler, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _type(self, name: str) -> str:
        if name in DATE_COLUMNS:
            return self.date_type
        return {"URLProceso": "url", "Valor del Contrato": "number"}.get(name, "text")

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        url = urlparse(h.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        self.requests.append(
            {"path": url.path, "params": params, "token": h.headers.get("X-App-Token")}
        )
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            return self._send(h, self.fail_code, {"error": "injected failure"})
        if url.path == f"/api/views/{self.dataset_id}.json":
            if self.metadata_status != 200:
                return self._send(h, self.metadata_status, {"error": "metadata unavailable"})
            cols = [
                {"name": name, "fieldName": field, "dataTypeName": self._type(name)}
                for name, field in self.fields.items()
            ]
            return self._send(
                h,
                200,
                {
                    "name": "SECOP II - Contratos Electrónicos (mock)",
                    "rowsUpdatedAt": 1700000000,
                    "columns": cols,
                },
            )
        if url.path == f"/resource/{self.dataset_id}.json":
            try:
                return self._send(h, 200, self._query(params))
            except ValueError as exc:  # Socrata answers 400 for unknown columns
                return self._send(h, 400, {"error": str(exc)})
        return self._send(h, 404, {"error": "not found"})

    def _query(self, p: dict[str, str]) -> list[dict]:
        df = self.rows
        for key, value in p.items():  # simple equality filters
            if not key.startswith("$"):
                df = df[df[key].astype(str) == value]
        if "$where" in p:
            # String comparison, like Socrata on TEXT columns (ISO strings sort correctly).
            for field, op, value in re.findall(r"([:\w]+) (>=|<) '([^']+)'", p["$where"]):
                df = df[df[field] >= value] if op == ">=" else df[df[field] < value]
        if p.get("$order") == ":id":
            df = df.sort_values(":id")
        offset, limit = int(p.get("$offset", 0)), int(p.get("$limit", 1000))
        df = df.iloc[offset : offset + limit]
        fields = (
            p["$select"].split(",")
            if "$select" in p
            else [c for c in df.columns if not c.startswith(":")]
        )
        unknown = [f for f in fields if f not in df.columns]
        if unknown:
            raise ValueError(f"unknown fields requested: {unknown}")
        out = []
        for rec in df[fields].to_dict(orient="records"):
            if rec.get("urlproceso") is not None:
                rec["urlproceso"] = {"url": rec["urlproceso"]}
            out.append({k: v for k, v in rec.items() if v is not None and v == v})
        return out
