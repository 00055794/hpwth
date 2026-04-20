"""
Tests for the FastAPI application (main.py).

Covers:
  GET  /           – index HTML page
  GET  /health     – health-check endpoint
  POST /predict    – single-property prediction
  POST /batch      – CSV / XLSX batch prediction
  POST /batch/download/xlsx – download predictions as XLSX
  GET  /template/csv  – download blank CSV template
  GET  /template/xlsx – download blank XLSX template
"""
import io

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        body = client.get("/health").json()
        assert "status" in body
        assert "model_loaded" in body

    def test_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_model_loaded_true(self, client):
        """After lifespan the model should be flagged as loaded."""
        body = client.get("/health").json()
        assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndex:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /predict – valid input
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "ROOMS": 2,
    "LONGITUDE": 76.9286,
    "LATITUDE": 43.2567,
    "TOTAL_AREA": 65.0,
    "FLOOR": 5,
    "TOTAL_FLOORS": 12,
    "FURNITURE": 3,
    "CONDITION": 5,
    "MATERIAL": 3,
    "YEAR": 2015,
}


class TestPredict:
    def test_returns_200_for_valid_input(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_success_flag(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body.get("success") is True

    def test_response_has_price_fields(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "price_kzt" in body
        assert "price_per_sqm" in body

    def test_price_kzt_positive(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["price_kzt"] > 0

    def test_response_has_region_and_segment(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "region_grid" in body
        assert "segment_code" in body

    def test_response_has_distances_and_stat(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "distances" in body
        assert "stat" in body

    def test_pipeline_assemble_called(self, client, mock_pipeline):
        mock_pipeline.assemble.reset_mock()
        client.post("/predict", json=VALID_PAYLOAD)
        mock_pipeline.assemble.assert_called_once()

    def test_nn_predict_called(self, client, mock_nn):
        mock_nn.predict_kzt.reset_mock()
        client.post("/predict", json=VALID_PAYLOAD)
        mock_nn.predict_kzt.assert_called_once()


class TestPredictValidation:
    """Pydantic validation errors should return HTTP 422."""

    def test_missing_required_field(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "ROOMS"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_rooms_below_min(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "ROOMS": 0}).status_code == 422

    def test_rooms_above_max(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "ROOMS": 21}).status_code == 422

    def test_longitude_out_of_range(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "LONGITUDE": 10.0}).status_code == 422

    def test_latitude_out_of_range(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "LATITUDE": 10.0}).status_code == 422

    def test_total_area_zero(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "TOTAL_AREA": 0}).status_code == 422

    def test_condition_out_of_range(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "CONDITION": 6}).status_code == 422

    def test_material_out_of_range(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "MATERIAL": 5}).status_code == 422

    def test_furniture_out_of_range(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "FURNITURE": 4}).status_code == 422


class TestPredictModelNotLoaded:
    """When the model globals are None the endpoint must return 503."""

    def test_503_when_pipeline_none(self, client):
        import main

        original_pipeline, original_nn = main._pipeline, main._nn
        try:
            main._pipeline = None
            resp = client.post("/predict", json=VALID_PAYLOAD)
            assert resp.status_code == 503
        finally:
            main._pipeline = original_pipeline
            main._nn = original_nn


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------

BATCH_COLS = [
    "ROOMS", "LATITUDE", "LONGITUDE", "TOTAL_AREA", "FLOOR",
    "TOTAL_FLOORS", "FURNITURE", "CONDITION", "MATERIAL", "YEAR",
]


def _make_csv_bytes(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


def _make_valid_rows(n: int = 3) -> list[dict]:
    return [
        {
            "ROOMS": 2, "LATITUDE": 43.2567, "LONGITUDE": 76.9286,
            "TOTAL_AREA": 65.0, "FLOOR": 5, "TOTAL_FLOORS": 12,
            "FURNITURE": 3, "CONDITION": 5, "MATERIAL": 3, "YEAR": 2015,
        }
    ] * n


class TestBatch:
    def test_returns_200_for_valid_csv(self, client):
        csv_bytes = _make_csv_bytes(_make_valid_rows())
        resp = client.post(
            "/batch",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200

    def test_returns_list(self, client):
        csv_bytes = _make_csv_bytes(_make_valid_rows(2))
        body = client.post(
            "/batch",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        ).json()
        assert isinstance(body, list)

    def test_result_has_prediction_columns(self, client):
        csv_bytes = _make_csv_bytes(_make_valid_rows(1))
        body = client.post(
            "/batch",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        ).json()
        assert len(body) == 1
        assert "pred_price_per_sqm" in body[0]
        assert "pred_price_kzt" in body[0]

    def test_400_on_missing_columns(self, client):
        df = pd.DataFrame({"ROOMS": [2], "LATITUDE": [43.2]})
        csv_bytes = df.to_csv(index=False).encode()
        resp = client.post(
            "/batch",
            files={"file": ("bad.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 400

    def test_400_on_unparseable_file(self, client):
        resp = client.post(
            "/batch",
            files={"file": ("garbage.csv", b"\x00\x01\x02", "text/csv")},
        )
        assert resp.status_code in (400, 500)


# ---------------------------------------------------------------------------
# POST /batch/download/xlsx
# ---------------------------------------------------------------------------

class TestBatchDownloadXlsx:
    def test_returns_200(self, client):
        rows = [
            {**row, "pred_price_per_sqm": 500000.0, "pred_price_kzt": 32500000.0}
            for row in _make_valid_rows(2)
        ]
        resp = client.post("/batch/download/xlsx", json=rows)
        assert resp.status_code == 200

    def test_content_type_xlsx(self, client):
        rows = [
            {**row, "pred_price_per_sqm": 500000.0, "pred_price_kzt": 32500000.0}
            for row in _make_valid_rows(1)
        ]
        resp = client.post("/batch/download/xlsx", json=rows)
        assert "spreadsheetml" in resp.headers["content-type"]

    def test_xlsx_is_readable(self, client):
        rows = [
            {**row, "pred_price_per_sqm": 500000.0, "pred_price_kzt": 32500000.0}
            for row in _make_valid_rows(2)
        ]
        resp = client.post("/batch/download/xlsx", json=rows)
        df = pd.read_excel(io.BytesIO(resp.content))
        assert len(df) == 2
        assert "pred_price_kzt" in df.columns


# ---------------------------------------------------------------------------
# GET /template/csv
# ---------------------------------------------------------------------------

class TestTemplateCsv:
    def test_returns_200(self, client):
        assert client.get("/template/csv").status_code == 200

    def test_content_type_csv(self, client):
        resp = client.get("/template/csv")
        assert "text/csv" in resp.headers["content-type"]

    def test_has_required_header_columns(self, client):
        resp = client.get("/template/csv")
        header_line = resp.text.splitlines()[0]
        for col in BATCH_COLS:
            assert col in header_line, f"Column '{col}' missing from CSV template header"

    def test_content_disposition_attachment(self, client):
        resp = client.get("/template/csv")
        assert "attachment" in resp.headers.get("content-disposition", "")


# ---------------------------------------------------------------------------
# GET /template/xlsx
# ---------------------------------------------------------------------------

class TestTemplateXlsx:
    def test_returns_200(self, client):
        assert client.get("/template/xlsx").status_code == 200

    def test_content_type_xlsx(self, client):
        resp = client.get("/template/xlsx")
        assert "spreadsheetml" in resp.headers["content-type"]

    def test_xlsx_has_data_sheet(self, client):
        resp = client.get("/template/xlsx")
        xl = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data")
        for col in BATCH_COLS:
            assert col in xl.columns, f"Column '{col}' missing from template XLSX Data sheet"

    def test_content_disposition_attachment(self, client):
        resp = client.get("/template/xlsx")
        assert "attachment" in resp.headers.get("content-disposition", "")
