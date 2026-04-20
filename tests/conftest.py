"""
Pytest configuration for FastAPI tests.

Heavy ML/geospatial libraries (geopandas, torch, lightgbm, …) are NOT required
to run these tests.  Before the app lifespan fires we insert lightweight mock
modules for ``feature_pipeline`` and ``nn_inference`` into ``sys.modules``.
The lifespan then calls our mock constructors instead of the real ones.
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Mock module setup (must happen before 'main' imports happen inside lifespan)
# ---------------------------------------------------------------------------

def _make_mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.assemble.return_value = pd.DataFrame(
        {
            "REGION": [10],
            "segment_code": [5],
            "price_index_current": [1.0],
        }
    )
    pipeline.get_display_info.return_value = {
        "distances": {
            "dist_to_pharmacy_km": 0.4,
            "dist_to_hospital_km": 1.2,
            "dist_to_kindergarten_km": 0.7,
            "dist_to_main_road_km": 0.1,
        },
        "stat": {"srednmes_zarplata": 350000},
    }
    return pipeline


def _make_mock_nn() -> MagicMock:
    nn = MagicMock()
    nn.predict_kzt.return_value = np.array([500_000.0])
    return nn


@pytest.fixture(scope="session", autouse=True)
def _patch_heavy_modules():
    """Insert mock feature_pipeline / nn_inference modules for the whole session."""
    mock_pipeline = _make_mock_pipeline()
    mock_nn = _make_mock_nn()

    fp_mod = MagicMock()
    fp_mod.FeaturePipeline.return_value = mock_pipeline

    nn_mod = MagicMock()
    nn_mod.NNInference.return_value = mock_nn

    sys.modules["feature_pipeline"] = fp_mod
    sys.modules["nn_inference"] = nn_mod

    yield mock_pipeline, mock_nn

    # Cleanup so other test sessions start clean
    sys.modules.pop("feature_pipeline", None)
    sys.modules.pop("nn_inference", None)


# ---------------------------------------------------------------------------
# App / TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client(_patch_heavy_modules):
    """Session-scoped TestClient that runs the app lifespan once."""
    from fastapi.testclient import TestClient
    import main  # noqa: PLC0415 – imported here after sys.modules patching

    with TestClient(main.app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="session")
def mock_pipeline(_patch_heavy_modules):
    pipeline, _ = _patch_heavy_modules
    return pipeline


@pytest.fixture(scope="session")
def mock_nn(_patch_heavy_modules):
    _, nn = _patch_heavy_modules
    return nn
