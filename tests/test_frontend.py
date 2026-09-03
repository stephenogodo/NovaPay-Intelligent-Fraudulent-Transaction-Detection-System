"""Tests for the Streamlit frontend.

Uses Streamlit's headless `AppTest` runner rather than a browser: it
actually executes frontend/app.py's script, exercises widget interactions,
and inspects the rendered element tree, so a broken import, an exception
inside a callback, or a form that silently fails to render all show up as
real test failures.

These tests require a live API (see conftest fixture `live_api`) --
frontend/app.py is a thin client with no scoring logic of its own, so
testing it against a live backend is testing what actually matters: does
the UI correctly call the API and render what comes back.
"""
from __future__ import annotations

import subprocess
import time

from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
API_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def live_api():
    """Start the FastAPI app in a subprocess for the duration of this
    test module, and tear it down afterward. Skips (rather than fails)
    if the API can't come up, since some CI environments may run frontend
    tests separately from the full stack.
    """
    proc = subprocess.Popen(
        ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(30):
            try:
                r = requests.get(f"{API_URL}/health", timeout=1.0)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.5)
        else:
            proc.terminate()
            pytest.skip("API did not become healthy in time")
        yield API_URL
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_app_loads_without_exception_when_api_is_down():
    """Frontend must degrade gracefully, not crash, if the backend is
    unreachable -- this is the state right after `streamlit run` if the
    API hasn't been started yet.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.sidebar.error) == 1
    assert "unreachable" in at.sidebar.error[0].value.lower()


def test_app_loads_and_connects_when_api_is_up(live_api):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.sidebar.success) == 1
    assert "connected" in at.sidebar.success[0].value.lower()
    assert len(at.tabs) == 4


def test_single_transaction_scoring_end_to_end(live_api):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.tabs[0].button[0].click().run()

    assert not at.exception
    metrics = {m.label: m.value for m in at.tabs[0].metric}
    assert "Decision" in metrics
    assert metrics["Model"] == "logistic_regression"

    charts = at.get("plotly_chart")
    assert len(charts) == 2  # gauge + SHAP reasons bar chart


def test_model_info_tab_shows_recall_uplift_requirement(live_api):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    metrics = {m.label: m.value for m in at.tabs[2].metric}
    assert "Recall uplift vs. rules baseline" in metrics
    assert "Meets ≥15% requirement" in metrics
    assert metrics["Meets ≥15% requirement"] == "✅ Yes"


def test_drift_monitoring_tab_runs_without_api(live_api):
    """Drift monitoring imports novapay_fraud.monitoring directly and
    does not depend on the API being reachable for its own computation,
    only for the health indicator in the sidebar.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    drift_tab_index = 3
    at.tabs[drift_tab_index].button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.get("metric")}
    assert "Drifted features" in metrics
    assert "Prediction drift score" in metrics


def test_batch_scoring_serialization_matches_api_contract(live_api):
    """Regression test for the exact records = json.loads(df.to_json(...))
    path frontend/app.py uses to turn an uploaded CSV into API payloads --
    this is what actually broke silently if the API schema and the
    frontend's CSV-to-JSON conversion ever drift apart.
    """
    import json
    import pandas as pd

    from novapay_fraud import config

    ref = pd.read_parquet(config.REFERENCE_DATA_PATH)
    sample = ref.head(5).copy()
    sample.insert(0, "transaction_id", [f"txn_test_{i}" for i in range(5)])
    sample.insert(1, "timestamp", pd.Timestamp.now(tz="UTC").isoformat())
    sample = sample.drop(columns=["fraud_probability", "actual"])

    records = json.loads(sample.to_json(orient="records", date_format="iso"))
    resp = requests.post(f"{live_api}/score/batch", json={"transactions": records}, timeout=30)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 5
