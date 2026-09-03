"""Tests for run.py -- the single end-to-end entry point.

These exercise run.py as a subprocess (the way a user actually invokes it),
not by importing its functions, since the whole point of this file is
process orchestration (spawning uvicorn/streamlit, health-polling,
signal-based shutdown) -- behavior that only shows up when it's actually
run as a process tree.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
RUN_PY = str(ROOT / "run.py")
API_URL = "http://localhost:18000"
FRONTEND_URL = "http://localhost:18501"


def _wait_for(url: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def test_help_runs_and_lists_subcommands():
    result = subprocess.run(
        [sys.executable, RUN_PY, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    for sub in ["all", "train", "api", "frontend", "test"]:
        assert sub in result.stdout


def test_train_subcommand_produces_artifacts():
    result = subprocess.run(
        [sys.executable, RUN_PY, "train"], cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "Selected model" in result.stdout
    assert (ROOT / "artifacts" / "fraud_model.joblib").exists()


def test_all_no_frontend_starts_api_and_shuts_down_cleanly():
    """Full process-tree test: launch `run.py all --no-frontend`, confirm
    the API becomes healthy and actually scores a transaction, then send
    SIGTERM and confirm the child process is gone (not orphaned).
    """
    proc = subprocess.Popen(
        [sys.executable, RUN_PY, "all", "--no-frontend",
         "--api-port", "18000", "--startup-timeout", "40"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        preexec_fn=os.setsid,
    )
    try:
        assert _wait_for(f"{API_URL}/health", timeout=45.0), "API never became healthy"

        health = requests.get(f"{API_URL}/health", timeout=5).json()
        assert health["model_loaded"] is True

        import json
        with open(ROOT / "sample_transaction.json") as f:
            sample = json.load(f)
        score_resp = requests.post(f"{API_URL}/score", json=sample, timeout=10)
        assert score_resp.status_code == 200
        assert 0.0 <= score_resp.json()["fraud_probability"] <= 1.0
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)

    # give the child uvicorn a moment to actually release the port
    time.sleep(1.0)
    with pytest.raises(requests.exceptions.RequestException):
        requests.get(f"{API_URL}/health", timeout=2.0)
