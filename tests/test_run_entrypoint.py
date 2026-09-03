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

sys.path.insert(0, str(ROOT))
import run as run_module  # noqa: E402  (import after sys.path fix, deliberately)


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


def test_connect_host_translates_bind_any_addresses():
    """Regression test for a real bug: '0.0.0.0' is a valid bind address
    (listen on every interface) but not a valid address to connect *to* on
    Windows -- unlike Linux, which quietly treats a connection to 0.0.0.0
    as a connection to localhost. This silently broke run.py's own health
    check (and any URL it printed) for Windows users, even though the
    server underneath had started and was healthy. _connect_host must
    translate '0.0.0.0'/'::' to a real loopback address for anything that
    connects out, while leaving an explicit host (e.g. a LAN IP) untouched.
    """
    assert run_module._connect_host("0.0.0.0") == "127.0.0.1"
    assert run_module._connect_host("::") == "127.0.0.1"
    assert run_module._connect_host("127.0.0.1") == "127.0.0.1"
    assert run_module._connect_host("localhost") == "localhost"
    assert run_module._connect_host("192.168.1.50") == "192.168.1.50"


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


def _kill_windows_port_owner(port: int) -> None:
    """Find whatever process is actually bound to `port` on Windows and
    kill it directly, by parsing `netstat -ano` -- sidestepping any
    assumption about Windows parent-child process-tree relationships
    entirely.

    Confirmed necessary on this project, not a hypothetical: on a Python
    3.14 venv (this project's newer per-version installer layout), the
    venv's own python.exe is a launcher/relay executable rather than a
    full standalone interpreter copy. A diagnostic session traced a real
    Windows failure to this directly -- run.py's own reported PID (its
    venv python.exe) and the PID actually found LISTENING on the API port
    (a differently-pathed base-install python.exe) were two distinct
    Windows processes, so `taskkill /F /T /PID <run.py's pid>` -- which
    walks Windows' own recorded parent-child bookkeeping -- did not
    reliably reach the one actually serving traffic. Finding the port's
    owner directly and killing that PID is immune to whether that
    bookkeeping chain holds, because it uses the one fact this test
    actually cares about (is anything still listening) as the source of
    truth, rather than an assumption about how it got there.
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return
    for line in result.stdout.splitlines():
        if "LISTENING" in line and f":{port} " in line.replace("\t", " "):
            pid = line.split()[-1]
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)


def test_all_no_frontend_starts_api_and_shuts_down_cleanly():
    """Full process-tree test: launch `run.py all --no-frontend`, confirm
    the API becomes healthy and actually scores a transaction, then
    terminate it and confirm the child process is gone (not orphaned).

    Process creation and cleanup are both platform-specific:
    - os.setsid/os.killpg (creation: preexec_fn=os.setsid; cleanup:
      killpg with the process group id) are POSIX-only and don't exist on
      Windows at all -- an AttributeError, not a graceful no-op.
    - On Windows, cleanup uses `taskkill /F /T /PID` on run.py's own PID
      as a first pass, THEN separately kills whatever is actually bound
      to the API port (see `_kill_windows_port_owner`'s docstring) --
      the tree-kill alone was confirmed insufficient on at least one real
      Windows/Python 3.14 setup where the process tree Windows records
      didn't include the process actually serving traffic.
    """
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(
        [sys.executable, RUN_PY, "all", "--no-frontend",
         "--api-port", "18000", "--startup-timeout", "40"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        **popen_kwargs,
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
        if os.name == "nt":
            # First pass: standard tree-kill by run.py's own PID.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
            # Second pass, unconditional: whatever is actually bound to
            # the API port, regardless of whether Windows' recorded
            # process tree included it (see _kill_windows_port_owner).
            _kill_windows_port_owner(18000)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        else:
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
