#!/usr/bin/env python3
"""NovaPay Fraud Detection — single entry point.

Runs the whole system end-to-end from one command: trains the model if
needed, starts the FastAPI scoring service, starts the Streamlit console,
waits for both to become healthy, and shuts them down cleanly on Ctrl+C.

    python run.py                  # train (if needed) + API + frontend
    python run.py train            # just train, write artifacts/, exit
    python run.py api              # just run the API in the foreground
    python run.py frontend         # just run the Streamlit console in the foreground
    python run.py test             # run the full pytest suite
    python run.py all --retrain    # force a fresh training run first
    python run.py all --no-frontend --api-port 9000

This file intentionally contains no modeling, API, or UI logic of its own
-- it only orchestrates the existing entry points (novapay_fraud.train,
uvicorn against api.main:app, streamlit against frontend/app.py) that are
each independently tested. If something here needs to change, the actual
behavior it's orchestrating almost certainly lives elsewhere in the repo.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"

# Make the local src/ package importable without requiring `pip install -e .`
# first -- lowers the bar for "clone and run" to zero setup steps.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _child_env() -> dict:
    """Environment for spawned uvicorn/streamlit subprocesses.

    sys.path.insert() at the top of this file only affects THIS process --
    a subprocess is a fresh Python interpreter that doesn't inherit it. Both
    api/main.py and frontend/app.py do `from novapay_fraud import ...`, so
    without either `pip install -e .` or PYTHONPATH pointing at src/, those
    subprocesses fail with ModuleNotFoundError. This was caught by an actual
    clean-room test (fresh venv, requirements.txt only, no editable
    install) -- the bug didn't show up in earlier testing because that
    session already had the package installed globally.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    return env


def _log(msg: str) -> None:
    print(f"[novapay] {msg}", flush=True)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(_args: argparse.Namespace) -> int:
    from novapay_fraud import train as train_module

    _log("Training pipeline starting (data cleaning -> features -> "
         "4 candidate models -> selection -> SHAP -> artifacts/)...")
    metadata = train_module.run()
    _log(f"Done. Selected model: {metadata['best_model']} "
         f"(recall uplift vs. rules baseline: "
         f"{metadata['recall_uplift_vs_rules_baseline_pct']:.1f}%, "
         f"meets >=15% requirement: {metadata['meets_min_recall_uplift_requirement']})")
    return 0


def _ensure_trained(force: bool) -> None:
    if force:
        _log("--retrain passed: training a fresh model before startup.")
        cmd_train(argparse.Namespace())
        return
    if MODEL_PATH.exists():
        _log(f"Found existing artifacts at {ARTIFACTS_DIR} -- skipping training. "
             f"(use --retrain to force a fresh run)")
        return
    _log("No trained model found -- training before startup.")
    cmd_train(argparse.Namespace())


# ---------------------------------------------------------------------------
# api / frontend (foreground, single-process use)
# ---------------------------------------------------------------------------

def cmd_api(args: argparse.Namespace) -> int:
    if not MODEL_PATH.exists():
        _ensure_trained(force=False)
    _log(f"Starting API on http://{args.host}:{args.api_port} (foreground, Ctrl+C to stop)")
    return subprocess.call([
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", args.host, "--port", str(args.api_port),
    ], cwd=ROOT, env=_child_env())


def cmd_frontend(args: argparse.Namespace) -> int:
    env = _child_env()
    env.setdefault("NOVAPAY_API_URL", f"http://{args.host}:{args.api_port}")
    _log(f"Starting Streamlit console on http://{args.host}:{args.frontend_port} "
         f"(expects the API at {env['NOVAPAY_API_URL']}, foreground, Ctrl+C to stop)")
    return subprocess.call([
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        f"--server.address={args.host}", f"--server.port={args.frontend_port}",
        "--server.headless=true",
    ], cwd=ROOT, env=env)


# ---------------------------------------------------------------------------
# all: train + API + frontend, orchestrated as child processes
# ---------------------------------------------------------------------------

def _wait_for_health(url: str, timeout: float, label: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    _log(f"{label} is healthy ({url})")
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    _log(f"{label} did NOT become healthy within {timeout:.0f}s ({url})")
    return False


def cmd_all(args: argparse.Namespace) -> int:
    _ensure_trained(force=args.retrain)

    procs: list[subprocess.Popen] = []

    def shutdown(*_a):
        _log("Shutting down...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        _log("Stopped.")

    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (shutdown(), sys.exit(0)))

    api_url = f"http://{args.host}:{args.api_port}"
    _log(f"Launching API on {api_url} ...")
    api_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", args.host, "--port", str(args.api_port),
    ], cwd=ROOT, env=_child_env())
    procs.append(api_proc)

    if not _wait_for_health(f"{api_url}/health", timeout=args.startup_timeout, label="API"):
        shutdown()
        return 1

    if args.no_frontend:
        _log(f"API ready. Docs at {api_url}/docs -- frontend skipped (--no-frontend). "
             f"Press Ctrl+C to stop.")
    else:
        env = _child_env()
        env["NOVAPAY_API_URL"] = api_url
        frontend_url = f"http://{args.host}:{args.frontend_port}"
        _log(f"Launching Streamlit console on {frontend_url} ...")
        frontend_proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "frontend/app.py",
            f"--server.address={args.host}", f"--server.port={args.frontend_port}",
            "--server.headless=true",
        ], cwd=ROOT, env=env)
        procs.append(frontend_proc)

        _wait_for_health(f"{frontend_url}/_stcore/health",
                          timeout=args.startup_timeout, label="Frontend")
        _log("")
        _log("=" * 60)
        _log(f"  API:      {api_url}  (docs at {api_url}/docs)")
        _log(f"  Frontend: {frontend_url}")
        _log("=" * 60)
        _log("Press Ctrl+C to stop both.")

    # Block until a child exits unexpectedly or the user hits Ctrl+C.
    try:
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    _log(f"Process (pid={p.pid}) exited unexpectedly with code {ret}.")
                    shutdown()
                    return ret or 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()
    return 0


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def cmd_test(args: argparse.Namespace) -> int:
    pytest_args = ["-q"] if not args.verbose else ["-v"]
    if args.filter:
        pytest_args += ["-k", args.filter]
    _log(f"Running test suite: pytest tests/ {' '.join(pytest_args)}")
    return subprocess.call([sys.executable, "-m", "pytest", "tests/", *pytest_args], cwd=ROOT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="NovaPay Fraud Detection -- single entry point (train / API / frontend / tests).",
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    common.add_argument("--api-port", type=int, default=8000, help="API port (default: 8000)")
    common.add_argument("--frontend-port", type=int, default=8501, help="Frontend port (default: 8501)")

    p_all = sub.add_parser("all", parents=[common], help="Train (if needed) + API + frontend [default]")
    p_all.add_argument("--retrain", action="store_true", help="Force retraining even if artifacts/ already exists")
    p_all.add_argument("--no-frontend", action="store_true", help="Start only the API, not the Streamlit console")
    p_all.add_argument("--startup-timeout", type=float, default=60.0, help="Seconds to wait for each service to become healthy")
    p_all.set_defaults(func=cmd_all)

    p_train = sub.add_parser("train", help="Run the training pipeline and exit")
    p_train.set_defaults(func=cmd_train)

    p_api = sub.add_parser("api", parents=[common], help="Run only the FastAPI service (foreground)")
    p_api.set_defaults(func=cmd_api)

    p_frontend = sub.add_parser("frontend", parents=[common], help="Run only the Streamlit console (foreground)")
    p_frontend.set_defaults(func=cmd_frontend)

    p_test = sub.add_parser("test", help="Run the automated test suite")
    p_test.add_argument("-v", "--verbose", action="store_true")
    p_test.add_argument("-k", "--filter", default=None, help="Only run tests matching this expression")
    p_test.set_defaults(func=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # `python run.py` with no subcommand == `python run.py all`
        args = parser.parse_args(["all", *(argv or [])])

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
