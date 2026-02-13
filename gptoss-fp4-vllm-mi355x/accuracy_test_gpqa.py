#!/usr/bin/env python3
"""
GPQA accuracy test script for vLLM OpenAI-compatible server.

This follows the approach in:
  competition_vllm/vllm/tests/evals/gpt_oss/test_gpqa_correctness.py

It runs:
  python -m gpt_oss.evals --eval gpqa ...

and parses the stdout for:
  'metric': <float>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Optional, Tuple


def check_server_health(base_url: str) -> bool:
    try:
        import requests
    except Exception:
        # Avoid hard dependency; health check is a convenience.
        return True

    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_v1_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def try_import_gpt_oss() -> bool:
    try:
        import gpt_oss  # noqa: F401

        return True
    except Exception:
        return False


def run_gpqa_eval(
    model_name: str,
    base_url_v1: str,
    reasoning_effort: str,
    n_threads: int,
    timeout_s: int,
) -> Tuple[Optional[float], str]:
    cmd = [
        sys.executable,
        "-m",
        "gpt_oss.evals",
        "--eval",
        "gpqa",
        "--model",
        model_name,
        "--reasoning-effort",
        reasoning_effort,
        "--base-url",
        base_url_v1,
        "--n-threads",
        str(n_threads),
    ]

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "dummy")

    # Use errors='replace' so that invalid UTF-8 from child (e.g. model output)
    # does not raise UnicodeDecodeError; we only need to parse 'metric': <float>.
    p = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=env,
        encoding="utf-8",
        errors="replace",
    )

    stdout = p.stdout or ""
    stderr = p.stderr or ""
    combined = stdout + ("\n" + stderr if stderr else "")

    if p.returncode != 0:
        return None, combined

    m = re.search(r"'metric':\s*([\d.]+)", stdout)
    if not m:
        return None, combined

    try:
        return float(m.group(1)), combined
    except Exception:
        return None, combined


def main() -> int:
    parser = argparse.ArgumentParser(description="GPQA accuracy test for vLLM")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://0.0.0.0:8888",
        help="Base URL of vLLM server (e.g. http://0.0.0.0:8888)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name used in OpenAI requests (must match server model)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=os.environ.get("REASONING_EFFORT", "low"),
        help="Reasoning effort passed to gpt_oss.evals (default: low)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=int(os.environ.get("GPQA_N_THREADS", "200")),
        help="Number of threads used by evaluator (default: 200)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("GPQA_TIMEOUT_S", "1800")),
        help="Timeout in seconds (default: 1800)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GPQA Accuracy Test (gpt_oss.evals)")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Base URL (v1): {ensure_v1_base_url(args.base_url)}")
    print(f"Model: {args.model}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"N threads: {args.n_threads}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 60)
    print()

    if not try_import_gpt_oss():
        print("ERROR: Python module 'gpt_oss' is not installed.", file=sys.stderr)
        print("INFO: Install it with:", file=sys.stderr)
        print("  git clone https://github.com/openai/gpt-oss.git", file=sys.stderr)
        print("  cd gpt-oss && pip install -e .", file=sys.stderr)
        return 1

    # Optional health check (best-effort)
    print("INFO: Checking server health...")
    if not check_server_health(args.base_url):
        print(f"ERROR: vLLM server is not responding at {args.base_url}/health", file=sys.stderr)
        return 1
    print("SUCCESS: Server is healthy")
    print()

    base_url_v1 = ensure_v1_base_url(args.base_url)
    metric, output = run_gpqa_eval(
        model_name=args.model,
        base_url_v1=base_url_v1,
        reasoning_effort=args.reasoning_effort,
        n_threads=args.n_threads,
        timeout_s=args.timeout,
    )

    print("Evaluation process output:\n", output)

    if metric is None:
        print("ERROR: Failed to obtain GPQA metric from evaluator output.", file=sys.stderr)
        return 1

    print()
    print("=" * 60)
    print("Accuracy Metrics")
    print("=" * 60)
    print(f"gpqa_metric: {metric:.4f}")
    print("=" * 60)

    # Format expected by the C++ benchmark parser
    print()
    print("Metrics for parsing:")
    print(f"  'gpqa_metric': {metric}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

