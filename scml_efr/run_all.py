"""
one-shot pipeline runner. from the scml_efr dir:
    python run_all.py                                 # defaults
    python run_all.py --iterations 500                # faster solve
    python run_all.py --n-configs 8 --n-steps 20      # bigger bench
    python run_all.py --skip-build --skip-solve       # reuse policy

each sub-script writes its own logs/<name>.latest.log so claude can
read them directly. this orchestrator writes logs/run_all.latest.log
with per-step banners + timing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def _run(name: str, cmd: list[str], env=None) -> None:
    print(f"\n===== [{name}] $ {' '.join(cmd)} =====", flush=True)
    t0 = time.time()
    # inherit stdout/stderr so sub-script progress streams live to terminal
    # AND to the subscript's own _runlog file
    result = subprocess.run(cmd, cwd=HERE, env=env)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[{name}] FAILED after {dt:.1f}s (rc={result.returncode})", flush=True)
        sys.exit(result.returncode)
    print(f"[{name}] done in {dt:.1f}s", flush=True)


def main() -> int:
    from _runlog import start as _start_log
    _start_log("run_all")

    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1000,
                    help="CFR+ iterations in solve.py")
    ap.add_argument("--report-every", type=int, default=200)
    ap.add_argument("--n-configs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=15)
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse existing games/scml_oneshot_v1.efg")
    ap.add_argument("--skip-solve", action="store_true",
                    help="reuse existing policies/scml_oneshot_v1.policy")
    args = ap.parse_args()

    jsonl = HERE / "logs" / "efr_agent.jsonl"
    if jsonl.exists():
        jsonl.unlink()
        print(f"[run_all] removed stale {jsonl}")

    py = sys.executable
    t_all = time.time()

    if not args.skip_build:
        _run("build", [py, "build_game.py"])
    else:
        print("[run_all] skipping build")

    if not args.skip_solve:
        _run("solve", [py, "solve.py",
                       "--iterations", str(args.iterations),
                       "--report-every", str(args.report_every)])
    else:
        print("[run_all] skipping solve")

    env = os.environ.copy()
    env["EFR_LOG_PATH"] = str(jsonl)
    _run(
        "bench",
        [py, "run_benchmark.py",
         "--n-configs", str(args.n_configs),
         "--n-steps", str(args.n_steps),
         "--out", "logs/benchmark.png"],
        env=env,
    )

    _run("summarize", [py, "summarize_logs.py", str(jsonl)])

    print(f"\n===== all done in {time.time()-t_all:.1f}s =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
