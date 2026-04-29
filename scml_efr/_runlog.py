"""
tiny helper: mirror stdout/stderr to predictable log files so the user
doesn't have to paste terminal output. claude can Read() the latest log.

usage (inside any entrypoint):
    from _runlog import start
    start("solve")          # writes to logs/solve.latest.log

behavior:
    - every on-disk line is timestamped; terminal output stays clean
    - logs/{name}.latest.log   — overwritten each run (easy to tail)
    - logs/{name}.{stamp}.log  — per-run archive (nothing lost)
"""

from __future__ import annotations

import atexit
import sys
import time
from pathlib import Path

_LOGS_DIR = Path(__file__).parent / "logs"


class _MultiTee:
    """write-forwards to terminal + N log files, timestamping each disk line."""

    def __init__(self, terminal, *files) -> None:
        self._term = terminal
        self._files = files
        self._at_line_start = True

    def write(self, data: str) -> int:
        self._term.write(data)
        for ch in data:
            if self._at_line_start:
                ts = time.strftime("%H:%M:%S")
                for f in self._files:
                    f.write(f"[{ts}] ")
                self._at_line_start = False
            for f in self._files:
                f.write(ch)
            if ch == "\n":
                self._at_line_start = True
        return len(data)

    def flush(self) -> None:
        self._term.flush()
        for f in self._files:
            f.flush()

    def isatty(self) -> bool:
        return self._term.isatty()

    def fileno(self):
        return self._term.fileno()


def start(name: str) -> Path:
    """begin mirroring stdout+stderr to logs/{name}.latest.log (and a
    timestamped archive). returns the rolling-latest Path."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    latest = _LOGS_DIR / f"{name}.latest.log"
    stamped = _LOGS_DIR / f"{name}.{time.strftime('%Y%m%d-%H%M%S')}.log"

    fh_latest = open(latest, "w", buffering=1)
    fh_stamped = open(stamped, "w", buffering=1)

    header = (
        f"# {name} run started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# cwd={Path.cwd()}\n"
        f"# argv={sys.argv}\n"
    )
    fh_latest.write(header)
    fh_stamped.write(header)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _MultiTee(orig_stdout, fh_latest, fh_stamped)
    sys.stderr = _MultiTee(orig_stderr, fh_latest, fh_stamped)

    def _close():
        # restore stdout/stderr BEFORE closing the log files so that any
        # atexit-triggered flush by the interpreter doesn't hit closed fds
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        for f in (fh_latest, fh_stamped):
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    atexit.register(_close)
    print(f"[runlog] mirroring output to {latest}")
    return latest
