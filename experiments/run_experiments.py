"""
experiment runner with resume support.

tracks completed runs in a json file so interrupted experiments can be
resumed without re-running anything. runs one experiment at a time to
keep laptop temps reasonable.

sweep dimensions: max_turns (MAX_TURNS_LIST), experiment matrix
(EXPERIMENTS), seeds (SEEDS). each (experiment, turns, seed) triple
produces one result file named '{name}_t{turns}_seed{seed}.txt' and one
entry in progress.json's completed list.

each run creates a timestamped directory under experiments/runs/.
re-running the script resumes the latest incomplete run.
pass --new to force a fresh run even if an incomplete one exists.

note: progress.json from before the max_turns sweep was added uses keys
of the form '{name}_seed{seed}' (no _t{turns} segment), which won't
match the new run_key format. start fresh runs with --new.
"""

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

EXECUTABLE = "./open_spiel-private/build.Release/bin/ltbr/run_corr_dist"
RUNS_DIR = Path("experiments/runs")

# game configuration. bargaining_small with header-default 6 instances,
# outcome sampling, exact distance metrics (no mc).
# sample every iteration into the correlation device (SAMPLE_FREQUENCY=1
# is enforced by the binary; see run_corr_dist.cc).
GAME_TEMPLATE = "bargaining_small(max_turns={turns})"
MAX_TURNS_LIST = [2, 3, 4]
ITERATIONS = 1000
# distance calc dominates per-run wall time on max_turns=4 (~9s/calc).
# 20 points along the curve is plenty for log-scale convergence plots,
# so report every 50 iters rather than every 5.
REPORT_INTERVAL = 50
NUM_SAMPLES = 10
SAMPLER = "outcome"
# single seed per (algo, equilibrium, turns) triple.
SEEDS = [0]

# concurrent subprocess workers. default is sequential (1) so the
# console output is one run at a time; pass --workers=N to parallelize.
# each run is single-threaded inside the binary, so on a multi-core mac
# you'll get near-linear speedup up to physical core count.
DEFAULT_WORKERS = 1

# per-run wall-clock budget. matches the latex runtime-risk note: if any
# single seed's run exceeds this, it is recorded as a failure rather than
# silently truncated.
RUN_TIMEOUT_SECONDS = 7200

# experiment matrix: (name, algo, equilibrium)
# matrix matches dealornodeal/latex/main.tex Table tab:planned_matrix.
# only entries supported by the c++ binary are included; EFR_BPS and
# external-regret EFR_CFPS are not in the binary's algo switch.
EXPERIMENTS = [
    # afcce — coarse, blind/external action transformations
    ("afcce_cfr",        "CFR",         "AFCCE"),
    ("afcce_efr_act",    "EFR_ACT",     "AFCCE"),
    # afce — correlated, internal action transformations
    ("afce_cfr_in",      "CFR_in",      "AFCE"),
    ("afce_efr_act_in",  "EFR_ACT_in",  "AFCE"),
    # efcce — coarse, sequence-level external/blind
    ("efcce_cfr",        "CFR",         "EFCCE"),
    ("efcce_efr_csps",   "EFR_CSPS",    "EFCCE"),
    ("efcce_efr_tips",   "EFR_TIPS",    "EFCCE"),
    ("efcce_efr_bhv",    "EFR_BHV",     "EFCCE"),
    # efce — correlated, sequence-level internal
    ("efce_cfr_in",      "CFR_in",      "EFCE"),
    ("efce_efr_cfps_in", "EFR_CFPS_in", "EFCE"),
    ("efce_efr_tips_in", "EFR_TIPS_in", "EFCE"),
    ("efce_efr_bhv_in",  "EFR_BHV_in",  "EFCE"),
    # cce — normal-form coarse correlated equilibrium distance
    ("cce_cfr",          "CFR",         "CCE"),
    ("cce_cfr_in",       "CFR_in",      "CCE"),
    # ce — normal-form correlated equilibrium distance
    ("ce_cfr",           "CFR",         "CE"),
    ("ce_cfr_in",        "CFR_in",      "CE"),
]


def find_latest_incomplete_run():
    """find the most recent run directory that isn't fully complete."""
    if not RUNS_DIR.exists():
        return None
    dirs = sorted(RUNS_DIR.iterdir(), reverse=True)
    for d in dirs:
        if not d.is_dir():
            continue
        progress_file = d / "progress.json"
        if not progress_file.exists():
            return d
        progress = json.loads(progress_file.read_text())
        total = len(EXPERIMENTS) * len(SEEDS) * len(MAX_TURNS_LIST)
        if len(progress.get("completed", [])) < total:
            return d
    return None


def create_new_run(name=None):
    """create a new run directory with config saved.

    if `name` is None, use a timestamp (YYYYMMDD_HHMMSS). otherwise use
    the explicit name. re-creates the directory if it already exists and
    is empty; errors if it exists with content.
    """
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "game_template": GAME_TEMPLATE,
        "max_turns_list": MAX_TURNS_LIST,
        "iterations": ITERATIONS,
        "report_interval": REPORT_INTERVAL,
        "num_samples": NUM_SAMPLES,
        "sampler": SAMPLER,
        "seeds": SEEDS,
        "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
        "experiments": [
            {"name": n, "algo": a, "equilibrium": e} for n, a, e in EXPERIMENTS
        ],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    return run_dir


def load_progress(run_dir):
    progress_file = run_dir / "progress.json"
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"completed": [], "failed": []}


def save_progress(run_dir, progress):
    progress_file = run_dir / "progress.json"
    progress_file.write_text(json.dumps(progress, indent=2) + "\n")


SUMMARY_HEADER = "key,name,algo,equilibrium,max_turns,seed,status,elapsed_s,final_iter,final_dist\n"


def append_summary(run_dir, key, name, algo, eq, turns, seed,
                   status, elapsed_s, final_iter, final_dist):
    """append a one-row csv summary line for a completed (or failed) run.

    creates the file with a header on first call. the file is grow-only;
    a row is appended once per finished run so the cumulative view is
    available even if the runner is interrupted.
    """
    summary_file = run_dir / "summary.csv"
    new_file = not summary_file.exists()
    with open(summary_file, "a") as f:
        if new_file:
            f.write(SUMMARY_HEADER)
        f.write(
            f"{key},{name},{algo},{eq},{turns},{seed},{status},"
            f"{elapsed_s:.2f},{final_iter},{final_dist}\n"
        )


def run_key(name, turns, seed):
    return f"{name}_t{turns}_seed{seed}"


def run_single(run_dir, name, algo, equilibrium, turns, seed,
               abort_flag=None, live_output=False):
    """run one experiment. returns a dict with fields:

      success      bool
      message      human-readable summary string (used in stdout line)
      final_iter   last reported iteration count, or '' if not available
      final_dist   last reported distance value, or '' if not available

    if live_output is True, the binary's stdout/stderr are inherited from
    the parent process — its progress bar writes directly to the terminal
    and you see iterations tick by in real time. only safe when there is
    exactly one run in flight; with parallel workers, multiple progress
    bars overwrite each other on the shared TTY.

    if live_output is False, stdout+stderr are redirected to a per-run
    logfile so concurrent workers don't clobber each other; the logfile
    is removed on success and kept on failure for diagnosis.
    """
    if abort_flag is not None and abort_flag.is_set():
        return {"success": False, "message": "skipped (aborted)",
                "final_iter": "", "final_dist": ""}

    key = run_key(name, turns, seed)
    outfile = run_dir / f"{key}.txt"
    logfile = run_dir / f"{key}.log"
    game = GAME_TEMPLATE.format(turns=turns)

    cmd = [
        EXECUTABLE,
        f"--game={game}",
        f"--algo={algo}",
        f"--equilibrium={equilibrium}",
        f"--output_file={outfile}",
        f"--t={ITERATIONS}",
        f"--report_interval={REPORT_INTERVAL}",
        f"--num_samples={NUM_SAMPLES}",
        f"--sampler={SAMPLER}",
        f"--random_seed={seed}",
    ]

    try:
        if live_output:
            # inherit parent stdout/stderr — progress bar streams to terminal
            result = subprocess.run(cmd, timeout=RUN_TIMEOUT_SECONDS)
        else:
            with open(logfile, "w") as logf:
                result = subprocess.run(
                    cmd,
                    timeout=RUN_TIMEOUT_SECONDS,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
        if result.returncode == 0 and outfile.exists():
            lines = outfile.read_text().strip().split("\n")
            if len(lines) > 1:
                last = lines[-1].split()
                # remove the (now-noisy) per-iter progress log on success
                logfile.unlink(missing_ok=True)
                return {
                    "success": True,
                    "message": f"final: iter={last[0]} dist={last[1]}",
                    "final_iter": last[0],
                    "final_dist": last[1],
                }
        log_hint = "" if live_output else f"; see {logfile.name}"
        return {
            "success": False,
            "message": f"failed (returncode={result.returncode}{log_hint})",
            "final_iter": "", "final_dist": "",
        }
    except subprocess.TimeoutExpired:
        log_hint = "" if live_output else f" (see {logfile.name})"
        return {
            "success": False,
            "message": f"timed out after {RUN_TIMEOUT_SECONDS}s{log_hint}",
            "final_iter": "", "final_dist": "",
        }


def parse_workers_arg(argv):
    """extract --workers=N (default DEFAULT_WORKERS). returns (workers, argv_without_flag)."""
    workers = DEFAULT_WORKERS
    remaining = []
    for a in argv:
        if a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
        else:
            remaining.append(a)
    return workers, remaining


def main():
    # usage:
    #   python3 experiments/run_experiments.py
    #     -> auto-pick latest incomplete run, or create a new timestamped one
    #   python3 experiments/run_experiments.py --new
    #     -> force a new timestamped run dir
    #   python3 experiments/run_experiments.py <name>
    #     -> use runs/<name> as the run dir. creates it if it doesn't exist,
    #        resumes it if it has a progress.json. bypasses the auto-resume
    #        logic that would otherwise pick up a different (alphabetically
    #        later) directory.
    #   --workers=N
    #     -> run up to N experiments concurrently. each is single-threaded
    #        inside the binary. defaults to DEFAULT_WORKERS.
    workers, argv = parse_workers_arg(sys.argv[1:])
    force_new = "--new" in argv
    positional = [a for a in argv if not a.startswith("-")]
    explicit_name = positional[0] if positional else None

    if explicit_name is not None:
        run_dir = RUNS_DIR / explicit_name
        if run_dir.exists() and (run_dir / "progress.json").exists() and not force_new:
            print(f"resuming run: {run_dir.name}")
        else:
            if run_dir.exists() and force_new:
                print(f"--new passed with existing name; overwriting {run_dir.name}")
            run_dir = create_new_run(name=explicit_name)
            print(f"starting run in explicit dir: {run_dir.name}")
    elif force_new:
        run_dir = create_new_run()
        print(f"starting new run: {run_dir.name}")
    else:
        run_dir = find_latest_incomplete_run()
        if run_dir:
            print(f"resuming run: {run_dir.name}")
        else:
            run_dir = create_new_run()
            print(f"starting new run: {run_dir.name}")

    progress = load_progress(run_dir)
    progress_lock = threading.Lock()

    # build full run list. order: turns outermost, then experiment, then seed.
    # the executor processes work in submission order when workers are free,
    # so faster (smaller-turns) runs naturally populate the run dir first.
    all_runs = []
    for turns in MAX_TURNS_LIST:
        for name, algo, eq in EXPERIMENTS:
            for seed in SEEDS:
                all_runs.append((name, algo, eq, turns, seed))

    remaining = [
        r for r in all_runs
        if run_key(r[0], r[3], r[4]) not in progress["completed"]
    ]

    total = len(all_runs)
    done = total - len(remaining)
    completed_count = [done]  # mutable holder for closure update

    print(f"experiment runner")
    print(f"  game template: {GAME_TEMPLATE}")
    print(f"  max_turns: {MAX_TURNS_LIST}")
    print(f"  iterations: {ITERATIONS}, samples: {NUM_SAMPLES} (exact dist)")
    print(f"  report_interval: {REPORT_INTERVAL}")
    print(f"  seeds: {SEEDS}")
    print(f"  workers: {workers}")
    print(f"  results: {run_dir}")
    print(f"  total runs: {total}, completed: {done}, remaining: {len(remaining)}")
    print()

    if not remaining:
        print("all experiments complete!")
        return

    sweep_start = time.time()
    started_count = [done]
    abort_flag = threading.Event()
    # only safe to inherit the binary's tty progress bar when nothing
    # else writes concurrently — i.e. workers == 1.
    live_output = workers == 1

    def run_and_record(spec):
        if abort_flag.is_set():
            return False
        name, algo, eq, turns, seed = spec

        with progress_lock:
            started_count[0] += 1
            sidx = started_count[0]
        # explicit start message so the user can see workers are alive
        # even when each individual run takes minutes.
        print(f"[start {sidx}/{total}] {algo:<14s} on {eq:<5s} t={turns} seed={seed}", flush=True)

        key = run_key(name, turns, seed)
        start = time.time()
        result = run_single(
            run_dir, name, algo, eq, turns, seed,
            abort_flag=abort_flag, live_output=live_output,
        )
        elapsed = time.time() - start
        success = result["success"]

        # if we were aborted mid-run, don't claim a slot in the completed
        # count or write a "FAIL" line — the run was interrupted, not a
        # genuine failure.
        if abort_flag.is_set() and not success:
            return False

        with progress_lock:
            if success:
                progress["completed"].append(key)
            else:
                progress["failed"].append(key)
            save_progress(run_dir, progress)
            append_summary(
                run_dir, key, name, algo, eq, turns, seed,
                "ok" if success else "fail",
                elapsed, result["final_iter"], result["final_dist"],
            )
            completed_count[0] += 1
            cidx = completed_count[0]

        status = "ok  " if success else "FAIL"
        print(f"[done  {cidx}/{total}] {status} {algo:<14s} on {eq:<5s} t={turns} seed={seed} ({elapsed:.0f}s) {result['message']}", flush=True)
        return success

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(run_and_record, spec) for spec in remaining]
    try:
        for _ in as_completed(futures):
            pass
    except KeyboardInterrupt:
        print("\ninterrupted; cancelling pending workers...", flush=True)
        abort_flag.set()
        # cancel queued (not-yet-started) futures so workers free up fast
        for fut in futures:
            fut.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        print(f"saved progress; {completed_count[0]}/{total} done.", flush=True)
        sys.exit(1)
    else:
        executor.shutdown(wait=True)

    sweep_elapsed = time.time() - sweep_start
    print()
    print(f"all {total} experiments complete in {sweep_elapsed/60:.1f} min")
    print(f"results in: {run_dir}")
    generate_plots(run_dir)


def generate_plots(run_dir, quiet=False):
    """run the plotting script on the (possibly partial) run."""
    if not quiet:
        print("\ngenerating convergence plots...")
    try:
        kwargs = {"check": True}
        if quiet:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        subprocess.run(
            [sys.executable, "experiments/plot_results.py", str(run_dir)],
            **kwargs,
        )
    except subprocess.CalledProcessError:
        print("  plot generation failed — run manually:")
        print(f"  python3 experiments/plot_results.py {run_dir}")


if __name__ == "__main__":
    main()
