#!/usr/bin/env python3
"""
bench.py - run the Part 1 experiments for ./parallel and save the results as CSV.

OS Report 2026, Part 1 - student ID 6610505420. Made for education only.
AI disclosure: written with the help of Claude Code (Anthropic, Claude Opus 5.5).

Usage:
    python3 bench.py --out results            # all experiments (about 30-40 minutes)
    python3 bench.py --out results --quick    # small sizes only, about 1-2 minutes
    python3 bench.py --out results --only scaling,osbind

Experiments (one CSV file each in --out):
    scaling   4 algorithms x n = 100..3200 (6400 for blocked) x threads 1..8, 12, 16
    parinit   blocked, serial vs parallel initialisation (for the Amdahl analysis)
    block     tile size sweep for the blocked algorithm
    pow2      n near powers of two (cache-set conflicts)
    schedule  OpenMP schedule kind and chunk size
    compiler  compiler optimisation flags (-O0 ... -O3 -march=native -ffast-math)
    osbind    OS / runtime settings: OMP_PROC_BIND, OMP_PLACES, OMP_WAIT_POLICY
    affinity  CPU affinity with taskset (fewer CPUs than threads -> oversubscription)
    thp       transparent huge pages (madvise on / off) for large n

Safety: runs one process at a time, never changes system settings (all OS
settings are per-process environment variables, taskset or madvise), and
./parallel itself refuses jobs that need more than 50% of RAM.
"""
import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "parallel.c")
NCPU = os.cpu_count() or 8

THREADS_FULL = sorted(set(list(range(1, NCPU + 1)) + [NCPU + NCPU // 2, 2 * NCPU]))
SIZES = [100, 200, 400, 800, 1600, 3200]


def reps_for(n):
    """Inner repetitions so that small sizes are timed over a long enough interval."""
    if n <= 100: return 200
    if n <= 200: return 50
    if n <= 400: return 20
    if n <= 800: return 5
    if n <= 1600: return 2
    return 1


def job(exp, args, env=None, prefix=None, binary=None, **labels):
    return {"exp": exp, "args": [str(a) for a in args], "env": env or {}, "prefix": prefix or [],
            "binary": binary, "labels": labels}


# --------------------------------------------------------------------------
# Experiment definitions
# --------------------------------------------------------------------------

def exp_scaling(quick):
    jobs = []
    sizes = [100, 200, 400, 800] if quick else SIZES + [6400]
    threads = [1, 2, 4, NCPU] if quick else THREADS_FULL
    for algo in ["naive", "ikj", "transpose", "blocked"]:
        for n in sizes:
            if algo == "naive" and n > 1600: continue        # naive 3200 takes > 40 s per run
            if n == 6400 and algo != "blocked": continue
            ts = [1, 2, 4, NCPU] if n == 6400 else threads
            for t in ts:
                jobs.append(job("scaling", ["-a", algo, "-n", n, "-t", t, "-r", reps_for(n)]))
    return jobs


def exp_parinit(quick):
    jobs = []
    sizes = [100, 400, 800] if quick else SIZES
    for n in sizes:
        for t in range(1, NCPU + 1):
            jobs.append(job("parinit", ["-a", "blocked", "-n", n, "-t", t, "-r", reps_for(n), "-p"]))
    return jobs


def exp_block(quick):
    jobs = []
    sizes = [800] if quick else [1600, 3200]
    blocks = [8, 16, 32, 64, 128] if quick else [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512]
    for n in sizes:
        for bs in blocks:
            for t in [1, NCPU]:
                jobs.append(job("block", ["-a", "blocked", "-n", n, "-b", bs, "-t", t, "-r", reps_for(n)]))
    return jobs


def exp_pow2(quick):
    jobs = []
    sizes = [500, 512, 520] if quick else [1000, 1016, 1024, 1032, 1048, 2000, 2032, 2048, 2064, 2096]
    for algo in ["naive", "ikj", "blocked"]:
        for n in sizes:
            if algo == "naive" and n > 1100: continue
            for t in [1, NCPU]:
                jobs.append(job("pow2", ["-a", algo, "-n", n, "-t", t, "-r", 2]))
    return jobs


def exp_schedule(quick):
    jobs = []
    sizes = [400] if quick else [400, 1600]
    for algo in ["ikj", "blocked"]:
        for n in sizes:
            for t in [NCPU - 1, NCPU]:
                for kind in ["static", "dynamic", "guided"]:
                    for chunk in [0, 1, 4, 16]:
                        jobs.append(job("schedule", ["-a", algo, "-n", n, "-t", t, "-s", kind, "-c", chunk,
                                                     "-r", reps_for(n)]))
    return jobs


COMPILER_VARIANTS = {
    "O0": ["-O0"],
    "O1": ["-O1"],
    "O2": ["-O2"],
    "O3": ["-O3"],
    "O3-native": ["-O3", "-march=native"],
    "O3-native-fastmath": ["-O3", "-march=native", "-ffast-math"],
}


def build_variants(outdir):
    bindir = os.path.join(outdir, "bin")
    os.makedirs(bindir, exist_ok=True)
    paths = {}
    for name, flags in COMPILER_VARIANTS.items():
        exe = os.path.join(bindir, "parallel_" + name)
        cmd = ["gcc", "-std=gnu11", "-fopenmp"] + flags + ["-o", exe, SOURCE, "-lm"]
        subprocess.run(cmd, check=True)
        paths[name] = exe
    return paths


def exp_compiler(quick, variants):
    jobs = []
    for name, exe in variants.items():
        for algo, n in [("naive", 800), ("ikj", 800 if quick else 1600), ("blocked", 800 if quick else 1600)]:
            for t in [1, NCPU]:
                jobs.append(job("compiler", ["-a", algo, "-n", n, "-t", t, "-r", reps_for(n)],
                                binary=exe, variant=name))
    return jobs


def exp_osbind(quick):
    jobs = []
    settings = [
        ("unset", {}),
        ("bind=false", {"OMP_PROC_BIND": "false"}),
        ("bind=close;places=cores", {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}),
        ("bind=spread;places=cores", {"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"}),
        ("bind=master;places=cores", {"OMP_PROC_BIND": "master", "OMP_PLACES": "cores"}),
        ("wait=active", {"OMP_WAIT_POLICY": "active"}),
        ("wait=passive", {"OMP_WAIT_POLICY": "passive"}),
    ]
    sizes = [200, 800] if quick else [200, 800, 3200]
    for label, env in settings:
        for n in sizes:
            for t in [NCPU // 2, NCPU, 2 * NCPU]:
                if label.startswith("bind=master") and n > 800: continue  # all threads on one core: slow
                jobs.append(job("osbind", ["-a", "blocked", "-n", n, "-t", t, "-r", reps_for(n)],
                                env=env, setting=label))
    return jobs


def exp_affinity(quick):
    jobs = []
    if not shutil.which("taskset"):
        print("taskset not found, skipping affinity experiment", file=sys.stderr)
        return jobs
    n = 800 if quick else 1600
    for ncpu in [1, 2, 4, NCPU]:
        cpus = f"0-{ncpu - 1}"
        for t in [1, 2, 4, NCPU, 2 * NCPU]:
            jobs.append(job("affinity", ["-a", "blocked", "-n", n, "-t", t, "-r", reps_for(n)],
                            prefix=["taskset", "-c", cpus], cpus=ncpu))
    return jobs


def exp_thp(quick):
    jobs = []
    sizes = [800, 1600] if quick else [800, 1600, 3200, 6400]
    for mode in ["default", "on", "off"]:
        for algo in ["ikj", "blocked"]:
            for n in sizes:
                if algo == "ikj" and n > 3200: continue
                for t in [1, NCPU]:
                    if n == 6400 and t == 1: continue
                    jobs.append(job("thp", ["-a", algo, "-n", n, "-t", t, "-r", reps_for(n), "--thp", mode]))
    return jobs


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_job(j, default_binary):
    binary = j["binary"] or default_binary
    cmd = j["prefix"] + [binary] + j["args"] + ["--csv"]
    env = dict(os.environ)
    for k in ["OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_NUM_THREADS", "OMP_SCHEDULE"]:
        env.pop(k, None)                      # start every run from a clean OpenMP environment
    env.update(j["env"])
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    return p.stdout.strip().splitlines()[-1]


def save_system_info(outdir, binary):
    lines = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"platform: {platform.platform()}",
             f"python: {platform.python_version()}"]
    for cmd in [["gcc", "--version"], ["lscpu"], ["free", "-h"], ["uname", "-a"],
                ["cat", "/sys/kernel/mm/transparent_hugepage/enabled"],
                ["cat", "/sys/kernel/mm/transparent_hugepage/defrag"]]:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
        except OSError:
            out = "(not available)"
        lines.append("$ " + " ".join(cmd) + "\n" + out)
    with open(os.path.join(outdir, "system.txt"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results", help="output directory for CSV files")
    ap.add_argument("--binary", default=os.path.join(HERE, "parallel"), help="path of the parallel program")
    ap.add_argument("--repeat", type=int, default=3, help="process-level repetitions per configuration")
    ap.add_argument("--quick", action="store_true", help="small sizes only (fast smoke test)")
    ap.add_argument("--only", default="", help="comma separated experiment names to run")
    a = ap.parse_args()

    if not os.path.exists(a.binary):
        sys.exit(f"{a.binary} not found - run 'make' first")
    os.makedirs(a.out, exist_ok=True)
    save_system_info(a.out, a.binary)
    header = subprocess.run([a.binary, "--header"], capture_output=True, text=True).stdout.strip()

    builders = {
        "scaling": lambda: exp_scaling(a.quick),
        "parinit": lambda: exp_parinit(a.quick),
        "block": lambda: exp_block(a.quick),
        "pow2": lambda: exp_pow2(a.quick),
        "schedule": lambda: exp_schedule(a.quick),
        "compiler": lambda: exp_compiler(a.quick, build_variants(a.out)),
        "osbind": lambda: exp_osbind(a.quick),
        "affinity": lambda: exp_affinity(a.quick),
        "thp": lambda: exp_thp(a.quick),
    }
    selected = [s for s in a.only.split(",") if s] or list(builders)
    for name in selected:
        if name not in builders:
            sys.exit(f"unknown experiment '{name}', choose from {', '.join(builders)}")

    repeat = 1 if a.quick else a.repeat
    for name in selected:
        jobs = builders[name]()
        if not jobs: continue
        label_keys = sorted({k for j in jobs for k in j["labels"]})
        path = os.path.join(a.out, f"{name}.csv")
        total = len(jobs) * repeat
        done, t_start = 0, time.time()
        with open(path, "w", newline="") as f:
            f.write(",".join(["exp", "rep"] + label_keys) + "," + header + "\n")
            # The repetition loop is outermost so that a short burst of background
            # activity affects one sample of many configurations, not every sample of one.
            for rep in range(repeat):
                for j in jobs:
                    line = run_job(j, a.binary)
                    labels = [str(j["labels"].get(k, "")).replace(",", ";") for k in label_keys]
                    f.write(",".join([name, str(rep)] + labels) + "," + line + "\n")
                    f.flush()
                    done += 1
                    el = time.time() - t_start
                    print(f"\r[{name}] {done}/{total}  elapsed {el:6.0f}s  eta {el / done * (total - done):6.0f}s ",
                          end="", file=sys.stderr, flush=True)
        print(f"\n[{name}] saved {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
