#!/usr/bin/env python3
"""
plot.py - draw the graphs and compute the Amdahl's-law analysis from bench.py results.

OS Report 2026, Part 1 - student ID 6610505420. Made for education only.
AI disclosure: written with the help of Claude Code (Anthropic, Claude Opus 5.5).

Usage:  python3 plot.py --data results          (needs python3-matplotlib and numpy)
Output: results/plots/*.png and results/summary.md (tables of the numbers behind the graphs)

Every point is the median of the process-level repetitions made by bench.py.

Amdahl's law: with a serial fraction s, the speedup on p threads is
    S(p) = 1 / (s + (1 - s) / p)
Rearranged, 1/S - 1/p = s * (1 - 1/p), so s is fitted by least squares through
the origin with x = 1 - 1/p and y = 1/S - 1/p (only p <= number of cores).
The Karp-Flatt metric e(p) = (1/S - 1/p) / (1 - 1/p) is the same quantity
computed separately for each p; if e grows with p the loss is overhead
(synchronisation, memory bandwidth, clock frequency), not a fixed serial part.
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---- colours (fixed order; a colour always means the same thing) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALGO_COLOR = {"blocked": SLOTS[0], "ikj": SLOTS[1], "transpose": SLOTS[2], "naive": SLOTS[3]}
ALGO_ORDER = ["blocked", "ikj", "transpose", "naive"]
BLUE_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]

NUMERIC = {"rep", "cpus", "n", "threads", "chunk", "block", "par_init", "reps", "t_alloc", "t_spawn", "t_init",
           "t_compute_min", "t_compute_avg", "t_checksum", "t_total", "gflops", "mem_theory_mb", "vm_hwm_mb",
           "vm_peak_mb", "anon_huge_mb", "minflt", "majflt", "nvcsw", "nivcsw", "checksum", "verify_err",
           "verify_ok", "t_verify"}
MEASURES = ["t_alloc", "t_spawn", "t_init", "t_compute_min", "t_compute_avg", "t_checksum", "t_total", "gflops",
            "mem_theory_mb", "vm_hwm_mb", "vm_peak_mb", "anon_huge_mb", "minflt", "majflt", "nvcsw", "nivcsw",
            "verify_ok"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2, "lines.markersize": 5,
        "font.size": 9.5, "figure.dpi": 150, "axes.axisbelow": True, "text.color": INK,
    })


def line(ax, x, y, color, label=None, marker="o", **kw):
    ax.plot(x, y, color=color, marker=marker, markeredgecolor=SURFACE, markeredgewidth=1.2,
            solid_capstyle="round", solid_joinstyle="round", label=label, **kw)


def save(fig, out, name, tight=True):
    path = os.path.join(out, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load(data_dir, name):
    path = os.path.join(data_dir, name + ".csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in NUMERIC and v not in ("", None):
                r[k] = float(v)
    return rows


def agg(rows, keys):
    """Median of every measure over the repetitions of the same configuration."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for key, rs in groups.items():
        d = dict(zip(keys, key))
        for m in MEASURES:
            vals = [r[m] for r in rs if isinstance(r.get(m), float)]
            d[m] = statistics.median(vals) if vals else float("nan")
        d["samples"] = len(rs)
        d["cv_total"] = (statistics.pstdev([r["t_total"] for r in rs]) / d["t_total"]) if len(rs) > 1 else 0.0
        out.append(d)
    return out


def pick(rows, **cond):
    return [r for r in rows if all(r.get(k) == v for k, v in cond.items())]


def series(rows, x, y):
    rows = sorted(rows, key=lambda r: r[x])
    return [r[x] for r in rows], [r[y] for r in rows]


def amdahl_fit(ps, speedups):
    """Least-squares serial fraction s from (p, S) pairs, p > 1."""
    x = np.array([1 - 1 / p for p in ps if p > 1])
    y = np.array([1 / s - 1 / p for p, s in zip(ps, speedups) if p > 1])
    if len(x) == 0:
        return float("nan"), float("nan")
    s = float(np.dot(x, y) / np.dot(x, x))
    s = min(max(s, 0.0), 1.0)
    pred = 1 / (s + (1 - s) / np.array([p for p in ps if p > 1]))
    meas = np.array([sp for p, sp in zip(ps, speedups) if p > 1])
    ss_res = float(np.sum((meas - pred) ** 2))
    ss_tot = float(np.sum((meas - meas.mean()) ** 2)) or 1e-12
    return s, 1 - ss_res / ss_tot


def amdahl_curve(s, p):
    p = np.asarray(p, dtype=float)
    return 1 / (s + (1 - s) / p)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_scaling(sc, out, ncpu, md):
    rows = pick(sc, algo="blocked", par_init=0.0)
    sizes = sorted({r["n"] for r in rows})
    ramp = BLUE_RAMP[-len(sizes):] if len(sizes) <= len(BLUE_RAMP) else BLUE_RAMP

    # time vs threads
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, c in zip(sizes, ramp):
        x, y = series(pick(rows, n=n), "threads", "t_total")
        line(ax, x, y, c, f"n = {int(n)}")
    ax.set_yscale("log")
    ax.set_xlabel("threads")
    ax.set_ylabel("total program time (s, log scale)")
    ax.set_title("Blocked algorithm: run time vs threads")
    ax.axvline(ncpu, color=AXIS, lw=1)
    ax.text(ncpu + 0.2, ax.get_ylim()[1], f"{ncpu} cores", color=MUTED, fontsize=8, va="top")
    ax.legend(ncol=2)
    save(fig, out, "01_time_vs_threads.png")

    # speedup: whole program and compute only
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, field, title in [(axes[0], "t_total", "whole program (alloc + init + compute + checksum)"),
                             (axes[1], "t_compute_min", "multiplication kernel only")]:
        for n, c in zip(sizes, ramp):
            rs = sorted(pick(rows, n=n), key=lambda r: r["threads"])
            base = next((r[field] for r in rs if r["threads"] == 1), None)
            if not base:
                continue
            line(ax, [r["threads"] for r in rs], [base / r[field] for r in rs], c, f"n = {int(n)}")
        maxp = max(r["threads"] for r in rows)
        ax.plot([1, ncpu], [1, ncpu], color=MUTED, lw=1, label="ideal (linear)")
        ax.axvline(ncpu, color=AXIS, lw=1)
        ax.set_xlabel("threads")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.5, maxp + 0.5)
    axes[0].set_ylabel("speedup  T(1) / T(p)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle("Speedup of the blocked algorithm", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "02_speedup.png", tight=False)

    # efficiency
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, c in zip(sizes, ramp):
        rs = sorted(pick(rows, n=n), key=lambda r: r["threads"])
        base = next((r["t_total"] for r in rs if r["threads"] == 1), None)
        if base:
            line(ax, [r["threads"] for r in rs], [base / r["t_total"] / r["threads"] for r in rs], c, f"n = {int(n)}")
    ax.axvline(ncpu, color=AXIS, lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("threads")
    ax.set_ylabel("parallel efficiency  S(p) / p")
    ax.set_title("Blocked algorithm: efficiency")
    ax.legend(ncol=2)
    save(fig, out, "03_efficiency.png")

    # table
    md.append("## Blocked algorithm, serial init: median total time (s) / speedup\n")
    threads = sorted({r["threads"] for r in rows})
    md.append("| n | " + " | ".join(f"p={int(t)}" for t in threads) + " |")
    md.append("|---|" + "---|" * len(threads))
    for n in sizes:
        rs = {r["threads"]: r for r in pick(rows, n=n)}
        base = rs.get(1.0, {}).get("t_total")
        cells = []
        for t in threads:
            r = rs.get(t)
            cells.append(f"{r['t_total']:.4g} / {base / r['t_total']:.2f}x" if r and base else "-")
        md.append(f"| {int(n)} | " + " | ".join(cells) + " |")
    md.append("")


def fig_algorithms(sc, out, ncpu, md):
    rows = pick(sc, par_init=0.0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, t in [(axes[0], 1.0), (axes[1], float(ncpu))]:
        for a in ALGO_ORDER:
            x, y = series(pick(rows, algo=a, threads=t), "n", "gflops")
            if x:
                line(ax, x, y, ALGO_COLOR[a], a)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["n"] for r in rows}))
        ax.set_xticklabels([str(int(v)) for v in sorted({r["n"] for r in rows})])
        ax.set_xlabel("n (log scale)")
        ax.set_ylabel("GFLOP/s  (2n³ / kernel time)")
        ax.set_title(f"{int(t)} thread{'s' if t > 1 else ''}")
    axes[0].legend()
    ax = axes[2]
    for a in ALGO_ORDER:
        pts = []
        for n in sorted({r["n"] for r in pick(rows, algo=a)}):
            r1 = pick(rows, algo=a, n=n, threads=1.0)
            rp = pick(rows, algo=a, n=n, threads=float(ncpu))
            if r1 and rp:
                pts.append((n, r1[0]["t_compute_min"] / rp[0]["t_compute_min"]))
        if pts:
            line(ax, [p[0] for p in pts], [p[1] for p in pts], ALGO_COLOR[a], a)
    ax.axhline(ncpu, color=MUTED, lw=1)
    ax.text(100, ncpu + 0.1, "ideal", color=MUTED, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["n"] for r in rows}))
    ax.set_xticklabels([str(int(v)) for v in sorted({r["n"] for r in rows})])
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel(f"kernel speedup on {ncpu} threads")
    ax.set_title(f"Speedup {ncpu} threads vs 1 thread")
    fig.suptitle("Comparison of the four algorithms", fontsize=11, fontweight="semibold", color=INK)
    save(fig, out, "04_algorithms.png")

    md.append("## Algorithms: median GFLOP/s (1 thread / %d threads)\n" % ncpu)
    sizes = sorted({r["n"] for r in rows})
    md.append("| algorithm | " + " | ".join(f"n={int(n)}" for n in sizes) + " |")
    md.append("|---|" + "---|" * len(sizes))
    for a in ALGO_ORDER:
        cells = []
        for n in sizes:
            r1 = pick(rows, algo=a, n=n, threads=1.0)
            rp = pick(rows, algo=a, n=n, threads=float(ncpu))
            cells.append(f"{r1[0]['gflops']:.1f} / {rp[0]['gflops']:.1f}" if r1 and rp else "-")
        md.append(f"| {a} | " + " | ".join(cells) + " |")
    md.append("")


def fig_amdahl(sc, pi, out, ncpu, md):
    serial = pick(sc, algo="blocked", par_init=0.0)
    par = pick(pi, algo="blocked", par_init=1.0)
    sizes = sorted({r["n"] for r in serial if r["n"] <= 3200})
    md.append("## Amdahl's law (blocked algorithm, p = 1..%d)\n" % ncpu)
    md.append("s_fit = serial fraction fitted to the measured speedups; s_meas = measured share of serial phases "
              "(alloc + init + checksum) in the 1-thread run; S_max = 1/s_fit = upper bound of the speedup; "
              "e(p) = Karp-Flatt metric.\n")
    md.append("| init | time used | n | s_fit | R² | s_meas (1 thread) | S(%d) measured | S_max = 1/s | e(2) | e(4) | e(%d) |"
              % (ncpu, ncpu))
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    fits = {}
    for label, rows in [("serial", serial), ("parallel", par)]:
        for field in ["t_total", "t_compute_min"]:
            for n in sizes:
                rs = sorted([r for r in pick(rows, n=n) if r["threads"] <= ncpu], key=lambda r: r["threads"])
                base = next((r for r in rs if r["threads"] == 1), None)
                if not base or len(rs) < 3:
                    continue
                ps = [r["threads"] for r in rs]
                sp = [base[field] / r[field] for r in rs]
                s, r2 = amdahl_fit(ps, sp)
                kf = {p: (1 / S - 1 / p) / (1 - 1 / p) for p, S in zip(ps, sp) if p > 1}
                s_meas = (base["t_alloc"] + base["t_init"] + base["t_checksum"]) / base["t_total"]
                fits[(label, field, n)] = (s, ps, sp, kf, s_meas)
                md.append(f"| {label} | {'whole program' if field == 't_total' else 'kernel only'} | {int(n)} | "
                          f"{s:.4f} | {r2:.3f} | {s_meas:.4f} | {sp[-1]:.2f} | {1 / s if s > 0 else float('inf'):.1f} | "
                          f"{kf.get(2, float('nan')):.4f} | {kf.get(4, float('nan')):.4f} | {kf.get(ncpu, float('nan')):.4f} |")
    md.append("")

    # Figure: measured speedup with fitted Amdahl curves, serial vs parallel init
    show = [n for n in sizes if n in (200, 800, 3200)] or sizes[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    pgrid = np.linspace(1, ncpu, 50)
    for ax, n in zip(axes, show):
        for label, col in [("serial", SLOTS[0]), ("parallel", SLOTS[1])]:
            f = fits.get((label, "t_total", n))
            if not f:
                continue
            s, ps, sp, kf, s_meas = f
            ax.plot(pgrid, amdahl_curve(s, pgrid), color=col, lw=1.2, alpha=0.6)
            line(ax, ps, sp, col, f"{label} init (s = {s:.3f})", linestyle="none")
        ax.plot([1, ncpu], [1, ncpu], color=MUTED, lw=1)
        ax.set_title(f"n = {int(n)}")
        ax.set_xlabel("threads p")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("whole-program speedup")
    fig.suptitle("Measured speedup (dots) and the fitted Amdahl curve (line)", fontsize=11,
                 fontweight="semibold", color=INK)
    save(fig, out, "05_amdahl_fit.png")

    # Figure: fitted serial fraction vs n
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    for (label, field), col, name in [(("serial", "t_total"), SLOTS[0], "whole program, serial init"),
                                      (("parallel", "t_total"), SLOTS[1], "whole program, parallel init"),
                                      (("serial", "t_compute_min"), SLOTS[2], "kernel only")]:
        pts = [(n, fits[(label, field, n)][0]) for n in sizes if (label, field, n) in fits]
        if pts:
            line(ax, [p[0] for p in pts], [p[1] for p in pts], col, name)
    pts = [(n, fits[("serial", "t_total", n)][4]) for n in sizes if ("serial", "t_total", n) in fits]
    if pts:
        line(ax, [p[0] for p in pts], [p[1] for p in pts], SLOTS[3], "measured serial phases (1 thread)",
             marker="s")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(v)) for v in sizes])
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("serial fraction s (log scale)")
    ax.set_title("Serial fraction vs problem size")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    # n = 100 and 200 are left out: their e(p) is above 0.5 (overhead dominates), see summary.md.
    big = [n for n in sizes if n >= 400]
    for n, c in zip(big, BLUE_RAMP[-len(big):]):
        f = fits.get(("serial", "t_total", n))
        if f:
            kf = f[3]
            line(ax, list(kf.keys()), list(kf.values()), c, f"n = {int(n)}")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel("threads p")
    ax.set_ylabel("Karp-Flatt e(p)")
    ax.set_title("Karp-Flatt metric, whole program (n ≥ 400)")
    ax.legend(ncol=2, fontsize=7.5)
    save(fig, out, "06_serial_fraction.png")


def fig_breakdown(sc, pi, out, ncpu):
    phases = [("t_alloc", "alloc (serial)", SLOTS[6]), ("t_spawn", "thread start-up", SLOTS[4]),
              ("t_init", "init", SLOTS[3]), ("t_compute_min", "compute (parallel)", SLOTS[0]),
              ("t_checksum", "checksum (serial)", SLOTS[1])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, (rows, title) in zip(axes, [(pick(sc, algo="blocked", par_init=0.0), "serial init"),
                                        (pick(pi, algo="blocked", par_init=1.0), "parallel init")]):
        labels, data = [], []
        for n in sorted({r["n"] for r in rows if r["n"] <= 3200}):
            for t in (1.0, float(ncpu)):
                r = pick(rows, n=n, threads=t)
                if not r:
                    continue
                r = r[0]
                tot = sum(r[k] for k, _, _ in phases)
                data.append([100 * r[k] / tot for k, _, _ in phases])
                labels.append(f"n={int(n)}, p={int(t)}")
        data = np.array(data)
        left = np.zeros(len(data))
        for i, (k, name, col) in enumerate(phases):
            ax.barh(range(len(data)), data[:, i], left=left, color=col, height=0.62, label=name,
                    edgecolor=SURFACE, linewidth=1)
            left += data[:, i]
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of total time (%)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=5, fontsize=8.5)
    fig.suptitle("Where the time goes: program phases (blocked algorithm)", fontsize=11,
                 fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "07_time_breakdown.png", tight=False)


def fig_memory(sc, out, ncpu, md):
    rows = pick(sc, par_init=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    for a in ALGO_ORDER:
        x, y = series(pick(rows, algo=a, threads=1.0), "n", "vm_hwm_mb")
        if x:
            line(ax, x, y, ALGO_COLOR[a], f"{a}: peak RSS")
    ns = np.array(sorted({r["n"] for r in rows}))
    ax.plot(ns, 3 * ns ** 2 * 8 / 2 ** 20, color=MUTED, lw=1, label="theory 3·n²·8 B (A, B, C)")
    ax.plot(ns, 4 * ns ** 2 * 8 / 2 ** 20, color=MUTED, lw=1, ls=(0, (1, 2)), label="theory 4·n²·8 B (+Bᵀ)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(v)) for v in ns])
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("MiB (log scale)")
    ax.set_title("Peak resident memory vs n (1 thread)")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for n, col in [(800.0, SLOTS[0]), (3200.0, SLOTS[1])]:
        rs = pick(rows, algo="blocked", n=n)
        x, y = series(rs, "threads", "vm_peak_mb")
        line(ax, x, y, col, f"n={int(n)}: peak virtual (VmPeak)")
        x, y = series(rs, "threads", "vm_hwm_mb")
        line(ax, x, y, col, f"n={int(n)}: peak resident (VmHWM)", marker="s", linestyle=(0, (4, 2)))
    ax.set_xlabel("threads")
    ax.set_ylabel("MiB")
    ax.set_title("Virtual vs resident memory vs threads")
    ax.legend(fontsize=7.5)
    save(fig, out, "08_memory.png")

    md.append("## Memory (blocked, serial init): theory vs measured (MiB)\n")
    md.append("| n | threads | theory 3n²·8 | VmHWM (peak RSS) | VmPeak (virtual) | minor faults |")
    md.append("|---|---|---|---|---|---|")
    for n in sorted({r["n"] for r in rows}):
        for t in (1.0, float(ncpu), 2.0 * ncpu):
            r = pick(rows, algo="blocked", n=n, threads=t)
            if r:
                r = r[0]
                md.append(f"| {int(n)} | {int(t)} | {r['mem_theory_mb']:.2f} | {r['vm_hwm_mb']:.2f} | "
                          f"{r['vm_peak_mb']:.2f} | {int(r['minflt'])} |")
    md.append("")


def fig_block(bl, out, ncpu, md):
    if not bl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=False)
    for ax, t in zip(axes, (1.0, float(ncpu))):
        for n, col in zip(sorted({r["n"] for r in bl}), SLOTS):
            x, y = series(pick(bl, n=n, threads=t), "block", "gflops")
            line(ax, x, y, col, f"n = {int(n)}")
        ax.set_xscale("log", base=2)
        bs = sorted({r["block"] for r in bl})
        ax.set_xticks(bs)
        ax.set_xticklabels([str(int(b)) for b in bs], fontsize=7.5)
        ax.set_xlabel("tile size bs (log scale)")
        ax.set_ylabel("GFLOP/s")
        ax.set_title(f"{int(t)} thread{'s' if t > 1 else ''}")
        ax.legend()
    fig.suptitle("Effect of the tile size (blocked algorithm)", fontsize=11, fontweight="semibold", color=INK)
    save(fig, out, "09_block_size.png")
    md.append("## Tile size: median GFLOP/s\n")
    bs = sorted({r["block"] for r in bl})
    md.append("| n | threads | " + " | ".join(f"bs={int(b)}" for b in bs) + " |")
    md.append("|---|---|" + "---|" * len(bs))
    for n in sorted({r["n"] for r in bl}):
        for t in (1.0, float(ncpu)):
            cells = [f"{pick(bl, n=n, threads=t, block=b)[0]['gflops']:.1f}" if pick(bl, n=n, threads=t, block=b)
                     else "-" for b in bs]
            md.append(f"| {int(n)} | {int(t)} | " + " | ".join(cells) + " |")
    md.append("")


def fig_pow2(pw, out, ncpu, md):
    if not pw:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, t in zip(axes, (1.0, float(ncpu))):
        for a in ALGO_ORDER:
            x, y = series(pick(pw, algo=a, threads=t), "n", "gflops")
            if x:
                line(ax, x, y, ALGO_COLOR[a], a)
        for p2 in (1024, 2048):
            ax.axvline(p2, color=AXIS, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n (log scale; vertical lines at 1024 and 2048)")
        ax.set_ylabel("GFLOP/s")
        ax.set_title(f"{int(t)} thread{'s' if t > 1 else ''}")
        ax.legend()
    fig.suptitle("Sizes near powers of two (cache set conflicts)", fontsize=11, fontweight="semibold", color=INK)
    save(fig, out, "10_power_of_two.png")
    md.append("## Sizes near powers of two: median GFLOP/s (1 thread / %d threads)\n" % ncpu)
    md.append("| algorithm | n | 1 thread | %d threads |" % ncpu)
    md.append("|---|---|---|---|")
    for a in ALGO_ORDER:
        for n in sorted({r["n"] for r in pick(pw, algo=a)}):
            r1, rp = pick(pw, algo=a, n=n, threads=1.0), pick(pw, algo=a, n=n, threads=float(ncpu))
            md.append(f"| {a} | {int(n)} | {r1[0]['gflops']:.1f} | {rp[0]['gflops']:.1f} |" if r1 and rp else "")
    md.append("")


def grouped_bars(ax, groups, series_names, values, colors):
    """values[g][s]; bars of one group touch with a 2px surface gap."""
    ng, ns = len(groups), len(series_names)
    width = 0.8 / ns
    for si, (name, col) in enumerate(zip(series_names, colors)):
        xs = np.arange(ng) - 0.4 + width * (si + 0.5)
        ax.bar(xs, [values[g][si] for g in range(ng)], width=width, color=col, label=name,
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(range(ng))
    ax.set_xticklabels(groups, fontsize=8)
    ax.grid(axis="x", visible=False)


def fig_schedule(sd, out, ncpu, md):
    if not sd:
        return
    combos = [("static", 0), ("static", 1), ("static", 4), ("static", 16), ("dynamic", 0), ("dynamic", 1),
              ("dynamic", 4), ("dynamic", 16), ("guided", 0), ("guided", 1), ("guided", 4), ("guided", 16)]
    names = [f"{k}\n{'default' if c == 0 else 'chunk ' + str(c)}" for k, c in combos]
    cases = [(a, n) for a in ("blocked", "ikj") for n in sorted({r["n"] for r in sd})]
    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 2.6 * len(cases)))
    axes = np.atleast_1d(axes)
    md.append("## OpenMP schedule: median kernel time (ms)\n")
    md.append("| algorithm | n | threads | " + " | ".join(n.replace("\n", " ") for n in names) + " |")
    md.append("|---|---|---|" + "---|" * len(names))
    for ax, (a, n) in zip(axes, cases):
        threads = sorted({r["threads"] for r in pick(sd, algo=a, n=n)})
        vals = []
        for kind, ch in combos:
            row = []
            for t in threads:
                r = pick(sd, algo=a, n=n, threads=t, schedule=kind, chunk=float(ch))
                row.append(1000 * r[0]["t_compute_min"] if r else 0)
            vals.append(row)
        for ti, t in enumerate(threads):
            md.append(f"| {a} | {int(n)} | {int(t)} | " + " | ".join(f"{v[ti]:.2f}" for v in vals) + " |")
        grouped_bars(ax, names, [f"{int(t)} threads" for t in threads], vals, SLOTS[:len(threads)])
        ax.set_ylabel("kernel time (ms)")
        ax.set_title(f"{a}, n = {int(n)}", fontsize=10)
    md.append("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels))
    fig.suptitle("OpenMP loop schedule and chunk size", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "11_schedule.png", tight=False)


def fig_compiler(cp, out, ncpu, md):
    if not cp:
        return
    variants = ["O0", "O1", "O2", "O3", "O3-native", "O3-native-fastmath"]
    variants = [v for v in variants if pick(cp, variant=v)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    md.append("## Compiler flags: median GFLOP/s\n")
    md.append("| algorithm | n | threads | " + " | ".join(variants) + " |")
    md.append("|---|---|---|" + "---|" * len(variants))
    for ax, t in zip(axes, (1.0, float(ncpu))):
        algos = [a for a in ALGO_ORDER if pick(cp, algo=a)]
        vals = []
        for v in variants:
            vals.append([pick(cp, variant=v, algo=a, threads=t)[0]["gflops"] if pick(cp, variant=v, algo=a, threads=t)
                         else 0 for a in algos])
        grouped_bars(ax, ["-" + v.replace("-native", " -march=native").replace("-fastmath", " -ffast-math")
                          for v in variants], algos, vals, [ALGO_COLOR[a] for a in algos])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel("GFLOP/s")
        ax.set_title(f"{int(t)} thread{'s' if t > 1 else ''}")
        ax.legend()
        for a in algos:
            n = pick(cp, algo=a)[0]["n"]
            md.append(f"| {a} | {int(n)} | {int(t)} | " + " | ".join(
                f"{pick(cp, variant=v, algo=a, threads=t)[0]['gflops']:.2f}" for v in variants) + " |")
    md.append("")
    fig.suptitle("Compiler optimisation level", fontsize=11, fontweight="semibold", color=INK)
    save(fig, out, "12_compiler.png")


def fig_osbind(ob, out, ncpu, md):
    if not ob:
        return
    settings = []
    for r in ob:
        if r["setting"] not in settings:
            settings.append(r["setting"])
    sizes = sorted({r["n"] for r in ob})
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.4 * len(sizes), 4.4))
    axes = np.atleast_1d(axes)
    md.append("## OS / OpenMP runtime settings: median kernel time (ms), blocked\n")
    md.append("| setting | n | " + " | ".join(f"p={int(t)}" for t in sorted({r['threads'] for r in ob})) +
              " | involuntary ctx switches (p=%d) |" % (2 * ncpu))
    md.append("|---|---|" + "---|" * (len({r['threads'] for r in ob}) + 1))
    for ax, n in zip(axes, sizes):
        threads = sorted({r["threads"] for r in pick(ob, n=n)})
        names = [s for s in settings if pick(ob, n=n, setting=s)]
        vals = []
        for s in names:
            row = []
            for t in threads:
                r = pick(ob, n=n, setting=s, threads=t)
                row.append(1000 * r[0]["t_compute_min"] if r else 0)
            vals.append(row)
            r16 = pick(ob, n=n, setting=s, threads=2.0 * ncpu)
            md.append(f"| {s} | {int(n)} | " + " | ".join(f"{v:.2f}" for v in row) +
                      f" | {int(r16[0]['nivcsw']) if r16 else '-'} |")
        y = np.arange(len(names))
        h = 0.8 / len(threads)
        for ti, (t, col) in enumerate(zip(threads, SLOTS)):
            ax.barh(y - 0.4 + h * (ti + 0.5), [v[ti] for v in vals], height=h, color=col,
                    label=f"{int(t)} threads", edgecolor=SURFACE, linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels([s.replace(";", "\n") for s in names], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("kernel time (ms, log scale)")
        ax.set_title(f"n = {int(n)}")
        ax.grid(axis="y", visible=False)
    md.append("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle("Thread binding and wait policy (environment variables of the OpenMP runtime)",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "13_os_binding.png", tight=False)


def fig_affinity(af, out, ncpu, md):
    if not af:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cpus = sorted({r["cpus"] for r in af})
    md.append("## CPU affinity (taskset): median kernel time (s), blocked n=%d\n" % int(af[0]["n"]))
    md.append("| CPUs allowed | " + " | ".join(f"p={int(t)}" for t in sorted({r['threads'] for r in af})) + " |")
    md.append("|---|" + "---|" * len({r['threads'] for r in af}))
    for c, col in zip(cpus, SLOTS):
        x, y = series(pick(af, cpus=c), "threads", "t_compute_min")
        line(ax, x, y, col, f"taskset: {int(c)} CPU{'s' if c > 1 else ''}")
        md.append(f"| {int(c)} | " + " | ".join(f"{v:.4f}" for v in y) + " |")
    ax.set_xscale("log", base=2)
    ts = sorted({r["threads"] for r in af})
    ax.set_xticks(ts)
    ax.set_xticklabels([str(int(t)) for t in ts])
    ax.set_yscale("log")
    ax.set_xlabel("OpenMP threads (log scale)")
    ax.set_ylabel("kernel time (s, log scale)")
    ax.set_title("Threads vs CPUs the OS allows (oversubscription)")
    ax.legend()
    md.append("")
    save(fig, out, "14_affinity.png")


def fig_thp(tp, out, ncpu, md):
    if not tp:
        return
    modes = ["default", "on", "off"]
    cases = sorted({(r["algo"], r["n"], r["threads"]) for r in tp}, key=lambda c: (c[0], c[2], c[1]))
    cases = [c for c in cases if c[1] >= 1600]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    labels = [f"{a}\nn={int(n)}\np={int(t)}" for a, n, t in cases]
    md.append("## Transparent huge pages (madvise): median values\n")
    md.append("| algorithm | n | threads | THP | init (s) | kernel (s) | GFLOP/s | AnonHugePages (MiB) | minor faults |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for ax, field, title, scale in [(axes[0], "t_init", "init time (s) - page faults happen here", 1),
                                    (axes[1], "t_compute_min", "kernel time (s) - TLB effect", 1),
                                    (axes[2], "minflt", "minor page faults", 1)]:
        vals = [[pick(tp, algo=a, n=n, threads=t, thp=m)[0][field] * scale if pick(tp, algo=a, n=n, threads=t, thp=m)
                 else 0 for m in modes] for a, n, t in cases]
        grouped_bars(ax, labels, [f"THP {m}" for m in modes], vals, SLOTS[:3])
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9.5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=3)
    for a, n, t in cases:
        for m in modes:
            r = pick(tp, algo=a, n=n, threads=t, thp=m)
            if r:
                r = r[0]
                md.append(f"| {a} | {int(n)} | {int(t)} | {m} | {r['t_init']:.4f} | {r['t_compute_min']:.4f} | "
                          f"{r['gflops']:.1f} | {r['anon_huge_mb']:.1f} | {int(r['minflt'])} |")
    md.append("")
    fig.suptitle("Transparent huge pages (2 MiB) via madvise()", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "15_huge_pages.png", tight=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results")
    ap.add_argument("--cpus", type=int, default=os.cpu_count() or 8)
    a = ap.parse_args()
    style()
    out = os.path.join(a.data, "plots")
    os.makedirs(out, exist_ok=True)
    ncpu = a.cpus
    md = ["# Benchmark summary (medians of repeated runs)\n"]

    keys_common = ["algo", "n", "threads", "schedule", "chunk", "block", "par_init", "thp", "reps"]
    sc = agg(load(a.data, "scaling"), keys_common)
    pi = agg(load(a.data, "parinit"), keys_common)
    bl = agg(load(a.data, "block"), keys_common)
    pw = agg(load(a.data, "pow2"), keys_common)
    sd = agg(load(a.data, "schedule"), keys_common)
    cp = agg(load(a.data, "compiler"), keys_common + ["variant"])
    ob = agg(load(a.data, "osbind"), keys_common + ["setting"])
    af = agg(load(a.data, "affinity"), keys_common + ["cpus"])
    tp = agg(load(a.data, "thp"), keys_common)

    all_rows = load(a.data, "scaling")
    bad = [r for r in all_rows if r.get("verify_ok") != 1.0]
    md.append(f"Correctness: {len(all_rows) - len(bad)} of {len(all_rows)} scaling runs passed Freivalds' check.\n")
    noisy = sorted(sc, key=lambda r: -r["cv_total"])[:5]
    md.append("Largest run-to-run variation (coefficient of variation of total time): " +
              ", ".join(f"{r['algo']} n={int(r['n'])} p={int(r['threads'])}: {100 * r['cv_total']:.1f}%" for r in noisy)
              + "\n")

    if sc:
        fig_scaling(sc, out, ncpu, md)
        fig_algorithms(sc, out, ncpu, md)
        fig_memory(sc, out, ncpu, md)
    if sc and pi:
        fig_amdahl(sc, pi, out, ncpu, md)
        fig_breakdown(sc, pi, out, ncpu)
    fig_block(bl, out, ncpu, md)
    fig_pow2(pw, out, ncpu, md)
    fig_schedule(sd, out, ncpu, md)
    fig_compiler(cp, out, ncpu, md)
    fig_osbind(ob, out, ncpu, md)
    fig_affinity(af, out, ncpu, md)
    fig_thp(tp, out, ncpu, md)

    with open(os.path.join(a.data, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", os.path.join(a.data, "summary.md"))


if __name__ == "__main__":
    main()
