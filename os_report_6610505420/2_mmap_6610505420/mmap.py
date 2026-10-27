#!/usr/bin/env python3
"""
mmap.py - a small memory-map visualizer for Linux processes.

OS Report 2026, Part 2 (Memory Map Visualization) - student ID 6610505420
Made for education only. This is not production software.
AI disclosure: written with the help of Claude Code (Anthropic, Claude Opus 5.5).

Input: the memory map that the Linux kernel publishes for every process
    /proc/<pid>/maps     address ranges, permissions and backing file of every region (VMA)
    /proc/<pid>/smaps    the same plus RSS / PSS / swap / huge pages of every region
    /proc/<pid>/pagemap  one 64-bit entry per virtual page ("present in RAM", "swapped", ...)
    /proc/<pid>/status   process totals; /proc/<pid>/task/* gives the threads
Output: a colour view in the terminal (one snapshot, or real time with --watch)
    and a self-contained interactive HTML report (snapshot or timeline).

Examples:
    python3 mmap.py --list                        list processes and their RSS
    python3 mmap.py self                          snapshot of this python process
    python3 mmap.py 1234                          snapshot of PID 1234
    python3 mmap.py 1234 --compact                one row per file/library instead of per region
    python3 mmap.py 1234 --watch 1                real time, refresh every second (Ctrl-C to stop)
    python3 mmap.py 1234 --html map.html          interactive HTML snapshot
    python3 mmap.py --run ./demo --watch 0.5      start a program and watch it live
    python3 mmap.py --run "./demo -s 1" --record 40 --html demo.html   record a timeline to HTML
    python3 mmap.py 1234 --pages 12               page-by-page residency of region #12
    python3 mmap.py --file saved_smaps.txt        draw a saved maps/smaps file

Safety: the tool only reads files under /proc. It never writes to another
process or sends it a signal, except the child it started itself with --run,
which is stopped (SIGTERM) when the tool finishes.

Note: this file is called mmap.py as required by the assignment, so it shadows
the standard-library module "mmap" for scripts in this folder. It does not
import any module that needs the real mmap module.
"""
import argparse
import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from array import array

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
MERGED_STACK_LIMIT = 16 << 20   # glibc thread stacks are 8 MiB by default; bigger = merged VMA

# Category key, display name, colour on a light background, colour on a dark background.
# Seven hues in a fixed order + neutral gray; the order never changes.
CATEGORIES = [
    ("program", "program (executable)", "#2a78d6", "#3987e5"),
    ("heap", "heap (brk)", "#eb6834", "#d95926"),
    ("anon", "anonymous mmap", "#1baf7a", "#199e70"),
    ("stack", "stack", "#eda100", "#c98500"),
    ("lib", "shared library", "#e87ba4", "#d55181"),
    ("file", "mapped file", "#008300", "#008300"),
    ("shm", "shared memory", "#6250d6", "#9085e9"),
    ("other", "guard / reserved / kernel", "#898781", "#898781"),
]
CAT_KEYS = [c[0] for c in CATEGORIES]
CAT_NAME = {c[0]: c[1] for c in CATEGORIES}
CAT_COLOR = {c[0]: c[2] for c in CATEGORIES}

HEADER_RE = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+([rwxsp-]{4})\s+([0-9a-f]+)\s+(\S+)\s+(\d+)\s*(.*)$")
FIELD_RE = re.compile(r"^(\w+):\s+(\d+) kB$")
SMAPS_FIELDS = {"Rss": "rss", "Pss": "pss", "Swap": "swap", "AnonHugePages": "huge",
                "Private_Dirty": "priv_dirty", "Anonymous": "anon_bytes"}
KERNEL_REGIONS = {
    "[vdso]": "vDSO: kernel code mapped into the process (fast gettimeofday etc.)",
    "[vvar]": "vvar: kernel data read by the vDSO",
    "[vvar_vclock]": "vvar: clock pages read by the vDSO",
    "[vsyscall]": "vsyscall: legacy fixed-address system call page",
    "[uprobes]": "uprobes: kernel tracing trampoline",
}


class ProcessGone(Exception):
    pass


# --------------------------------------------------------------------------
# Reading /proc
# --------------------------------------------------------------------------

def read_text(path):
    with open(path, "r", errors="replace") as f:
        return f.read()


def parse_maps_text(text):
    """Parse /proc/PID/maps or /proc/PID/smaps text into a list of region dicts."""
    regions, cur = [], None
    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            start, end = int(m[1], 16), int(m[2], 16)
            cur = {"start": start, "end": end, "size": end - start, "perms": m[3], "offset": int(m[4], 16),
                   "dev": m[5], "inode": int(m[6]), "path": m[7].strip(), "rss": None, "pss": None,
                   "swap": None, "huge": None, "priv_dirty": None, "anon_bytes": None, "flags": "", "res": None}
            regions.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("VmFlags:"):
            cur["flags"] = line[8:].strip()
            continue
        f = FIELD_RE.match(line)
        if f and f[1] in SMAPS_FIELDS:
            cur[SMAPS_FIELDS[f[1]]] = int(f[2]) * 1024
    regions.sort(key=lambda r: r["start"])
    return regions


def read_status(pid):
    st = {}
    for line in read_text(f"/proc/{pid}/status").splitlines():
        key, _, val = line.partition(":")
        val = val.strip()
        if val.endswith(" kB"):
            st[key] = int(val[:-3]) * 1024
        elif key in ("Name", "State", "Threads", "Pid", "PPid"):
            st[key] = val
    try:
        st["cmdline"] = read_text(f"/proc/{pid}/cmdline").replace("\0", " ").strip()
    except OSError:
        st["cmdline"] = ""
    return st


def read_threads(pid):
    """Return [{tid, name, sp}]. sp (stack pointer) comes from /proc/PID/task/TID/syscall,
    which is readable for our own children or as root; otherwise it is None."""
    threads = []
    try:
        tids = sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return threads
    for tid in tids:
        base = f"/proc/{pid}/task/{tid}"
        try:
            name = read_text(base + "/comm").strip()
        except OSError:
            continue
        sp = None
        try:
            fields = read_text(base + "/syscall").split()
            if len(fields) >= 3 and fields[0] != "running":
                sp = int(fields[-2], 16)    # "... sp pc" are the last two fields
        except (OSError, ValueError):
            pass
        threads.append({"tid": tid, "name": name, "sp": sp})
    return threads


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def perm_kind(perms):
    return {"r-x": "code", "--x": "code", "r--": "read-only data", "rw-": "read-write data",
            "rwx": "code + data (rwx)", "---": "no access"}.get(perms[:3], perms[:3])


def is_library(name):
    return ".so" in name


def elf_bss_ends(regions):
    """For every mapped ELF file (program or library) return the address where its
    zero-filled .bss ends, computed from the ELF program headers: the writable
    PT_LOAD segment covers p_vaddr .. p_vaddr + p_memsz, and the part after p_filesz
    is the .bss. The kernel/loader maps that part as anonymous memory right after
    the file-backed data, and may merge it with a neighbouring anonymous region."""
    ends = {}
    for r in regions:
        p = r["path"]
        if not p.startswith("/") or r["offset"] != 0 or p in ends:
            continue
        ends[p] = None
        try:
            with open(p, "rb") as f:
                hdr = f.read(64)
                if hdr[:4] != b"\x7fELF" or hdr[4] != 2 or hdr[5] != 1:   # 64-bit little-endian only
                    continue
                phoff = int.from_bytes(hdr[0x20:0x28], "little")
                phentsize = int.from_bytes(hdr[0x36:0x38], "little")
                phnum = int.from_bytes(hdr[0x38:0x3a], "little")
                f.seek(phoff)
                ph = f.read(phentsize * phnum)
        except OSError:
            continue
        first_vaddr, bss_end = None, None
        for i in range(phnum):
            e = ph[i * phentsize:(i + 1) * phentsize]
            if len(e) < 56 or int.from_bytes(e[0:4], "little") != 1:          # PT_LOAD
                continue
            flags = int.from_bytes(e[4:8], "little")
            vaddr = int.from_bytes(e[16:24], "little")
            memsz = int.from_bytes(e[40:48], "little")
            if first_vaddr is None:
                first_vaddr = vaddr
            if flags & 2:                                                       # PF_W
                bss_end = max(bss_end or 0, vaddr + memsz)
        if first_vaddr is None or bss_end is None:
            continue
        base = r["start"] - (first_vaddr & ~(PAGE - 1))
        ends[p] = (base + bss_end + PAGE - 1) & ~(PAGE - 1)
    return ends


def classify(regions, exe, threads):
    """Give every region a category, a description ("label") and an owning object."""
    exe_name = os.path.basename(exe) if exe else None
    bss_ends = elf_bss_ends(regions)
    for i, r in enumerate(regions):
        prev = regions[i - 1] if i > 0 else None
        nxt = regions[i + 1] if i + 1 < len(regions) else None
        r["cat"], r["label"], r["obj"] = classify_one(r, prev, nxt, exe, exe_name, bss_ends)

    # Exact thread stacks: a region that contains a thread's stack pointer.
    owners = {}
    for th in threads:
        if th["sp"] is None:
            continue
        for r in regions:
            if r["start"] <= th["sp"] < r["end"]:
                owners.setdefault(id(r), (r, []))[1].append(f"{th['tid']} ({th['name']})")
                break
    for r, tids in owners.values():
        who = ", ".join(tids)
        if r["path"] == "[stack]":
            r["label"] = f"main thread stack (grows down), tid {who}"
        elif r["cat"] in ("anon", "stack"):
            if r["size"] > MERGED_STACK_LIMIT:
                # Neighbouring anonymous mappings with the same permissions are merged by
                # the kernel into one VMA, so one region can hold a stack and other data.
                r["cat"] = "anon"
                r["label"] = f"anonymous memory merged into one VMA; holds the stack of tid {who}"
            else:
                r["cat"] = "stack"
                r["label"] = f"thread stack of tid {who}"


def classify_one(r, prev, nxt, exe, exe_name, bss_ends):
    p, perms = r["path"], r["perms"]
    kind = perm_kind(perms)
    adjacent_prev = prev is not None and prev["end"] == r["start"]
    adjacent_next = nxt is not None and nxt["start"] == r["end"]

    if p == "[heap]":
        return "heap", "heap: small malloc() blocks, grows with brk()", None
    if p == "[stack]":
        return "stack", "main thread stack (grows down)", None   # tid added later if known
    if p.startswith("[stack:"):
        return "stack", f"thread stack (tid {p[7:-1]})", None
    if p in KERNEL_REGIONS:
        return "other", KERNEL_REGIONS[p], None
    if p.startswith("[anon_shmem:"):
        return "shm", f"named shared anonymous memory '{p[12:-1]}'", None
    if p.startswith("[anon:"):
        return "anon", f"named anonymous memory '{p[6:-1]}' ({kind})", None
    if p.startswith(("/dev/shm/", "/memfd:", "/SYSV", "/dev/zero")) or (perms[3] == "s" and not p):
        return "shm", f"shared memory {p} ({kind})", p
    if p.startswith("/"):
        name = os.path.basename(p.replace(" (deleted)", ""))
        note = " - file was deleted" if p.endswith(" (deleted)") else ""
        if perms[3] == "s":
            note += " - MAP_SHARED"
        if exe and p == exe:
            section = {"code": ".text (machine code)",
                       "read-only data": ".rodata / ELF headers / RELRO",
                       "read-write data": ".data (initialised globals)"}.get(kind, kind)
            return "program", f"{name}: {section}", p
        if is_library(name):
            return "lib", f"{name}: {kind}", p
        return "file", f"file {name}: {kind}{note}", p
    if p.startswith("["):
        return "other", p, None

    # Anonymous regions (no file): use the neighbours to recognise common layouts.
    if perms[:3] == "---":
        if adjacent_prev and not prev["path"] and prev["perms"][:3] == "rw-" \
                and (prev["size"] + r["size"]) % (64 << 20) == 0:
            return "anon", "malloc arena: reserved, not yet usable", None
        return "other", "guard page / reserved address space (no access)", None
    if adjacent_prev and prev["path"].startswith("/") and prev["perms"][:3] == "rw-":
        name = os.path.basename(prev["path"])
        owner = "program" if exe and prev["path"] == exe else "lib" if is_library(name) else None
        bss_end = bss_ends.get(prev["path"])
        if owner and bss_end is not None and r["start"] < bss_end:
            if r["end"] <= bss_end:
                return owner, f"{name}: .bss (zero-initialised globals)", prev["path"]
            # The kernel merged the .bss with a neighbouring anonymous mapping (same permissions).
            return "anon", (f"{name}: .bss ({human(bss_end - r['start'])}) merged with anonymous memory "
                            f"({human(r['end'] - bss_end)})"), None
        if owner and bss_end is None:            # ELF file not readable: fall back to a guess
            return owner, f"{name}: probably .bss (anonymous region right after its data)", prev["path"]
    if adjacent_next and not nxt["path"] and nxt["perms"][:3] == "---" \
            and (r["size"] + nxt["size"]) % (64 << 20) == 0:
        return "anon", "malloc arena: heap of a non-main thread", None
    if adjacent_prev and not prev["path"] and prev["perms"][:3] == "---" and prev["size"] <= 64 * 1024 \
            and perms[:3] == "rw-":
        return "stack", "thread stack? (guard page right below it)", None
    return "anon", f"anonymous memory: mmap() / large malloc() ({kind})", None


# --------------------------------------------------------------------------
# Page residency from /proc/PID/pagemap
# --------------------------------------------------------------------------

def read_pagemap(pid, start, npages):
    """Return an array of 64-bit pagemap entries, or None if not readable."""
    try:
        with open(f"/proc/{pid}/pagemap", "rb") as f:
            f.seek((start // PAGE) * 8)
            data = f.read(npages * 8)
    except OSError:
        return None
    a = array("Q")
    a.frombytes(data[: len(data) // 8 * 8])
    return a


def add_residency(pid, regions, buckets=48, max_region=1 << 30, budget_pages=1 << 23):
    """For each region store 'res': the percentage of resident pages in each of
    `buckets` equal slices of its address range (low address first)."""
    try:
        f = open(f"/proc/{pid}/pagemap", "rb")
    except OSError:
        return False
    with f:
        for r in regions:
            npages = r["size"] // PAGE
            if r["perms"][:3] == "---" or r["size"] > max_region or npages > budget_pages or npages == 0:
                continue
            try:
                f.seek((r["start"] // PAGE) * 8)
                data = f.read(npages * 8)
            except OSError:
                continue
            budget_pages -= npages
            a = array("Q")
            a.frombytes(data[: len(data) // 8 * 8])
            if not a:
                continue
            n = len(a)
            k = min(buckets, n)
            res = []
            for b in range(k):
                lo, hi = b * n // k, (b + 1) * n // k
                present = sum(v >> 63 for v in a[lo:hi])
                res.append(round(100 * present / (hi - lo)))
            r["res"] = res
    return True


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------

def take_snapshot(pid, pagemap=True):
    base = f"/proc/{pid}"
    try:
        try:
            text, detailed = read_text(base + "/smaps"), True
        except PermissionError:
            text, detailed = read_text(base + "/maps"), False
        status = read_status(pid)
    except (FileNotFoundError, ProcessLookupError):
        raise ProcessGone(pid)
    regions = parse_maps_text(text)
    if not regions:                               # zombie: the process has exited
        raise ProcessGone(pid)
    try:
        exe = os.readlink(base + "/exe")
    except OSError:
        exe = None
    classify(regions, exe, read_threads(pid))
    if pagemap:
        add_residency(pid, regions)
    return {"t": time.time(), "pid": pid, "regions": regions, "status": status, "detailed": detailed}


def snapshot_from_file(path):
    regions = parse_maps_text(read_text(path))
    if not regions:
        sys.exit(f"{path}: no maps/smaps lines found")
    # Guess the executable: the first file-backed r-xp mapping that is not a library.
    exe = next((r["path"] for r in regions if r["path"].startswith("/") and "x" in r["perms"]
                and not is_library(os.path.basename(r["path"]))), None)
    classify(regions, exe, [])
    return {"t": os.path.getmtime(path), "pid": None, "regions": regions,
            "status": {"Name": os.path.basename(path), "cmdline": f"(from file {path})"},
            "detailed": any(r["rss"] is not None for r in regions)}


def totals(snap):
    regs = snap["regions"]
    return {"vsz": sum(r["size"] for r in regs),
            "rss": sum(r["rss"] or 0 for r in regs),
            "pss": sum(r["pss"] or 0 for r in regs),
            "swap": sum(r["swap"] or 0 for r in regs),
            "n": len(regs)}


def summarize(regions):
    out = {k: {"n": 0, "size": 0, "rss": 0, "pss": 0, "swap": 0} for k in CAT_KEYS}
    for r in regions:
        c = out[r["cat"]]
        c["n"] += 1
        c["size"] += r["size"]
        c["rss"] += r["rss"] or 0
        c["pss"] += r["pss"] or 0
        c["swap"] += r["swap"] or 0
    return out


def group_regions(regions):
    """Merge neighbouring regions that belong to the same file or library."""
    groups = []
    for r in regions:
        g = groups[-1] if groups else None
        if g and r["obj"] and g["obj"] == r["obj"] and g["end"] == r["start"] and g["cat"] == r["cat"]:
            g["end"] = r["end"]
            g["size"] += r["size"]
            for k in ("rss", "pss", "swap", "huge"):
                if r[k] is not None:
                    g[k] = (g[k] or 0) + r[k]
            g["count"] += 1
            g["perm_list"].append(r["perms"])
            g["res"] = None
        else:
            g = dict(r, count=1, perm_list=[r["perms"]])
            groups.append(g)
    for g in groups:
        if g["count"] > 1:
            name = os.path.basename(g["obj"])
            g["label"] = f"{name} ({g['count']} regions: {' '.join(p[:3] for p in g['perm_list'])})"
    return groups


# --------------------------------------------------------------------------
# Terminal rendering
# --------------------------------------------------------------------------

def human(n):
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "K", "M", "G", "T", "P", "E"):
        if abs(n) < 1024 or unit == "E":
            if unit == "B":
                return f"{int(n)}B"
            return f"{n:.1f}{unit}" if abs(n) < 100 else f"{n:.0f}{unit}"
        n /= 1024


class Style:
    def __init__(self, color):
        self.color = color

    def fg(self, hexcol, s):
        if not self.color:
            return s
        r, g, b = int(hexcol[1:3], 16), int(hexcol[3:5], 16), int(hexcol[5:7], 16)
        return f"\033[38;2;{r};{g};{b}m{s}\033[0m"

    def bold(self, s):
        return f"\033[1m{s}\033[0m" if self.color else s

    def dim(self, s):
        return f"\033[2m{s}\033[0m" if self.color else s

    def swatch(self, cat):
        return self.fg(CAT_COLOR[cat], "■")


def shade(frac):
    if frac >= 0.66: return "█"
    if frac >= 0.33: return "▓"
    if frac > 0: return "▒"
    return "░"


def header_lines(snap, st):
    s, t = snap["status"], totals(snap)
    who = f"PID {snap['pid']} ({s.get('Name', '?')})" if snap["pid"] else s.get("Name", "?")
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap["t"]))
    cmd = s.get("cmdline", "")
    lines = [st.bold(f"Memory map of {who}") + "  " + st.dim(f"{cmd[:70]}  {when}")]
    rss = s.get("VmRSS", t["rss"] if snap["detailed"] else None)
    parts = [f"virtual {human(s.get('VmSize', t['vsz']))}", f"resident {human(rss)}"]
    if "RssAnon" in s:
        parts[-1] += f" (anon {human(s['RssAnon'])}, file {human(s.get('RssFile'))}, shmem {human(s.get('RssShmem'))})"
    parts.append(f"PSS {human(t['pss']) if snap['detailed'] else '-'}")
    parts.append(f"swap {human(s.get('VmSwap', t['swap']))}")
    if "Threads" in s:
        parts.append(f"threads {s['Threads']}")
    parts.append(f"regions {t['n']}")
    lines.append(" · ".join(parts))
    if not snap["detailed"] and snap["pid"]:
        lines.append(st.dim("(no permission to read smaps: RSS per region unknown, run as the same user or root)"))
    elif not snap["detailed"]:
        lines.append(st.dim("(this file has no RSS data: save /proc/PID/smaps instead of maps to see it)"))
    return lines


def overview_lines(regions, st, width):
    """One-line picture of the whole address space: low addresses on the left.
    Neighbouring regions of the same category form one segment; its width grows
    with log(size); the shade shows how much of it is resident; ┊ marks a hole."""
    segs = []
    for r in regions:
        s = segs[-1] if segs else None
        if s and s["cat"] == r["cat"] and s["end"] == r["start"]:
            s["end"], s["size"], s["rss"] = r["end"], s["size"] + r["size"], s["rss"] + (r["rss"] or 0)
        else:
            if s is not None:
                s["gap"] = r["start"] > s["end"]
            segs.append({"cat": r["cat"], "end": r["end"], "size": r["size"], "rss": r["rss"] or 0, "gap": False})
    usable = max(10, min(width, 140) - 12 - sum(1 for s in segs if s["gap"]))
    weights = [math.log2(s["size"] / PAGE + 1) for s in segs]
    total_w = sum(weights) or 1
    out = []
    for s, w in zip(segs, weights):
        cells = max(1, round(usable * w / total_w))
        out.append(st.fg(CAT_COLOR[s["cat"]], shade(s["rss"] / s["size"] if s["size"] else 0) * cells))
        if s["gap"]:
            out.append(st.dim("┊"))
    legend = "  ".join(f"{st.swatch(k)} {CAT_NAME[k]}" for k in CAT_KEYS)
    return [st.dim("low ") + "".join(out) + st.dim(" high"),
            st.dim("     █ ≥66% resident  ▓ ≥33%  ▒ some  ░ none   ┊ unmapped hole   width ~ log(size)"),
            legend]


def region_lines(regions, st, width, index_of, prev_index=None, max_rows=None):
    """One row per region, highest address first (like the usual textbook picture).
    index_of maps a start address to the region number used by --pages."""
    bar_w = 22
    maxlog = max((math.log2(r["size"] / PAGE + 1) for r in regions), default=1) or 1
    head = f"{'#':>4} {'address range':<25} {'perm':<5} {'size':>7} {'RSS':>7}  {'size (log) · █ resident':<{bar_w}}  what"
    lines = [st.bold(head)]
    rows = list(reversed(regions))
    for i, r in enumerate(rows):
        idx = index_of[r["start"]]
        L = max(1, round(bar_w * math.log2(r["size"] / PAGE + 1) / maxlog))
        rss = r["rss"]
        filled = 0 if not rss else max(1, round(L * rss / r["size"]))
        bar = st.fg(CAT_COLOR[r["cat"]], "█" * filled + "░" * (L - filled)) + " " * (bar_w - L)
        perms = r["perms"] if r.get("count", 1) == 1 else f"{r['count']}×"
        mark = " "
        if prev_index is not None:
            old = prev_index.get(r["start"])
            if old is None:
                mark = st.bold("+")
            elif old["end"] != r["end"] or (r["rss"] or 0) != (old["rss"] or 0):
                mark = st.bold("*")
        addr = f"{r['start']:012x}-{r['end']:012x}"
        fixed = f"{idx:>4} {addr:<25} {perms:<5} {human(r['size']):>7} {human(rss):>7}"
        room = max(10, width - len(fixed) - bar_w - 7)
        lines.append(f"{fixed}{mark} {bar}  {st.swatch(r['cat'])} {r['label'][:room]}")
        if i + 1 < len(rows):
            gap = r["start"] - rows[i + 1]["end"]
            if gap > 0:
                lines.append(st.dim(f"{'':>4} {'':<25} {'':<5} {'┆ hole':>7} {human(gap):>7}  unmapped"))
        if max_rows and len(lines) >= max_rows:
            left = len(rows) - i - 1
            if left:
                lines.append(st.dim(f"     ... {left} more rows (make the terminal taller or use --compact)"))
            break
    return lines


def summary_lines(regions, st, prev_regions=None):
    summ = summarize(regions)
    prev = summarize(prev_regions) if prev_regions is not None else None
    total_rss = sum(c["rss"] for c in summ.values()) or 1
    lines = [st.bold(f"  {'category':<27}{'regions':>8}{'virtual':>9}{'RSS':>9}{'PSS':>9}{'swap':>8}"
                     + ("   ΔRSS" if prev else "") + "   share of RSS")]
    for k in CAT_KEYS:
        c = summ[k]
        if c["n"] == 0 and not (prev and prev[k]["n"]):
            continue
        share = c["rss"] / total_rss
        bar = st.fg(CAT_COLOR[k], "█" * round(24 * share))
        delta = ""
        if prev:
            d = c["rss"] - prev[k]["rss"]
            delta = f"{('+' if d > 0 else '') + human(d) if d else '':>7}"
        lines.append(f"{st.swatch(k)} {CAT_NAME[k]:<27}{c['n']:>8}{human(c['size']):>9}{human(c['rss']):>9}"
                     f"{human(c['pss']):>9}{human(c['swap']):>8}" + (f" {delta}" if prev else "")
                     + f"   {bar} {100 * share:4.1f}%")
    return lines


def sparkline(values, width):
    values = values[-width:]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    ticks = "▁▂▃▄▅▆▇█"
    if hi == lo:
        return ticks[0] * len(values)
    return "".join(ticks[min(7, int((v - lo) / (hi - lo) * 8))] for v in values)


def print_pages(pid, regions, idx, st, width):
    if not 0 <= idx < len(regions):
        sys.exit(f"region #{idx} does not exist (0..{len(regions) - 1})")
    r = regions[idx]
    npages = r["size"] // PAGE
    if npages > (1 << 22):
        sys.exit(f"region #{idx} has {npages} pages, too many to draw (limit {1 << 22})")
    a = read_pagemap(pid, r["start"], npages)
    if a is None or len(a) == 0:
        sys.exit("cannot read /proc/PID/pagemap (needs the same user or root)")
    cols = max(16, min(64, width - 20))
    cells = min(len(a), cols * 32)
    per = len(a) / cells
    print(st.bold(f"Region #{idx}: {r['start']:012x}-{r['end']:012x} {r['perms']} {human(r['size'])}  ")
          + f"{st.swatch(r['cat'])} {r['label']}")
    print(st.dim(f"{len(a)} pages of {PAGE} B; each cell = {per:.1f} page(s); "
                 "█ all resident ▓ ≥50% ▒ some · none S swapped; low address top-left"))
    present = swapped = exclusive = filesh = 0
    for v in a:
        present += v >> 63
        swapped += (v >> 62) & 1
        filesh += (v >> 61) & 1
        exclusive += (v >> 56) & 1
    color = CAT_COLOR[r["cat"]]
    for row_start in range(0, cells, cols):
        line = []
        for c in range(row_start, min(cells, row_start + cols)):
            lo, hi = int(c * per), max(int(c * per) + 1, int((c + 1) * per))
            chunk = a[lo:hi]
            p = sum(v >> 63 for v in chunk) / len(chunk)
            if any((v >> 62) & 1 for v in chunk):
                line.append("S")
            elif p >= 1:
                line.append(st.fg(color, "█"))
            elif p >= 0.5:
                line.append(st.fg(color, "▓"))
            elif p > 0:
                line.append(st.fg(color, "▒"))
            else:
                line.append(st.dim("·"))
        addr = r["start"] + int(row_start * per) * PAGE
        print(f"{addr:012x} " + "".join(line))
    n = len(a)
    print(f"resident {present}/{n} pages ({100 * present / n:.1f}%) = {human(present * PAGE)}, "
          f"swapped {swapped}, file-page or shared-anon {filesh}, exclusively mapped {exclusive}")


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------

def snapshot_to_json(snap):
    regs = snap["regions"]
    out = []
    for i, r in enumerate(regs):
        gap = (regs[i + 1]["start"] - r["end"]) if i + 1 < len(regs) else 0
        out.append({"s": f"{r['start']:016x}", "e": f"{r['end']:016x}", "z": r["size"], "p": r["perms"],
                    "f": r["path"], "c": r["cat"], "l": r["label"], "o": r["obj"], "r": r["rss"],
                    "q": r["pss"], "w": r["swap"], "h": r["huge"], "x": r["res"], "g": float(gap)})
    st = {k: v for k, v in snap["status"].items()}
    return {"t": snap["t"], "status": st, "detailed": snap["detailed"], "regions": out}


def write_html(path, snaps, events, title):
    data = {"title": title, "page": PAGE, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": [{"key": k, "name": n, "light": l, "dark": d} for k, n, l, d in CATEGORIES],
            "snapshots": [snapshot_to_json(s) for s in snaps],
            "events": [{"t": t, "k": k, "text": txt} for t, k, txt in events]}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = HTML_TEMPLATE.replace("__TITLE__", html_escape(title)).replace("__DATA__", payload)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# --------------------------------------------------------------------------
# Watching / recording
# --------------------------------------------------------------------------

def start_child(cmd, capture):
    try:
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE if capture else None,
                                 stderr=subprocess.STDOUT if capture else None, text=True, bufsize=1)
    except OSError as e:
        sys.exit(f"cannot start {' '.join(cmd)}: {e}")
    lines = queue.Queue()
    if capture:
        def pump():
            for line in child.stdout:
                lines.put((time.time(), line.rstrip()))
        threading.Thread(target=pump, daemon=True).start()
    return child, lines


def stop_child(child):
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def diff_events(prev, cur):
    """Describe what changed between two snapshots."""
    ev = []
    old = {r["start"]: r for r in prev["regions"]}
    new = {r["start"]: r for r in cur["regions"]}
    # A region that grows downwards (the stack) keeps its end address, not its start.
    old_by_end = {r["end"]: r for r in prev["regions"] if r["start"] not in new}
    moved = set()
    for s, r in new.items():
        o = old.get(s)
        if o is None and r["end"] in old_by_end and old_by_end[r["end"]]["path"] == r["path"]:
            o = old_by_end[r["end"]]
            moved.add(o["start"])
            ev.append(f"~ region ending at {r['end']:x} grew down {human(o['size'])} -> {human(r['size'])}: "
                      f"{r['label']}")
            continue
        if o is None:
            ev.append(f"+ new {human(r['size'])} region at {s:x}: {r['label']}")
        elif o["end"] != r["end"]:
            ev.append(f"~ region at {s:x} resized {human(o['size'])} -> {human(r['size'])}: {r['label']}")
        elif abs((r["rss"] or 0) - (o["rss"] or 0)) >= (1 << 20):
            d = (r["rss"] or 0) - (o["rss"] or 0)
            ev.append(f"{'▲' if d > 0 else '▼'} RSS {'+' if d > 0 else ''}{human(d)} in {r['label']}")
    for s, o in old.items():
        if s not in new and s not in moved:
            ev.append(f"- unmapped {human(o['size'])} region at {s:x}: {o['label']}")
    return ev


def watch(pid, args, st, child=None, child_lines=None):
    snaps, events, prev = [], [], None
    history = []
    t0 = time.time()
    clear = st.color and sys.stdout.isatty()
    try:
        while True:
            if child_lines is not None:
                while not child_lines.empty():
                    t, line = child_lines.get()
                    events.append((t, "out", line))     # a line printed by the watched program
            try:
                snap = take_snapshot(pid, pagemap=not args.no_pagemap)
            except ProcessGone:
                print(st.bold(f"\nprocess {pid} has exited"))
                break
            snaps.append(snap)
            history.append(totals(snap)["rss"])
            if prev is not None:
                for e in diff_events(prev, snap):
                    events.append((snap["t"], "map", e))  # a change seen in the memory map
            if args.watch:
                draw_live(snap, prev, events, history, st, clear, args.compact)
            elif len(snaps) % 10 == 1:
                print(f"recording... {len(snaps)} snapshots, {time.time() - t0:.0f} s", file=sys.stderr)
            prev = snap
            if args.record and time.time() - t0 >= args.record:
                break
            if child is not None and child.poll() is not None:
                break
            time.sleep(args.watch if args.watch else args.interval)
    except KeyboardInterrupt:
        print()
    return snaps, events


def draw_live(snap, prev, events, history, st, clear, compact):
    size = shutil.get_terminal_size((120, 40))
    width, height = size.columns, size.lines
    out = header_lines(snap, st)
    out.append(f"RSS history {st.fg(CAT_COLOR['program'], sparkline(history, 40))} "
               f"now {human(history[-1])}, max {human(max(history))}")
    out += overview_lines(snap["regions"], st, width)
    out += summary_lines(snap["regions"], st, prev["regions"] if prev else None)
    out.append(st.bold("recent events"))
    out += [st.dim(time.strftime("%H:%M:%S ", time.localtime(t))) + e[: width - 10] for t, _, e in events[-6:]] \
        or [st.dim("  (none yet)")]
    regions = group_regions(snap["regions"]) if compact else snap["regions"]
    index_of = {r["start"]: i for i, r in enumerate(snap["regions"])}
    prev_index = None
    if prev:
        prev_rows = group_regions(prev["regions"]) if compact else prev["regions"]
        prev_index = {r["start"]: r for r in prev_rows}
    room = max(5, height - len(out) - 3)
    out += region_lines(regions, st, width, index_of, prev_index, max_rows=room)
    out.append(st.dim("+ new region   * changed size/RSS   Ctrl-C to stop"))
    text = "\n".join(line + ("\033[K" if clear else "") for line in out)
    if clear:
        sys.stdout.write("\033[H" + text + "\n\033[J")
    else:
        sys.stdout.write(text + "\n\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------
# --list
# --------------------------------------------------------------------------

def list_processes(st):
    uid = os.getuid()
    rows = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if uid != 0 and os.stat(f"/proc/{d}").st_uid != uid:
                continue
            s = read_status(int(d))
        except OSError:
            continue
        rows.append((s.get("VmRSS", 0), s.get("VmSize", 0), int(d), s.get("Name", "?"), s.get("cmdline", "")))
    rows.sort(reverse=True)
    print(st.bold(f"{'PID':>7} {'RSS':>8} {'virtual':>8}  name / command line"))
    for rss, vsz, pid, name, cmd in rows[:40]:
        print(f"{pid:>7} {human(rss):>8} {human(vsz):>8}  {name:<16} {st.dim(cmd[:60])}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualise the memory map of a Linux process.",
                                 epilog="examples: mmap.py self | mmap.py 1234 --watch 1 | "
                                        "mmap.py --run ./demo --record 40 --html demo.html")
    ap.add_argument("pid", nargs="?", help="process id, or 'self'")
    ap.add_argument("--run", metavar="CMD",
                    help='start CMD and visualise it; quote it if it has arguments, e.g. --run "./demo -s 1"')
    ap.add_argument("--file", help="read a saved maps or smaps file instead of a live process")
    ap.add_argument("--list", action="store_true", help="list processes you can inspect")
    ap.add_argument("--watch", type=float, metavar="SEC", help="real-time terminal view, refresh every SEC s")
    ap.add_argument("--record", type=float, metavar="SEC", help="take snapshots for SEC seconds (with --html)")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between snapshots for --record")
    ap.add_argument("--delay", type=float, default=1.0, help="with --run and no --watch/--record: wait before the snapshot")
    ap.add_argument("--html", metavar="FILE", help="write an interactive HTML report")
    ap.add_argument("--json", metavar="FILE", help="write the snapshot(s) as JSON")
    ap.add_argument("--compact", action="store_true", help="one row per file/library instead of one per region")
    ap.add_argument("--pages", type=int, metavar="N", help="draw page residency of region #N (from pagemap)")
    ap.add_argument("--no-pagemap", action="store_true", help="do not read /proc/PID/pagemap")
    ap.add_argument("--no-color", action="store_true", help="plain text output")
    args = ap.parse_args()

    st = Style(not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
    # When the output goes to a file or a pipe, do not cut the descriptions short.
    width = shutil.get_terminal_size((120, 40)).columns if sys.stdout.isatty() else 200

    if not args.file and not os.path.exists("/proc/self/maps"):
        sys.exit("this tool needs Linux /proc (use Linux or WSL), or pass --file with a saved maps file")
    if args.list:
        list_processes(st)
        return
    if args.watch is not None and args.watch <= 0 or args.interval <= 0:
        sys.exit("--watch and --interval must be positive")

    child, child_lines = None, None
    if args.file:
        snaps, events = [snapshot_from_file(args.file)], []
        pid = None
    else:
        if args.run:
            child, child_lines = start_child(shlex.split(args.run), capture=bool(args.watch or args.record))
            pid = child.pid
        elif args.pid in (None, "self"):
            pid = os.getpid()
        elif args.pid.isdigit():
            pid = int(args.pid)
        else:
            sys.exit("pid must be a number or 'self' (use --list to find one)")
        if not os.path.exists(f"/proc/{pid}"):
            sys.exit(f"no process with PID {pid}")
        try:
            if args.watch or args.record:
                snaps, events = watch(pid, args, st, child, child_lines)
            else:
                if child is not None:
                    time.sleep(args.delay)
                snaps, events = [take_snapshot(pid, pagemap=not args.no_pagemap)], []
        except ProcessGone:
            sys.exit(f"process {pid} exited before it could be read")
        except PermissionError as e:
            sys.exit(f"permission denied: {e.filename} (inspect your own processes, or run as root)")
        finally:
            stop_child(child)
        if not snaps:
            sys.exit("no snapshot was taken")

    last = snaps[-1]
    if args.pages is not None:
        if pid is None:
            sys.exit("--pages needs a live process")
        print_pages(pid, last["regions"], args.pages, st, width)
        return
    if not args.watch:
        regions = group_regions(last["regions"]) if args.compact else last["regions"]
        index_of = {r["start"]: i for i, r in enumerate(last["regions"])}
        out = header_lines(last, st) + [""]
        out += region_lines(regions, st, width, index_of) + [""]
        out += overview_lines(last["regions"], st, width) + [""]
        out += summary_lines(last["regions"], st)
        print("\n".join(out))

    name = last["status"].get("Name", "process")
    title = f"{name} (PID {last['pid']})" if last["pid"] else name
    if args.html:
        write_html(args.html, snaps, events, title)
        print(f"wrote {args.html} ({len(snaps)} snapshot{'s' if len(snaps) > 1 else ''}) - open it in a web browser")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"title": title, "snapshots": [snapshot_to_json(s) for s in snaps],
                       "events": [{"t": t, "k": k, "text": e} for t, k, e in events]}, f, indent=1)
        print(f"wrote {args.json}")


# --------------------------------------------------------------------------
# HTML template (self-contained: no external files or libraries)
# --------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory map: __TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --hover: rgba(11,11,11,0.04);
  --c-program: #2a78d6; --c-heap: #eb6834; --c-anon: #1baf7a; --c-stack: #eda100;
  --c-lib: #e87ba4; --c-file: #008300; --c-shm: #6250d6; --c-other: #898781;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.05);
    --c-program: #3987e5; --c-heap: #d95926; --c-anon: #199e70; --c-stack: #c98500;
    --c-lib: #d55181; --c-file: #008300; --c-shm: #9085e9; --c-other: #898781;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.05);
  --c-program: #3987e5; --c-heap: #d95926; --c-anon: #199e70; --c-stack: #c98500;
  --c-lib: #d55181; --c-file: #008300; --c-shm: #9085e9; --c-other: #898781;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 2px; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--ink2); font-size: 13px; }
.muted { color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--ink2); font-size: 12px; }
.tile .value { font-size: 22px; font-weight: 600; margin-top: 2px; }
.tile .delta { font-size: 12px; color: var(--muted); }
.cols { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 16px; }
@media (max-width: 860px) { .cols { grid-template-columns: minmax(0, 1fr); } }
.controls { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; margin-top: 10px; font-size: 13px; color: var(--ink2); }
.controls input[type=range] { flex: 1 1 240px; }
button { font: inherit; font-size: 13px; color: var(--ink); background: var(--page); border: 1px solid var(--border);
         border-radius: 6px; padding: 4px 10px; cursor: pointer; }
button:hover { background: var(--hover); }
input[type=search] { font: inherit; font-size: 13px; color: var(--ink); background: var(--page); border: 1px solid var(--border);
         border-radius: 6px; padding: 4px 8px; min-width: 220px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px; font-size: 12px; color: var(--ink2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.legend span.off { opacity: 0.35; }
.sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; flex: none; }
svg text { fill: var(--ink2); font-size: 11px; }
svg .addr { fill: var(--muted); font-variant-numeric: tabular-nums; font-family: ui-monospace, Consolas, monospace; font-size: 10.5px; }
svg .lbl { fill: var(--ink); font-size: 12px; }
svg .gaptext { fill: var(--muted); font-size: 10.5px; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
.mapwrap { max-height: 900px; overflow-y: auto; margin-top: 8px; }
.row.dim { opacity: 0.18; }
.row .hit { fill: transparent; }
.row:hover .hit, .row.sel .hit { fill: var(--hover); }
.sumrow { display: grid; grid-template-columns: 150px minmax(0, 1fr) 72px; gap: 10px; align-items: center; padding: 5px 0;
          font-size: 13px; border-bottom: 1px solid var(--grid); }
.sumrow:last-child { border-bottom: 0; }
.sumrow .name { display: flex; gap: 8px; align-items: center; }
.sumrow .track { height: 10px; position: relative; }
.sumrow .bar { position: absolute; left: 0; top: 0; height: 10px; border-radius: 0 4px 4px 0; }
.sumrow .num { text-align: right; font-variant-numeric: tabular-nums; }
.sumrow .detail { grid-column: 2 / 4; font-size: 11.5px; color: var(--muted); margin-top: -4px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th { text-align: left; font-weight: 600; color: var(--ink2); border-bottom: 1px solid var(--axis); padding: 6px 8px; cursor: pointer; white-space: nowrap; }
td { border-bottom: 1px solid var(--grid); padding: 4px 8px; vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.mono { font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; white-space: nowrap; }
tr.sel td { background: var(--hover); }
.tablewrap { overflow-x: auto; max-height: 560px; overflow-y: auto; margin-top: 8px; }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface); color: var(--ink);
       border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; max-width: 360px;
       box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; }
#tip b { font-weight: 600; }
#tip .k { color: var(--muted); }
.note { font-size: 12px; color: var(--muted); margin-top: 6px; }
</style>
</head>
<body>
<main>
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap">
    <div>
      <h1>Memory map: __TITLE__</h1>
      <div class="sub" id="subtitle"></div>
    </div>
    <button id="theme" title="Switch light / dark">Theme</button>
  </div>

  <div class="tiles" id="tiles"></div>

  <section class="card" id="timelineCard" hidden>
    <h2>Resident memory over time, by category</h2>
    <div class="sub">Stacked RSS of each region category. Click or drag on the chart, or use the slider, to choose the snapshot shown below.</div>
    <div id="timeline" style="margin-top:8px"></div>
    <div class="controls">
      <button id="play">Play</button>
      <input type="range" id="slider" min="0" max="0" value="0" aria-label="snapshot">
      <span id="snapLabel" class="muted"></span>
    </div>
    <div id="eventNow" class="note"></div>
  </section>

  <div class="cols">
    <section class="card">
      <h2>Address space</h2>
      <div class="sub">High addresses at the top. Block height grows with log(size); the filled part of each horizontal slice is the share of its pages that are resident in RAM (from pagemap, or RSS/size). Holes between regions are shortened.</div>
      <div class="controls">
        <label><input type="checkbox" id="group"> one block per file / library</label>
        <label><input type="checkbox" id="hideGuard"> hide guard / reserved regions</label>
      </div>
      <div class="legend" id="legend"></div>
      <div class="mapwrap"><div id="map"></div></div>
    </section>
    <section class="card">
      <h2>Where the resident memory is</h2>
      <div class="sub">RSS per category for the selected snapshot (bar length), with virtual size and region count below.</div>
      <div id="summary" style="margin-top:8px"></div>
      <h2 style="margin-top:18px">Events</h2>
      <div id="events" class="note"></div>
    </section>
  </div>

  <section class="card">
    <h2>All regions</h2>
    <div class="controls"><input type="search" id="filter" placeholder="Filter by name, permission, category..."><span class="muted" id="count"></span></div>
    <div class="tablewrap"><table id="table"></table></div>
  </section>
  <div class="note">Generated by mmap.py on <span id="created"></span>. Source: /proc/PID/smaps, /proc/PID/pagemap, /proc/PID/status.</div>
</main>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const PAGE = DATA.page;
const CATS = DATA.categories;
const CAT = Object.fromEntries(CATS.map(c => [c.key, c]));
const snaps = DATA.snapshots;
const t0 = snaps[0].t;
let cur = snaps.length - 1, hidden = new Set(), selected = null, sortKey = 's', sortDir = -1, timer = null;

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function human(n) {
  if (n === null || n === undefined) return '–';
  const u = ['B','KiB','MiB','GiB','TiB','PiB','EiB']; let i = 0; n = Number(n);
  while (Math.abs(n) >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return i === 0 ? `${Math.round(n)} B` : `${n < 10 ? n.toFixed(2) : n < 100 ? n.toFixed(1) : n.toFixed(0)} ${u[i]}`;
}
const color = k => `var(--c-${k})`;
const shortAddr = h => h.replace(/^0+(?=.{12})/, '');

function totals(s) {
  let v = 0, r = 0, p = 0, w = 0;
  for (const x of s.regions) { v += x.z; r += x.r || 0; p += x.q || 0; w += x.w || 0; }
  return {vsz: v, rss: r, pss: p, swap: w, n: s.regions.length};
}
function byCat(s) {
  const o = Object.fromEntries(CATS.map(c => [c.key, {n: 0, z: 0, r: 0, q: 0, w: 0}]));
  for (const x of s.regions) { const c = o[x.c]; c.n++; c.z += x.z; c.r += x.r || 0; c.q += x.q || 0; c.w += x.w || 0; }
  return o;
}

/* ---------- tooltip ---------- */
const tip = $('tip');
function showTip(ev, html) {
  tip.innerHTML = html; tip.style.display = 'block';
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
}
const hideTip = () => { tip.style.display = 'none'; };
function regionTip(x) {
  const pct = x.z ? (100 * (x.r || 0) / x.z).toFixed(1) + '%' : '–';
  return `<div style="display:flex;gap:6px;align-items:center"><span class="sw" style="background:${color(x.c)}"></span><b>${esc(CAT[x.c].name)}</b></div>
    <div style="margin:4px 0">${esc(x.l)}</div>
    <div><span class="k">range</span> ${shortAddr(x.s)}–${shortAddr(x.e)}</div>
    <div><span class="k">perms</span> ${esc(x.p)} &nbsp; <span class="k">size</span> ${human(x.z)}</div>
    <div><span class="k">RSS</span> ${human(x.r)} (${pct}) &nbsp; <span class="k">PSS</span> ${human(x.q)}</div>
    <div><span class="k">swap</span> ${human(x.w)} &nbsp; <span class="k">huge pages</span> ${human(x.h)}</div>
    ${x.f ? `<div><span class="k">backing</span> ${esc(x.f)}</div>` : ''}`;
}

/* ---------- items for the map / table (optionally grouped) ---------- */
function items() {
  let regs = snaps[cur].regions.map((x, i) => Object.assign({idx: i, count: 1}, x));
  if ($('group').checked) {
    const out = [];
    for (const x of regs) {
      const g = out[out.length - 1];
      if (g && x.o && g.o === x.o && g.e === x.s && g.c === x.c) {
        g.x = mergeRes(g, x); g.e = x.e; g.z += x.z; g.count++;
        for (const k of ['r','q','w','h']) if (x[k] !== null) g[k] = (g[k] || 0) + x[k];
        g.perms.push(x.p); g.g = x.g;
      } else out.push(Object.assign({}, x, {perms: [x.p]}));
    }
    for (const g of out) if (g.count > 1) {
      g.l = `${g.o.split('/').pop()} (${g.count} regions: ${g.perms.map(p => p.slice(0, 3)).join(' ')})`;
      g.p = `${g.count} maps`;
    }
    regs = out;
  }
  if ($('hideGuard').checked) regs = regs.filter(x => !(x.c === 'other' && x.p.startsWith('---')));
  return regs;
}
function resOf(x) { return x.x || [x.z ? Math.round(100 * (x.r || 0) / x.z) : 0]; }
function mergeRes(a, b) {          /* concatenate two residency profiles weighted by size */
  const ra = resOf(a), rb = resOf(b), n = 48, out = [];
  const total = a.z + b.z;
  for (let i = 0; i < n; i++) {
    const pos = (i + 0.5) / n * total;
    if (pos < a.z) out.push(ra[Math.min(ra.length - 1, Math.floor(pos / a.z * ra.length))]);
    else out.push(rb[Math.min(rb.length - 1, Math.floor((pos - a.z) / b.z * rb.length))]);
  }
  return out;
}
function resample(res, k) {
  if (res.length <= k) return res;
  const out = [];
  for (let i = 0; i < k; i++) {
    const lo = Math.floor(i * res.length / k), hi = Math.max(lo + 1, Math.floor((i + 1) * res.length / k));
    let s = 0; for (let j = lo; j < hi; j++) s += res[j];
    out.push(s / (hi - lo));
  }
  return out;
}

/* ---------- address-space map ---------- */
function renderMap() {
  const regs = items().slice().reverse();          /* high addresses first */
  const W = 620, addrW = 96, colX = addrW + 8, colW = 150, lblX = colX + colW + 12;
  let y = 6; const parts = [];
  regs.forEach((x, i) => {
    const h = Math.max(20, Math.round(6.5 * Math.log2(x.z / PAGE + 1)));
    const inner = h - 2;                               /* 2px surface gap between blocks */
    const res = resample(resOf(x), Math.max(1, Math.floor(inner / 2)));
    const k = res.length, sh = inner / k;
    const dim = hidden.has(x.c) ? ' dim' : '';
    const sel = selected === x.idx ? ' sel' : '';
    let g = `<g class="row${dim}${sel}" data-i="${x.idx}">
      <rect class="hit" x="0" y="${y}" width="${W}" height="${h}" rx="4"></rect>
      <rect x="${colX}" y="${y}" width="${colW}" height="${inner}" rx="3" fill="${color(x.c)}" fill-opacity="0.16"></rect>`;
    for (let b = 0; b < k; b++) {
      const w = colW * res[b] / 100;
      if (w <= 0) continue;
      const sy = y + inner - (b + 1) * sh;          /* bucket 0 = lowest address = bottom */
      g += `<rect x="${colX}" y="${sy.toFixed(2)}" width="${w.toFixed(1)}" height="${(sh + 0.02).toFixed(2)}" fill="${color(x.c)}"></rect>`;
    }
    const pct = x.z ? Math.round(100 * (x.r || 0) / x.z) : 0;
    const label = x.l.length > 46 ? x.l.slice(0, 45) + '…' : x.l;
    g += `<text class="addr" x="${addrW}" y="${y + inner - 2}" text-anchor="end">${shortAddr(x.s)}</text>
      <text class="lbl" x="${lblX}" y="${y + Math.min(14, inner - 3)}">${esc(label)}</text>`;
    if (inner >= 26) g += `<text x="${lblX}" y="${y + 28}">${x.p} · ${human(x.z)} · RSS ${human(x.r)}${x.r !== null ? ` (${pct}%)` : ''}</text>`;
    g += `</g>`;
    parts.push(g);
    y += h;
    const next = regs[i + 1];
    if (next) {
      /* addresses are 64-bit hex strings: BigInt keeps them exact */
      const gap = Number(BigInt('0x' + x.s) - BigInt('0x' + next.e));
      if (gap > 0) {
        parts.push(`<g><line class="axis" x1="${colX + 30}" y1="${y + 4}" x2="${colX + 42}" y2="${y + 10}"></line>
          <line class="axis" x1="${colX + 36}" y1="${y + 4}" x2="${colX + 48}" y2="${y + 10}"></line>
          <text class="gaptext" x="${colX + 56}" y="${y + 11}">unmapped hole ${human(gap)}</text></g>`);
        y += 16;
      }
    }
  });
  $('map').innerHTML = `<svg viewBox="0 0 ${W} ${y + 6}" width="100%" style="max-width:${W}px" role="img" aria-label="address space map">${parts.join('')}</svg>`;
  const regsById = Object.fromEntries(regs.map(x => [x.idx, x]));
  $('map').querySelectorAll('.row').forEach(el => {
    const x = regsById[el.dataset.i];
    el.addEventListener('mousemove', ev => showTip(ev, regionTip(x)));
    el.addEventListener('mouseleave', hideTip);
    el.addEventListener('click', () => { selected = x.idx; renderMap(); renderTable(true); });
  });
}

/* ---------- legend ---------- */
function renderLegend() {
  const used = new Set(snaps.flatMap(s => s.regions.map(x => x.c)));
  $('legend').innerHTML = CATS.filter(c => used.has(c.key)).map(c =>
    `<span data-k="${c.key}" class="${hidden.has(c.key) ? 'off' : ''}" title="click to highlight / dim"><i class="sw" style="background:${color(c.key)}"></i>${esc(c.name)}</span>`).join('');
  $('legend').querySelectorAll('span').forEach(el => el.addEventListener('click', () => {
    const k = el.dataset.k; hidden.has(k) ? hidden.delete(k) : hidden.add(k); renderLegend(); renderMap();
  }));
}

/* ---------- tiles ---------- */
function renderTiles() {
  const s = snaps[cur], t = totals(s), st = s.status;
  const p = cur > 0 ? totals(snaps[cur - 1]) : null;
  const d = (a, b) => p ? `${a - b >= 0 ? '+' : '−'}${human(Math.abs(a - b))} vs previous` : '';
  const tiles = [
    ['Virtual size', human(st.VmSize ?? t.vsz), d(t.vsz, p?.vsz)],
    ['Resident (RSS)', s.detailed ? human(st.VmRSS ?? t.rss) : '–', d(t.rss, p?.rss)],
    ['Proportional (PSS)', s.detailed ? human(t.pss) : '–', ''],
    ['Swapped out', human(st.VmSwap ?? t.swap), ''],
    ['Regions (VMAs)', String(t.n), p ? `${t.n - p.n >= 0 ? '+' : '−'}${Math.abs(t.n - p.n)} vs previous` : ''],
    ['Threads', st.Threads ?? '–', ''],
  ];
  $('tiles').innerHTML = tiles.map(([l, v, dl]) => `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="delta">${dl || '&nbsp;'}</div></div>`).join('');
  const when = new Date(s.t * 1000).toLocaleString();
  $('subtitle').textContent = `${st.cmdline || ''} · snapshot ${cur + 1} of ${snaps.length} · ${when}`;
}

/* ---------- summary ---------- */
function renderSummary() {
  const c = byCat(snaps[cur]);
  const max = Math.max(1, ...Object.values(c).map(v => v.r));
  $('summary').innerHTML = CATS.filter(k => c[k.key].n > 0).map(k => {
    const v = c[k.key];
    return `<div class="sumrow"><div class="name"><span class="sw" style="background:${color(k.key)}"></span>${esc(k.name)}</div>
      <div class="track"><div class="bar" style="width:${(100 * v.r / max).toFixed(1)}%;background:${color(k.key)}"></div></div>
      <div class="num">${human(v.r)}</div>
      <div class="detail">virtual ${human(v.z)} · ${v.n} region${v.n > 1 ? 's' : ''}${v.w ? ' · swap ' + human(v.w) : ''}</div></div>`;
  }).join('');
}

/* ---------- table ---------- */
const COLS = [
  ['i', '#', 'n'], ['s', 'start', 'mono'], ['e', 'end', 'mono'], ['p', 'perms', 'mono'], ['z', 'size', 'n'],
  ['r', 'RSS', 'n'], ['q', 'PSS', 'n'], ['w', 'swap', 'n'], ['c', 'category', ''], ['l', 'description', '']];
function renderTable(scroll) {
  const q = $('filter').value.trim().toLowerCase();
  let regs = items().map(x => Object.assign({i: x.idx}, x));
  if (q) regs = regs.filter(x => `${x.l} ${x.f} ${x.p} ${CAT[x.c].name}`.toLowerCase().includes(q));
  regs.sort((a, b) => {
    let va = a[sortKey], vb = b[sortKey];
    if (sortKey === 'c') { va = CAT[a.c].name; vb = CAT[b.c].name; }
    if (va === null) va = -1; if (vb === null) vb = -1;
    return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir;
  });
  const head = '<tr>' + COLS.map(([k, n, cls]) => `<th class="${cls === 'n' ? 'n' : ''}" data-k="${k}">${n}${sortKey === k ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('') + '</tr>';
  const body = regs.map(x => `<tr data-i="${x.idx}" class="${selected === x.idx ? 'sel' : ''}">
    <td class="n">${x.idx}</td><td class="mono">${shortAddr(x.s)}</td><td class="mono">${shortAddr(x.e)}</td><td class="mono">${esc(x.p)}</td>
    <td class="n">${human(x.z)}</td><td class="n">${human(x.r)}</td><td class="n">${human(x.q)}</td><td class="n">${human(x.w)}</td>
    <td><span style="display:inline-flex;gap:6px;align-items:center;white-space:nowrap"><span class="sw" style="background:${color(x.c)}"></span>${esc(CAT[x.c].name)}</span></td>
    <td>${esc(x.l)}${x.f && !x.l.includes(x.f) ? `<div class="muted" style="font-size:11px">${esc(x.f)}</div>` : ''}</td></tr>`).join('');
  $('table').innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  $('count').textContent = `${regs.length} row${regs.length === 1 ? '' : 's'}`;
  $('table').querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
    const k = th.dataset.k; if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = ['s','e','i','p','c','l'].includes(k) ? 1 : -1; }
    renderTable();
  }));
  $('table').querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => {
    selected = Number(tr.dataset.i); renderMap(); renderTable();
  }));
  if (scroll) { const tr = $('table').querySelector('tr.sel'); if (tr) tr.scrollIntoView({block: 'nearest'}); }
}

/* ---------- events ---------- */
function renderEvents() {
  const evs = DATA.events;
  if (!evs.length) { $('events').textContent = snaps.length > 1 ? 'No events were recorded.' : 'Events appear when a timeline is recorded (--record or --watch).'; return; }
  const tNow = snaps[cur].t;
  const recent = evs.filter(e => e.t <= tNow + 0.001).slice(-12);
  $('events').innerHTML = recent.map(e => `<div><span style="font-variant-numeric:tabular-nums">${(e.t - t0).toFixed(1)} s</span> · ${esc(e.text)}</div>`).join('') || 'No events yet at this time.';
  const last = evs.filter(e => e.t <= tNow + 0.001 && e.k === 'out').slice(-1)[0];
  $('eventNow').textContent = last ? `Program output at this time: ${last.text}` : '';
}

/* ---------- timeline (stacked area of RSS by category) ---------- */
function niceStep(max, n) {
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
}
function renderTimeline() {
  if (snaps.length < 2) return;
  $('timelineCard').hidden = false;
  const W = 900, H = 230, L = 64, R = 16, T = 10, B = 28;
  const xs = snaps.map(s => s.t - t0), tmax = Math.max(...xs) || 1;
  const cats = CATS.map(c => c.key).filter(k => snaps.some(s => s.regions.some(x => x.c === k && (x.r || 0) > 0)));
  const stacks = snaps.map(s => { const c = byCat(s); let acc = 0; return cats.map(k => { const lo = acc; acc += c[k].r; return [lo, acc]; }); });
  const ymaxRaw = Math.max(...stacks.map(s => s.length ? s[s.length - 1][1] : 0), 1);
  const MiB = 1048576, step = niceStep(ymaxRaw / MiB, 4) * MiB, ymax = Math.ceil(ymaxRaw / step) * step;
  const X = t => L + (W - L - R) * t / tmax, Y = v => T + (H - T - B) * (1 - v / ymax);
  let g = '';
  for (let v = 0; v <= ymax + 1; v += step) g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"></line><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${human(v)}</text>`;
  const xstep = niceStep(tmax, 8);
  for (let t = 0; t <= tmax + 1e-9; t += xstep) g += `<text x="${X(t)}" y="${H - 8}" text-anchor="middle">${+t.toFixed(1)} s</text>`;
  g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}"></line>`;
  cats.forEach((k, ci) => {
    const top = snaps.map((s, i) => `${X(xs[i]).toFixed(1)},${Y(stacks[i][ci][1]).toFixed(1)}`);
    const bot = snaps.map((s, i) => `${X(xs[i]).toFixed(1)},${Y(stacks[i][ci][0]).toFixed(1)}`).reverse();
    g += `<polygon points="${top.concat(bot).join(' ')}" fill="${color(k)}" fill-opacity="0.85" stroke="var(--surface)" stroke-width="1.5" stroke-linejoin="round"></polygon>`;
  });
  for (const e of DATA.events.filter(e => e.k === 'out')) {
    const x = X(Math.min(tmax, Math.max(0, e.t - t0)));
    g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${T + 6}" stroke="var(--muted)" stroke-width="1.5"></line>`;
  }
  g += `<line id="cursor" x1="${X(xs[cur])}" x2="${X(xs[cur])}" y1="${T}" y2="${Y(0)}" stroke="var(--ink)" stroke-width="1.5"></line>`;
  g += `<rect id="hitT" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent" style="cursor:crosshair"></rect>`;
  $('timeline').innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="RSS over time">${g}</svg>
    <div class="legend">${cats.map(k => `<span><i class="sw" style="background:${color(k)}"></i>${esc(CAT[k].name)}</span>`).join('')}</div>`;
  const svg = $('timeline').querySelector('svg'), hit = $('hitT');
  const nearest = ev => {
    const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
    const t = (x - L) / (W - L - R) * tmax; let best = 0;
    xs.forEach((v, i) => { if (Math.abs(v - t) < Math.abs(xs[best] - t)) best = i; }); return best;
  };
  let dragging = false;
  hit.addEventListener('mousemove', ev => {
    const i = nearest(ev), c = byCat(snaps[i]);
    const rows = cats.slice().reverse().map(k => `<div style="display:flex;gap:6px;align-items:center"><span class="sw" style="background:${color(k)}"></span>${esc(CAT[k].name)}<span style="margin-left:auto;padding-left:12px">${human(c[k].r)}</span></div>`).join('');
    showTip(ev, `<b>${xs[i].toFixed(1)} s</b> · total RSS ${human(totals(snaps[i]).rss)}${rows}`);
    if (dragging) select(i);
  });
  hit.addEventListener('mouseleave', () => { hideTip(); dragging = false; });
  hit.addEventListener('mousedown', ev => { dragging = true; select(nearest(ev)); });
  addEventListener('mouseup', () => { dragging = false; });
}

function select(i) {
  cur = Math.max(0, Math.min(snaps.length - 1, i));
  $('slider').value = cur;
  $('snapLabel').textContent = `snapshot ${cur + 1}/${snaps.length} · ${(snaps[cur].t - t0).toFixed(1)} s`;
  const c = document.getElementById('cursor');
  if (c) { const W = 900, L = 64, R = 16, tmax = Math.max(...snaps.map(s => s.t - t0)) || 1; const x = L + (W - L - R) * (snaps[cur].t - t0) / tmax; c.setAttribute('x1', x); c.setAttribute('x2', x); }
  renderTiles(); renderMap(); renderSummary(); renderTable(); renderEvents();
}

$('slider').max = snaps.length - 1;
$('slider').addEventListener('input', e => select(Number(e.target.value)));
$('play').addEventListener('click', () => {
  if (timer) { clearInterval(timer); timer = null; $('play').textContent = 'Play'; return; }
  if (cur >= snaps.length - 1) select(0);
  $('play').textContent = 'Pause';
  timer = setInterval(() => { if (cur >= snaps.length - 1) { clearInterval(timer); timer = null; $('play').textContent = 'Play'; } else select(cur + 1); }, 250);
});
$('group').addEventListener('change', () => { renderMap(); renderTable(); });
$('hideGuard').addEventListener('change', () => { renderMap(); renderTable(); });
$('filter').addEventListener('input', () => renderTable());
$('theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
});
$('created').textContent = DATA.created;
renderLegend(); renderTimeline(); select(cur);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
