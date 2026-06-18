"""Collect + extract the per-pipe msprof numbers for the sparse_attn_sharedkv
kernel (cube/vector time + mac/scalar/mte/fixpipe/vec ratios), for TileLang vs
Ascend C bottleneck localisation.

The decisive question this answers: is the kernel compute-bound, memory-bound, or
overlap/sync-bound, and on which pipe -- so the next change can target the real
cost instead of guessing.

Two ways to use it (run on the NPU container):

  A) Collect AND parse in one shot (runs msprof for you):
       python msprof_pipe.py --impl tilelang  --scenario swa_prefill
       python msprof_pipe.py --impl ascendc   --scenario swa_prefill

  B) Parse a CSV you already produced your own way (most robust if your msprof
     CLI differs):
       # produce op_summary CSV however you normally do, with PipeUtilization, then:
       python msprof_pipe.py --csv /path/to/op_summary_*.csv

What it prints, for every op row whose name matches --match (default
"sparse_attn_sharedkv"): the op name + every column whose header mentions
time / ratio / cycles. It also lists ALL distinct op names found, so if the
filter misses, you can see the real op name and re-run with --match.

Key derived check (do it by eye on the printed numbers):
  * device kernel time ~= max(aicore_time, aiv_time)  -> cube/vector overlap is GOOD
  * device kernel time ~= aicore_time + aiv_time       -> cube/vector run SERIALLY
  * aic_scalar_ratio vs the old 0.63                    -> did the scalar load drop
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys

# Columns we care about: any header containing one of these (case-insensitive).
_WANT = ("time", "ratio", "cycle", "mac", "scalar", "mte", "fixpipe", "vec", "dur")
# How to find the op-name column.
_NAME_HINTS = ("op name", "op_name", "opname", "op type", "name")


def _norm(s: str) -> str:
    """Lowercase and drop non-alphanumerics, so 'sparse_attn_sharedkv' and the
    Ascend C 'SparseAttnSharedkv' compare equal."""
    return "".join(c for c in s.lower() if c.isalnum())


def _find_name_col(header: list[str]) -> int:
    low = [h.strip().lower() for h in header]
    for hint in _NAME_HINTS:
        for i, h in enumerate(low):
            if hint in h:
                return i
    return 0


def _want_cols(header: list[str]) -> list[int]:
    out = []
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if any(w in hl for w in _WANT):
            out.append(i)
    return out


def parse_csv(path: str, match: str, exclude: str = "metadata") -> None:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print(f"  (empty CSV: {path})")
        return
    header = rows[0]
    name_col = _find_name_col(header)
    cols = _want_cols(header)
    print(f"=== {os.path.basename(path)} ===")
    print(f"op-name column: [{name_col}] {header[name_col]!r}")

    nmatch = _norm(match)
    nexcl = _norm(exclude) if exclude else ""
    names = []
    hits = []
    for r in rows[1:]:
        if name_col >= len(r):
            continue
        nm = r[name_col].strip()
        names.append(nm)
        nn = _norm(nm)
        # Match on normalized name (underscore/case-insensitive); skip the
        # separate metadata op which also contains "SharedkvMetadata".
        if nmatch in nn and (not nexcl or nexcl not in nn):
            hits.append(r)

    print(f"distinct op names ({len(set(names))}): {sorted(set(names))[:40]}")
    if not hits:
        print(f"  !! no row matched --match={match!r}; pick the real name above.")
        return
    print(f"\nmatched rows for {match!r} ({len(hits)}):")
    for r in hits:
        nm = r[name_col].strip() if name_col < len(r) else "?"
        print(f"\n  op: {nm}")
        for c in cols:
            if c < len(r) and r[c].strip() != "":
                print(f"    {header[c].strip():28s} = {r[c].strip()}")


def collect(impl: str, scenario: str, outdir: str) -> str | None:
    """Run msprof with PipeUtilization, return the op_summary CSV path (or None)."""
    app = (
        f"python sparse_attn_sharedkv_perf_compare.py "
        f"--scenarios {scenario} --only {impl} --warmup 2 --iters 5"
    )
    cmd = [
        "msprof",
        f"--output={outdir}",
        f"--application={app}",
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
    ]
    print("running:", " ".join(cmd))
    print("(if this errors, run msprof your own way with PipeUtilization, then")
    print(" re-run:  python msprof_pipe.py --csv <op_summary csv>)")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nmsprof failed: {e}")
        return None
    # Auto-export usually drops CSVs under <outdir>/PROF_*/**/. Glob broadly.
    cands = sorted(
        glob.glob(os.path.join(outdir, "**", "*op_summary*.csv"), recursive=True)
    )
    if not cands:
        print(f"\nno op_summary CSV under {outdir}. Try exporting:")
        print(f"  msprof --export=on --output={outdir}/PROF_*")
        print(f"then: python msprof_pipe.py --csv {outdir}/PROF_*/.../op_summary_*.csv")
        return None
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["tilelang", "ascendc"], default="tilelang")
    ap.add_argument("--scenario", default="swa_prefill")
    ap.add_argument("--csv", default=None, help="parse an existing op_summary CSV")
    ap.add_argument("--match", default="sparse_attn_sharedkv", help="op-name filter")
    ap.add_argument(
        "--exclude",
        default="metadata",
        help="skip op names containing this (normalized); '' to disable",
    )
    ap.add_argument("--outdir", default=None, help="msprof output dir")
    args = ap.parse_args()

    if args.csv:
        parse_csv(args.csv, args.match, args.exclude)
        return

    outdir = args.outdir or f"./prof_{args.impl}_{args.scenario}"
    os.makedirs(outdir, exist_ok=True)
    csv_path = collect(args.impl, args.scenario, outdir)
    if csv_path:
        print()
        parse_csv(csv_path, args.match, args.exclude)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
