#!/usr/bin/env python3
"""
Convert a public LLM inference trace into a demand vector on the corpus grid.

Target trace: Azure LLM inference traces (AzurePublicDataset, "AzureLLMInferenceTrace_*").
Columns: TIMESTAMP, ContextTokens, GeneratedTokens. Download it yourself and
pass the CSV path; this script does not fetch.

Each request is binned to the nearest (prompt, output) grid point. Demand per
class is requests/s over the chosen window. SLOs are supplied once per output
bin because the trace carries no latency targets; state them in HYPOTHESIS.md.

Usage:
    python dispatch/trace_to_demand.py trace.csv --window-min 60 --out dispatch/demand.trace.yaml
"""
import argparse
from pathlib import Path

import polars as pl
import yaml

GRID_PROMPT = [512, 2048, 7680]
GRID_OUTPUT = [128, 256]
DEFAULT_SLO = {128: {"ttft_max_ms": 1000, "tpot_max_ms": 40}, 256: {"ttft_max_ms": 2000, "tpot_max_ms": 40}}


def nearest(x, grid):
    return min(grid, key=lambda g: abs(g - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--window-min", type=float, default=60.0, help="use the first N minutes")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    df = pl.read_csv(a.trace, try_parse_dates=True)
    df = df.rename({c: c.lower() for c in df.columns})
    t0 = df["timestamp"].min()
    df = df.filter(pl.col("timestamp") <= t0 + pl.duration(minutes=a.window_min))
    seconds = a.window_min * 60

    df = df.with_columns(
        pl.col("contexttokens").map_elements(lambda x: nearest(x, GRID_PROMPT), return_dtype=pl.Int64).alias("prompt"),
        pl.col("generatedtokens").map_elements(lambda x: nearest(x, GRID_OUTPUT), return_dtype=pl.Int64).alias("output"),
    )
    counts = df.group_by(["prompt", "output"]).len().sort(["prompt", "output"])
    classes = [
        {"name": f"p{r['prompt']}-o{r['output']}", "prompt": r["prompt"], "output": r["output"],
         "rps": round(r["len"] / seconds, 4), **DEFAULT_SLO[r["output"]]}
        for r in counts.iter_rows(named=True)
    ]
    a.out.write_text(yaml.safe_dump({"source": str(a.trace), "window_min": a.window_min, "classes": classes}, sort_keys=False))
    print(f"{len(df)} requests over {a.window_min} min -> {len(classes)} classes -> {a.out}")


if __name__ == "__main__":
    main()
