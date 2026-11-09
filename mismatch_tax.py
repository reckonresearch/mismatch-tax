#!/usr/bin/env python3
"""
Mismatch Tax: achieved decode throughput vs. the roofline implied by each cell's
own measured microbench ceilings.

    mismatch_tax = 1 - (roofline_step_ms / measured_tpot_ms)

A tax of 0.57 means the silicon delivered 43% of what its measured GEMM and
bandwidth ceilings say it could, at that batch and context. The tax is NOT a
vendor-vs-vendor claim; it is the gap between a card's own ceilings and what the
serving stack extracted from it. Serving-stack version is a known >2x variable on
identical hardware, so cells are only comparable within the caveats in
MEASURED_CORPUS.md.

Why roofline, not spec sheet: ceilings are measured on the rented box at capture
time (GEMM bf16 n=8192, device-to-device bandwidth). Spec-sheet numbers would
inflate the tax with the vendor's marketing margin.

Usage:
    python mismatch_tax.py data/measured/            # all cells in a directory
    python mismatch_tax.py data/measured/mi300x_llama3-8b.jsonl

Reads schema-5 JSONL only. Fails loud on any other schema, matching the corpus
loader's policy.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

SCHEMA = 5
BYTES_BF16 = 2

# ----------------------------------------------------------------------------
# Measured ceilings per accelerator, transcribed from MEASURED_CORPUS.md.
# GEMM in TFLOPS (bf16, n=8192). BW in GB/s (device-to-device copy).
# These are the cell's OWN microbench results, not spec sheet.
# ----------------------------------------------------------------------------
CEILINGS: dict[str, tuple[float, float]] = {
    "l40s": (211.0, 652.0),
    "h100-pcie": (398.0, 1546.0),
    "h100-sxm": (728.0, 2997.0),
    "mi300x": (645.0, 3693.0),
    "a100-80g": (263.0, 1742.0),  # corpus reports 1737 to 1748; midpoint
}

# Vendor spec-sheet HBM/GDDR bandwidth, GB/s. NOT measured. Used only for the
# --bw spec sensitivity run. First measured run (2026-11-09) produced negative
# tax on L40S, which falsifies the d2d-copy microbench as a decode ceiling:
# a copy reports bytes written, decode is a pure read stream and can exceed it.
# Until the harness carries a read-only bandwidth microbench, both bounds are
# reported and the measured-copy figure is labeled as understating read BW.
SPEC_BW: dict[str, float] = {
    "l40s": 864.0,
    "h100-pcie": 2000.0,
    "h100-sxm": 3350.0,
    "mi300x": 5300.0,
    "a100-80g": 2039.0,
}

# Provenance of the silicon identity, from the corpus. Carried into the output
# so nobody reads a self_reported cell as verified.
SILICON_PROVENANCE: dict[str, str] = {
    "l40s": "self_reported",
    "h100-pcie": "self_reported",
    "h100-sxm": "captured",
    "mi300x": "self_reported",
    "a100-80g": "captured",
}


@dataclass(frozen=True)
class ModelArch:
    """Dense decoder-only transformer parameters needed for a decode roofline."""
    name: str
    params: float          # total parameters
    n_layers: int
    n_kv_heads: int
    head_dim: int

    @property
    def weight_bytes(self) -> float:
        return self.params * BYTES_BF16

    @property
    def kv_bytes_per_token(self) -> float:
        # K and V, per layer, per token, bf16
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * BYTES_BF16


# Llama-3-8B: 8.03B params, 32 layers, 8 KV heads (GQA), head_dim 128.
# Qwen3-30B-A3B is a sparse MoE; a dense roofline misstates its weight traffic,
# so that cell is excluded here rather than approximated. See EXCLUDED below.
MODELS: dict[str, ModelArch] = {
    "llama3-8b": ModelArch("llama3-8b", 8.03e9, 32, 8, 128),
}
EXCLUDED_MODELS = {"qwen3-30b-a3b": "sparse MoE; dense roofline not valid"}


def roofline_decode_step_ms(
    arch: ModelArch, batch: int, context_tokens: int, gemm_tflops: float, bw_gbs: float
) -> tuple[float, str]:
    """
    One decode step for a batch of `batch` sequences, each at `context_tokens`.

    Memory side: weights read once per step + KV cache read for every sequence.
    Compute side: 2 FLOPs per parameter per token, times batch.
    Step time is the max of the two (classic roofline, no overlap penalty).
    Returns (step_ms, bound) where bound is "memory" or "compute".
    """
    bytes_moved = arch.weight_bytes + batch * context_tokens * arch.kv_bytes_per_token
    t_mem = bytes_moved / (bw_gbs * 1e9)
    flops = 2.0 * arch.params * batch
    t_cmp = flops / (gemm_tflops * 1e12)
    if t_mem >= t_cmp:
        return t_mem * 1e3, "memory"
    return t_cmp * 1e3, "compute"


def load_cell(path: Path, allow_mock: bool = False) -> pl.DataFrame:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("schema") != SCHEMA:
                raise ValueError(f"{path}:{i}: schema {rec.get('schema')!r}, expected {SCHEMA}")
            if rec.get("source") != "measured" and not (allow_mock and rec.get("source") == "mock"):
                raise ValueError(f"{path}:{i}: source {rec.get('source')!r}, corpus admits only 'measured'")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return pl.DataFrame(rows)


def analyze_cell(df: pl.DataFrame, bw_source: str = "measured") -> pl.DataFrame | None:
    silicon = df["silicon"][0]
    model = df["model_name"][0]
    if model in EXCLUDED_MODELS:
        print(f"  skip {silicon}/{model}: {EXCLUDED_MODELS[model]}", file=sys.stderr)
        return None
    if silicon not in CEILINGS:
        raise KeyError(f"no measured ceilings for silicon {silicon!r}; add to CEILINGS")
    arch = MODELS[model]
    gemm, bw = CEILINGS[silicon]
    if bw_source == "spec":
        bw = SPEC_BW[silicon]

    # Context during decode grows from prompt to prompt+output; use the midpoint
    # so KV traffic is neither the optimistic start nor the pessimistic end.
    out = []
    for r in df.iter_rows(named=True):
        ctx = r["avg_prompt_tokens"] + r["avg_output_tokens"] / 2
        step_ms, bound = roofline_decode_step_ms(arch, r["batch"], ctx, gemm, bw)
        tax = 1.0 - step_ms / r["measured_tpot_ms"]
        out.append({
            "silicon": silicon,
            "silicon_provenance": SILICON_PROVENANCE.get(silicon, r.get("silicon_provenance")),
            "bw_ceiling": bw_source,
            "model": model,
            "batch": r["batch"],
            "prompt": r["avg_prompt_tokens"],
            "output": r["avg_output_tokens"],
            "measured_tpot_ms": r["measured_tpot_ms"],
            "roofline_step_ms": round(step_ms, 3),
            "bound": bound,
            "mismatch_tax": round(tax, 4),
            # Achieved decode tokens/s for the whole batch; used by the dispatcher.
            "achieved_tok_s": round(r["batch"] * 1000.0 / r["measured_tpot_ms"], 2),
        })
    return pl.DataFrame(out)


def summarize(all_rows: pl.DataFrame) -> pl.DataFrame:
    # Average over the two repetitions first, then report per-cell medians and
    # the best (lowest) tax the stack reached anywhere in the grid.
    per_cfg = (
        all_rows.group_by(["silicon", "silicon_provenance", "bw_ceiling", "model", "batch", "prompt", "output"])
        .agg(pl.col("mismatch_tax").mean(), pl.col("bound").first())
    )
    return (
        per_cfg.group_by(["silicon", "silicon_provenance", "bw_ceiling", "model"])
        .agg(
            pl.col("mismatch_tax").median().alias("tax_median"),
            pl.col("mismatch_tax").min().alias("tax_best"),
            pl.col("mismatch_tax").max().alias("tax_worst"),
            (pl.col("bound") == "memory").mean().alias("frac_memory_bound"),
            pl.len().alias("n_configs"),
        )
        .sort("tax_median")
    )


def main(argv: list[str]) -> int:
    allow_mock = "--allow-mock" in argv
    bw_source = "spec" if "--bw" in argv and argv[argv.index("--bw") + 1] == "spec" else "measured"
    args = [a for a in argv[1:] if a not in ("--allow-mock", "--bw", "spec", "measured")]
    if len(args) != 1:
        print(__doc__)
        return 2
    if allow_mock:
        print("WARNING: --allow-mock set; results carry no claim", file=sys.stderr)
    if bw_source == "spec":
        print("NOTE: --bw spec; bandwidth ceiling is vendor spec sheet, not measured", file=sys.stderr)
    target = Path(args[0])
    files = sorted(target.glob("*.jsonl")) if target.is_dir() else [target]
    frames = []
    for f in files:
        res = analyze_cell(load_cell(f, allow_mock=allow_mock), bw_source=bw_source)
        if res is not None:
            frames.append(res)
    if not frames:
        print("no analyzable cells", file=sys.stderr)
        return 1
    all_rows = pl.concat(frames)
    suffix = "" if bw_source == "measured" else "_specbw"
    all_rows.write_csv(f"mismatch_tax_rows{suffix}.csv")
    summary = summarize(all_rows)
    summary.write_csv(f"mismatch_tax_summary{suffix}.csv")
    with pl.Config(tbl_rows=50, tbl_width_chars=160, float_precision=3):
        print(summary)
    print(f"\nrows -> mismatch_tax_rows{suffix}.csv, summary -> mismatch_tax_summary{suffix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
