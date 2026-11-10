# mismatch-tax

Two analyses built on the Berth measured corpus (`data/measured/`, schema 5,
six cells, five accelerators). Both read only `source: measured` rows and fail
loud on any other schema or source. Hypotheses are pre-registered in
HYPOTHESIS.md before either tool runs on measured data.

## 1. Mismatch Tax

`mismatch_tax.py` computes, per measured configuration, the gap between a
card's own microbench-implied decode roofline and what the serving stack
actually delivered.

    python mismatch_tax.py data/measured/

Output: `mismatch_tax_rows.csv` (every config) and `mismatch_tax_summary.csv`
(per-cell median, best, worst, share memory-bound). Provenance flags are
carried into every row.

The Qwen3-30B-A3B cell is excluded: a dense roofline misstates MoE weight
traffic. The H100 SXM fp8 cell is excluded from the bf16 comparison per the
corpus policy.

## 2. served-token-dispatch

`dispatch/solve.py` allocates a demand vector across the five silicon at
minimum $/hour subject to per-class latency SLOs, using only measured TTFT and
TPOT for throughput. Compares against two baselines.

    python dispatch/solve.py --cells data/measured \
        --prices dispatch/prices.yaml --demand dispatch/demand.example.yaml

`dispatch/prices.yaml` ships as `status: placeholder`. No $/served-token figure
is a claim until it is replaced with the invoiced price of each rented box and
set to `status: measured`.

To backtest on a public trace:

    python dispatch/trace_to_demand.py AzureLLMInferenceTrace_conv.csv \
        --window-min 60 --out dispatch/demand.trace.yaml

## Smoke test without measured data

    python make_synthetic_cells.py          # writes data/mock, source=mock
    python mismatch_tax.py data/mock --allow-mock
    python dispatch/solve.py --cells data/mock --prices dispatch/prices.yaml \
        --demand dispatch/demand.example.yaml --allow-mock

Mock numbers carry no claim and are rejected by the loader without the flag.

## Environment

    uv venv && uv pip install -r requirements.txt

Python 3.11+. No GPU needed; everything here runs on the traces.

Built with Claude Code. Measurements, hypotheses, and interpretation are the
author's.

## Companion paper

This repo is the first empirical companion to *The Mismatch Tax* (Scott, August 2026),
which defines the tax as the loss when allocated, busy silicon runs outside its design
point, and makes four dated predictions. The paper is the framework; this repo is the
first pre-registered measurement of it on a public corpus.

- Paper: https://www.sidney.fyi/papers/the-mismatch-tax.pdf
- Corpus: https://github.com/reckonresearch/reckon (`data/measured/`)
