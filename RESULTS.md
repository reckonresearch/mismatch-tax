# Results: first measured Mismatch Tax run

Run date: 2026-11-09. Pre-registration: HYPOTHESIS.md (commit 60f53dd).
Corpus: five bf16 Llama-3-8B cells, 30 configurations x 2 repetitions each,
served by vLLM. Qwen3-30B-A3B (MoE) and H100 SXM fp8 cells excluded per policy.

Three separate statements. They are kept apart on purpose.

## 1. The timing instrument is validated

TTFT and TPOT are internally consistent across 150 configurations on five
accelerators from two vendors. Every configuration is memory-bound under the
roofline (frac_memory_bound = 1.0 on all cells), which is the expected regime
for an 8B bf16 model at batch 32 and below. One anomaly (L40S, section 2) is
explained by the ceiling, not the trace.

## 2. The bandwidth microbench is falsified as a decode ceiling

Against the corpus's own measured device-to-device copy bandwidth, L40S
returns a negative tax on all 30 configurations (median -0.10, best -0.27).
A roofline cannot be beaten, so the ceiling is wrong. The copy microbench
counts bytes read plus written (`bench/microbench.py`), but a copy's
interleaved read/write traffic pays DRAM bus-turnaround and write-allocate
costs, so its aggregate figure lands well below what decode's pure read stream
achieves. On L40S the copy reached 652 of 864 GB/s spec; a read stream reaches
over 90%. The copy figure understates the read ceiling by a card-dependent
margin.

This is Defect 5 in the reckon register recurring outside the package: the
type guard that prevents the copy figure being used as a peak inside `berth`
does not travel with the number once it is published as a bare float. Filed
as Defect 17 in `reckonresearch/reckon`.

Against vendor spec-sheet bandwidth, L40S flips positive (best 0.043), meaning
vLLM on L40S reaches roughly 96% of spec read bandwidth at its best
configuration. That both confirms the roofline model (weights read once per
step plus KV traffic) and closes the anomaly.

Consequence: the measured-copy figure is retained in the corpus as what it is,
a copy benchmark, but is no longer used as a decode ceiling. The next harness
upgrade is a read-only stream bandwidth microbench. Until it lands, tax figures
are reported against spec bandwidth and labeled `bw_ceiling: spec`.

## 3. Mismatch Tax, spec-bandwidth ceiling

| Silicon | Provenance | Median tax | Best | Worst |
|---|---|---|---|---|
| L40S | self_reported | 0.17 | 0.04 | 0.24 |
| H100 PCIe | self_reported | 0.24 | 0.20 | 0.34 |
| A100 80GB | captured | 0.27 | 0.24 | 0.77 |
| MI300X | self_reported | 0.46 | 0.35 | 0.86 |
| H100 SXM | captured | 0.55 | 0.38 | 0.70 |

Reading: the tax rises with the card's bandwidth. The two highest-bandwidth
parts (HBM3) leave the most on the table; GDDR6 leaves the least. At this
model size and batch range the serving stack, not the silicon, is the
bottleneck on the newest hardware.

The earlier unreproduced "57% on MI300X" figure is retired. The measured MI300X
median is 0.46 against spec bandwidth, 0.22 against the (invalid) copy
ceiling. The number nearest 57% in this corpus belongs to H100 SXM.

## Confounds, declared before the run and still standing

- MI300X, L40S, and H100 PCIe are `self_reported` silicon provenance.
- Serving-stack version varies by cell. The H100 SXM environment record lists
  an older vLLM than its capture date implies; an old stack on the newest card
  is the leading candidate explanation for its 0.55 median and is directly
  testable by re-serving that cell on current vLLM. This is the highest-value
  re-measurement in the corpus.
- Spec-sheet bandwidth is a vendor figure, so the spec-ceiling tax is an upper
  bound on stack inefficiency. The true ceiling lies between copy and spec.
- Roofline assumes no compute/memory overlap. This biases the roofline
  optimistic and the tax high by a small, unquantified margin.

## What this does not say

- Nothing about Berth's estimator accuracy. That claim is tested in each cell's
  `validate.txt` and summarized separately in VALIDATION.md.
- Nothing vendor-vs-vendor. A card's tax is the gap between its own ceiling and
  what one serving stack extracted from it on one day.

## Reproduce

    python mismatch_tax.py data/measured/            # copy ceiling (section 2)
    python mismatch_tax.py data/measured/ --bw spec  # spec ceiling (section 3)
