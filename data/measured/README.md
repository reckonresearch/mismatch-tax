# Measured corpus

Raw hardware traces behind berth's validation. One cell per file, schema 5,
every record `source: measured`. Consolidated 2026-11-08 from the P0 sweep
outputs. Each file passes `python -m bench.check_contributed`.

| cell | silicon | model | precision | n | median TPOT (ms) | silicon_provenance |
|---|---|---|---|---|---|---|
| a100-80g_llama3-8b.jsonl | a100-80g | llama3-8b | bf16 | 60 | 11.96 | captured |
| a100-80g_qwen3-30b-a3b.jsonl | a100-80g | qwen3-30b-a3b | bf16 | 60 | 12.34 | captured |
| h100-pcie_llama3-8b.jsonl | h100-pcie | llama3-8b | bf16(assumed, pre-schema4) | 60 | 12.18 | self_reported |
| h100-sxm_llama3-8b.jsonl | h100-sxm | llama3-8b | bf16 | 60 | 13.31 | captured |
| h100-sxm_llama3-8b_fp8.jsonl | h100-sxm | llama3-8b | fp8 | 60 | 11.69 | captured |
| l40s_llama3-8b.jsonl | l40s | llama3-8b | bf16(assumed, pre-schema4) | 60 | 25.75 | self_reported |
| mi300x_llama3-8b.jsonl | mi300x | llama3-8b | bf16 | 60 | 6.37 | self_reported |

## Provenance notes

- `self_reported` cells (l40s, h100-pcie, mi300x) had their silicon identity asserted
  by the operator, not captured by on-box nvidia-smi/rocm-smi. Accepted by the corpus
  gate; flagged for a reviewer. l40s and h100-pcie predate schema 3 (captured hand-labelled).
- `captured` cells (h100-sxm, a100-80g) had silicon verified on the box.
- Two h100-sxm precision variants are kept: bf16 (headline, matches the rest of the
  corpus) and fp8 (`_fp8`). Only bf16 belongs in like-for-like cross-silicon comparison.
- Precision is recovered from the schema-4 `w_bytes` field (2.0=bf16, 1.0=fp8), which
  schema 5 drops; l40s and h100-pcie predate it and are bf16 by run convention.

## Headline bf16 llama3-8b set (like-for-like cross-silicon)

l40s, h100-pcie, h100-sxm, mi300x, a100-80g. Five accelerators, two vendors,
three memory technologies (GDDR6, HBM2e, HBM3). Plus a sixth cell on a second
model (a100-80g, qwen3-30b-a3b) and an h100-sxm fp8 variant.

## raw_md5 (source file before migration)

- a100-80g_llama3-8b.jsonl: 817515ef5f68735f15d7fab1ed7011c2
- a100-80g_qwen3-30b-a3b.jsonl: 1b07de2d0afa3310583eb3fed722dc4a
- h100-pcie_llama3-8b.jsonl: 7bd0e65f6a38337bb2a1cf2460f7faba
- h100-sxm_llama3-8b.jsonl: b13a63efb064d5084105b90438393bab
- h100-sxm_llama3-8b_fp8.jsonl: 3fec180ca2b322d96b5756712fddaceb
- l40s_llama3-8b.jsonl: 472e5decfe7a6b9182f96fafdf151b03
- mi300x_llama3-8b.jsonl: 753c0a3bd873cfefedd6db11e4aaf7a3
