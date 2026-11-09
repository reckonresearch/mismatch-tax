#!/usr/bin/env python3
"""
Generate schema-5 JSONL cells with plausible shapes for testing the pipeline
WITHOUT the measured corpus. Every row is tagged source="mock" so the real
loader rejects it; this script's output is for local smoke tests only and
must never land in data/measured/.

Shapes are roofline x (1 + inefficiency) with inefficiency rising at small
batch and long context, which is what vLLM traces typically look like. The
numbers are not measurements and carry no claim.
"""
import json
import random
from pathlib import Path

from mismatch_tax import CEILINGS, MODELS, roofline_decode_step_ms

GRID_BATCH = [1, 4, 8, 16, 32]
GRID_PROMPT = [512, 2048, 7680]
GRID_OUTPUT = [128, 256]


def main(out_dir: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    arch = MODELS["llama3-8b"]
    for silicon, (gemm, bw) in CEILINGS.items():
        rows = []
        n = 0
        for rep in range(2):
            for b in GRID_BATCH:
                for p in GRID_PROMPT:
                    for o in GRID_OUTPUT:
                        step, _ = roofline_decode_step_ms(arch, b, p + o / 2, gemm, bw)
                        # inefficiency: worse at batch 1, worse at long prompt
                        ineff = 0.6 + 1.2 / b + 0.0001 * p + rng.gauss(0, 0.03)
                        tpot = step * (1 + ineff)
                        ttft = (2 * arch.params * b * p) / (gemm * 1e12) * 1e3 * (1.3 + rng.gauss(0, 0.05))
                        rows.append({
                            "schema": 5, "silicon": silicon, "model_name": "llama3-8b",
                            "batch": b, "avg_prompt_tokens": p, "avg_output_tokens": o,
                            "measured_ttft_ms": round(ttft, 3), "measured_tpot_ms": round(tpot, 3),
                            "t": round(n / 59, 4), "source": "mock",
                            "silicon_provenance": "mock", "power_w": None, "power_provenance": None,
                        })
                        n += 1
        with (out_dir / f"{silicon}_llama3-8b.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {len(CEILINGS)} mock cells to {out_dir}")


if __name__ == "__main__":
    main(Path("data/mock"))
