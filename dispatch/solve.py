#!/usr/bin/env python3
"""
served-token-dispatch: minimum-cost fleet allocation across heterogeneous
silicon, driven by Berth's measured cells.

Problem
-------
Given
  * measured throughput curves per (silicon, batch, prompt, output) from the
    corpus (TTFT and TPOT, so a slot's request rate is known, not modeled),
  * a demand vector: requests/s per workload class, each with a latency SLO
    (max TTFT ms, max TPOT ms),
  * a price per node-hour per silicon,
choose how many nodes of each silicon to run, and at which batch, for each
class, to meet demand at minimum $/hour subject to the SLO.

This is a small integer program: n[c,s,b] = number of nodes serving class c on
silicon s at batch b. Solved with CP-SAT. Five silicon x five batches x a
handful of classes is trivial for the solver; the value is in the measured
inputs, not solver cleverness.

A "served token" is one output token delivered inside its declared time bound.
Configurations that violate the SLO produce zero served tokens and are excluded
from the feasible set rather than penalized.

Baselines
---------
  homogeneous-best : every class on the single silicon with the best median
                     throughput, at the best SLO-feasible batch. The "just buy
                     H100s" default.
  uniform          : demand split equally across every silicon that can meet
                     the SLO. The "we have a mixed fleet and round-robin" default.

Usage
-----
    python dispatch/solve.py --cells data/measured --prices dispatch/prices.yaml \
        --demand dispatch/demand.example.yaml

Pre-registration
----------------
Write the hypothesis (expected % saving vs homogeneous-best) in HYPOTHESIS.md
with a date BEFORE running on measured cells. The backtest result goes beneath
it, whatever it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import yaml
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mismatch_tax import load_cell  # noqa: E402

GRID_PROMPT = [512, 2048, 7680]
GRID_OUTPUT = [128, 256]
SCALE = 1000  # CP-SAT needs integers; rates are scaled by this


def nearest(x: float, grid: list[int]) -> int:
    return min(grid, key=lambda g: abs(g - x))


def load_curves(cells_dir: Path, allow_mock: bool) -> pl.DataFrame:
    """
    One row per (silicon, batch, prompt, output), repetitions averaged.
    Adds slot_rps: requests/s one node at that batch can complete for that
    (prompt, output) shape, from measured wall-clock timing only.
    """
    frames = [load_cell(p, allow_mock=allow_mock) for p in sorted(cells_dir.glob("*.jsonl"))]
    df = pl.concat(frames)
    df = df.filter(pl.col("model_name") == "llama3-8b")  # single model for v1
    df = (
        df.group_by(["silicon", "batch", "avg_prompt_tokens", "avg_output_tokens"])
        .agg(pl.col("measured_ttft_ms").mean(), pl.col("measured_tpot_ms").mean())
    )
    batch_s = (pl.col("measured_ttft_ms") + pl.col("avg_output_tokens") * pl.col("measured_tpot_ms")) / 1000.0
    return df.with_columns(
        (pl.col("batch") / batch_s).alias("slot_rps"),
        pl.col("avg_prompt_tokens").map_elements(lambda x: nearest(x, GRID_PROMPT), return_dtype=pl.Int64).alias("prompt_bin"),
        pl.col("avg_output_tokens").map_elements(lambda x: nearest(x, GRID_OUTPUT), return_dtype=pl.Int64).alias("output_bin"),
    )


def feasible_options(curves: pl.DataFrame, cls: dict) -> pl.DataFrame:
    """Configurations for this class that meet its SLO."""
    return curves.filter(
        (pl.col("prompt_bin") == cls["prompt"])
        & (pl.col("output_bin") == cls["output"])
        & (pl.col("measured_ttft_ms") <= cls["ttft_max_ms"])
        & (pl.col("measured_tpot_ms") <= cls["tpot_max_ms"])
    )


def solve(curves: pl.DataFrame, prices: dict[str, float], demand: list[dict]) -> dict:
    m = cp_model.CpModel()
    n = {}
    nodes_by_silicon: dict[str, list] = {s: [] for s in prices}

    for ci, cls in enumerate(demand):
        opts = feasible_options(curves, cls)
        if opts.is_empty():
            raise RuntimeError(f"class {cls['name']}: no SLO-feasible configuration on any silicon")
        supply_terms = []
        for r in opts.iter_rows(named=True):
            s, b = r["silicon"], r["batch"]
            if s not in prices:
                continue
            v = m.NewIntVar(0, 10_000, f"n_{ci}_{s}_{b}")
            n[(ci, s, b)] = v
            supply_terms.append(int(round(r["slot_rps"] * SCALE)) * v)
            nodes_by_silicon[s].append(v)
        m.Add(sum(supply_terms) >= int(round(cls["rps"] * SCALE)))

    # price in milli-dollars per hour to stay integer
    m.Minimize(sum(int(round(prices[s] * 1000)) * sum(vs) for s, vs in nodes_by_silicon.items() if vs))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("solver returned no solution")

    alloc = [
        {"class": demand[ci]["name"], "silicon": s, "batch": b, "nodes": solver.Value(v)}
        for (ci, s, b), v in n.items() if solver.Value(v) > 0
    ]
    nodes = {s: sum(solver.Value(v) for v in vs) for s, vs in nodes_by_silicon.items()}
    cost_hr = sum(prices[s] * k for s, k in nodes.items())
    return {"status": solver.StatusName(status), "cost_per_hour": cost_hr, "nodes": nodes, "allocation": alloc}


def served_tokens_per_hour(demand: list[dict]) -> float:
    return sum(c["rps"] * c["output"] for c in demand) * 3600


def baseline_homogeneous_best(curves, prices, demand) -> dict:
    best = None
    for s in prices:
        sub = curves.filter(pl.col("silicon") == s)
        try:
            res = solve(sub, {s: prices[s]}, demand)
        except RuntimeError:
            continue
        if best is None or res["cost_per_hour"] < best["cost_per_hour"]:
            best = res | {"silicon": s}
    if best is None:
        raise RuntimeError("no single silicon can meet every SLO")
    return best


def baseline_uniform(curves, prices, demand) -> dict:
    """Each class split equally across all silicon that can serve it, each at
    its best SLO-feasible batch. Integer nodes rounded up per silicon."""
    import math
    nodes = {s: 0 for s in prices}
    for cls in demand:
        opts = feasible_options(curves, cls).filter(pl.col("silicon").is_in(list(prices)))
        sil = opts["silicon"].unique().to_list()
        if not sil:
            raise RuntimeError(f"class {cls['name']}: infeasible")
        share = cls["rps"] / len(sil)
        for s in sil:
            best_rps = opts.filter(pl.col("silicon") == s)["slot_rps"].max()
            nodes[s] += math.ceil(share / best_rps)
    return {"cost_per_hour": sum(prices[s] * k for s, k in nodes.items()), "nodes": nodes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--demand", type=Path, required=True)
    ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("dispatch_result.json"))
    a = ap.parse_args()

    prices_doc = yaml.safe_load(a.prices.read_text())
    if prices_doc.get("status") != "measured":
        print(f"WARNING: prices.yaml status={prices_doc.get('status')!r}; cost figures carry no claim", file=sys.stderr)
    prices = {k: float(v["usd_per_hour"]) for k, v in prices_doc["silicon"].items()}
    demand = yaml.safe_load(a.demand.read_text())["classes"]

    curves = load_curves(a.cells, a.allow_mock)
    opt = solve(curves, prices, demand)
    hb = baseline_homogeneous_best(curves, prices, demand)
    un = baseline_uniform(curves, prices, demand)
    stph = served_tokens_per_hour(demand)

    report = {
        "served_tokens_per_hour": stph,
        "optimal": opt | {"usd_per_1m_served_tokens": opt["cost_per_hour"] / stph * 1e6},
        "baseline_homogeneous_best": hb | {"usd_per_1m_served_tokens": hb["cost_per_hour"] / stph * 1e6},
        "baseline_uniform": un | {"usd_per_1m_served_tokens": un["cost_per_hour"] / stph * 1e6},
        "saving_vs_homogeneous_best": 1 - opt["cost_per_hour"] / hb["cost_per_hour"],
        "saving_vs_uniform": 1 - opt["cost_per_hour"] / un["cost_per_hour"],
    }
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "optimal"} | {"optimal_nodes": opt["nodes"], "optimal_cost_per_hour": opt["cost_per_hour"]}, indent=2))
    print(f"\nfull result -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
