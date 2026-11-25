# Measurement note for "The Binding Constraint on Intelligence"

Sidney Scott, Reckon Research. Compiled November 24, 2026 from the lab's public repositories `reckonresearch/reckon` (main at 28bd4c2) and `reckonresearch/mismatch-tax` (main at 602b3e3), both fetched that day. Labels: measured, exploratory, assumed, missing.

## 1. Evidence table

| # | Claim | Number | Label | Provenance | Traces retained |
|---|---|---|---|---|---|
| 1 | The serving stack leaves a card-dependent share of spec memory bandwidth unused in decode | Median decode gap against vendor spec bandwidth: L40S 0.17, H100 PCIe 0.24, A100 80GB 0.27, MI300X 0.46, H100 SXM 0.55 | Exploratory | Llama-3-8B, bf16, vLLM, 30 configurations x 2 repetitions per card, batch 1 to 32, prompts of about 513, 2,049 and 7,681 tokens. Run date 2026-11-09. `mismatch-tax/mismatch_tax_summary_specbw.csv` and `RESULTS.md` section 3, commit ad10987, relabeled in 9484795 | Yes, per configuration: `data/measured/*.jsonl` with the md5 of each source file |
| 2 | The median hides the tail on one card | A100 80GB: median 0.27, worst configuration 0.77. MI300X: median 0.46, worst 0.86 | Exploratory | Same file and commit | Yes, per configuration |
| 3 | Repeating a run exposes instability a single run hides | 210 configurations measured twice. Each repetition's deviation from its pair mean: 0.19 percent at the median and 21.0 percent at worst for per-token latency; 0.62 and 12.9 percent for first-token latency | Measured, derived | Computed November 23, 2026 from `reckon/data/measured` at 28bd4c2. The script is not committed | Yes, per configuration |
| 4 | Tokens are produced by configurations that miss an interactive deadline | 183 of 300 records on the five bf16 Llama-3-8B cells have a median that misses a 500 ms first-token or 30 ms per-token bound. 88 of 300 miss 2,000 ms or 100 ms. All 183 miss on first-token time | Exploratory, derived | Computed November 24, 2026 from `reckon/data/measured` at 28bd4c2. Bounds are the MLPerf Inference v5.1 Llama3.1-8B interactive and server constraints as recorded in a tier table that is not yet published. The script is not committed and the analysis is not registered | Yes, per configuration |
| 5 | Share of tokens that miss a time bound | Missing | Missing | Each record stores the median across the streams of one batch. No per-request timing is retained, so no token share can be computed | No |
| 6 | Share of tokens below a quality floor | Missing | Missing | The corpus has never measured a quality floor. It is declared by the buyer | No |
| 7 | Overhead of verification, monitoring or retries | Missing | Missing | Neither repository holds such a measurement | No |
| 8 | A microbenchmark ceiling was falsified by the first run | L40S: negative gap on all 30 configurations against the corpus's own copy-bandwidth ceiling, median -0.10, best -0.27. The copy reached 652 of 864 GB/s spec | Measured | `mismatch-tax/RESULTS.md` section 2 and `mismatch_tax_summary.csv`, commit ad10987. `reckon/DEFECTS.md`, Defect 17, commit 6910b3b | Yes, per configuration |

**Conditions missing from the published record for rows 1, 2 and 8:** the vLLM version of each cell and the capture date of each cell. The corpus README says only that it was consolidated on 2026-11-08. `RESULTS.md` says the H100 SXM cell ran an older vLLM than its capture date implies. Silicon identity is operator-reported for the L40S, H100 PCIe and MI300X cells.

**Use in the essay.** Rows 1, 2 and 8 are committed and reproducible, and the rewritten section uses only those. Rows 3 and 4 become usable once their scripts are committed.

## 2. Audit history, checked against the public record

| Statement in the draft | Finding | Corrected wording |
|---|---|---|
| The analysis was relabeled from pre-registered to exploratory in November 2026 because the hypotheses file was committed with blank intervals four minutes before the first results | The relabel is confirmed: November 18, 2026, commits 9484795 and 602b3e3 in mismatch-tax and ed351ec in reckon. The blank intervals are confirmed. "Four minutes" is not verifiable: the record cites commit 60f53dd, which is absent from public history. In public history the hypotheses file and the results arrive in one commit, ad10987, at 15:10:54 on November 9. Defect 17 was committed to the register at 13:56:24 the same day, 74 minutes earlier | "The hypotheses file and the first results entered the public repository in one commit, with every interval blank." |
| The first run falsified one of the corpus's own microbenchmark ceilings, logged as Defect 17 | Confirmed. See row 8 | No change |
| The mismatch tax itself remains unmeasured | Confirmed. The repository README says the run "does not measure the mismatch tax itself" | No change |

## 3. A first-person detail, from the record

On the first run my L40S beat its own ceiling on all 30 configurations. A roofline cannot be beaten, so the ceiling was wrong: I had published a copy benchmark as a bare number, 652 GB/s against an 864 GB/s spec, and then used it as a read ceiling. A type guard inside the package had blocked that mistake since Defect 5, and it did not travel with the number once the number was published.

## 4. Moved out of the essay: what control costs, and who pays

A stylized model, with my assumptions stated. Add an independent verifier that catches 90 percent of step errors. Per-step reliability moves from 99 to 99.9 percent, and the 100-step chain moves from 36.6 to 90.5 percent completion (0.99^100 = 0.366; 0.999^100 = 0.905). If the verifier consumes 30 percent of the step's tokens, the bill rises 1.3x. That is cheap, on one condition: the verifier's errors are independent of the actor's. A model checking its own work shares its own blind spot, which returns the problem to the correlated tail.

The 90 percent catch rate and the 30 percent token cost are assumptions. The corpus holds no measured overhead for verification, monitoring or retries.

Three parties pay. The buyer pays in tokens and latency. The operator pays for monitors, fallbacks, and the people who staff them. The lab pays in compute diverted from training to interpretability and evals. Control is a compute allocation decision, made against scale, every budget cycle.

## 5. Moved out of the essay: the open-weight rival

Open weights are the main rival route to legibility, and the route cuts both ways. The advantage is real. Anyone can inspect the weights, third parties can build instruments, and a deployer can run several different models to decorrelate the tail. The costs are also real. Weights are not an explanation, the instruments are scarce and compute-hungry, and the deployer inherits hosting, monitoring, and the liability, with no counterparty to certify anything. The market's current verdict is 19 percent of usage and 4 percent of spend (Menlo Ventures, November 3, 2026).

If regulators and insurers come to accept deployer-run monitoring on open weights as evidence, the premium moves from the labs to an independent assurance layer. Call 1 loses in that world and the thesis survives in a different form. My claim concerns the method. Google DeepMind and OpenAI run interpretability teams too, and the winner is whoever industrializes the full stack first.
