# Pre-registered hypotheses

Fill in and commit this file BEFORE running either tool on `data/measured/`.
The commit timestamp is the registration. Results go in RESULTS.md afterward,
whatever they are.

## H1. Mismatch Tax (mismatch_tax.py)

Registered: YYYY-MM-DD by Sidney Scott

Claim: on the five bf16 Llama-3-8B cells, the median decode mismatch tax
(1 - roofline_step / measured_tpot) will fall between ____ and ____ across
silicon, and the MI300X cell will show a median tax of ____ ± ____.

Known confounds, declared in advance:
- MI300X, L40S, H100 PCIe cells are `self_reported` silicon provenance.
- Serving-stack version varies across cells; a >2x version effect on identical
  hardware has been observed previously.
- Roofline assumes no compute/memory overlap and bf16 weights read once per
  step; both bias the roofline optimistic, so the tax is an upper bound on
  stack inefficiency and a lower bound on nothing.

Falsified if: the measured median falls outside the stated interval.

## H2. Dispatch saving (dispatch/solve.py)

Registered: YYYY-MM-DD by Sidney Scott

Setup: demand vector from the first ____ minutes of ____ (public trace),
SLOs as stated in dispatch/demand.trace.yaml, prices from rental invoices at
capture time (dispatch/prices.yaml, status: measured).

Claim: measured-curve dispatch across the five silicon will cost ____% less
per served token than the homogeneous-best baseline, and ____% less than the
uniform baseline.

Falsified if: saving vs homogeneous-best is below ____%.

What a null result would mean: at this fleet size and SLO mix, silicon
heterogeneity does not pay for its routing complexity, which bounds the
Mismatch Tax thesis rather than supporting it. That is a publishable result.
