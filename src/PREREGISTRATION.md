
---

# experiments_v3 — freeze log

## Frozen config (after L confirms)
- Edit: **attn (q,k,v,o), rank 16, binary sign + per-tensor scale**. q/v retired.
  Capacity sweep basis: on CrowS disability, attn r16 removes 96% in-budget vs
  q/v's 0%; rank 64 and MLP targets both worse.
- Elicitation: likelihood-based (Schick). Free-gen / forced-choice banned.
- Both nulls (data-partition + sign-shuffle) mandatory per axis; designer
  numbers above the partition null. Strict collateral budget unchanged.
- Resume keys include {arm, axis, target-set} (v2 lesson).

## P0 results (existing data, pre-registered thresholds)
- **K1 — REFUTED (informative).** Per-axis endogenous removal is NOT predicted
  by the per-axis elicitation gap (Pearson r=-0.14, ns, n=15 axes). CrowS
  elicitation gaps are normal (-0.02..+0.10), comparable to templated axes, yet
  CrowS removal is ~0. So the templated-vs-naturalistic boundary is NOT an
  elicitation collapse; it is governed by REMOVABILITY (the ceiling), a third
  factor orthogonal to the elicitation gap. This elevates M (structural test)
  and motivates ceiling_a in K2.
- **HY — clean.** Items-per-half floor = 40 gates out disability (30) and
  physical-appearance (31). Pooled CrowS blind spot over floor-passing axes at
  attn = -0.003 (still null after hygiene).

## Items-per-half floor
Set to **40** (plan's ~100 unreachable for most CrowS axes; 40 is where the
odd/even split still yields a stable profile). Applied as a formal axis gate.
