
## The third factor, identified: ENDOGENOUS EFFICIENCY

G and K1 both pointed to a missing "removability/structure" factor. Isolating
it cleanly — endogenous removal as a fraction of the exogenous ceiling, at
matched config, on removable axes:

| axis | endo | exo ceiling | efficiency | kind |
|---|---|---|---|---|
| gen_fm | 0.531 | 0.280 | 1.89 | templated |
| occ_gender | 0.466 | 0.535 | 0.87 | templated |
| crows physical-appearance | 0.057 | 0.098 | 0.59 | naturalistic |
| crows age | 0.010 | 0.081 | 0.12 | naturalistic |
| crows disability | 0.004 | 0.138 | 0.03 | naturalistic |

**templated mean efficiency 1.38 vs naturalistic 0.25.**

This sharpens the scope boundary into a precise claim. CrowS axes ARE removable
— the exogenous (ground-truth) signal removes 8–14% of the bias at attn. What
collapses on naturalistic content is **endogenous efficiency**: the elicited
biased/debiased partition fails to yield a direction that generalizes to
held-out items. It is not the elicitation *gap* (K1: normal on CrowS), not the
axis's *removability* (exo works), and not the designer *family* (all
endogenous designers fail equally, v2 B3). It is that a slightly-better-than-
random partition of *heterogeneous* items produces a non-generalizing edit,
whereas the same partition quality on *templated* items (shared frames) yields
a robust group-slot direction.

**Frame-overlap** is the pre-registered structural cause; a first cut correlates
+0.44 (p=0.10, n=15) with removability but is confounded by config and by a
slot-unmasked n-gram metric. The clean causal test is **M** (templatize CrowS /
de-templatize occ-gender): it is now elevated from optional to the primary
scope experiment, because the boundary is mechanistic (elicited-direction
generalization), not descriptive (templated vs not).

## Two-factor law → three-factor law

Removal = f(elicitation gap [designer quality], same-family penalty [relational,
NOT mediated by pairwise sharing per G], endogenous efficiency [structural,
governs the templated↔naturalistic boundary]). The v2 paper's two-factor claim
is incomplete; v3's P0 tier establishes the third factor and refutes the
pairwise-sharing mediation of the second.
