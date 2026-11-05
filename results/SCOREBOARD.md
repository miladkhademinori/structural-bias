# Synthetic verification scoreboard

54 assertions from 6 suite records: 33 PASS, 7 MECHANISM-FAIL, 13 CALIBRATION-EXCURSION, 1 PAPER-QUESTION.
Claims (the verdict unit): 39 claims: 24 PASS, 6 MECHANISM-FAIL, 8 CALIBRATION-EXCURSION, 1 PAPER-QUESTION.

Vocabulary (docs/VERDICTS.md): MECHANISM-FAIL, the mechanism the claim asserts is not exhibited (an exact arm misses its numerical tolerance, a finite-sample arm misses by more than the battery's family-wise bound, or the spec design cannot exhibit the claim at any n, marked [design] with the variant arm printed beside it); CALIBRATION-EXCURSION, the mechanism is exhibited and one gate number is missed within the battery's bound, size in standard errors reported; PAPER-QUESTION, a deterministic finding about a printed paper number, carried to the owner's Q-list with the measured value.

## Claims

| paper | claim | arms | verdict |
|---|---|---|---|
| B | E1.1 | E1.1 (spec): MECHANISM-FAIL [design]; E1.1 (variant_wellcond): PASS | **MECHANISM-FAIL [design; variant: E1.1: PASS]** |
| B | E1.2 | E1.2 (spec): MECHANISM-FAIL [design]; E1.2 (variant_wellcond): CALIBRATION-EXCURSION | **MECHANISM-FAIL [design; variant: E1.2: CALIBRATION-EXCURSION]** |
| B | E1.3 | E1.3 (spec): MECHANISM-FAIL [design]; E1.3 (variant_wellcond): CALIBRATION-EXCURSION | **MECHANISM-FAIL [design; variant: E1.3: CALIBRATION-EXCURSION]** |
| B | E2.1a | E2.1a (variant/lam=0/n=50000): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |
| B | E2.1b | E2.1b (variant/estimated/n=1000/seeds[0, 1, 2, 3, 4]): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |
| B | E2.2 | E2.2 (variant/lam=0/n=50000): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |
| B | E2.3b | E2.3b (variant/lam=0/n=50000 trained): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |
| B | E2.4 | E2.4 (variant): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |
| B | E4.1b | E4.1b (paper protocol n=2000, MC 1e6): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |
| B | E4.1c | E4.1c (empirical means): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |
| B | E4.2c | E4.2c (paper protocol): PAPER-QUESTION | **PAPER-QUESTION** |
| B | E5.1 | E5.1[spec|exact] (exact): MECHANISM-FAIL [design]; E5.1[spec|empirical MC 100000] (empirical MC 100000): PASS; E5.1[variant_wellcond|exact] (exact): PASS; E5.1[variant_wellcond|empirical MC 100000] (empirical MC 100000): PASS | **MECHANISM-FAIL [design; variant: E5.1[variant_wellcond|exact]: PASS, E5.1[variant_wellcond|empirical MC 100000]: PASS]** |
| B | E5.2 | E5.2[spec|exact] (exact): MECHANISM-FAIL [design]; E5.2[spec|empirical MC 100000] (empirical MC 100000): MECHANISM-FAIL [design]; E5.2[variant_wellcond|exact] (exact): CALIBRATION-EXCURSION; E5.2[variant_wellcond|empirical MC 100000] (empirical MC 100000): CALIBRATION-EXCURSION | **MECHANISM-FAIL [design; variant: E5.2[variant_wellcond|exact]: CALIBRATION-EXCURSION, E5.2[variant_wellcond|empirical MC 100000]: CALIBRATION-EXCURSION]** |
| B | E5.3 | E5.3[spec] (exact + empirical): MECHANISM-FAIL [design]; E5.3[variant_wellcond] (exact + empirical): CALIBRATION-EXCURSION | **MECHANISM-FAIL [design; variant: E5.3[variant_wellcond]: CALIBRATION-EXCURSION]** |
| B | E5.4 | E5.4 (empirical MC control): CALIBRATION-EXCURSION | **CALIBRATION-EXCURSION** |

PASS: E1.4, E2.3a, E3.1, E3.2a, E3.2b, E3.3a, E3.3b, E4.1a, E4.2a, E4.2a2, E4.2b, E4.3a, E4.3b, E4.4, E5.0, E5.0b, E5.5, E6.0, E6.1, E6.2, E6.3, E6.4a, E6.4b, E6.5

Tier 1 (mechanism: algebraic and population identities, exact in all inputs) must be all PASS for the code to be trusted.
Tier 2 (calibration: Monte-Carlo and finite-sample estimates, including exact functionals of finite-sample heads) lists excursions with the suites' own diagnosis; they are re-gated by standard error (tests/zbattery.py and core.gates), never by widening tolerances.

## Tier 1: mechanism

17/17 PASS

| run | id | arm | verdict | value | threshold |
|---|---|---|---|---|---|

PASS: E2.3a, E3.1, E3.2a, E3.3a, E4.1a, E4.2a, E4.2b, E4.4, E5.0[spec], E5.0[variant_wellcond], E5.0b[spec], E5.0b[variant_wellcond], E6.2, E6.3, E6.4a, E6.4b, E6.0

## Tier 2: calibration

16/37 PASS

| run | id | arm | verdict | value | threshold |
|---|---|---|---|---|---|
| runs\B\synthetic_bias\B_synthetic_bias_E1.json | E1.1 | spec | **MECHANISM-FAIL [design]** | min=0.882414 (max 0.999506) | > 0.999 |
| runs\B\synthetic_bias\B_synthetic_bias_E1.json | E1.2 | spec | **MECHANISM-FAIL [design]** | ratio in (0.3500, 1.2139); max per-task std 2.49e-01 | (0.98, 1.0); std < 1e-3 |
| runs\B\synthetic_bias\B_synthetic_bias_E1.json | E1.3 | spec | **MECHANISM-FAIL [design]** | max diff 13.9578 | < 5e-3 |
| runs\B\synthetic_bias\B_synthetic_bias_E1.json | E1.2 | variant_wellcond | **CALIBRATION-EXCURSION** | ratio in (0.9749, 1.0190); max per-task std 1.36e-02 | (0.98, 1.0); std < 1e-3 |
| runs\B\synthetic_bias\B_synthetic_bias_E1.json | E1.3 | variant_wellcond | **CALIBRATION-EXCURSION** | max diff 0.0752 | < 5e-3 |
| runs\B\synthetic_bias\B_synthetic_bias_E2.json | E2.1a | variant/lam=0/n=50000 | **CALIBRATION-EXCURSION** | min cos 0.997610 (max 0.999992) | > 0.999 |
| runs\B\synthetic_bias\B_synthetic_bias_E2.json | E2.1b | variant/estimated/n=1000/seeds[0, 1, 2, 3, 4] | **CALIBRATION-EXCURSION** | valid cos min 0.0153, median 0.9713, max 0.9991; 0/1500 undirected (NaN) pairs | > 0.99 |
| runs\B\synthetic_bias\B_synthetic_bias_E2.json | E2.2 | variant/lam=0/n=50000 | **CALIBRATION-EXCURSION** | task-pair means in (1.0003, 1.0018) | (0.97, 1.00) |
| runs\B\synthetic_bias\B_synthetic_bias_E2.json | E2.3b | variant/lam=0/n=50000 trained | **CALIBRATION-EXCURSION** | min cos 0.993327; max CV 0.0280 | cos > 0.9999; CV < 1e-3 |
| runs\B\synthetic_bias\B_synthetic_bias_E2.json | E2.4 | variant | **CALIBRATION-EXCURSION** | per-arm breakdown in details | cos > 0.999 (n_hi arms), > 0.99 (n_lo ordering arms); norm c |
| runs\B\synthetic_bias\B_synthetic_bias_E4.json | E4.1b | paper protocol n=2000, MC 1e6 | **CALIBRATION-EXCURSION** | max dev 0.6625; violations 332/360 | <= 0.001 each |
| runs\B\synthetic_bias\B_synthetic_bias_E4.json | E4.1c | empirical means | **CALIBRATION-EXCURSION** | {'lam=1e-05': 0.31467, 'lam=0.001': 0.06801, 'lam=0.1': 0.00599} | <= 0.01 |
| runs\B\synthetic_bias\B_synthetic_bias_E4.json | E4.2c | paper protocol | **PAPER-QUESTION** | {'lam=1e-05': {'mean': 0.8513, 'min': 0.8377, 'max': 0.8632}, 'lam=0.001': {'mea | >= 0.5 (paper threshold; expect ~0.94) |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.1[spec|exact] | exact | **MECHANISM-FAIL [design]** | slope -0.9393 (95% CI [-0.8788, -0.2430], R^2 0.6376) | [-0.31, -0.19] |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.2[spec|exact] | exact | **MECHANISM-FAIL [design]** | slope -0.2268 (95% CI [-0.2583, -0.1902]); gap(WT-FIX) -0.7125 | slope <= -0.40; WT-FIX >= 0.15 |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.2[spec|empirical MC 100000] | empirical MC 100000 | **MECHANISM-FAIL [design]** | slope -0.2227 (95% CI [-0.2553, -0.1854]); gap(WT-FIX) -0.0159 | slope <= -0.40; WT-FIX >= 0.15 |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.3[spec] | exact + empirical | **MECHANISM-FAIL [design]** | {'full_exact': -0.9393, 'tail4_exact': -1.9969, 'full_mc': -0.2385, 'tail4_mc':  | tail4 < full (both arms) |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.2[variant_wellcond|exact] | exact | **CALIBRATION-EXCURSION** | slope -0.3873 (95% CI [-0.4198, -0.3510]); gap(WT-FIX) 0.1672 | slope <= -0.40; WT-FIX >= 0.15 |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.2[variant_wellcond|empirical MC 100000] | empirical MC 100000 | **CALIBRATION-EXCURSION** | slope -0.3879 (95% CI [-0.4249, -0.3515]); gap(WT-FIX) 0.1679 | slope <= -0.40; WT-FIX >= 0.15 |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.3[variant_wellcond] | exact + empirical | **CALIBRATION-EXCURSION** | {'full_exact': -0.2201, 'tail4_exact': -0.1953, 'full_mc': -0.22, 'tail4_mc': -0 | tail4 < full (both arms) |
| runs\B\synthetic_rate\B_synthetic_rate_E5.json | E5.4 | empirical MC control | **CALIBRATION-EXCURSION** | {'min_frac_within_3se': 0.985, 'cal': {'spec': {'wt': {'max_se_units': 2.8086638 | frac >= 0.99 (gated cells) |

PASS: E1.4, E1.1, E1.4, E3.2b, E3.3b, E4.2a2, E4.3a, E4.3b, E5.1[spec|empirical MC 100000], E5.1[variant_wellcond|exact], E5.1[variant_wellcond|empirical MC 100000], E5.5, E6.1[kappa=1.5], E6.1[kappa=2], E6.1[kappa=3], E6.5

## Diagnoses recorded by the suites for the excursions and paper questions

- runs\B\synthetic_bias\B_synthetic_bias_E1.json E1.2: Ratio spread is finite-sample MLE noise (shrinkage (1-lam*alpha_t) ~1-1e-4 is far below the noise floor at this n; achieving std<1e-3 needs ~n=1e6/class, beyond CPU budget).
- runs\B\synthetic_bias\B_synthetic_bias_E1.json E1.3: Same finite-sample noise source as E1.2 (b scale O(1); noise ~1%).
- runs\B\synthetic_bias\B_synthetic_bias_E2.json E2.1a: Finite-sample head noise sqrt-level; bias signal sqrt(gamma)~0.5-0.7.
- runs\B\synthetic_bias\B_synthetic_bias_E2.json E2.1b: At n_t=1000 (d=32) the estimated-centroid noise sqrt(tr(Sigma)/n) is comparable to the weakest bias signals sqrt(gamma_min)~1, so weak task pairs are direction-noise dominated (NaN = |pred_hat| ~ 0). Theorem 4.1 is exact in the population arm (E2.3a) and holds to cos 0.998 on trained heads with true stats (E2.1a).
- runs\B\synthetic_bias\B_synthetic_bias_E2.json E2.2: Finite-sample noise (shrinkage-only-lowers argument is a population statement); per-pair spread in details.
- runs\B\synthetic_bias\B_synthetic_bias_E2.json E2.3b: Trained-arm noise floor: per-pair B_hat noise/signal ~1-2 pct at n=50000 exceeds the 0.1 pct CV budget (the assertion is stated for the population arm, which is algebraically identical and passes exactly; see E2.3a).
- runs\B\synthetic_bias\B_synthetic_bias_E2.json E2.4: Per-arm values in details: n_lo arms sit at the sqrt(d/n) noise floor (0.99-0.999); lambda=1e-2 shrinkage attenuates ||B|| by (1-lam*alpha_t), which is the theorem's own prediction, not a bias-identity failure.
- runs\B\synthetic_bias\B_synthetic_bias_E4.json E4.1b: The 0.001 tolerance is the paper's MC-precision budget; the trained head adds n^{-1/2} estimation noise (measured spread up to ~0.05 per seed at lam=1e-5, large a; ~0.005 at lam=1e-1). Per-lambda mean P13 agrees with 0.2398 to <=0.001 at lam in {1e-3, 1e-1}; the exact flatness claim itself passes algebraically (E4.1a).
- runs\B\synthetic_bias\B_synthetic_bias_E4.json E4.1c: lam=1e-5 mean drifts at large a (finite-sample e1-direction noise of task-1's head grows with a; ridge at lam>=1e-3 stabilizes it).
- runs\B\synthetic_rate\B_synthetic_rate_E5.json E5.2[variant_wellcond|exact]: Measured slope just misses the -0.40 gate on the full 8-point grid: the small-n FIX cells carry plug-in-stats noise on top of the lambda-residue, biasing the OLS upward; the bootstrap CI reaches the gate at its lower edge. The paper's signature claim (FIX decays clearly faster than WT) PASSES via the gap assertion on this arm.
- runs\B\synthetic_rate\B_synthetic_rate_E5.json E5.2[variant_wellcond|empirical MC 100000]: Measured slope just misses the -0.40 gate on the full 8-point grid: the small-n FIX cells carry plug-in-stats noise on top of the lambda-residue, biasing the OLS upward; the bootstrap CI reaches the gate at its lower edge. The paper's signature claim (FIX decays clearly faster than WT) PASSES via the gap assertion on this arm.
- runs\B\synthetic_rate\B_synthetic_rate_E5.json E5.3[variant_wellcond]: Measured tail slope is SHALLOWER than the full-grid fit: the lambda(n) term is a clean n^-1/4 across the grid while the estimation-noise term (steeper, n^-1/2) only lifts the small-n end, so the tail converges toward -0.25 and the full fit sits below it -- the flattening direction G06 predicted is inverted when the schedule term dominates (measured mechanism, honest FAIL).
- runs\B\synthetic_rate\B_synthetic_rate_E5.json E5.4: Aggregate calibration is sound: mean |dev|/SE across all 800 cells is 0.818 vs the theoretical E|Z| = sqrt(2/pi) = 0.798 for a correctly calibrated MC. The gate fails on one arm-method (3/200 cells outside 3 SE vs binomial expectation 0.5/200; probability of >= 3 tail draws ~ 1.4%) -- an honest tail outcome of the pinned MC streams, not a systematic defect; the EXACT arm (deterministic, no MC noise) is unaffected and carries the rate assertions.
