# Verdicts: what the verification suites measured

The verdict vocabulary of the scoreboard (`results/SCOREBOARD.md`, generated
by `experiments/scoreboard.py`) and the per-table verdict summary, stated in
the paper's own measurement voice: every number below is a stored measurement,
printed beside the paper's printed value wherever one exists.

## Vocabulary

- **PASS** — the assertion held: at machine precision in tier 1, at the
  battery's calibrated tolerance in tier 2.
- **MECHANISM-FAIL** — the mechanism the claim asserts is not exhibited: an
  exact arm misses its numerical tolerance, a finite-sample arm misses by more
  than the battery's family-wise bound, or the spec design cannot exhibit the
  claim at any n (marked [design], with the variant arm printed beside it).
- **CALIBRATION-EXCURSION** — the mechanism is exhibited and one gate number
  is missed within the battery's bound; the size of the miss is reported in
  standard errors. Excursions are re-gated by standard error
  (`tests/zbattery.py`, `core/gates.py`), never by widening tolerances.
- **PAPER-QUESTION** — a deterministic finding about a printed paper number,
  carried to the owner's Q-list with the measured value.

Tier 1 (mechanism: algebraic and population identities, exact in all inputs)
must be all PASS for the code to be trusted. Tier 2 (calibration: Monte-Carlo
and finite-sample estimates, including exact functionals of finite-sample
heads) lists excursions with the suites' own diagnoses.

## The scoreboard as measured (Paper B store)

54 assertions from 6 suite records: 33 PASS, 7 MECHANISM-FAIL,
13 CALIBRATION-EXCURSION, 1 PAPER-QUESTION. Claims (the verdict unit): 39
claims — 24 PASS, 6 MECHANISM-FAIL, 8 CALIBRATION-EXCURSION, 1
PAPER-QUESTION. Tier 1: 17/17 PASS. Tier 2: 16/37 PASS, each excursion with
the suite's own diagnosis in the record.

## Verdicts by suite family

### E1 — the closed-form structural bias (`runs/B/synthetic_bias/B_synthetic_bias_E1.json`)

- E1.4 PASS. E1.1, E1.2, E1.3: the spec arms are MECHANISM-FAIL [design] (the
  spec design cannot exhibit the claim at any n), with the variant arms
  printed beside them: the E1.1 variant PASSes; the E1.2 and E1.3 variants are
  CALIBRATION-EXCURSIONs at a measured finite-sample MLE noise floor (ratio
  spread in (0.9749, 1.0190), per-task std 1.36e-02 at this n; achieving
  std < 1e-3 needs ~n=1e6/class, beyond CPU budget).

### E2 — the bias direction (`B_synthetic_bias_E2.json`)

- The population arm E2.3a PASSes exactly — Theorem 4.1 holds as an identity.
- The trained and estimated arms (E2.1a, E2.1b, E2.2, E2.3b, E2.4) are
  CALIBRATION-EXCURSIONs, each at its measured noise floor: Theorem 4.1 holds
  to cos 0.998 on trained heads with true statistics (E2.1a measures min cos
  0.997610); at n_t=1000 the estimated-centroid noise is comparable to the
  weakest bias signals, so weak task pairs are direction-noise dominated; the
  per-pair B_hat noise/signal of 1-2 percent at n=50000 exceeds the 0.1
  percent CV budget of E2.3b; the lambda-shrinkage attenuation the E2.4 arm
  measures is the theorem's own prediction.

### E3 — the falsification constructions (`B_synthetic_bias_E3.json`)

- E3.1, E3.2a, E3.2b, E3.3a, E3.3b PASS (paper Figure 2's model setting).

### E4 — the flatness of the corrected head (`B_synthetic_bias_E4.json`)

- The exact flatness claim passes algebraically (E4.1a); E4.2a, E4.2a2,
  E4.2b, E4.3a, E4.3b, E4.4 PASS.
- E4.1b and E4.1c are CALIBRATION-EXCURSIONs with diagnoses: the 0.001
  tolerance is the paper's Monte-Carlo precision budget, and the trained head
  adds n^{-1/2} estimation noise (per-lambda mean P13 agrees with 0.2398 to
  <= 0.001 at lam in {1e-3, 1e-1}).
- E4.2c PAPER-QUESTION (carried to the owner's Q-list): the paper-protocol
  rate measures mean 0.8513 at lam=1e-5 (0.8377-0.8632) beside the paper
  threshold 0.5 (expected ~0.94).

### E5 — the rate comparison (`runs/B/synthetic_rate/B_synthetic_rate_E5.json`)

- E5.0, E5.0b, E5.5 PASS.
- E5.1, E5.2, E5.3: the spec arms are MECHANISM-FAIL [design]; the variant
  arms are printed beside them. On the variant arms the paper's signature
  claim PASSES via the gap assertion (FIX decays clearly faster than WT; gap
  WT-FIX 0.1672 and 0.1679 against the 0.15 gate), while the full-grid slope
  measures -0.3873 / -0.3879 against the -0.40 gate — CALIBRATION-EXCURSIONs
  with the plug-in-stats noise diagnosis; the E5.3 variant measures the tail
  slope shallower than the full-grid fit (the lambda(n) term is a clean
  n^-1/4 while the estimation-noise term lifts the small-n end).
- E5.4 CALIBRATION-EXCURSION: aggregate calibration is sound — mean |dev|/SE
  0.818 across all 800 cells against the theoretical 0.798 for a correctly
  calibrated Monte-Carlo; the gate outcome is one arm-method's honest tail
  (3/200 cells outside 3 SE, probability ~1.4 percent), and the exact arm
  carries the rate assertions.

### E6 — the kappa sweep (`B_synthetic_rate_E6.json`)

- E6.0 through E6.5 PASS (kappa arms 1.5, 2, 3 included).

### The non-vision arm and controls (`runs/B/nonvision/`)

- `B_nonvision_smoke.json` — the 20 Newsgroups smoke record (local caches).
- `CI3_diagnostic_controls.json` PASS, verdict identity=True: the smoke
  cos=1.0000 is an algebraic gauge identity (a per-task-centered random head
  also gives cos 1), so it checks the gauge — the finding is recorded in the
  run record's verdict block.

### The LDA/NCM comparison (`runs/B/lda_ncm/B_lda_ncm_demo.json`)

- The closed-form demo preview of `tab:lda-ncm-comparison` (synthetic
  provenance); the released fragment under `tables/out/B/` is the synthetic
  selftest preview, marked NOT A PAPER NUMBER.

## The withdrawals, as the ledger records them

The camera-ready change summary's three withdrawals are recorded verbatim in
`results/CORRECTIONS.md` (RX4 rows, 2026-11-05), in the paper's own disclosure
wording; each row names what replaces the withdrawn claim:

- **The gap-recovery range (82-94 percent, mean 90.5) is withdrawn to the
  provenance ledger.** The benchmark is reported per row with per-row
  provenance: on the rows the released protocol regenerates, the ordering
  WT < FIX < J holds on the two DomainNet configurations (50.6 < 53.1 < 68.3
  on DINOv2; 53.8 < 62.9 < 71.5 on CLIP), while on CIFAR-100 the corrected
  head lands 46.2-62.5 against the printed 85.7-86.5, below the uncorrected
  head; the rows that do not regenerate sit in the ledger with their reasons
  (four ImageNet-100 rows: dataset root not distributed; two ImageNet-1K
  rows: no released builder). Scale-matched recovery is reported as
  backbone-conditional (DINOv2 93 percent of the gap; FIX_c 67.0 against
  J 68.3).
- **The bias-direction agreement statistic (cosine at least 0.92) is
  withdrawn.** Direction agreement is reported per pair: the regenerated
  per-pair cosine is 0.22-0.35 with cross-pair coefficient of variation
  0.21-0.32, and the pair-averaged cosine 0.986-0.9997 is a gauge identity.
- **The within-5-percent rate-agreement sentence is withdrawn.** The
  regenerated slopes are negative on every backbone and shallower than the
  kappa-derived prediction by factors of 1.7-3.3 (R^2 0.32-0.81), so
  prediction and measurement are reported side by side.

## Where the findings live

- Every verdict, value, threshold and diagnosis: the run records under
  `results/runs/B/` and the scoreboard `results/SCOREBOARD.md` / `.json`.
- The ledger closes and the withdrawal rows: `results/CORRECTIONS.md` (with
  the `ledger_close`/note fields of `results/INDEX.json`); the same status is
  carried in the coverage matrix of `docs/REPLICATION.md`.
- Printed-versus-regenerated table cells: `tables/out/DIFF.md` and
  `TEX_AUDIT_B.md` (from `python replicate.py audit-tex --paper B`).

No suite verdict was flipped, no measurement rewritten, no withdrawn value
re-asserted to produce this summary; the theory spine stands as reviewed, and
each withdrawn comparison is replaced by the measured statement recorded
beside it in the ledger.
