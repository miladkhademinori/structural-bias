# Corrections to results records

Running log of corrections applied to the store after registration. Each entry
names the record touched, the before state, the after state, and the reason.
Store convention respected: re-read before write, sibling rows preserved.

## 2026-10-22, agent CI5: three pending ledger rows closed

The replication letter of 2026-10-22 carried three rows whose status column
read "log pending" while the underlying store rows carried FAIL verdicts.
Claude's reviewer-lens ruling (reply section 12): "pending" next to a FAIL is
where a reader stops trusting the table; close those three first. Closed by
annotation, not by re-run: in each case the computation exists, is stored, and
is final; the only outstanding item is the unwritten I03/I04 exec-log
narrative, which is a documentation task assigned to the J wave. Re-running
the suites would reproduce the stored bytes (seed policy: SHA-256 over
(seed, purpose, salt); see H06/I01/I02 determinism records).

### Row 1: drift asset row (letter sec 2.4 inventory)

- Before: "fig:drift-trajectory + tab:drift-slopes | Linear drift, t* values |
  EX9-EX11 stored; log pending"
- After: EX9-EX11 stored and final. EX9: 8 of 9 assertions PASS; the FAIL
  (A23a) is an owner-flag with full diagnosis in the run record: the printed
  -0.164 slope is recovered only as a stages-1-5 pre-acceleration window fit
  (-0.1620); the stated stages-1-10 OLS is arithmetically -0.2191. EX10: A27
  PASS; the FAIL (A26) is an owner-flag: the printed r=0.998 is not
  recoverable from the printed point set (measured 0.9931). EX11 PASS.
- Records touched: `INDEX.json` row `A_drift_EX9` gained a `ledger_close`
  field; run JSONs unchanged (already final).

### Row 2: claims-ledger drift row (letter sec 6 crosswalk)

- Before: "A: critical horizon t* and accuracy crossover | drift suites
  EX9-EX11 | Store rows only so far (EX11 PASS); log pending"
- After: the horizon claim itself is verified: t* 14.1953 / 2.341 / 0.2341
  against printed 14.2 / 2.35 / 0.21 (A23c1-c3 PASS), time-to-floor 0.474 vs
  0.48 (A23d PASS), secant identity exact to 1e-12 (A24 PASS), t* ratio 6.06
  in the sub-linear band (A25 PASS). The two suite FAILs are the owner-flags
  A23a and A26 above, not missing computations. I04 log owed to the J wave.
- Records touched: `INDEX.json` row `A_drift_EX10` gained a `ledger_close`
  field.

### Row 3: fleet-roster margin row (letter appendix D)

- Before: "I03-I04 | Paper A suites: synthetic_margin (EX1-EX5), drift
  (EX9-EX11); logs pending"
- After (margin arm): EX1, EX2, EX4, EX5 PASS; EX3 FAIL on 2 of 8
  sub-assertions, both exact-arm with diagnoses in the run record: EX3.4
  (interpolating exponent -(1-nu)/8 vs aligned -(1-nu/2)^2/8, difference
  nu^2/32 exceeds the 0.005 tolerance for nu > 0.4) and EX3.7 (Gaussian-tail
  prefactor contributes ~2^-0.25 per doubling, exceeding the 1.2 lower slack
  of the bracket). The 1/8 and 1/16 tightness anchors (EX3.1, EX3.2) PASS.
  I03 log owed to the J wave.
- Records touched: `INDEX.json` row `A_synthetic_margin_EX3` gained a
  `ledger_close` field.

### What was NOT changed

- No suite verdict was flipped, no measurement rewritten, no FAIL softened.
  The FAILs are findings about the papers' printed numbers and tolerance
  bands; they stand.
- The letter files themselves are frozen correspondence and were not edited;
  this note plus the `ledger_close` fields are the store-side record of the
  closes, and `docs/REPLICATION.md` carries the same status in the coverage
  matrix.

## 2026-11-05, agent RX4: Paper B withdrawal ledger (change-summary rows 1-3)

The Paper B camera-ready (12500) states the outcome of its verification pass
in its change summary; rows 1-3 are the withdrawals. This entry records them
in this ledger in the paper's own disclosure wording (rows quoted from the
greenpass-staged change-summary block; every number verified against
`freeze_packs/B/12500_final.pdf`, pymupdf extraction, 2026-11-05). No suite
verdict is flipped and no measurement rewritten; each row names the
withdrawn claim and what replaces it.

### Row 1: gap-recovery range withdrawn

- Before: the accepted abstract and introduction state that the correction
  recovers 82 to 94 percent of the joint-training gap (mean 90.5).
- After (paper's wording): "Under the released protocol this range is not
  reproduced: only 4 of the 10 benchmark rows regenerate, and on CIFAR-100
  the corrected head lands 46.2 to 62.5 against the accepted 85.7 to 86.5,
  below the uncorrected head, while on DomainNet the ordering uncorrected,
  corrected, joint-training holds (50.6 < 53.1 < 68.3 on DINOv2; 53.8 <
  62.9 < 71.5 on CLIP)." The camera-ready reports the benchmark per row
  with per-row provenance; the accepted values of the rows that do not
  regenerate are withdrawn to the provenance ledger -- the four ImageNet-100
  rows because the dataset root is not distributed with the released code,
  and the two ImageNet-1K rows because the released code has no builder for
  that dataset. The scale-matched recovery result enters as an appendix
  addition with its own provenance: recovery is backbone-conditional,
  holding on 1 of 3 backbones on DomainNet (DINOv2, 93%), and the withdrawn
  blanket recovery claim it replaces is named as withdrawn; the
  scale-matched correction was not computed on the CIFAR-100 rows and is
  reported as such.
- Store side: no released payload carries the accepted benchmark values
  (the shipped `tables/out/B/` fragments are synthetic selftest previews
  marked NOT A PAPER NUMBER), so nothing in the store needs striking; this
  row is the ledger the README points at. The `figures/figure8_gap_recovery.png`
  asset, which asserted the withdrawn 82-94% range with no withdrawal
  marking, was removed from the release the same day (README Figure 8 is a
  text notice).

### Row 2: bias-direction agreement withdrawn

- Before: the accepted version reports a bias-direction cosine of at least
  0.92.
- After (paper's wording): "the accepted cosine of at least 0.92 corresponds
  to neither regenerated statistic family: the pair-averaged cosine is 0.986
  to 0.9997 and is a gauge identity (at any head satisfying the softmax
  gauge it equals 1 by construction, so it checks the gauge, not the
  theorem), and the per-pair cosine is 0.22 to 0.35 with cross-pair
  coefficient of variation 0.21 to 0.32." Direction agreement is reported
  per pair; the averaged-cosine exhibit of the accepted version no longer
  appears; the appendix table prints its accepted ranges named as withdrawn
  beside the regenerated statistics. The definition behind the accepted
  values could not be established and is stated as such.

### Row 3: rate agreement withdrawn

- Before: the accepted version states that the predicted exponential rate
  matches the OLS slopes within 5 percent.
- After (paper's wording): "the regenerated slopes are negative on every
  backbone and shallower than the kappa-derived prediction by factors of 1.7
  to 3.3 (CLIP -0.045 against -0.0755; ViT -0.053 against -0.176; R^2 0.32
  to 0.81, about 900 pairs each), so prediction and measurement are
  reported side by side. The pooled pair (empirical -0.208 against
  predicted -0.192) prints as the accepted exhibit's bootstrap inset, named
  as the accepted values, and the agreement sentence is withdrawn; a
  per-pair comparison, each pair at its own kappa, is stated in its place."

### What was NOT changed

- No suite verdict was flipped, no measurement rewritten, no withdrawn value
  re-asserted. The rows above are the paper's own disclosures, quoted; the
  theory spine (theorems 3.3, 4.1, 5.1, 5.2, 6.2, 7.2 and the finite-sample
  restatement) is untouched by these verdicts and stands as reviewed.

