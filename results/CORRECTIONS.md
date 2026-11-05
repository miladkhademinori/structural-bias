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
