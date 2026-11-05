# How to regenerate the results store

Every family in `results/` regenerates from the code in this repository by one
of the `replicate.py` subcommands below (or the named experiment driver);
nothing in the store is hand-typed. Feature caches live under
`$REPLICATION_FEATURES` (default `./features`); results under `./results`.
The seed policy is deterministic (SHA-256 over (seed, purpose, salt);
H06/I01/I02 determinism records), so re-running a suite reproduces its stored
bytes. This is the Paper B (Structural Bias) repository; the Paper A suites
(margin, slope fits, drift) ship with the separate companion A repository.

## The store, family by family

| Family | What it holds | How to regenerate |
|---|---|---|
| `runs/B/synthetic_bias/` (E1-E4) | The closed-form structural-bias verification suites: one self-describing JSON envelope per run (configuration, seeds, `metrics.assertions` with each assertion's value, threshold and diagnosis; provenance `synthetic`). | `python replicate.py synthetic --run` (re-runs every suite, then scores), or `python -m experiments.synthetic_bias --suites E1,E2,E3,E4` |
| `runs/B/synthetic_rate/` (E5, E6) | The rate-comparison suites (slow versus parametric; exact and Monte-Carlo arms; `--full-grid` is the released grid). | `python replicate.py synthetic --run`, or `python -m experiments.synthetic_rate --suites E5,E6 --full-grid` |
| `runs/B/nonvision/` (`B_nonvision_smoke.json`) | The 20 Newsgroups non-vision arm, smoke scale, local caches (provenance `smoke-local-cached`). The full run is a separate command. | smoke: cached; full run: `python replicate.py nonvision --allow-download` (Colab; `results/INDEX.json` update is opt-in via `--update-index`) |
| `runs/B/nonvision/` (`CI3_diagnostic_controls.json`) | The CI-3 controls battery on the bias-direction diagnostic (provenance `ci3-controls-local-cached`); the record's verdict block carries the gauge-identity finding. | `python experiments/diagnostic_controls.py [--seeds 0,1,2,3,4]` |
| `runs/B/lda_ncm/` (`B_lda_ncm_demo.json`) | The closed-form LDA/NCM comparison (the `tab:lda-ncm-comparison` cell family) — the stored record is the synthetic demo preview (provenance `synthetic`); MISSING is reported, never estimated. | `python experiments/lda_ncm_eval.py` (closed-form, from the `data/B/features/` caches) |
| `INDEX.json` | The run registry: run id, path, provenance, suite status, created time, and notes (the `ledger_close` annotations of the corrections ledger). | Updated by each suite writer at run time (same commands as above; nonvision with `--update-index`). |
| `SCOREBOARD.md` / `SCOREBOARD.json` | The claim-by-claim scoreboard over the suite records: 54 assertions / 39 claims with tiers and verdicts (vocabulary and summary: `docs/VERDICTS.md`). | `python replicate.py synthetic` (scores the stored records; `--run` re-runs the suites first) |
| `CORRECTIONS.md` | The corrections ledger: every correction applied after registration, named before/after/reason — the CI5 ledger closes and the Paper B withdrawal rows of the camera-ready change summary. Hand-kept: it records the annotations, the commands above regenerate the bytes. | edited per correction |
| `_selftest/`, `_selftest_features/` | The end-to-end selftest on synthetic caches (drivers and table emitters). Synthetic provenance; nothing here is a paper number. | `python replicate.py selftest` |
| `README.md` | The store guide. | hand-maintained |

## The real-feature families (feature caches required)

- `python replicate.py features --paper B` or `colab/02_features_paperB.ipynb`
  — the frozen-feature caches for the B protocol (adds `mae_b16`, `resnet50`;
  ImageNet-100 needs `--imagenet100-root`; GPU).
- `python replicate.py analyze --paper B [--seeds 0,1,2,3,4]` — the benchmark
  rows into `results/paperB_results.json` (FIX and FIX_c fields travel with
  every row), then the tables.
- `python replicate.py nonvision --allow-download` — the 20 Newsgroups full
  run (Colab); `colab/05_nonvision_paperB.ipynb` is the notebook form.
- `python replicate.py figures` — every figure, each with a `.data.json`
  sidecar.
- `python replicate.py audit-tex --paper B --tex <neurips_2026.tex>` — the
  printed-versus-regenerated ledger (`TEX_AUDIT_B.md` / `.json`).
- `python replicate.py tables [--check]` — the released table fragments under
  `tables/out/B/` and `tables/out/DIFF.md` from the store; `--check` is the
  freeze test (byte for byte). The three released fragments are synthetic
  selftest previews of the paper's tables, marked NOT A PAPER NUMBER.

## Coverage matrix (the corrections-ledger status)

Carrying the same rows as `results/CORRECTIONS.md`:

| Record / ledger row | Stored | Status |
|---|---|---|
| `B_synthetic_bias_E1` | final | FAIL (spec arms [design]; variant arms recorded beside them; E1.4 PASS) — diagnoses in the record |
| `B_synthetic_bias_E2` | final | FAIL (E2.3a population arm PASSes exactly; the trained/estimated arms are excursions with measured noise floors) |
| `B_synthetic_bias_E3` | final | PASS |
| `B_synthetic_bias_E4` | final | FAIL (E4.2c PAPER-QUESTION; excursions diagnosed in the record) |
| `B_synthetic_rate_E5` | final | FAIL (spec arms [design]; variant arms recorded beside them; E5.0, E5.0b, E5.5 PASS) |
| `B_synthetic_rate_E6` | final | PASS |
| `CI3_diagnostic_controls` | final | PASS (verdict identity=True; the smoke cos=1.0000 is an algebraic gauge identity — see the record's verdict block) |
| CI5 ledger rows 1-3 (letter of 2026-10-22) | closed by annotation | the three "log pending" rows of the replication letter; the computations exist, are stored and are final — the Paper A run records they name (`A_drift_EX9`, `A_drift_EX10`, `A_synthetic_margin_EX3`) ship with the companion Paper A repository, and this ledger entry carries the same close wording |
| RX4 ledger rows 1-3 (change-summary withdrawals, 2026-11-05) | recorded | the gap-recovery range, the bias-direction agreement statistic and the rate-agreement sentence are withdrawn to the provenance ledger, each row naming what replaces them (per-row benchmark provenance; direction agreement per pair; prediction and measurement side by side). Store side: no released payload carries the withdrawn benchmark values — the shipped `tables/out/B/` fragments are synthetic selftest previews marked NOT A PAPER NUMBER, and the figure that asserted the withdrawn range was removed from the release |

No suite verdict was flipped, no measurement rewritten, no withdrawn value
re-asserted; re-running the suites reproduces the stored bytes.
