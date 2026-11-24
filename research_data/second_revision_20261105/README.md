# Final Docking and Three-Seed Transfer Records

This small addition supplies the final independent-docking records (Table S31)
and three-seed transfer statistics (Tables S11 and S32). It does not duplicate
the existing benchmark or deployment-model package.

- `independent_docking_final/`: 30 prespecified candidates, stage outcomes,
  sanitization audit, and final counts. All 30 Vina searches completed; each
  endpoint group had 14/15 sanitizable outputs. Its README distinguishes the
  final cohort from the superseded pilot.
- `transfer_three_seed/transfer_top10_rates_by_seed_and_pocket.csv.gz`:
  per-pocket top-10% endpoint-positive rates for seeds 11, 22, and 33, for
  zero-shot and double-holdout evaluation. Columns identify evaluation,
  generator, pocket, method, selected count, rate, and seed.
- `transfer_three_seed/transfer_seed_variability_summary.csv`: recorded
  three-seed means and sample SDs (ddof=1), including paired gains over QED/SA.
- `transfer_three_seed/transfer_three_seed_paired_bootstrap.csv`: recorded
  intervals after averaging seeds within each pocket; 5000 paired pocket
  resamples with bootstrap seed 20261105. Pooled results weight each
  generator-pocket equally. W/T/L uses a 0.01 absolute-rate tolerance.
- `summarize_transfer_bootstrap.py`: original statistical analysis script.
- `reproduce_statistics.py`: regenerates and checks the S11/S32 summaries
  and S31 Wilson/Newcombe intervals and Fisher test against recorded results.

## Reproduce the Summaries

From the repository root, using Python 3 with NumPy, pandas, and SciPy:

```sh
python research_data/second_revision_20261105/reproduce_statistics.py --output reproduced_revision_statistics
```

The command fails if any regenerated numerical summary differs from its
recorded result by more than floating-point tolerance. No training, docking,
or molecular preparation is rerun: these scripts reproduce statistical
summaries from the supplied recorded outcomes. The original benchmark
workflow, feature definitions, and fitted model remain in the existing
repository directories. Raw molecular structures and host-local paths are
not distributed in this addition. Source structures remain subject to the
licenses of their cited datasets.
