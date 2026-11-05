# Final Independent Docking Validation (Table S31)

This directory is the final record used in the second revision. The candidate
sample was fixed before docking, using seed 20261105 and the union of the primary
QED/SA and RF top-10% shortlists. The primary ligand score is QED minus 0.25 times
the SA proxy. Each endpoint stratum contains 15 candidates from distinct pockets;
the combined sample covers 28 pockets.

| Outcome | Endpoint-Positive | Endpoint-Negative |
| --- | ---: | ---: |
| Candidates | 15 | 15 |
| Ligand/receptor preparation completed | 15 | 15 |
| Vina search completed | 15 | 15 |
| Readable and sanitizable SDF output | 14 | 14 |

The two output failures are R2D13 (endpoint-negative) and R2D29
(endpoint-positive). Each candidate produced nine converted modes, none of which
passed RDKit sanitization because of invalid nitrogen valence. Their 18
mode-level checks and source-file checksums are in `output_sanitization_audit.csv`.

Each usable-output proportion is 0.9333 (Wilson 95% CI: 0.7018-0.9881).
The positive-minus-negative difference is 0.0000 (Newcombe 95% CI:
-0.2379 to 0.2379; two-sided Fisher P = 1.0000). These data do not show a
difference in basic docking-workflow completion between the endpoint groups.
Global docking completion and preservation of a supplied pose after local
relaxation are different outcomes.

## Files

- `candidate_results.csv`: all 30 candidate-level outcomes and recorded stages.
- `prespecified_candidate_manifest.csv`: candidate identities, original endpoint
  groups, shortlist membership, and sampling information before docking.
- `docking_validation_group_summary.csv`: stage counts by endpoint group.
- `docking_validation_statistics.json`: final completion statistics and intervals.
- `docking_validation_failure_reasons.csv`: the two final output failures.
- `output_sanitization_audit.csv`: all nine failed modes for each affected output.
- `FINAL_RECORD.json`: final-status declaration and primary analysis settings.
- `SHA256SUMS.json`: checksums of the public derived records.

Host-local input paths are omitted from the public CSV files. Candidate IDs and
all numerical measurements are retained. Source structures are identified by
the same source dataset and candidate IDs as the manuscript.

## Relation to the Earlier Pilot

This record supersedes the preliminary 30/30 usable-output report. That pilot
sample used QED minus 0.05 times SA rather than the primary benchmark score and
is not the final Table S31 cohort. It is not included in this directory. The
final matched-shortlist cohort has 30/30 completed Vina searches but only 28/30
readable, sanitizable outputs; those quantities must not be interchanged.

This record does not modify the endpoint definition or the primary benchmark
performance estimates.
