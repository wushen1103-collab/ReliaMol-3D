# Revision analysis addendum

The CSV files contain the pocket-level docking-adjunct comparison, double-holdout
paired intervals, Figure 6 shortlist values, rule-ranking timings, and
generator-specific permutation diagnostics reported in the revised manuscript.
`close_remaining_reviewer_items.py` regenerates these tables from the
candidate-level intermediate frames and score files listed at its top. Run it
from the ReliaMol-3D repository root, or set `RELIAMOL_OUTPUT_ROOT` to the
directory containing those output folders. The script also requires the
Figure 6 shortlist and case-manifest paths via `--shortlists` and `--manifest`.

The sibling ReliaMol3D_Benchmark_v1.0/ directory contains derived
candidate-level endpoints and other benchmark data. This directory does not
contain raw molecular structures or the intermediate docking-score file.
