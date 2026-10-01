# Derived data for the BIB revision

This directory holds numerical data supporting the revised
Downstream Computational Usability in Pocket-Conditioned 3D Molecular
Generation manuscript. It is not a copy of the submission files.

- ReliaMol3D_Benchmark_v1.0/data/ contains compressed candidate-level
  descriptors and derived endpoint outcomes for 10,000 DiffSBDD and 38,073
  target-generator candidates.
  The full score-free representation has 41 features; direct-excluded
  analyses use a 34-feature subset.
- ReliaMol3D_Benchmark_v1.0/splits/ records the fixed homology-aware
  assignments and receptor metadata.
- ReliaMol3D_Benchmark_v1.0/results/ contains aggregate and pocket-level
  statistics, protocol-sensitivity analyses, and transfer diagnostics.
- ReliaMol3D_Benchmark_v1.0/data_dictionary.csv defines the published
  columns and evidence-access roles.
- remaining_reviewer_20261001/ contains revision-specific docking-adjunct,
  double-holdout, Figure 6 shortlist, timing, and feature-importance tables.
  close_remaining_reviewer_items.py records their analysis logic.

The numerical candidate tables do not include raw ligand or receptor
coordinates. Obtain source structures from the cited public datasets under
their original licenses. Some analyses require intermediate model scores and
structure files that are not redistributed here; the diagnostic script is
therefore not a self-contained end-to-end reproduction pipeline. See the
repository scripts/ and docs/ directories for the available workflow code
and data sources.
