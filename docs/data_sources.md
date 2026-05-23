# Data Sources

This repository does not track raw molecular data or docking work directories. The scripts expect local copies of public resources and expose path arguments so the same code can run outside the original remote server.

## DiffSBDD Candidates

Purpose: primary 10,000-candidate DiffSBDD exact-CrossDocked audit.

Expected input:

- official DiffSBDD sampled SDF files for CrossDocked;
- one SDF file per pocket, with generated candidate conformations.

Relevant scripts:

- `scripts/audit_diffsbdd_public_samples.py`
- `scripts/audit_diffsbdd_crossdocked_gninatypes.py`
- `scripts/run_diffsbdd_crossdocked_reliamol.py`

Key arguments:

```bash
--sdf-root /path/to/diffsbdd/crossdocked_fullatom_cond
```

## CrossDocked Receptor Coordinates

Purpose: exact receptor-coordinate provenance audit and receptor-aware feature construction.

Expected input:

- CrossDocked2020 receptor archive containing `.gninatypes` files;
- a tar index listing archive members;
- an extraction directory for selected receptors.

Relevant scripts:

- `scripts/audit_diffsbdd_crossdocked_gninatypes.py`
- `scripts/sweep_diffsbdd_crossdocked_thresholds.py`
- `scripts/score_diffsbdd_crossdocked_vina_gnina.py`

Key arguments:

```bash
--archive /path/to/CrossDocked2020_receptors.tgz
--index /path/to/tar_index.txt
--extract-dir /path/to/selected_receptors
```

## TargetDiff-Family Generated Results

Purpose: multi-generator transfer across TargetDiff, Pocket2Mol, CVAE, and autoregressive outputs.

Expected input:

- official TargetDiff-family sampling-result files;
- CrossDocked pocket PDB files aligned to the benchmark coordinate frame.

Relevant scripts:

- `scripts/audit_targetdiff_official_meta.py`
- `scripts/run_targetdiff_official_multigen_reliamol.py`

Key arguments:

```bash
--input-dir /path/to/targetdiff_sampling_results
--pocket-dir /path/to/crossdocked_pocket10
```

## DUD-E Controlled-Failure Stress Test

Purpose: external public stress test with active/decoy poses and deliberately corrupted reliability variants.

Expected input:

- public DUD-E target packages;
- active and decoy SDF/MOL2 files for the selected targets.

Relevant scripts:

- `scripts/audit_dude_public_pool.py`
- `scripts/run_dude_docking_smoke.py`
- `scripts/run_dude_public_reliability.py`

Key arguments:

```bash
--dude-root /path/to/dude_subset
```

## Docking and Pose Tools

Some baselines require local executables:

- AutoDock Vina;
- GNINA;
- Open Babel;
- Meeko receptor/ligand preparation scripts;
- PoseBusters for physical-validity checks.

Relevant scripts expose explicit executable arguments such as:

```bash
--vina /path/to/vina
--gnina /path/to/gnina
--obabel /path/to/obabel
--mk-prepare-ligand /path/to/mk_prepare_ligand.py
--mk-prepare-receptor /path/to/mk_prepare_receptor.py
```

## Included Derived Tables

Small derived tables are included under `paper_results/` for manuscript-number verification. These files are not raw data. They are compact summaries copied from the remote output reports after excluding large feature frames, raw candidate tables, predictions, figures, and docking work directories.
