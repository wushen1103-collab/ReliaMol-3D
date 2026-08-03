# ReliaMol-3D

ReliaMol-3D is a post-generation reliability-fusion framework for auditing and reranking 3D structure-based molecular generation outputs. The code in this repository supports the manuscript experiments on DiffSBDD, CrossDocked receptor-coordinate provenance, PoseBusters physical-validity gating, Vina/GNINA score baselines, OpenMM fixed-receptor post-relaxation endpoint analysis, aligned public-PDB coordinate-stress calibration, CASF redocking boundary analysis, TargetDiff-family multi-generator transfer, and DUD-E controlled-failure stress testing.

This is a lightweight reproducibility repository. It contains source code, experiment scripts, configs, paper-level summary tables, and small derived result files. It does not contain raw molecular datasets, generated SDF/MOL2/PT files, docking work directories, figures, PDFs, virtual environments, or large prediction tables.

## Repository Layout

```text
src/reliamol3d/            Core metrics and model utilities
scripts/                   Audit, training, scoring, and summarization scripts
configs/                   Small YAML configs for proxy and smoke experiments
reports/                   Compact manuscript-facing summary reports
paper_results/             Curated small derived tables from remote output reports
docs/                      Reproducibility, data-source, and paper-result notes
requirements.txt           Minimal pip dependencies
environment.yml            Conda environment recipe
REPRODUCIBILITY_SOURCE.txt Remote source and commit used to build this release
```

## Installation

The most stable route is Conda because RDKit and docking-related dependencies are easier to resolve there.

```bash
conda env create -f environment.yml
conda activate reliamol3d
pip install -e .
```

For a lighter CPU-only setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Some scripts additionally require external command-line tools such as AutoDock Vina, GNINA, Open Babel, or PoseBusters. Their executable paths are exposed as command-line arguments in the corresponding scripts.

## Data

Raw data are intentionally not tracked in Git. Reproduction requires downloading or preparing the public resources described in `docs/data_sources.md`, including:

- official DiffSBDD generated SDF samples;
- CrossDocked receptor coordinates in `.gninatypes` or aligned pocket PDB form;
- official TargetDiff-family sampling-result files;
- public DUD-E targets for the controlled-failure stress test.

Use local paths through script arguments such as `--sdf-root`, `--archive`, `--input-dir`, `--dude-root`, `--vina`, and `--gnina`. The original remote paths are preserved in defaults to document the exact run environment, but users should override them on a new machine.

## Main Reproduction Entry Points

The following commands reproduce the main evidence blocks after the required public data and tools are available.

```bash
# 1. DiffSBDD exact-CrossDocked receptor-coordinate audit
python scripts/audit_diffsbdd_crossdocked_gninatypes.py \
  --sdf-root /path/to/diffsbdd/crossdocked_fullatom_cond \
  --archive /path/to/CrossDocked2020_receptors.tgz \
  --index /path/to/tar_index.txt \
  --extract-dir /path/to/selected_receptors \
  --output-dir outputs/diffsbdd_crossdocked_gninatypes_audit

# 2. Main ReliaMol reranking on DiffSBDD exact-CrossDocked candidates
python scripts/run_diffsbdd_crossdocked_reliamol.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit/reports \
  --output-dir outputs/diffsbdd_crossdocked_reliamol

# 3. Feature-group ablation
python scripts/run_diffsbdd_crossdocked_feature_ablation.py \
  --feature-frame outputs/diffsbdd_crossdocked_reliamol/reports/feature_frame.csv \
  --output-dir outputs/diffsbdd_crossdocked_feature_ablation

# 4. Threshold sensitivity
python scripts/sweep_diffsbdd_crossdocked_thresholds.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit/reports \
  --pred-dir outputs/diffsbdd_crossdocked_reliamol/reports \
  --output-dir outputs/diffsbdd_crossdocked_threshold_sensitivity

# 5. PoseBusters gate
python scripts/run_diffsbdd_crossdocked_posebusters.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit/reports \
  --pred-dir outputs/diffsbdd_crossdocked_reliamol/reports \
  --output-dir outputs/diffsbdd_crossdocked_posebusters_molfast

# 6. Vina/GNINA score-only and ensemble baselines
python scripts/score_diffsbdd_crossdocked_vina_gnina.py \
  --audit-dir outputs/diffsbdd_crossdocked_gninatypes_audit/reports \
  --pred-dir outputs/diffsbdd_crossdocked_reliamol/reports \
  --posebusters-dir outputs/diffsbdd_crossdocked_posebusters_molfast/reports \
  --output-dir outputs/diffsbdd_crossdocked_vina_gnina_100pocket \
  --vina /path/to/vina \
  --gnina /path/to/gnina

# 7. TargetDiff-family official multi-generator audit and reranking
python scripts/audit_targetdiff_official_meta.py \
  --input-dir /path/to/targetdiff_sampling_results \
  --pocket-dir /path/to/crossdocked_pocket10 \
  --output-dir outputs/targetdiff_official_multigen_audit

python scripts/run_targetdiff_official_multigen_reliamol.py \
  --audit-dir outputs/targetdiff_official_multigen_audit/reports \
  --output-dir outputs/targetdiff_official_multigen_reliamol

# 8. DUD-E controlled-failure external stress test
python scripts/audit_dude_public_pool.py \
  --dude-root /path/to/dude_subset \
  --output-dir outputs/dude_public_pool

python scripts/run_dude_public_reliability.py \
  --input-root outputs/dude_docking_public6 \
  --output-dir outputs/dude_public_reliability_external_ood

# 9. JCTC post-relaxation endpoint and coordinate-stress analyses
python scripts/revision_r1_openmm_relaxation.py --help
python scripts/revision_r1_analyze_physical_endpoints.py --help
python scripts/jctc_make_aligned_receptor_plan.py --help
python scripts/jctc_aligned_physical_cv.py --help
python scripts/jctc_derive_priority_tables.py --help

# 10. CASF redocking boundary analysis
python scripts/revision_r1_casf_redocking.py --help
python scripts/revision_r1_casf_evaluate.py --help
```

## Included Results

`paper_results/` contains small derived report tables copied from the remote experiment output folders. These files are included to let readers verify manuscript-level numbers without downloading multi-GB raw files. The JCTC submission update additionally tracks compact post-relaxation endpoint summaries in `paper_results/outputs/jctc_priority_tables_v1/` and aligned public-PDB calibration summaries in `paper_results/outputs/jctc_aligned_physical_cv_v1/`. See `docs/paper_result_map.md` for a mapping between manuscript claims and result files.

The original lightweight release was built from:

```text
Remote source: /home/test/wsk/16ReliaMol-3D
Remote commit: d78abc2
```

The JCTC submission update adds OpenMM post-relaxation endpoint, aligned public-PDB calibration, and CASF redocking-boundary scripts and compact result tables in the current Git commit.

## What Is Excluded

The following are intentionally excluded by `.gitignore`:

- raw `outputs/` work directories;
- `logs/`;
- `.venv/` and `.conda_env/`;
- generated caches and bytecode;
- raw molecular files such as `.sdf`, `.mol2`, `.pdbqt`, `.pt`, and compressed archives;
- figures and manuscript PDFs.

This keeps the repository small enough for normal GitHub use while preserving the code and derived evidence needed for reproducibility.
