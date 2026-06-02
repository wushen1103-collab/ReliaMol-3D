# DUD-E Public Candidate Pool Audit

Targets: adrb2, bace1, cp3a4
Max molecules per class: 100

Purpose: verify whether public DUD-E SDF candidates have usable 3D coordinates near the receptor pocket before using pocket-conditioned reliability labels.

- adrb2 active: n=100, sanitize=1.000, median center-to-crystal=54.00 A, within12A=0.000, pocket-contact=0.000 -> needs docking/alignment
- adrb2 decoy: n=100, sanitize=1.000, median center-to-crystal=52.54 A, within12A=0.000, pocket-contact=0.000 -> needs docking/alignment
- bace1 active: n=100, sanitize=1.000, median center-to-crystal=36.98 A, within12A=0.000, pocket-contact=0.000 -> needs docking/alignment
- bace1 decoy: n=100, sanitize=1.000, median center-to-crystal=34.65 A, within12A=0.000, pocket-contact=0.000 -> needs docking/alignment
- cp3a4 active: n=100, sanitize=1.000, median center-to-crystal=48.52 A, within12A=0.000, pocket-contact=0.000 -> needs docking/alignment
- cp3a4 decoy: n=100, sanitize=1.000, median center-to-crystal=48.80 A, within12A=0.000, pocket-contact=0.000 -> needs docking/alignment

Decision:
- Do not use raw DUD-E SDF coordinates directly as pocket poses.
- Next step should dock a small active/decoy subset into each public receptor, then compute pocket/geometric/scoring failures on docked poses.
- The current audit remains useful as a clean molecule-quality and public-source traceability check.
