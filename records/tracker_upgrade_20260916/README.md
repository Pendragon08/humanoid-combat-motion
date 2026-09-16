# Tracker upgrade development diagnostics

See [report and current algorithm](../../docs/tracker_upgrade_and_algorithm_20260916.md).

24 new physics executions:16 TWIST2 combat,4 SONIC combat,4 standing witnesses.
Summary additionally includes4 previous TWIST reference runs, so its24 rows
are not24 new combat executions. No policy training occurred.

TWIST2 source: https://github.com/amazon-far/TWIST2 at
d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a (MIT). Per-run executable fragments
are extracted from that source; original license retained in LICENSE_TWIST2.
SONIC observation/action code comes from HumanTracker at
79d637da5e07b8e2c588ca613d3254f9d8acc0f7 (Apache2.0),
https://github.com/GalaxyGeneralRobotics/HumanTracker.
NVIDIA model weights: https://huggingface.co/nvidia/GEAR-SONIC at
6733128a3d8a523b1418b06bca3cdf61c8b0987f, NVIDIA Open Model License.
Weights are not redistributed in this repository.

PBHC-derived input references retain their original source and noncommercial
license record under records/pbhc_20260916. Binary states and references remain
in the server/local results_archive.zip. verification_receipt.json records its
hash; per-motion hashes in summary.json were checked against actual NPZ bytes.

Different native model, simulator version, reset and gain settings mean this
is a controller-system diagnostic, not isolated algorithm superiority or a
paper benchmark reproduction. No HumanScore, contact success, real-robot or
new synthetic-data training utility is claimed. All failures are retained.
