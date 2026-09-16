# BM1 completed evaluation archive

Run: `bm1_20260915_v1`; completed 2026-09-16 01:45 China time.

- `suite_result.json`: 20 per-repetition values, means, standard deviations, sampling uncertainty and paired differences.
- `suite_result.md`: compact Chinese summary.
- `artifacts/{baseline,caption_adapter}/evaluation.log`: original upstream metric logs.
- `generation_inputs.jsonl.gz`: complete per-call caption hashes, lengths and CUDA RNG hashes (no original captions).
- `frozen_inputs.json`: input file paths and SHA256; no original motion arrays.
- `provenance.json`: sample order, model/evaluator/adapter and environment provenance.
- `EVALUATION_COMPLETE.json`: original completion receipts and output hashes.
- `archive_index.json`: original server-relative paths, byte lengths and uncompressed hashes.
- `archive_verification.json`: independent local recomputation and checksum verification.

Recompute from repository root with `python -m scripts.verify_bm1_archive`. Python floating-point summary comparison uses 1e-12 tolerance across versions; raw log values and SHA256 comparisons are exact.

The available official InterHuman subset contains 1099 paired samples; this is not a complete original-dataset replication. Both arms use fixed pretrained weights and an existing caption-only adapter. Repetitions vary generation/evaluation randomness, not independently trained models. No positive combat contact, robot executability, or downstream imitation utility claim follows from these metrics.

Large pretrained models and source motion arrays remain at their recorded server locations. This archive stores the evidence needed to recompute the reported statistics without redistributing third-party datasets.
