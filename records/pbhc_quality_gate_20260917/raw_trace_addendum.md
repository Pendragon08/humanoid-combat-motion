# Quality-arm raw trajectory recomputation addendum

Date: 2026-09-17. Review independence: same-family. Acceptance status: provisional.
**PASS for all four quality-arm numerical recomputations. Historical overall integrity verdict remains WARN.**

The supplied archive SHA256 was independently verified: b6de5461b99d89835b437cec3aa23e197bb103d72e7bb08cd4006d91b5eac85a.
Read directly from the archive without extracting it. Recomputed from raw body/joint positions, environment origins, termination flags and original/quality references using an independent NumPy implementation. Did not call the experiment evaluator or summarizer, and did not run or modify training.

|Motion|Completion|Quality-target strict|Original-target strict|Original root mean (m)|Original body8 mean (m)|Original joint29 mean (rad)|Observed time range (s)|
|---|---:|---:|---:|---:|---:|---:|---:|
|Hooks_punch|100%|0%|0%|0.134669989|0.149116859|0.111673988|5.80-5.80|
|Horse-stance_punch|100%|0%|0%|0.087035924|0.104807273|0.106165051|6.62-6.62|
|Roundhouse_kick|0%|0%|0%|0.118630938|0.126474202|0.135643676|1.66-1.66|
|Side_kick|100%|0%|0%|0.483136058|0.484851986|0.103496023|5.92-5.92|

All four quality-arm summary rows match: strict/completion exactly; original root/body/joint means within 1e-6; raw trace error arrays and metric means within 2e-6. Alive masks exactly match first-termination censoring, including terminal frames. Reference-frame indices, horizon, joint/body ordering, 50 Hz and 32 environments were checked. Trace/reference hashes match manifests/metrics; archived summary matches COMPLETE and the local summary.

This resolves the earlier missing-quality-trace limitation only. Raw/baseline traces are absent, so raw-arm results remain unverified by independent recomputation. Checkpoint bytes and installed simulator source are absent. No claim of physical feasibility, training gain or generalization follows. The matched-training queue was not modified or evaluated.

Historical integrity_audit.md/json are preserved. Exact member hashes and calculations are recorded in raw_trace_addendum.json; reproducible read-only computation is in raw_trace_recompute.py.
