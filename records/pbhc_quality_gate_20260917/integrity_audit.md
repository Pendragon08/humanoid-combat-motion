# Quality gate experiment integrity audit

Date: 2026-09-17
Auditor: fresh read-only Codex reviewer
Review independence: same-family
Acceptance status: provisional
Overall verdict: WARN
Evaluation type: simulation_only

All evidence paths below are relative to D:/具身智能/humanoid-combat-motion. No training or evaluation jobs were launched by this reviewer. This preserves the review at its original evidence cutoff: the local quality-gate and raw diagnostic run directories were absent, so independent numerical verification from raw trajectories was unavailable.

## A. Ground-truth provenance — PASS with proxy qualification

Geometry comes from original reference trajectories plus URDF FK; contact masks are upstream heuristics. These limitations are explicit at scripts/audit_pbhc_temporal_quality.py:32-38,70-74. Original-target rescoring loads the untouched original reference (scripts/summarize_pbhc_quality_gate.py:22-31). Neither geometric penetration nor simulated contact is physical ground truth.

## B. Score normalization — PASS

Errors retain radians/metres and strict acceptance uses fixed thresholds (scripts/evaluate_pbhc_support_diagnostic.py:97-108). Source-relative derivative envelopes and penetration reductions are clearly gate criteria, not inflated performance scores (scripts/prepare_pbhc_quality_gate.py:49-76). No output-max performance normalization was found.

## C. Results and provenance — WARN

Report matches summary: all four quality-gated strict scores are zero; Horse-stance drops from raw 100% to 0% (records/pbhc_quality_gate_20260917/report.md:9-16). All ten recorded script/config/selection hash comparisons and COMPLETE-to-summary hash matched local bytes.

Both runs/pbhc_quality_gate_20260917 and runs/pbhc_support_diagnostic_20260917 were absent locally at the review cutoff. Trace/reference/checkpoint bytes and numerical recomputation were therefore unavailable to this reviewer. This is an independent-verification limit, not evidence of phantom results.

## D. Actual calls — PASS at code level; execution recomputation unavailable

Scheduler invokes evaluator and summarizer (scripts/run_pbhc_quality_gate.py:54-67); selection and verification invoke the gate (scripts/select_pbhc_quality_candidates.py:36-40; scripts/verify_pbhc_quality_gate.py:29-47). Shared gate implementation is correctly disclosed (records/pbhc_quality_gate_20260917/verification.json:225).

## E. Scope, selection and censoring — PASS with existing limits

Four development motions, seed 123, 32 nominal environments, zero training updates (records/pbhc_quality_gate_20260917/execution_manifest.json:56-59). All four sources are retained. Source-only selection contains no reads of new execution outcomes.

Means include termination frames and exclude subsequent resets (scripts/evaluate_pbhc_support_diagnostic.py:106,117-128). Roundhouse means compare different observed horizons, 1.66 versus 1.70 seconds (records/pbhc_quality_gate_20260917/summary.json:79,95); its smaller mean cannot establish improved execution. Native termination still depends on supplied references (summary.json:3).

## F. Evaluation type

simulation_only. Source-relative geometric checks are explicitly labeled proxies. There is one development evaluation seed and no training-gain, generalization, robustness, or synthetic-data-utility conclusion.

## Substantive clarifications and corrections

1. QP guarantees must be stated narrowly. P = 2(I + 20 D1^T D1 + 500 D2^T D2) is positive definite, so a feasible problem has a unique minimizer. Constraints bound discrete correction differences, with conditional source-relative velocity/acceleration envelopes (scripts/prepare_pbhc_quality_gate.py:20-40). They do not guarantee feasibility, continuous-time smoothness, smaller total acceleration, contact correctness, torque feasibility, or execution. Absolute peaks may increase to 1.05 times the source peak plus tolerance. Post-solve zeroing/clipping means final gate checks, not solver status alone, establish acceptance (:110,119-121). All reported QPs solved; largest reported constraint violation was approximately 1.93e-10.

2. Finite search does not relax metrics. It scales toward raw and reruns every unchanged gate check (scripts/select_pbhc_quality_candidates.py:16-40). Nonlinear geometry is rechecked. It finds the largest passing member of {1,.75,.5,.25}, not the largest possible correction. Report wording “最大可接受修正幅度” should say “预定离散候选中的最大通过幅度” (report.md:4). Selection is data-adaptive development, correctly disclosed as added after v1 inspection (selection_protocol.json:10).

3. Dependency SHA chain is incomplete. Preparation records contact hashes (scripts/prepare_pbhc_quality_gate.py:126), but selection/verification reload contact files without asserting those hashes (scripts/select_pbhc_quality_candidates.py:31-33; scripts/verify_pbhc_quality_gate.py:26-28). Imported gate/geometry implementations, URDF, inventory/source FPS, and frozen-config bytes are not all constrained by the execution freeze (scripts/run_pbhc_quality_gate.py:35-37). Record and verify these dependencies for reproducibility. Current records do not establish that any dependency changed.

4. Data-quality gains are limited. Reported mean geometric penetration reductions are 1.54%, 1.78%, 4.99%, and 4.25%; maximum penetration is unchanged in every case (verification.json:40-43,88-91,136-139,184-187). These support modest source-relative geometry improvement only. Current execution evidence does not support dynamics or training improvement.

## Claim impact

- Unchanged-threshold source-only finite selection: supported by inspected code and records.
- Four sources retained, including negative execution outcomes: supported.
- Modest mean geometric penetration reduction: supported by recorded metrics; independent trajectory recomputation unavailable.
- Largest possible accepted correction: unsupported; qualify as largest passing value on the finite grid.
- Dynamics, learning, generalization, or data-utility improvement: unsupported by this fixed-policy pilot.
