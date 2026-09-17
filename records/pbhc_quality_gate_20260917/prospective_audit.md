# Prospective matched-data pilot audit

Date: 2026-09-17
Review independence: same-family
Acceptance status: provisional
Verdict: WARN — no confirmed fatal startup or evaluation defect found by static inspection
Evaluation type: simulation_only
This reviewer did not launch or modify training. Local production manifests, matched smoke receipts, installed simulator runtime, or matched-training results were not available for independent runtime verification.

## Files inspected

configs/pbhc_matched_data_pilot_v1.json; scripts/train_pbhc_matched_data.py; scripts/prepare_pbhc_matched_data.py; scripts/run_pbhc_matched_data.py; scripts/summarize_pbhc_matched_data.py. Relevant evaluator, bundle, environment specification and launch-wrapper dependencies were also inspected.

## Fairness and budget

- Two selected development motions, raw and quality_gated arms, same motion on the same GPU. Selection after the fixed-policy outcomes is explicit, so no outcome-independent or held-out claim is warranted (config:2-5,17).
- Per arm: 512 environments x 24 steps x 1000 updates = 12,288,000 environment steps; four jobs total 49,152,000. Final checkpoint is fixed to model_999.pt, and receipt/exposure checks enforce the budget (config:6-20; run script:51-53; summary script:22).
- Quality-arm actor/critic and normalizer tensor hashes must match raw before learn (train script:73-90,114). The same seed is set before environment/runner construction (:35,41,44,72). A mismatch aborts rather than silently permitting unequal initialization.
- Both models are evaluated on both references under randomized and nominal regimes, producing 16 evaluations (run script:54-63; summary script:29-36). The primary endpoint uses the same untouched raw reference and randomized regime (config:15).
- Matching weights does not force identical later environment random draws; reference-dependent trajectories and adaptive sampling can diverge. This is correctly qualified (summary script:40). One training seed supports development diagnosis only.
- Fixed metric definitions are consistent between evaluators: strict per-frame joint/root/body/height and native termination checks. Mean errors are censored at first termination and must be interpreted with completion (evaluate_pbhc_beyondmimic.py:72-108; evaluate_pbhc_followup.py:82-118; summary script:41).

## Concrete readiness gates and corrections

1. **Require real two-update smoke receipts before launch.** The launcher expects TRAINING_COMPLETE.json with exactly 16 x 24 x 2 = 768 steps for both arms (prepare script:23-28). A run using --smoke-steps writes SMOKE_COMPLETE.json and cannot satisfy this check (train script:105-117). No smoke runtime was independently witnessed here; this is an unverified readiness gate, not a proven runtime defect.

2. **Bind quality references to the prior approved output.** Raw reference SHA is compared against selection provenance, but quality_gated reference SHA is merely taken from the current file (prepare script:45-54). Compare each quality reference against records/pbhc_quality_gate_20260917/execution_manifest.json or the approved conversion manifest before preparation. Otherwise an accidentally replaced quality reference can become the newly frozen input without detection. No mismatch is established from current local evidence.

3. **Bind smoke evidence to exact inputs and trainer.** Launch checks step counts and initialization digests, but not the smoke config bundle SHA, arm, seed, or trainer SHA (prepare script:23-28). Both raw and quality bundles and the frozen training script should be verified against the smoke records; otherwise unrelated/stale smoke runs could satisfy the gate.

4. **Record runtime source identity and validate 24 steps before spending the budget.** The environment-spec JSON names package versions and source commits, but matching its JSON hash does not establish the installed editable-source bytes (prepare script:36-37; env-isaaclab.json:9,33,36-40,59). The queue freezes local wrapper/evaluator files but not the imported upstream task/reward/runner source (prepare script:55-59). Validate the actual installed version/commit plus dirty state and assert agent.num_steps_per_env == 24 before learn. Current code checks exposure against 24 only after the full job (run script:51-53); a changed upstream default could waste a job before failing.

5. **Keep claims bounded and report all arms.** Two sources were chosen after earlier outcomes, one seed, no independent motions. Same-reference randomized results are the main comparison; own-reference or nominal improvements alone cannot establish general data utility (config:4,15; summary script:38-41,47).

## Checked potential startup issues

The pinned rsl-rl-lib 2.3.1 runner uses alg.policy, initializes logger_type inside learn, and saves the final checkpoint with the final zero-based iteration index. Thus train script:79,91-104 and config:model_999.pt are consistent with the pinned implementation. This is source-level compatibility evidence, not a witness of the installed server version.

Official source checked: [RSL-RL v2.3.1 OnPolicyRunner](https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v2.3.1/rsl_rl/runners/on_policy_runner.py).

Single-reference bundles preserve all required reference arrays and add metadata without interpolation (combat/reference_bundle.py:20-35). Scheduler chooses evaluator flags consistent with each parser (run script:57-60). The initial random checkpoint is taken before learning; the production final-checkpoint policy forbids cherry-picking.

## Blocking disposition

No confirmed fatal code-level blocker found. Execution readiness remains unverified until the matched two-update smoke and exact source/reference provenance checks are available. The audit does not authorize jobs and does not treat absent local runtime evidence as a failed experiment.
