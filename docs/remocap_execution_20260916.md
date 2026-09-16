# ReMoCap / Ninjutsu execution protocol

The user withdrew DuoBox and authorized the ReMoCap recommendation. This document replaces the future BM2 data requirement; previous BM1 results and protocol v1 are historical evidence and remain unchanged.

## Research question and comparison boundary

Construct useful humanoid combat demonstration data from a limited real seed set. Ninjutsu supplies martial-arts human interaction motion, not native robot trajectories or a complete boxing hit/block benchmark. Actor/reactor identity must not be silently interpreted as attacker/defender ground truth.

Published ReMoS Table 2 comparisons include InterGen, ReMoS, RAIG, ComMDM, InterFormer and MixNMatch. We prioritize the first two for implementation. Our common 22-joint body-only protocol is an adaptation, not an exact Table 2 reproduction. Existing InterHuman TIMotion FID numbers are not comparable to ReMoCap FID.

## Ordered execution

1. Download the official archive, preserve license text locally, check CRC and SHA256. Keep raw and derived data off GitHub.
2. Audit CSV headers, BVH frame rate, units, axis, official train/test source identities and duplicate motions. Reserve validation source recordings from training before constructing windows. Do not concatenate frames from different shots.
3. Adapt upstream ReMoS body and InterGen denoisers to the same body position representation, actor-only inputs, training examples and update budget. Keep upstream sources unmodified and pinned.
4. Pilot 200 updates, inspect validation and runtime, then train 5000 updates per seed (0,1,2). Select checkpoints exclusively on validation data. Store actual memory/runtime, complete logs and source hashes.
5. Evaluate fixed test conditions with identical seeds and 50-step DDIM. Report body reconstruction, velocity, geometric quality and diversity. Only report official FID when its feature extractor/protocol has been reproduced.
6. Compare raw, temporal, contact and combined refinement on the first chronological window from each test source (all three draws). Reuse the project's bone-preserving refinement. A weak training-library proposal retrieves the nearest training actor using 10 sampled body frames and transfers its longest geometric near-contact episode; no test reactor or test contact labels enter this step. When no plan exists, contact-only remains raw and combined reduces to temporal-only. This is a retrieval-conditioned component baseline, not the completed proposed algorithm. Independent contact annotation remains a separate task.
7. Proceed to MoCap dose and robot utility only after the preceding gates pass. Fixed robot controller and equal training budgets are inherited unchanged.

## Upstream readiness and exact invocation

ReMoS README currently marks Ninjutsu training/testing as "coming soon". Its supplied Ninjutsu loader also references absent skeleton/visualizer modules. We therefore import the published body denoiser through our own source-preserving data adapter. This excludes the hand cascade and must be labelled ReMoS-body-adapted.

InterGen uses its upstream denoiser, with 66 body-position input channels instead of 262 original motion channels. Text conditions are masked. Joint unconditional denoising training followed by actor clamping is reported as InterGen-body-adapted.

Warm environment: `docs/timotion_environment.md`, unchanged `env-timotion.json` hash bbd2963d. No environment rebuild is planned. Upstream repositories live in ignored `third_party/`.

Run from `/data/code/ouyanghao/oyh/humanoid-combat-motion`:

```sh
bash scripts/run_timotion.sh -m scripts.smoke_remocap_models
```

Expected: both upstream denoisers complete a seeded CUDA forward/backward with finite loss and emit `REMOCAP_MODELS_WITNESS`. This validates model integration only, not dataset quality or scientific performance.

## Evidence and role

- ReMoS, ECCV 2024: Supporting Interaction Work / martial-arts reaction baseline. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05358.pdf
- ReMoS supplement: baseline adaptation details. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05358-supp.pdf
- ReMoS source and noncommercial dataset license: https://github.com/anindita127/ReMoS
- InterGen, IJCV 2024: Supporting Interaction Work. https://github.com/tr3e/InterGen

## Remaining limits

Reduced skeleton, training budget and losses differ from the published benchmarks. No new result establishes precise impact force, tactical causality, full-body robot dynamics, or downstream learning utility by itself. Report negative results as observed. A lower smoothing error is not evidence of better contact or robot learning.

## Acquisition and validation results

Archive acquired and CRC checked: 1,479,264,924 bytes, SHA256 `d79af424e1ac12a36e8afb9cf85dec9965a4c4b33a5611c6cf17d133707c7e72`. The original Ninjutsu split has 51 training and 28 test recordings. Training is divided into 41 training / 10 validation sources before cutting windows: 682 / 186 / 378 non-overlapping 40-frame windows at 20 fps. Actor identity is not enough to certify a subject-disjoint split.

The downloaded CSV timestamps and BVH frame-time metadata are inspected per recording. An observed raw rate is 50 fps, while the upstream preprocessing script hardcodes 25 fps. We use timestamp interpolation, not that assumption; keep this difference explicit in any comparison.

Independent model-integration check passed in the unchanged warm environment. The 200-step pilots reduce validation denoising loss but have large generated-position errors (ReMoS adaptation about 1026 mm, InterGen adaptation about 1403 mm on four validation windows). These are pipeline checks, not useful-motion claims. Full training starts from fresh seeds, not from these pilot checkpoints.
