/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# ADR-0002: Defer image-detector selection

- Status: Accepted; prototype integration blocked
- Date: 2026-08-31
- Scope: Static human-versus-AI-generated image classification

## Context

N-Güven needs an image-origin signal that remains useful after common social-media
transformations and does not produce an unsafe number of high-confidence false
positives. The reviewed candidates were AIorNot SigLIP2
(`prithivMLmods/AIorNot-SigLIP2`) and CapCheck ViT
(`capcheck/ai-image-detection`). This project did not train or fine-tune either model.

Both candidates were loaded from hash-pinned safetensors with remote code and network
access disabled. They were evaluated on the same source-separated SynCred-Bench
revision: 450 human and 600 synthetic originals, each represented by canonical,
JPEG-90, JPEG-70, resize, center-crop, and screenshot variants. This produced 6,300
predictions per candidate on the same Apple M2 Pro MPS runtime.

The protocol fixed the acceptance targets and ranking order before inference. A model
had to pass every target; ranking alone could not authorize integration.

| Measure | Target | ViT | SigLIP2 |
|---|---:|---:|---:|
| Overall Macro F1 | ≥ 0.80 | 0.531738 | 0.340523 |
| Worst-transformation Macro F1 | ≥ 0.70 | 0.508897 | 0.317235 |
| High-confidence false-positive rate | ≤ 0.05 | 0.306667 | 0.040000 |
| P95 inference latency | ≤ 3,000 ms | 63.590 ms | 68.345 ms |

ViT ranks first because worst-transformation Macro F1 is the primary ordering measure,
but it fails both quality targets and the high-confidence false-positive target.
SigLIP2 passes the high-confidence false-positive and latency targets, but fails both
quality targets. The screenshot transformation is the weakest slice for both models.

## Decision

Do not select either candidate for N-Güven prototype integration.

Treat ViT only as the ranking leader of this experiment, not as an accepted model.
Keep the image-analysis service unconfigured until a candidate passes the unchanged
acceptance gate on comparable evidence.

Canonical evidence:

- [public comparison](../../ml/evaluation/image/comparisons/image-origin-robustness-v1.json)
- [hash-bound evidence summary](../../ml/evaluation/image/benchmarks/evidence/image-origin-robustness-v1.json)
- [frozen benchmark lock](../../ml/evaluation/image/benchmarks/image-origin-robustness-v1.json)

## Consequences

- No image-detector weights may be wired into an application service or presented as a
  working N-Güven feature from this comparison.
- The UI and backend must use an explicit unavailable/not-configured state for image
  origin analysis until a later ADR accepts a model.
- The frozen test set must not be used to tune thresholds, prompts, preprocessing, or
  candidate hyperparameters. A separate development/validation set is required.
- The next candidate study should retain source separation, the six transformation
  slices, false-positive controls, hash-pinned safetensors, and the same acceptance
  semantics. Stronger licensed architectures or a validated ensemble may be assessed.
- The low latency of both candidates is operationally encouraging but cannot compensate
  for insufficient classification quality.

## Rejected alternatives

- **Select ViT because it ranks first:** rejected because rank does not override failed
  acceptance targets, especially the 30.67% high-confidence false-positive rate.
- **Select SigLIP2 because its high-confidence false-positive rate passes:** rejected
  because its overall and worst-transformation Macro F1 values are far below target.
- **Lower the targets after seeing the results:** rejected as post-hoc protocol drift.
- **Reverse labels or choose the better result manually:** rejected because both label
  maps match their pinned upstream configs and the comparison contract.
- **Create an ensemble from the frozen-test outputs:** rejected because this would tune
  against the final test evidence without a separate validation protocol.

## Limitations

SynCred-Bench is external to the candidates' declared training sources, but one dataset
cannot establish production generalization. The results do not cover video, audio,
general image manipulation, attribution to a particular generator, or future model
families. Raw images, weights, and per-record predictions remain private; public
artifacts expose metrics, non-sensitive environment metadata, and cryptographic hashes.
