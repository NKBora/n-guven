# N-Güven Turkish Text-Origin Prototype Model Card

## Model decision

BERTurk is the selected candidate family for prototype integration after a controlled
comparison with ModernBERT-TR. No weight bundle is distributed by this repository and
no model is approved for production.

## Intended use

- Provide one calibrated signal in an offline Turkish human-versus-synthetic text
  analysis prototype.
- Support engineering evaluation and service integration behind the existing
  model-independent adapter.
- Return evidence for human review; never act as the sole basis for moderation,
  attribution, punishment, or authorship claims.

## Out-of-scope use

- Claiming that a person or named model authored a text.
- Applying the score to languages or domains not represented by reviewed evidence.
- Using the detector as an autonomous enforcement decision.
- Treating confidence as proof of origin.

## Training and evaluation

- Upstream: `dbmdz/bert-base-turkish-cased` at revision
  `b6e1de16c983e0f2c70664591ea3f22810072608`.
- Task adaptation: full fine-tuning selected over a linear probe for seeds 17, 42, 71.
- Dataset: private `text-origin-tr-v1`, 12,000 balanced records with leakage-safe
  80/10/10 splits and two synthetic generator families.
- Preprocessing: `tr-text-v1`, NFC normalization, LF newlines, maximum 128 tokens.
- Calibration: independent temperature scaling per seed, fitted on validation only.
- Final evaluation: one frozen 1,200-record test split after both candidate experiments
  were completed.

The three-seed mean test Macro F1 is `0.999722`; mean high-confidence FPR is `0.0`,
mean inference time is `38.412 ms`, and mean p95 inference is `45.816 ms` on the reviewed
local CPU backend. Full metrics, variance, confidence intervals, acceptance checks, and
artifact hashes are in [`comparisons/text-origin-tr-v1.json`](comparisons/text-origin-tr-v1.json).

## Risks and limitations

- Source-family and writing-style differences can provide shortcut signals.
- The benchmark does not demonstrate unseen-generator, adversarial, paraphrase,
  translation, OCR, or distribution-shift robustness.
- Very high benchmark scores increase the need for external validation; they do not
  eliminate uncertainty.
- Latency is host- and runtime-specific and must be remeasured in the deployed service.

## Release gate

Before production use, add external-domain and adversarial test sets, approve a single
hash-pinned model/calibration pair, run service load and rollback tests, define drift
alerts, and document human-review behavior for uncertain outputs.
