# ADR-0001: Select BERTurk for the text-origin prototype

- Status: Accepted for prototype integration
- Date: 2026-08-29
- Scope: Turkish human-versus-synthetic text-origin classification

## Context

N-Güven requires a Turkish text-origin signal with reproducible provenance, calibrated
confidence, bounded false positives, and practical CPU latency. The reviewed candidates
were BERTurk (`dbmdz/bert-base-turkish-cased`) and ModernBERT-TR
(`ytu-ce-cosmos/modernbert-tr-base`). Neither model was pretrained by this project.

Both candidates used the same `text-origin-tr-v1` dataset identity, preprocessing,
80/10/10 leakage-safe partition, seeds 17/42/71, one-epoch MPS compute protocol,
validation-only temperature calibration, and frozen 1,200-record test manifest. Raw
text, weights, calibrations, predictions, and per-seed results remain private; public
evidence contains metrics and cryptographic hashes only.

## Decision

Select the BERTurk candidate family for the next prototype integration step. Do not
promote a model artifact to production yet.

The candidates tied on the primary and first tie-break metrics:

| Metric (three-seed mean) | BERTurk | ModernBERT-TR |
|---|---:|---:|
| Macro F1 | 0.999722 | 0.999722 |
| Accuracy | 0.999722 | 0.999722 |
| Mean inference | 38.412 ms | 55.244 ms |
| p95 inference | 45.816 ms | 66.264 ms |
| Brier score | 0.000147 | 0.000365 |
| ECE | 0.000510 | 0.001032 |
| High-confidence FPR | 0.000000 | 0.000556 |

The frozen protocol ranks Macro F1 first, accuracy second, and mean inference latency
third. BERTurk therefore wins by the declared latency tie-breaker. Both candidates pass
the minimum Macro F1, maximum high-confidence FPR, and maximum p95 latency targets.

Canonical evidence: [`ml/evaluation/comparisons/text-origin-tr-v1.json`](../../ml/evaluation/comparisons/text-origin-tr-v1.json).

## Consequences

- The text service may next integrate a verified local BERTurk safetensors artifact
  behind the existing model-independent adapter.
- ModernBERT-TR remains a reproducible fallback candidate; it is not rejected on quality.
- A single release artifact and calibration must be selected using validation evidence,
  then verified through service contract, load, and rollback tests.
- Production approval remains blocked on external-domain and adversarial Turkish data,
  robustness slices, monitoring thresholds, and a documented rollback path.

## Limitations

The human and synthetic classes come from distinct reviewed source families. The
near-perfect scores can therefore reflect source or style shortcuts. The result supports
candidate selection inside this prototype benchmark only; it does not establish that
arbitrary AI-generated Turkish text can be detected reliably.
