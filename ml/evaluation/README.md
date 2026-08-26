# N-Güven Evaluation Foundation

This directory reserves controlled locations for future dataset traceability, evaluation scripts, and generated result artifacts.

**No benchmark has been performed yet.**

No metric, threshold, dataset selection, dataset license conclusion, or model comparison is recorded as an N-Güven result. ModernBERT-TR and BERTurk remain future candidates and have not been integrated or selected.

## Layout

```text
ml/evaluation/
├── manifests/
│   └── schema.json
├── results/
│   └── .gitkeep
└── scripts/
    └── .gitkeep
```

- `manifests/schema.json` defines the traceability contract for future dataset records.
- `results/` is reserved for reproducible outputs after an evaluation protocol is approved and executed.
- `scripts/` is reserved for evaluation and validation tooling.

## Schema validation

Validate the schema definition itself from the repository root after installing the text service development dependencies:

```bash
cd services/text-ai
python -m pip install -e ".[dev]"
cd ../..
python -c 'import json; from jsonschema import Draft202012Validator; Draft202012Validator.check_schema(json.load(open("ml/evaluation/manifests/schema.json", encoding="utf-8")))'
```

Dataset records must not be added until their source, access date, license, content hash, label provenance, intended use, and split are known. Records must reflect verified metadata rather than placeholders.
