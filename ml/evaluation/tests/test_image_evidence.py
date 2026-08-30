/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import hashlib
import json
from pathlib import Path


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    EVALUATION_ROOT
    / "image"
    / "benchmarks"
    / "evidence"
    / "image-origin-robustness-v1.json"
)
COMPARISON_PATH = (
    EVALUATION_ROOT / "image" / "comparisons" / "image-origin-robustness-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_image_evidence_binds_reviewed_artifacts_without_selecting_model() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(
        (EVALUATION_ROOT / evidence["benchmarkLock"]["path"]).read_text(encoding="utf-8")
    )

    assert evidence["benchmarkLock"]["sha256"] == _sha256(
        EVALUATION_ROOT / evidence["benchmarkLock"]["path"]
    )
    assert evidence["candidateRegistry"]["sha256"] == _sha256(
        EVALUATION_ROOT / evidence["candidateRegistry"]["path"]
    )
    assert evidence["comparison"]["sha256"] == _sha256(COMPARISON_PATH)
    assert comparison["selection"] == {
        "leaders": [],
        "status": "no-qualified-candidate",
    }
    assert evidence["comparison"]["selectedCandidate"] is None
    assert all(
        candidate["acceptance"]["status"] == "failed"
        for candidate in comparison["candidates"]
    )
    result_hashes = {
        candidate["adapterId"]: candidate["artifacts"]["resultSha256"]
        for candidate in comparison["candidates"]
    }
    assert {
        run["adapterId"]: run["resultSha256"] for run in evidence["runs"]
    } == result_hashes
    assert evidence["materialization"]["recordCount"] == 1050
    assert evidence["materialization"]["variantCount"] == 6300
    assert evidence["source"]["totalBytes"] == sum(
        artifact["sizeBytes"] for artifact in benchmark["source"]["artifacts"]
    )
    for artifact in evidence["implementation"]["files"]:
        assert artifact["sha256"] == _sha256(EVALUATION_ROOT / artifact["path"])
