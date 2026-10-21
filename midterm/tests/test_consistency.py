from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema


EXPECTED = {
    "genetic": ["preprocess", "encode-decode", "initial-fitness", "selection", "crossover", "mutation", "replacement", "comparison"],
    "fuzzy": ["preprocess", "memberships", "fuzzification", "rule-evaluation", "implication", "aggregation", "centroid", "interpretation"],
    "mlp": ["preprocess-split", "sample-normalization", "initialization", "forward", "loss", "backward", "update", "post-update-forward"],
}


def test_all_manifests_are_traceable(midterm_root: Path, manifest_schema: dict) -> None:
    for technique, expected in EXPECTED.items():
        result = json.loads((midterm_root / "results" / technique / "result.json").read_text())
        jsonschema.validate(result, manifest_schema)
        assert [stage["stage_id"] for stage in result["stages"]] == expected
        for stage in result["stages"]:
            assert stage["title_es"] and stage["operation"] and stage["code_objects"]
            assert isinstance(stage["inputs"], dict) and isinstance(stage["outputs"], dict)
        assert result["pdf_mapping"] and result["citations"] and result["limitations"]


def test_content_selectors_and_code_objects_resolve(midterm_root: Path) -> None:
    sys.path.insert(0, str(midterm_root / "presentation"))
    from generate_presentation import _detail_value, _selector_items

    content = json.loads((midterm_root / "presentation/content.json").read_text())
    manifests = {name: json.loads((midterm_root / "results" / name / "result.json").read_text()) for name in EXPECTED}
    for slide in content["slides"]:
        manifest = manifests[slide["section"]]
        assert slide["citations"] and slide["narration_es"]
        for item in _selector_items(slide):
            assert _detail_value(manifest, item)
        for responsibility in slide.get("code_objects", []):
            assert any(responsibility in stage["code_objects"] for stage in manifest["stages"])


def test_script_matches_pptx_notes(midterm_root: Path) -> None:
    from pptx import Presentation

    content = json.loads((midterm_root / "presentation/content.json").read_text())
    deck = Presentation(midterm_root / "artifacts/midterm.pptx")
    notes = [slide.notes_slide.notes_text_frame.text.strip() for slide in deck.slides]
    expected = [slide["narration_es"].strip() for slide in content["slides"]]
    assert notes == expected
    script = (midterm_root / "artifacts/guion.md").read_text(encoding="utf-8")
    assert all(text in script for text in expected)
