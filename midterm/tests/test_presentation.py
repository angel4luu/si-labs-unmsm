from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation


def test_content_structure_and_timing(midterm_root: Path) -> None:
    content = json.loads((midterm_root / "presentation/content.json").read_text())
    slides = content["slides"]
    assert [section["id"] for section in content["sections"]] == ["genetic", "fuzzy", "mlp"]
    assert len(slides) == 12
    assert len({slide["id"] for slide in slides}) == 12
    assert 510 <= sum(section["duration_seconds"] for section in content["sections"]) <= 570
    for section in content["sections"]:
        assert 160 <= section["duration_seconds"] <= 200
        assert section["theory_seconds"] >= 60
        assert section["code_seconds"] >= 60
        section_slides = [slide for slide in slides if slide["section"] == section["id"]]
        assert sum(slide["duration_seconds"] for slide in section_slides) == section["duration_seconds"]
        assert next(slide for slide in section_slides if slide["layout"] == "code")["duration_seconds"] >= 60

    assert [slide["layout"] for slide in slides] == ["overview", "contract", "process", "code"] * 3

    for slide in slides:
        assert slide["intro_es"].strip()
        assert len(slide["intro_es"].split()) <= 24

    for slide in slides[0::4]:
        assert slide["family"]["details"] and slide["dataset"]["details"]
        assert slide["pipeline"] and slide["exam_rows"]

    for slide in slides[1::4]:
        assert [block["role"] for block in slide["blocks"]] == [
            "problem",
            "inputs",
            "model",
            "parameters",
            "output",
        ]

    for slide in slides[2::4]:
        assert len(slide["steps"]) == 5
        assert slide["result_selectors"]

    for slide in slides[3::4]:
        assert 8 <= len(slide["code"]["lines"]) <= 18
        assert set(slide["code"]["markers"].values()) == {callout["marker"] for callout in slide["callouts"]}


def test_code_snippets_match_source(midterm_root: Path) -> None:
    sys.path.insert(0, str(midterm_root / "presentation"))
    from generate_presentation import check_snippet

    content = json.loads((midterm_root / "presentation/content.json").read_text())
    for slide in content["slides"]:
        if slide["layout"] == "code":
            check_snippet(slide["code"], midterm_root)


def test_authored_slide_copy_is_concise(midterm_root: Path) -> None:
    content = json.loads((midterm_root / "presentation/content.json").read_text())

    def authored_words(value: object) -> list[str]:
        if isinstance(value, str):
            return [word for word in value.split() if any(char.isalpha() for char in word)]
        if isinstance(value, list):
            return [word for item in value for word in authored_words(item)]
        if isinstance(value, dict):
            excluded = {"narration_es", "citations", "selector", "format", "code_objects", "code", "values"}
            return [word for key, item in value.items() if key not in excluded for word in authored_words(item)]
        return []

    for slide in content["slides"]:
        assert len(authored_words(slide)) <= 100, slide["id"]


def test_generated_deck_contract(midterm_root: Path) -> None:
    path = midterm_root / "artifacts/midterm.pptx"
    deck = Presentation(path)
    assert len(deck.slides) == 12
    assert deck.slide_width / deck.slide_height == 16 / 9
    for slide in deck.slides:
        assert slide.notes_slide.notes_text_frame.text.strip()
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= deck.slide_width
            assert shape.top + shape.height <= deck.slide_height
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if not run.text.strip():
                        continue
                    name = getattr(shape, "name", "")
                    if name.startswith("code-"):
                        assert run.font.name == "Liberation Mono"
                        minimum = 14
                    else:
                        assert run.font.name == "Liberation Sans"
                        minimum = 10 if name.startswith("citation-") else 18
                    if name.startswith("heading-"):
                        minimum = 28
                    assert run.font.size and run.font.size.pt >= minimum

    for index in range(len(deck.slides)):
        names = {shape.name for shape in deck.slides[index].shapes}
        assert any(name.startswith("intro-") for name in names)


def test_rendered_outputs(midterm_root: Path) -> None:
    assert shutil.which("libreoffice") and shutil.which("pdfinfo") and shutil.which("pdftoppm")
    for command in (["libreoffice", "--version"], ["pdfinfo", "-v"], ["pdftoppm", "-v"]):
        version = subprocess.run(command, text=True, capture_output=True, check=True)
        assert (version.stdout + version.stderr).strip()
    pdf = midterm_root / "artifacts/midterm.pdf"
    result = subprocess.run(["pdfinfo", str(pdf)], text=True, capture_output=True, check=True)
    pages = int(next(line.split(":", 1)[1] for line in result.stdout.splitlines() if line.startswith("Pages:")))
    assert pages == 12
    rendered = sorted((midterm_root / "artifacts/rendered").glob("slide-*.png"))
    assert len(rendered) == 12
    for path in rendered:
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            assert image.width > 1000 and image.height > 500
