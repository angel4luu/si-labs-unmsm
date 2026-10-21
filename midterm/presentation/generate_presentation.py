"""Genera el PPTX y el guion del examen parcial desde manifiestos canónicos."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


FONT = "Liberation Sans"
CODE_FONT = "Liberation Mono"
NAVY = RGBColor(0x0F, 0x17, 0x2A)
SLATE = RGBColor(0x33, 0x41, 0x55)
MUTED = RGBColor(0x64, 0x74, 0x8B)
PALE = RGBColor(0xF1, 0xF5, 0xF9)
CODE_BG = RGBColor(0xF8, 0xFA, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x0E, 0xA5, 0xE9)
VIOLET = RGBColor(0x7C, 0x3A, 0xED)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
RED = RGBColor(0xDC, 0x26, 0x26)
KEYWORD = RGBColor(0x6D, 0x28, 0xD9)

SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 7.5
CONTENT_LEFT = 0.65
CONTENT_WIDTH = 12.03
CODE_SIZE = 14
EXPECTED_LAYOUTS = ["overview", "contract", "process", "code"] * 3
PYTHON_KEYWORDS = re.compile(r"\b(def|return|for|in|if|else|class|lambda|float|int|list|dict|str)\b")


class PresentationConsistencyError(Exception):
    """Error legible de entrada, trazabilidad o generación."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PresentationConsistencyError(f"No se pudo leer JSON válido: {path}") from exc


def resolve(manifest: dict[str, Any], selector: str) -> Any:
    parts = selector.split(".")
    current: Any = manifest
    if parts[0] == "stages":
        try:
            current = next(stage for stage in manifest["stages"] if stage["stage_id"] == parts[1])
        except (KeyError, StopIteration) as exc:
            raise PresentationConsistencyError(f"No existe la etapa de {selector}") from exc
        parts = parts[2:]
    try:
        for part in parts:
            current = current[int(part)] if isinstance(current, list) else current[part]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PresentationConsistencyError(f"Selector inválido: {selector}") from exc
    return current


def _detail_value(manifest: dict[str, Any], detail: dict[str, Any]) -> str:
    """Devuelve el texto de un detalle: literal, selector con formato o plantilla con varios selectores."""
    if "text" in detail:
        return detail["text"]
    try:
        if "template" in detail:
            values = {key: resolve(manifest, selector) for key, selector in detail["values"].items()}
            return detail["template"].format(**values)
        return detail.get("format", "{}").format(resolve(manifest, detail["selector"]))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise PresentationConsistencyError(f"Formato inválido en {detail}") from exc


def _detail_text(manifest: dict[str, Any], detail: dict[str, Any]) -> str:
    value = _detail_value(manifest, detail)
    return f"{detail['label']}: {value}" if detail.get("label") else value


def _selector_items(slide: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from slide.get("result_selectors", [])
    groups = [slide.get("family", {}), slide.get("dataset", {}), *slide.get("blocks", []), *slide.get("steps", [])]
    for group in groups:
        yield from (detail for detail in group.get("details", []) if "text" not in detail)
    if "limitation_selector" in slide:
        yield {"selector": slide["limitation_selector"]}


def _normalize_code(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\\n", "\n"))


def check_snippet(code: dict[str, Any], project_root: Path) -> None:
    """Cada tramo contiguo del fragmento (separado por '…') debe existir literalmente en el código fuente."""
    path = project_root / code["source"]
    try:
        source = _normalize_code(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PresentationConsistencyError(f"No se pudo leer el código fuente: {path}") from exc
    chunk: list[str] = []
    for line in [*code["lines"], "…"]:
        if "…" not in line:
            chunk.append(line)
            continue
        normalized = _normalize_code("\n".join(chunk))
        if normalized and normalized not in source:
            raise PresentationConsistencyError(f"El fragmento no coincide con {code['source']}: {chunk[0].strip()}")
        chunk = []


def _set_run(run: Any, size: int, color: RGBColor, bold: bool = False, font: str = FONT) -> None:
    run.font.name = font
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def add_text(
    slide: Any,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: int = 18,
    color: RGBColor = SLATE,
    bold: bool = False,
    name: str = "body",
    align: PP_ALIGN | None = None,
    vertical: MSO_ANCHOR = MSO_ANCHOR.TOP,
    font: str = FONT,
) -> Any:
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    shape.name = name
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = vertical
    frame.margin_left = frame.margin_right = Inches(0.05)
    frame.margin_top = frame.margin_bottom = Inches(0.03)
    paragraph = frame.paragraphs[0]
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    _set_run(run, size, color, bold, font)
    return shape


def _accent(section: str) -> RGBColor:
    return {"genetic": VIOLET, "fuzzy": AMBER, "mlp": BLUE}.get(section, BLUE)


def add_heading(slide: Any, title: str, number: int, accent: RGBColor) -> None:
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SLIDE_WIDTH), Inches(0.16))
    band.fill.solid()
    band.fill.fore_color.rgb = accent
    band.line.fill.background()
    band.name = f"accent-{number}"
    add_text(slide, title, 0.65, 0.34, 11.95, 0.62, size=28, color=NAVY, bold=True, name=f"heading-{number}")


def add_intro(slide: Any, text: str, number: int) -> None:
    add_text(slide, text, CONTENT_LEFT, 1.02, CONTENT_WIDTH, 0.43, size=18, color=MUTED, name=f"intro-{number}")


def add_footer(slide: Any, deck_data: dict[str, Any], citations: list[str], number: int) -> None:
    parts = [f"{deck_data['author']} · {deck_data['course']}"]
    if citations:
        parts.append("Fuentes: " + " · ".join(citations))
    add_text(slide, "   |   ".join(parts), 0.65, 7.06, 11.65, 0.22, size=10, color=MUTED, name=f"citation-{number}")
    add_text(slide, str(number), 12.2, 7.06, 0.48, 0.22, size=10, color=MUTED, name=f"citation-page-{number}", align=PP_ALIGN.RIGHT)


def add_card(
    slide: Any,
    title: str,
    body: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    accent: RGBColor,
    name: str,
    fill: RGBColor = PALE,
) -> Any:
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.name = name
    card.adjustments[0] = 0.08
    card.fill.solid()
    card.fill.fore_color.rgb = fill
    card.line.color.rgb = accent
    card.line.width = Pt(1.2)
    frame = card.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.15)
    frame.margin_top = frame.margin_bottom = Inches(0.05)
    p1 = frame.paragraphs[0]
    r1 = p1.add_run()
    r1.text = title
    _set_run(r1, 18, NAVY, True)
    for line in body.split("\n"):
        paragraph = frame.add_paragraph()
        run = paragraph.add_run()
        run.text = line
        _set_run(run, 18, SLATE)
    return card


def _metric_card(slide: Any, label: str, value: str, left: float, top: float, width: float, height: float, accent: RGBColor, index: int) -> None:
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.name = f"body-metric-row-{index}"
    card.adjustments[0] = 0.12
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = accent
    frame = card.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.06)
    frame.margin_top = frame.margin_bottom = Inches(0.03)
    p1 = frame.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = label
    _set_run(r1, 18, SLATE, True)
    p2 = frame.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = value
    _set_run(r2, 18, NAVY)


def add_metric_row(slide: Any, metrics: list[tuple[str, str, float]], left: float, top: float, width: float, accent: RGBColor) -> None:
    if not metrics:
        return
    gap = 0.12
    total_weight = sum(weight for _, _, weight in metrics)
    available = width - gap * (len(metrics) - 1)
    x = left
    for index, (label, value, weight) in enumerate(metrics):
        card_width = available * weight / total_weight
        _metric_card(slide, label, value, x, top, card_width, 1.0, accent, index)
        x += card_width + gap


def add_metric_column(slide: Any, metrics: list[tuple[str, str, float]], left: float, top: float, width: float, height: float, accent: RGBColor) -> None:
    if not metrics:
        return
    gap = 0.1
    card_height = (height - gap * (len(metrics) - 1)) / len(metrics)
    for index, (label, value, _) in enumerate(metrics):
        _metric_card(slide, label, value, left, top + index * (card_height + gap), width, card_height, accent, index)


def add_picture_contained(slide: Any, path: Path, left: float, top: float, width: float, height: float) -> Any:
    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    frame.name = "body-figure-frame"
    frame.adjustments[0] = 0.05
    frame.fill.solid()
    frame.fill.fore_color.rgb = WHITE
    frame.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    with Image.open(path) as image:
        image_ratio = image.width / image.height
    box_ratio = width / height
    if image_ratio >= box_ratio:
        picture_width = width - 0.16
        picture_height = picture_width / image_ratio
    else:
        picture_height = height - 0.16
        picture_width = picture_height * image_ratio
    x = left + (width - picture_width) / 2
    y = top + (height - picture_height) / 2
    picture = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(picture_width), height=Inches(picture_height))
    picture.name = "body-figure"
    return picture


def _artifact_path(manifest: dict[str, Any], results_dir: Path, purpose: str) -> Path:
    for artifact in manifest["artifacts"]:
        if artifact["purpose"] == purpose:
            path = results_dir / manifest["technique"] / artifact["path"]
            if not path.is_file():
                raise PresentationConsistencyError(f"Falta el artefacto: {path}")
            return path
    raise PresentationConsistencyError(f"No existe artefacto con propósito: {purpose}")


def _metrics(slide_data: dict[str, Any], manifest: dict[str, Any]) -> list[tuple[str, str, float]]:
    return [(item["label"], _detail_value(manifest, item), item.get("weight", 1)) for item in slide_data.get("result_selectors", [])]


def _set_cell(cell: Any, text: str, *, fill: RGBColor, color: RGBColor, bold: bool = False) -> None:
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.margin_left = cell.margin_right = Inches(0.1)
    cell.margin_top = cell.margin_bottom = Inches(0.04)
    frame = cell.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    _set_run(run, 18, color, bold)


def add_table(slide: Any, rows: list[tuple[str, ...]], header: tuple[str, ...], widths: tuple[float, ...], top: float, row_height: float, accent: RGBColor, name: str) -> None:
    table_shape = slide.shapes.add_table(len(rows) + 1, len(header), Inches(CONTENT_LEFT), Inches(top), Inches(CONTENT_WIDTH), Inches(row_height * (len(rows) + 1)))
    table_shape.name = name
    table = table_shape.table
    for column, width in enumerate(widths):
        table.columns[column].width = Inches(width)
    for row in table.rows:
        row.height = Inches(row_height)
    for column, heading in enumerate(header):
        _set_cell(table.cell(0, column), heading, fill=accent, color=WHITE, bold=True)
    for row_index, values in enumerate(rows, 1):
        fill = WHITE if row_index % 2 else PALE
        for column, value in enumerate(values):
            _set_cell(table.cell(row_index, column), value, fill=fill, color=NAVY if column == 0 else SLATE, bold=column == 0)


def add_pipeline(slide: Any, stages: list[str], top: float, accent: RGBColor) -> None:
    width = CONTENT_WIDTH / len(stages)
    for index, label in enumerate(stages):
        shape_type = MSO_SHAPE.PENTAGON if index == 0 else MSO_SHAPE.CHEVRON
        shape = slide.shapes.add_shape(shape_type, Inches(CONTENT_LEFT + index * width), Inches(top), Inches(width + 0.05), Inches(0.72))
        shape.name = f"body-pipeline-{index}"
        shape.fill.solid()
        shape.fill.fore_color.rgb = accent if index % 2 == 0 else NAVY
        shape.line.fill.background()
        frame = shape.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        frame.margin_left = frame.margin_right = Inches(0.02)
        paragraph = frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = label
        _set_run(run, 18, WHITE, True)


def render_overview(slide: Any, slide_data: dict[str, Any], manifest: dict[str, Any], number: int) -> None:
    accent = _accent(slide_data["section"])
    add_heading(slide, slide_data["title"], number, accent)
    add_intro(slide, slide_data["intro_es"], number)
    for key, left in (("family", 0.65), ("dataset", 6.78)):
        group = slide_data[key]
        body = "\n".join(_detail_text(manifest, detail) for detail in group["details"])
        add_card(slide, group["title"], body, left, 1.55, 5.9, 1.58, accent=accent, name=f"body-overview-{key}")
    add_pipeline(slide, slide_data["pipeline"], 3.3, accent)
    rows = []
    for row in slide_data["exam_rows"]:
        mapping = manifest["pdf_mapping"][row["mapping"]]
        rows.append((row["label"], str(mapping["original_value"]), str(mapping["replacement"])))
    add_table(slide, rows, ("Elemento", "Examen (sintético)", "Esta versión (dato real)"), (2.6, 3.9, 5.53), 4.2, 0.44, accent, "body-exam-table")
    limitation = str(resolve(manifest, slide_data["limitation_selector"]))
    add_text(slide, f"Alcance: {limitation}", CONTENT_LEFT, 6.2, CONTENT_WIDTH, 0.7, size=18, color=RED, name="body-limitation")


def render_contract(slide: Any, slide_data: dict[str, Any], manifest: dict[str, Any], results_dir: Path, number: int) -> None:
    accent = _accent(slide_data["section"])
    add_heading(slide, slide_data["title"], number, accent)
    add_intro(slide, slide_data["intro_es"], number)
    blocks = slide_data["blocks"]
    artifact = slide_data.get("artifact_purpose")
    if artifact:
        for index, block in enumerate(blocks):
            body = "\n".join(_detail_text(manifest, detail) for detail in block["details"])
            add_card(slide, block["title"], body, 0.65, 1.52 + index * 1.06, 5.2, 1.0, accent=accent, name=f"body-contract-{block['role']}")
        image = _artifact_path(manifest, results_dir, artifact)
        add_picture_contained(slide, image, 6.05, 1.52, 6.63, 3.9)
        add_metric_row(slide, _metrics(slide_data, manifest), 6.05, 5.62, 6.63, accent)
        return

    positions = [
        (0.65, 1.6, 3.85, 1.65),
        (4.74, 1.6, 3.85, 1.65),
        (8.83, 1.6, 3.85, 1.65),
        (0.65, 3.48, 5.9, 1.65),
        (6.78, 3.48, 5.9, 1.65),
    ]
    for block, (left, top, width, height) in zip(blocks, positions, strict=True):
        body = "\n".join(_detail_text(manifest, detail) for detail in block["details"])
        add_card(slide, block["title"], body, left, top, width, height, accent=accent, name=f"body-contract-{block['role']}")
    add_metric_row(slide, _metrics(slide_data, manifest), 0.65, 5.55, 12.03, accent)


def render_process(slide: Any, slide_data: dict[str, Any], manifest: dict[str, Any], results_dir: Path, number: int) -> None:
    accent = _accent(slide_data["section"])
    add_heading(slide, slide_data["title"], number, accent)
    add_intro(slide, slide_data["intro_es"], number)
    rows = [(step["title"], " · ".join(_detail_text(manifest, detail) for detail in step["details"])) for step in slide_data["steps"]]
    add_table(slide, rows, ("Paso", "Cálculo con los datos"), (4.35, 7.68), 1.52, 0.44, accent, "body-process-table")
    image = _artifact_path(manifest, results_dir, slide_data["artifact_purpose"])
    add_picture_contained(slide, image, 0.65, 4.3, 6.9, 2.62)
    add_metric_column(slide, _metrics(slide_data, manifest), 7.75, 4.3, 4.93, 2.62, accent)


def _add_code_line(paragraph: Any, line: str) -> None:
    position = 0
    for match in PYTHON_KEYWORDS.finditer(line):
        if match.start() > position:
            _set_run(paragraph.add_run(), CODE_SIZE, NAVY, font=CODE_FONT)
            paragraph.runs[-1].text = line[position:match.start()]
        _set_run(paragraph.add_run(), CODE_SIZE, KEYWORD, True, CODE_FONT)
        paragraph.runs[-1].text = match.group(0)
        position = match.end()
    if position < len(line) or not line:
        _set_run(paragraph.add_run(), CODE_SIZE, MUTED if "…" in line else NAVY, font=CODE_FONT)
        paragraph.runs[-1].text = line[position:] or " "


def render_code(slide: Any, slide_data: dict[str, Any], number: int) -> None:
    accent = _accent(slide_data["section"])
    add_heading(slide, slide_data["title"], number, accent)
    add_intro(slide, slide_data["intro_es"], number)
    code = slide_data["code"]
    add_text(slide, code["caption"], CONTENT_LEFT, 1.48, CONTENT_WIDTH, 0.34, size=CODE_SIZE, color=MUTED, name="code-caption", font=CODE_FONT)
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(CONTENT_LEFT), Inches(1.88), Inches(8.3), Inches(4.87))
    box.name = "code-block"
    box.fill.solid()
    box.fill.fore_color.rgb = CODE_BG
    box.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.12)
    frame.margin_top = frame.margin_bottom = Inches(0.08)
    markers = code.get("markers", {})
    for index, line in enumerate(code["lines"]):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        marker = markers.get(str(index))
        gutter = paragraph.add_run()
        gutter.text = f"{marker} " if marker else "  "
        _set_run(gutter, CODE_SIZE, accent, True, CODE_FONT)
        _add_code_line(paragraph, line)
    gap = 0.08
    callouts = slide_data["callouts"]
    height = min(0.9, (4.87 - gap * (len(callouts) - 1)) / len(callouts))
    for index, callout in enumerate(callouts):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.13), Inches(1.88 + index * (height + gap)), Inches(3.55), Inches(height))
        card.name = f"body-callout-{index}"
        card.adjustments[0] = 0.12
        card.fill.solid()
        card.fill.fore_color.rgb = PALE
        card.line.color.rgb = accent
        text = card.text_frame
        text.clear()
        text.word_wrap = True
        text.vertical_anchor = MSO_ANCHOR.MIDDLE
        text.margin_left = text.margin_right = Inches(0.1)
        text.margin_top = text.margin_bottom = Inches(0.03)
        paragraph = text.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT
        mark = paragraph.add_run()
        mark.text = f"{callout['marker']}  "
        _set_run(mark, 18, accent, True)
        body = paragraph.add_run()
        body.text = callout["text"]
        _set_run(body, 18, SLATE)


def validate(content: dict[str, Any], manifests: dict[str, dict[str, Any]], results_dir: Path, strict: bool, project_root: Path) -> None:
    slides = content.get("slides", [])
    if content.get("deck", {}).get("aspect_ratio") != "16:9" or len(slides) != len(EXPECTED_LAYOUTS):
        raise PresentationConsistencyError(f"El contenido debe definir exactamente {len(EXPECTED_LAYOUTS)} diapositivas 16:9")
    if [slide.get("layout") for slide in slides] != EXPECTED_LAYOUTS:
        raise PresentationConsistencyError("La secuencia de layouts no cumple el contrato overview-contract-process-code")
    sections = content.get("sections", [])
    if [section.get("id") for section in sections] != ["genetic", "fuzzy", "mlp"]:
        raise PresentationConsistencyError("Las secciones deben estar en orden genetic, fuzzy, mlp")
    if any(section.get("duration_seconds") != 180 for section in sections):
        raise PresentationConsistencyError("Cada técnica debe durar exactamente 180 segundos")
    for section in sections:
        planned = sum(slide["duration_seconds"] for slide in slides if slide.get("section") == section["id"])
        if planned != section["duration_seconds"]:
            raise PresentationConsistencyError(f"Las diapositivas de {section['id']} suman {planned} s, no 180 s")
    known_citations = {item["label"] for manifest in manifests.values() for item in manifest.get("citations", [])}
    ids: set[str] = set()
    for slide in slides:
        if slide.get("id") in ids or not slide.get("narration_es") or not slide.get("title") or not slide.get("intro_es"):
            raise PresentationConsistencyError("ID, título, introducción o narración faltante")
        ids.add(slide["id"])
        if re.search(r"\bTODO\b|\bPLACEHOLDER\b|\{\{", json.dumps(slide)):
            raise PresentationConsistencyError(f"Marcador sin resolver en {slide['id']}")
        manifest = manifests.get(slide.get("section"))
        if not manifest:
            raise PresentationConsistencyError(f"Sección desconocida en {slide['id']}")
        if not slide.get("citations") or not manifest.get("pdf_mapping"):
            raise PresentationConsistencyError(f"Falta cita o mapeo PDF en {slide['id']}")
        for item in _selector_items(slide):
            _detail_value(manifest, item)
        layout = slide["layout"]
        if layout == "overview":
            if not slide.get("pipeline") or not slide.get("exam_rows"):
                raise PresentationConsistencyError(f"Panorama incompleto en {slide['id']}")
            for row in slide["exam_rows"]:
                if not 0 <= row["mapping"] < len(manifest["pdf_mapping"]):
                    raise PresentationConsistencyError(f"Mapeo PDF inexistente en {slide['id']}")
        elif layout == "contract":
            roles = [block.get("role") for block in slide.get("blocks", [])]
            if roles != ["problem", "inputs", "model", "parameters", "output"]:
                raise PresentationConsistencyError(f"Contrato incompleto en {slide['id']}")
        elif layout == "process" and len(slide.get("steps", [])) != 5:
            raise PresentationConsistencyError(f"El proceso debe tener cinco fases en {slide['id']}")
        elif layout == "code":
            if not slide.get("callouts") or not slide.get("code", {}).get("lines"):
                raise PresentationConsistencyError(f"Fragmento de código incompleto en {slide['id']}")
            markers = set(slide["code"].get("markers", {}).values())
            if markers != {callout["marker"] for callout in slide["callouts"]}:
                raise PresentationConsistencyError(f"Marcadores y notas no coinciden en {slide['id']}")
            check_snippet(slide["code"], project_root)
        if slide.get("artifact_purpose"):
            _artifact_path(manifest, results_dir, slide["artifact_purpose"])
        for citation in slide.get("citations", []):
            if strict and citation not in known_citations:
                raise PresentationConsistencyError(f"Cita desconocida: {citation}")


def generate(results_dir: Path, content_path: Path, output: Path, script_output: Path, strict: bool = False) -> None:
    content = load_json(content_path)
    project_root = content_path.resolve().parent.parent
    manifests = {name: load_json(results_dir / name / "result.json") for name in ("genetic", "fuzzy", "mlp")}
    validate(content, manifests, results_dir, strict, project_root)
    deck_data = content["deck"]
    deck = Presentation()
    deck.slide_width = 12_192_000
    deck.slide_height = 6_858_000
    blank = deck.slide_layouts[6]
    total = sum(section["duration_seconds"] for section in content["sections"])
    script_parts = [
        "# Guion — Examen parcial de Software Inteligente",
        "",
        f"Duración planificada: {total // 60}:{total % 60:02d} (3:00 por técnica, sin portada ni cierre).",
        "",
    ]
    for number, slide_data in enumerate(content["slides"], 1):
        slide = deck.slides.add_slide(blank)
        background = slide.background.fill
        background.solid()
        background.fore_color.rgb = WHITE
        layout = slide_data["layout"]
        manifest = manifests[slide_data["section"]]
        if layout == "overview":
            render_overview(slide, slide_data, manifest, number)
        elif layout == "contract":
            render_contract(slide, slide_data, manifest, results_dir, number)
        elif layout == "process":
            render_process(slide, slide_data, manifest, results_dir, number)
        elif layout == "code":
            render_code(slide, slide_data, number)
        else:
            raise PresentationConsistencyError(f"Layout no soportado en {slide_data['id']}")
        add_footer(slide, deck_data, slide_data["citations"], number)
        if number == len(content["slides"]):
            add_text(slide, "Declaración de IA: " + deck_data["ai_disclosure"], 0.65, 6.8, 12.03, 0.24, size=10, color=MUTED, name="citation-ai")
        notes = slide.notes_slide.notes_text_frame
        notes.clear()
        notes.text = slide_data["narration_es"]
        script_parts.extend([
            f"## {number}. {slide_data['title']} [{slide_data['id']}]",
            "",
            f"**Tiempo:** {slide_data['duration_seconds']} s",
            "",
            slide_data["narration_es"],
            "",
        ])
    script_parts.extend(["## Declaración de asistencia de IA", "", deck_data["ai_disclosure"], ""])
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        script_output.parent.mkdir(parents=True, exist_ok=True)
        deck.save(output)
        script_output.write_text("\n".join(script_parts), encoding="utf-8")
    except OSError as exc:
        raise PresentationConsistencyError("No se pudieron escribir el PPTX o el guion") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script-output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        generate(args.results_dir, args.content, args.output, args.script_output, args.strict)
    except PresentationConsistencyError as exc:
        print(f"Error de consistencia: {exc}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
