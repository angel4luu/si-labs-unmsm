from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest


CASES = {
    "genetic": ("genetic_algorithm.py", "data/derived/online_retail_85123a_daily.csv", {"result.json", "generation_comparison.png", "generation_table.csv"}),
    "fuzzy": ("fuzzy_logic.py", "data/raw/nasa_power_lima_2024.json", {"result.json", "memberships.png", "aggregation.png"}),
    "mlp": ("mlp.py", "data/raw/student-mat.csv", {"result.json", "network_update.png", "parameter_update.csv"}),
}


def validate_artifacts(manifest: dict, output: Path) -> None:
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        assert path.is_file() and path.stat().st_size > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


@pytest.mark.parametrize("technique", CASES)
def test_canonical_manifests_follow_schema(midterm_root: Path, manifest_schema: dict, technique: str) -> None:
    output = midterm_root / "results" / technique
    manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
    jsonschema.validate(manifest, manifest_schema, format_checker=jsonschema.FormatChecker())
    assert manifest["schema_version"] == "1.0"
    assert manifest["technique"] == technique
    validate_artifacts(manifest, output)


@pytest.mark.parametrize("technique", CASES)
def test_cli_is_deterministic_and_records_seed(midterm_root: Path, tmp_path: Path, technique: str) -> None:
    script, data, outputs = CASES[technique]
    out = tmp_path / technique
    command = [
        sys.executable, str(midterm_root / "src" / script),
        "--data", str(midterm_root / data),
        "--metadata", str(midterm_root / "data/metadata" / f"{technique}.json"),
        "--config", str(midterm_root / "configs" / f"{technique}.json"),
        "--output-dir", str(out), "--seed", "123",
    ]
    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    manifest_a = json.loads((out / "result.json").read_text())
    assert manifest_a["execution"]["seed"] == 123
    assert {p.name for p in out.iterdir()} == outputs
    second = subprocess.run(command, text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    assert json.loads((out / "result.json").read_text()) == manifest_a


@pytest.mark.parametrize("_technique,case", CASES.items())
def test_modules_are_import_safe(midterm_root: Path, tmp_path: Path, _technique: str, case: tuple) -> None:
    script = midterm_root / "src" / case[0]
    spec = importlib.util.spec_from_file_location(f"safe_{script.stem}", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    before = set(tmp_path.iterdir())
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    assert set(tmp_path.iterdir()) == before
    assert callable(module.run_exercise)


@pytest.mark.parametrize("technique", CASES)
def test_missing_data_returns_status_three(midterm_root: Path, tmp_path: Path, technique: str) -> None:
    script, _, _ = CASES[technique]
    result = subprocess.run([
        sys.executable, str(midterm_root / "src" / script), "--data", str(tmp_path / "missing"),
        "--metadata", str(midterm_root / "data/metadata" / f"{technique}.json"),
        "--config", str(midterm_root / "configs" / f"{technique}.json"),
        "--output-dir", str(tmp_path / "out"),
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 3
    assert result.stderr.strip() and "Traceback" not in result.stderr
