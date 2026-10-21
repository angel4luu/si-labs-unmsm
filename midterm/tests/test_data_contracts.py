from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pytest


TECHNIQUES = ("genetic", "fuzzy", "mlp")
HASH = re.compile(r"^[a-f0-9]{64}$")


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_metadata_and_configuration_contract(midterm_root: Path, technique: str) -> None:
    metadata_path = midterm_root / "data/metadata" / f"{technique}.json"
    config_path = midterm_root / "configs" / f"{technique}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert metadata["dataset_id"]
    assert metadata["technique"] == technique
    assert metadata["title"] and metadata["provider"]
    assert metadata["source_url"].startswith("https://")
    assert metadata["license"]
    date.fromisoformat(metadata["retrieved_on"])
    assert metadata["source_sha256"] is None or HASH.fullmatch(metadata["source_sha256"])
    snapshot = midterm_root.parent / metadata["snapshot_path"]
    assert snapshot.is_file()
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert HASH.fullmatch(metadata["snapshot_sha256"])
    assert digest == metadata["snapshot_sha256"]
    assert metadata["schema"] and metadata["transformations"] and metadata["limitations"]

    assert config["config_version"].startswith("1.0")
    assert config["technique"] == technique
    assert isinstance(config["seed"], int)
    assert config["dataset_metadata"] == f"midterm/data/metadata/{technique}.json"
    assert config["algorithm"] and config["presentation_case"] and config["pdf_mapping"]
    seen: set[str] = set()
    for item in config["pdf_mapping"]:
        assert item["pdf_element"] not in seen
        seen.add(item["pdf_element"])
        assert item["treatment"] in {"retained", "replaced", "unnecessary"}
        assert item["justification"]
        if item["treatment"] == "replaced":
            assert "replacement" in item and item["replacement"] is not None


def test_retail_snapshot_has_no_identifiers_and_expected_size(midterm_root: Path) -> None:
    import pandas as pd

    data = pd.read_csv(midterm_root / "data/derived/online_retail_85123a_daily.csv")
    assert len(data) == 305
    assert "CustomerID" not in data.columns
    assert list(data.columns) == ["date", "quantity_units", "capped_units", "demand_dozens"]


def test_power_and_student_snapshot_sizes(midterm_root: Path) -> None:
    import pandas as pd

    power = json.loads((midterm_root / "data/raw/nasa_power_lima_2024.json").read_text())
    assert len(power["properties"]["parameter"]["T2M"]) == 366
    students = pd.read_csv(midterm_root / "data/raw/student-mat.csv", sep=";")
    assert len(students) == 395
