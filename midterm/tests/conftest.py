from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path

import pytest


MIDTERM_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(MIDTERM_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def midterm_root(repo_root: Path) -> Path:
    return repo_root / "midterm"


@pytest.fixture(scope="session")
def manifest_schema(repo_root: Path) -> dict:
    path = repo_root / "specs/001-midterm-presentation/contracts/result-manifest.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    path = tmp_path / "results"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("La red está deshabilitada durante las pruebas")

    monkeypatch.setattr(socket, "create_connection", denied)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


