from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient


EXPECTED = {"genetic_algorithm.ipynb", "fuzzy_logic.ipynb", "mlp.ipynb"}
FORBIDDEN_TEXT = ("%run", "pip install", "uv add", "requests.", "urllib", "input(", "midterm/results")


def _tree_hash(path: Path) -> dict[str, str]:
    return {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in path.rglob("*") if file.is_file()}


def test_exact_notebook_set_and_thin_policy(midterm_root: Path) -> None:
    paths = list((midterm_root / "notebooks").glob("*.ipynb"))
    assert {path.name for path in paths} == EXPECTED
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown").lower()
        for heading in ("procedencia", "teoría", "responsabilidades oop", "conclusión", "limitaciones"):
            assert heading in markdown
        code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        assert "run_exercise" in code and "Path(\"..\").resolve()" in code
        assert not any(fragment in code for fragment in FORBIDDEN_TEXT)
        tree = ast.parse(code)
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))


def test_clean_kernel_execution_and_internal_parity(midterm_root: Path) -> None:
    before = _tree_hash(midterm_root / "results")
    for path in sorted((midterm_root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        client = NotebookClient(
            notebook,
            timeout=60,
            kernel_name="python3",
            allow_errors=False,
            store_widget_state=False,
            resources={"metadata": {"path": str(path.parent)}},
        )
        executed = client.execute()
        assert all(cell.get("execution_count") is not None for cell in executed.cells if cell.cell_type == "code")
        assert any("Paridad verificada" in str(output) for cell in executed.cells for output in cell.get("outputs", []))
    assert _tree_hash(midterm_root / "results") == before
