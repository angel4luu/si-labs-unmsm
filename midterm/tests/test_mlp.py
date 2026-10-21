from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from mlp import BinaryCrossEntropy, GradeNormalizer, MLP221, StudentDataset, run_exercise


def test_split_normalization_and_update(midterm_root: Path) -> None:
    dataset = StudentDataset.load(midterm_root / "data/raw/student-mat.csv", midterm_root / "data/metadata/mlp.json")
    train, test = dataset.stratified_split(0.2, 42)
    assert (len(train), len(test)) == (316, 79)
    assert 3 in train and 3 not in test
    x = GradeNormalizer().transform(15, 14)
    assert x.tolist() == pytest.approx([0.75, 0.70])
    model = MLP221.from_parameters(
        [[0.10, 0.20], [-0.10, 0.10]], [0.10, -0.10], [0.30, -0.20], 0.05,
    )
    trace = model.train_one(x, 1.0, 0.2)
    assert trace.prediction_before == pytest.approx(0.532124874)
    assert trace.loss_before == pytest.approx(0.630877092)
    assert trace.prediction_after == pytest.approx(0.568593022)
    assert trace.loss_after == pytest.approx(0.564590352)
    assert model.hidden.weights.shape == (2, 2)
    assert model.output.weights.shape == (1, 2)


def test_run_is_fast_and_has_oop_structure(midterm_root: Path, tmp_path: Path) -> None:
    started = time.perf_counter()
    manifest = run_exercise(midterm_root / "data/raw/student-mat.csv", midterm_root / "data/metadata/mlp.json", midterm_root / "configs/mlp.json", tmp_path)
    assert time.perf_counter() - started < 10
    assert [s["stage_id"] for s in manifest["stages"]] == ["preprocess-split", "sample-normalization", "initialization", "forward", "loss", "backward", "update", "post-update-forward"]
    tree = ast.parse((midterm_root / "src/mlp.py").read_text())
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert {"StudentDataset", "GradeNormalizer", "DenseLayer", "Sigmoid", "BinaryCrossEntropy", "MLP221", "TrainingTrace"} <= classes
