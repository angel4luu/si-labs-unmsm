from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from fuzzy_logic import FuzzyVariable, MamdaniController, MamdaniRule, MembershipFunction, PowerDataset, run_exercise


def test_memberships_and_centroid(midterm_root: Path) -> None:
    dataset = PowerDataset.load(
        midterm_root / "data/raw/nasa_power_lima_2024.json",
        midterm_root / "data/metadata/fuzzy.json",
    )
    assert dataset.values["20240514"] == pytest.approx(19.4)
    temperature = FuzzyVariable({
        "low": MembershipFunction("trapezoid", [16.09, 16.09, 17.86, 18.685]),
        "medium": MembershipFunction("triangle", [17.86, 18.685, 20.0875]),
        "high": MembershipFunction("trapezoid", [18.685, 20.0875, 22.39, 22.39]),
    })
    fan = FuzzyVariable({
        "slow": MembershipFunction("triangle", [0, 20, 40]),
        "moderate": MembershipFunction("triangle", [30, 50, 70]),
        "fast": MembershipFunction("triangle", [60, 80, 100]),
    })
    controller = MamdaniController(temperature, fan, [MamdaniRule("low", "slow"), MamdaniRule("medium", "moderate"), MamdaniRule("high", "fast")], 0.01)
    trace = controller.infer(19.4)
    assert all(0 <= value <= 1 for value in trace.memberships.values())
    assert trace.memberships == pytest.approx({"low": 0, "medium": 0.4901960784, "high": 0.5098039216})
    assert trace.centroid_denominator > 0
    assert trace.crisp_output == pytest.approx(65.20463096)


def test_run_is_fast_and_has_oop_structure(midterm_root: Path, tmp_path: Path) -> None:
    started = time.perf_counter()
    manifest = run_exercise(midterm_root / "data/raw/nasa_power_lima_2024.json", midterm_root / "data/metadata/fuzzy.json", midterm_root / "configs/fuzzy.json", tmp_path)
    assert time.perf_counter() - started < 10
    assert [s["stage_id"] for s in manifest["stages"]] == ["preprocess", "memberships", "fuzzification", "rule-evaluation", "implication", "aggregation", "centroid", "interpretation"]
    tree = ast.parse((midterm_root / "src/fuzzy_logic.py").read_text())
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert {"PowerDataset", "MembershipFunction", "FuzzyVariable", "MamdaniRule", "MamdaniController", "InferenceTrace"} <= classes
