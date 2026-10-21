from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from genetic_algorithm import DemandProblem, GeneticAlgorithm, Individual, RetailDataset, run_exercise


def test_encoding_and_bounds() -> None:
    assert Individual.from_value(0).bits == "00000"
    assert Individual.from_value(31).bits == "11111"
    assert Individual.from_value(7).decode() == 7
    with pytest.raises(ValueError):
        Individual.from_value(32)


def test_first_generation_fixture(midterm_root: Path) -> None:
    dataset = RetailDataset.load(
        midterm_root / "data/derived/online_retail_85123a_daily.csv",
        midterm_root / "data/metadata/genetic.json",
    )
    problem = DemandProblem(dataset.demands, 0, 31)
    assert problem.fitness(4) == pytest.approx(0.1502463054)
    ga = GeneticAlgorithm(problem, [4, 11, 14, 25], [(4, 25), (11, 14)], 3, 0.05, 2, 1)
    trace = ga.run_generation()
    assert trace.selected == [4, 11]
    assert trace.crossover_children == [7, 8]
    assert trace.mutated_children == [7, 9]
    assert trace.new_population == [4, 11, 7, 9]
    assert trace.improvement_percent == pytest.approx(9.20, abs=0.01)


def test_run_is_fast_and_has_oop_structure(midterm_root: Path, tmp_path: Path) -> None:
    started = time.perf_counter()
    manifest = run_exercise(
        midterm_root / "data/derived/online_retail_85123a_daily.csv",
        midterm_root / "data/metadata/genetic.json",
        midterm_root / "configs/genetic.json",
        tmp_path,
    )
    assert time.perf_counter() - started < 10
    assert [s["stage_id"] for s in manifest["stages"]] == [
        "preprocess", "encode-decode", "initial-fitness", "selection", "crossover",
        "mutation", "replacement", "comparison",
    ]
    tree = ast.parse((midterm_root / "src/genetic_algorithm.py").read_text())
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert {"RetailDataset", "DemandProblem", "Individual", "GeneticAlgorithm", "GenerationTrace"} <= classes
