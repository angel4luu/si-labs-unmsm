"""Algoritmo genético didáctico sobre demanda diaria real de UCI Online Retail."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataValidationError(Exception):
    pass


class ConfigurationError(Exception):
    pass


class ComputationError(Exception):
    pass


class OutputError(Exception):
    pass


class ConsistencyError(Exception):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, error_type: type[Exception]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise error_type(f"No se pudo leer JSON válido: {path}") from exc


@dataclass(frozen=True)
class RetailDataset:
    demands: np.ndarray
    frame: pd.DataFrame
    metadata: dict[str, Any]

    @classmethod
    def load(cls, data_path: Path, metadata_path: Path) -> "RetailDataset":
        if not data_path.is_file() or not metadata_path.is_file():
            raise DataValidationError("Falta la instantánea o su metadata")
        metadata = _read_json(metadata_path, DataValidationError)
        if metadata.get("technique") != "genetic":
            raise DataValidationError("La metadata no corresponde al algoritmo genético")
        if _sha256(data_path) != metadata.get("snapshot_sha256"):
            raise DataValidationError("El hash de la instantánea Retail no coincide")
        try:
            frame = pd.read_csv(data_path)
        except Exception as exc:
            raise DataValidationError("No se pudo leer la instantánea Retail") from exc
        required = {"date", "quantity_units", "capped_units", "demand_dozens"}
        if not required.issubset(frame.columns) or "CustomerID" in frame.columns or len(frame) != 305:
            raise DataValidationError("El esquema Retail no cumple el contrato")
        demands = frame["demand_dozens"].to_numpy(dtype=float)
        if not np.isfinite(demands).all() or (demands < 0).any():
            raise DataValidationError("La demanda contiene valores inválidos")
        return cls(demands=demands, frame=frame, metadata=metadata)


@dataclass(frozen=True)
class DemandProblem:
    demands: np.ndarray
    lower_bound: int
    upper_bound: int

    def mae(self, value: int) -> float:
        self.validate(value)
        return float(np.mean(np.abs(self.demands - value)))

    def fitness(self, value: int) -> float:
        return 1.0 / (1.0 + self.mae(value))

    def validate(self, value: int) -> None:
        if not self.lower_bound <= value <= self.upper_bound:
            raise ValueError(f"Valor fuera de [{self.lower_bound}, {self.upper_bound}]")


@dataclass(frozen=True)
class Individual:
    bits: str

    @classmethod
    def from_value(cls, value: int, width: int = 5) -> "Individual":
        if not 0 <= value < 2**width:
            raise ValueError("El entero no cabe en el cromosoma")
        return cls(format(value, f"0{width}b"))

    def decode(self) -> int:
        return int(self.bits, 2)


@dataclass(frozen=True)
class GenerationTrace:
    initial_population: list[int]
    initial_fitness: list[float]
    tournament_pairs: list[list[int]]
    selected: list[int]
    crossover_children: list[int]
    mutation_masks: list[str]
    mutated_children: list[int]
    elites: list[int]
    new_population: list[int]
    new_fitness: list[float]
    initial_best: int
    final_best: int
    improvement_percent: float


class GeneticAlgorithm:
    def __init__(
        self,
        problem: DemandProblem,
        initial_population: list[int],
        tournament_pairs: list[tuple[int, int]] | list[list[int]],
        crossover_point: int,
        mutation_probability: float,
        elite_count: int,
        seed: int,
    ) -> None:
        if len(initial_population) != 4 or len(tournament_pairs) != 2:
            raise ConfigurationError("La demostración requiere cuatro individuos y dos torneos")
        if not 0 <= mutation_probability <= 1 or not 0 < crossover_point < 5:
            raise ConfigurationError("Parámetros genéticos fuera de rango")
        for value in initial_population:
            problem.validate(value)
        self.problem = problem
        self.initial_population = list(initial_population)
        self.tournament_pairs = [list(pair) for pair in tournament_pairs]
        self.crossover_point = crossover_point
        self.mutation_probability = mutation_probability
        self.elite_count = elite_count
        self.seed = seed

    def _winner(self, pair: list[int]) -> int:
        return max(pair, key=lambda value: (self.problem.fitness(value), -value))

    def run_generation(self) -> GenerationTrace:
        initial_fitness = [self.problem.fitness(value) for value in self.initial_population]
        selected = [self._winner(pair) for pair in self.tournament_pairs]
        left, right = (Individual.from_value(value).bits for value in selected)
        crossed = [
            Individual(left[: self.crossover_point] + right[self.crossover_point :]).decode(),
            Individual(right[: self.crossover_point] + left[self.crossover_point :]).decode(),
        ]
        rng = np.random.default_rng(self.seed)
        mutated: list[int] = []
        masks: list[str] = []
        for value in crossed:
            bits = list(Individual.from_value(value).bits)
            mask = rng.random(len(bits)) < self.mutation_probability
            for index, flip in enumerate(mask):
                if flip:
                    bits[index] = "1" if bits[index] == "0" else "0"
            masks.append("".join("1" if flip else "0" for flip in mask))
            mutated.append(Individual("".join(bits)).decode())
        ranked = sorted(self.initial_population, key=self.problem.fitness, reverse=True)
        elites = ranked[: self.elite_count]
        new_population = elites + mutated
        new_fitness = [self.problem.fitness(value) for value in new_population]
        initial_best = max(self.initial_population, key=self.problem.fitness)
        final_best = max(new_population, key=self.problem.fitness)
        improvement = 100 * (self.problem.fitness(final_best) / self.problem.fitness(initial_best) - 1)
        return GenerationTrace(
            self.initial_population,
            initial_fitness,
            self.tournament_pairs,
            selected,
            crossed,
            masks,
            mutated,
            elites,
            new_population,
            new_fitness,
            initial_best,
            final_best,
            improvement,
        )


def _artifact(path: Path, media_type: str, purpose: str) -> dict[str, str]:
    return {"path": path.name, "media_type": media_type, "sha256": _sha256(path), "purpose": purpose}


def _write_outputs(output_dir: Path, problem: DemandProblem, trace: GenerationTrace) -> list[dict[str, str]]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        table = output_dir / "generation_table.csv"
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["generation", "value_dozens", "chromosome", "mae", "fitness"])
            for generation, population in (("initial", trace.initial_population), ("new", trace.new_population)):
                for value in population:
                    writer.writerow([generation, value, Individual.from_value(value).bits, f"{problem.mae(value):.9f}", f"{problem.fitness(value):.9f}"])
        figure = output_dir / "generation_comparison.png"
        fig, ax = plt.subplots(figsize=(10, 5.6))
        positions = np.arange(4)
        ax.bar(positions - 0.18, trace.initial_fitness, 0.36, label="Inicial", color="#64748b")
        ax.bar(positions + 0.18, trace.new_fitness, 0.36, label="Nueva", color="#0ea5e9")
        ax.set_xticks(positions, [str(i + 1) for i in positions])
        ax.set_ylabel("Aptitud 1/(1+MAE)", fontsize=18)
        ax.set_xlabel("Posición en la población", fontsize=18)
        ax.set_title("Primera generación: comparación de aptitud", fontsize=20)
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=18)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figure, dpi=150, metadata={"Software": "si-midterm"})
        plt.close(fig)
        return [_artifact(figure, "image/png", "comparación de generaciones"), _artifact(table, "text/csv", "tabla auditable de aptitud")]
    except OSError as exc:
        raise OutputError("No se pudieron escribir los artefactos genéticos") from exc


def run_exercise(data_path: Path, metadata_path: Path, config_path: Path, output_dir: Path, seed: int | None = None) -> dict[str, Any]:
    data_path, metadata_path, config_path, output_dir = map(Path, (data_path, metadata_path, config_path, output_dir))
    dataset = RetailDataset.load(data_path, metadata_path)
    if not config_path.is_file():
        raise ConfigurationError("No existe la configuración genética")
    config = _read_json(config_path, ConfigurationError)
    if config.get("technique") != "genetic":
        raise ConfigurationError("La configuración no corresponde a genetic")
    algorithm = config.get("algorithm", {})
    try:
        effective_seed = config["seed"] if seed is None else seed
        problem = DemandProblem(dataset.demands, int(algorithm["lower_bound"]), int(algorithm["upper_bound"]))
        ga = GeneticAlgorithm(problem, algorithm["initial_population"], algorithm["tournament_pairs"], int(algorithm["crossover_point"]), float(algorithm["mutation_probability"]), int(algorithm["elite_count"]), int(effective_seed))
        trace = ga.run_generation()
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("Configuración genética inválida") from exc
    artifacts = _write_outputs(output_dir, problem, trace)
    initial_rows = [{"value": v, "bits": Individual.from_value(v).bits, "mae": problem.mae(v), "fitness": problem.fitness(v)} for v in trace.initial_population]
    final_rows = [{"value": v, "bits": Individual.from_value(v).bits, "mae": problem.mae(v), "fitness": problem.fitness(v)} for v in trace.new_population]
    stages = [
        {"stage_id": "preprocess", "title_es": "Preparación de demanda", "inputs": {"rows": len(dataset.frame), "stock_code": "85123A"}, "operation": "Filtrar ventas válidas, agregar por día, recortar a 331 unidades y convertir con ceil(unidades/12).", "outputs": {"observations": len(dataset.demands), "median_dozens": float(np.median(dataset.demands))}, "code_objects": ["RetailDataset"]},
        {"stage_id": "encode-decode", "title_es": "Codificación binaria", "inputs": {"bounds": [problem.lower_bound, problem.upper_bound], "bits": 5}, "operation": "Representar cada lote entero x como cromosoma binario de cinco bits.", "outputs": {"population": [{"value": v, "bits": Individual.from_value(v).bits} for v in trace.initial_population]}, "code_objects": ["Individual", "DemandProblem"]},
        {"stage_id": "initial-fitness", "title_es": "Aptitud inicial", "inputs": {"population": trace.initial_population}, "operation": "Calcular MAE(x)=promedio(|x-d_t|) y fitness=1/(1+MAE).", "outputs": {"individuals": initial_rows}, "code_objects": ["DemandProblem", "GenerationTrace"]},
        {"stage_id": "selection", "title_es": "Selección por torneo", "inputs": {"pairs": trace.tournament_pairs}, "operation": "Elegir en cada par el individuo con mayor aptitud.", "outputs": {"selected": trace.selected}, "code_objects": ["GeneticAlgorithm"]},
        {"stage_id": "crossover", "title_es": "Cruce de un punto", "inputs": {"parents": trace.selected, "point": ga.crossover_point}, "operation": "Intercambiar sufijos después del tercer bit.", "outputs": {"children": trace.crossover_children, "bits": [Individual.from_value(v).bits for v in trace.crossover_children]}, "code_objects": ["GeneticAlgorithm", "Individual"]},
        {"stage_id": "mutation", "title_es": "Mutación", "inputs": {"children": trace.crossover_children, "probability": ga.mutation_probability, "seed": effective_seed}, "operation": "Invertir cada bit cuando el sorteo uniforme es menor que 0.05.", "outputs": {"masks": trace.mutation_masks, "mutated_children": trace.mutated_children}, "code_objects": ["GeneticAlgorithm"]},
        {"stage_id": "replacement", "title_es": "Reemplazo elitista", "inputs": {"elites": trace.elites, "offspring": trace.mutated_children}, "operation": "Conservar los dos mejores padres e incorporar los dos descendientes.", "outputs": {"new_population": trace.new_population, "individuals": final_rows}, "code_objects": ["GeneticAlgorithm", "GenerationTrace"]},
        {"stage_id": "comparison", "title_es": "Comparación", "inputs": {"initial_best": trace.initial_best, "final_best": trace.final_best}, "operation": "Comparar la mejor aptitud antes y después de una generación.", "outputs": {"initial_best_fitness": problem.fitness(trace.initial_best), "final_best_fitness": problem.fitness(trace.final_best), "improvement_percent": trace.improvement_percent}, "code_objects": ["DemandProblem", "GenerationTrace"]},
    ]
    manifest = {
        "schema_version": "1.0", "technique": "genetic",
        "dataset": {"dataset_id": dataset.metadata["dataset_id"], "metadata_path": config["dataset_metadata"], "snapshot_sha256": _sha256(data_path)},
        "execution": {"seed": int(effective_seed), "input_sha256": _sha256(data_path), "config_sha256": _sha256(config_path), "command": "run_exercise(data, metadata, config, output_dir, seed)"},
        "pdf_mapping": config["pdf_mapping"], "stages": stages,
        "summary": {"primary_value": trace.final_best, "units": "docenas/día", "conclusion_es": f"La mejor solución pasa de {trace.initial_best} a {trace.final_best} docenas y mejora la aptitud {trace.improvement_percent:.2f}%."},
        "artifacts": artifacts,
        "citations": [{"label": "UCI Online Retail", "url": dataset.metadata["source_url"], "license": dataset.metadata["license"]}, {"label": "DOI 10.24432/C5BW33", "url": "https://doi.org/10.24432/C5BW33", "license": "CC BY 4.0"}],
        "limitations": dataset.metadata["limitations"],
    }
    result_path = output_dir / "result.json"
    if result_path.exists():
        old = _read_json(result_path, ConsistencyError)
        if old.get("execution", {}).get("input_sha256") != manifest["execution"]["input_sha256"] or old.get("execution", {}).get("config_sha256") != manifest["execution"]["config_sha256"]:
            raise ConsistencyError("El resultado existente corresponde a otras entradas")
    try:
        result_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise OutputError("No se pudo escribir result.json") from exc
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_exercise(args.data, args.metadata, args.config, args.output_dir, args.seed)
    except ConfigurationError as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr); return 2
    except DataValidationError as exc:
        print(f"Error de datos: {exc}", file=sys.stderr); return 3
    except ComputationError as exc:
        print(f"Error numérico: {exc}", file=sys.stderr); return 4
    except OutputError as exc:
        print(f"Error de salida: {exc}", file=sys.stderr); return 5
    except ConsistencyError as exc:
        print(f"Error de consistencia: {exc}", file=sys.stderr); return 6
    for stage in result["stages"]:
        print(f"[{stage['stage_id']}] {stage['title_es']}")
    print(result["summary"]["conclusion_es"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
