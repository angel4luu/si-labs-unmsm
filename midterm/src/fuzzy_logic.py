"""Control Mamdani didáctico con temperatura real de NASA POWER."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class DataValidationError(Exception): pass
class ConfigurationError(Exception): pass
class ComputationError(Exception): pass
class OutputError(Exception): pass
class ConsistencyError(Exception): pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, error_type: type[Exception]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise error_type(f"No se pudo leer JSON válido: {path}") from exc


@dataclass(frozen=True)
class PowerDataset:
    values: dict[str, float]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, data_path: Path, metadata_path: Path) -> "PowerDataset":
        if not data_path.is_file() or not metadata_path.is_file():
            raise DataValidationError("Falta la instantánea NASA POWER o su metadata")
        metadata = _read_json(metadata_path, DataValidationError)
        if metadata.get("technique") != "fuzzy" or _sha256(data_path) != metadata.get("snapshot_sha256"):
            raise DataValidationError("Metadata o hash NASA POWER inválido")
        payload = _read_json(data_path, DataValidationError)
        try:
            raw = payload["properties"]["parameter"]["T2M"]
            values = {key: float(value) for key, value in raw.items() if float(value) != -999}
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError("El esquema NASA POWER no contiene T2M válido") from exc
        if len(values) != 366 or not all(np.isfinite(list(values.values()))):
            raise DataValidationError("La serie NASA POWER debe contener 366 observaciones")
        return cls(values=values, metadata=metadata)


@dataclass(frozen=True)
class MembershipFunction:
    shape: str
    points: list[float]

    def __post_init__(self) -> None:
        expected = 3 if self.shape == "triangle" else 4 if self.shape == "trapezoid" else 0
        if len(self.points) != expected or any(a > b for a, b in zip(self.points, self.points[1:])):
            raise ConfigurationError("Función de membresía inválida")

    def evaluate(self, value: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(value, dtype=float)
        if self.shape == "triangle":
            a, b, c = self.points
            left = np.ones_like(x) if a == b else (x - a) / (b - a)
            right = np.ones_like(x) if b == c else (c - x) / (c - b)
            result = np.maximum(np.minimum(left, right), 0.0)
            result = np.where((x < a) | (x > c), 0.0, result)
        else:
            a, b, c, d = self.points
            left = np.ones_like(x) if a == b else (x - a) / (b - a)
            right = np.ones_like(x) if c == d else (d - x) / (d - c)
            result = np.maximum(np.minimum(np.minimum(left, 1.0), right), 0.0)
            result = np.where((x < a) | (x > d), 0.0, result)
        result = np.clip(result, 0.0, 1.0)
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class FuzzyVariable:
    sets: dict[str, MembershipFunction]

    def memberships(self, value: float) -> dict[str, float]:
        return {name: float(function.evaluate(value)) for name, function in self.sets.items()}


@dataclass(frozen=True)
class MamdaniRule:
    antecedent: str
    consequent: str


@dataclass(frozen=True)
class InferenceTrace:
    input_value: float
    memberships: dict[str, float]
    activations: dict[str, float]
    universe: np.ndarray
    implicated: dict[str, np.ndarray]
    aggregated: np.ndarray
    centroid_numerator: float
    centroid_denominator: float
    crisp_output: float


class MamdaniController:
    def __init__(self, input_variable: FuzzyVariable, output_variable: FuzzyVariable, rules: list[MamdaniRule], step: float) -> None:
        if step <= 0:
            raise ConfigurationError("El paso del centroide debe ser positivo")
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.rules = rules
        self.step = step

    def infer(self, input_value: float) -> InferenceTrace:
        memberships = self.input_variable.memberships(input_value)
        universe = np.arange(0.0, 100.0 + self.step / 2, self.step)
        implicated: dict[str, np.ndarray] = {}
        activations: dict[str, float] = {}
        for rule in self.rules:
            strength = memberships[rule.antecedent]
            activations[f"{rule.antecedent}->{rule.consequent}"] = strength
            consequent = np.asarray(self.output_variable.sets[rule.consequent].evaluate(universe))
            implicated[rule.consequent] = np.minimum(strength, consequent)
        aggregated = np.maximum.reduce(list(implicated.values()))
        denominator = float(np.sum(aggregated))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ComputationError("El centroide tiene denominador nulo")
        numerator = float(np.sum(universe * aggregated))
        crisp = numerator / denominator
        return InferenceTrace(input_value, memberships, activations, universe, implicated, aggregated, numerator, denominator, crisp)


def _artifact(path: Path, media_type: str, purpose: str) -> dict[str, str]:
    return {"path": path.name, "media_type": media_type, "sha256": _sha256(path), "purpose": purpose}


def _build_variable(spec: dict[str, Any]) -> FuzzyVariable:
    return FuzzyVariable({name: MembershipFunction(item["shape"], [float(x) for x in item["points"]]) for name, item in spec.items()})


def _write_plots(output_dir: Path, controller: MamdaniController, trace: InferenceTrace) -> list[dict[str, str]]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        memberships_path = output_dir / "memberships.png"
        temp_x = np.linspace(16.09, 22.39, 600)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = {"low": "#38bdf8", "medium": "#f59e0b", "high": "#ef4444", "slow": "#38bdf8", "moderate": "#f59e0b", "fast": "#ef4444"}
        for name, function in controller.input_variable.sets.items():
            axes[0].plot(temp_x, function.evaluate(temp_x), label=name, color=colors[name])
        axes[0].axvline(trace.input_value, color="#0f172a", linestyle="--", label=f"{trace.input_value:.2f} °C")
        axes[0].set_title("Membresías de temperatura", fontsize=20)
        axes[0].set_xlabel("Temperatura (°C)", fontsize=18); axes[0].set_ylabel("μ", fontsize=18)
        axes[0].tick_params(labelsize=18); axes[0].legend(fontsize=18)
        for name, function in controller.output_variable.sets.items():
            axes[1].plot(trace.universe, function.evaluate(trace.universe), label=name, color=colors[name])
        axes[1].set_title("Membresías de salida", fontsize=20)
        axes[1].set_xlabel("Comando del ventilador (%)", fontsize=18); axes[1].set_ylabel("μ", fontsize=18)
        axes[1].tick_params(labelsize=18); axes[1].legend(fontsize=18)
        fig.tight_layout()
        fig.savefig(memberships_path, dpi=150, metadata={"Software": "si-midterm"})
        plt.close(fig)

        aggregation_path = output_dir / "aggregation.png"
        fig, ax = plt.subplots(figsize=(10, 5.6))
        for name, curve in trace.implicated.items():
            ax.plot(trace.universe, curve, alpha=0.55, label=f"{name} implicado")
        ax.fill_between(trace.universe, trace.aggregated, color="#7c3aed", alpha=0.35, label="agregación máx.")
        ax.axvline(trace.crisp_output, color="#111827", linestyle="--", linewidth=2, label=f"centroide {trace.crisp_output:.2f}%")
        ax.set_title("Implicación, agregación y centroide", fontsize=20)
        ax.set_xlabel("Comando del ventilador (%)", fontsize=18); ax.set_ylabel("μ", fontsize=18)
        ax.tick_params(labelsize=18); ax.legend(loc="upper left", fontsize=18)
        fig.tight_layout()
        fig.savefig(aggregation_path, dpi=150, metadata={"Software": "si-midterm"})
        plt.close(fig)
        return [_artifact(memberships_path, "image/png", "funciones de membresía"), _artifact(aggregation_path, "image/png", "agregación y centroide")]
    except OSError as exc:
        raise OutputError("No se pudieron escribir los gráficos difusos") from exc


def run_exercise(data_path: Path, metadata_path: Path, config_path: Path, output_dir: Path, seed: int | None = None) -> dict[str, Any]:
    data_path, metadata_path, config_path, output_dir = map(Path, (data_path, metadata_path, config_path, output_dir))
    dataset = PowerDataset.load(data_path, metadata_path)
    if not config_path.is_file():
        raise ConfigurationError("No existe la configuración difusa")
    config = _read_json(config_path, ConfigurationError)
    if config.get("technique") != "fuzzy":
        raise ConfigurationError("La configuración no corresponde a fuzzy")
    try:
        algorithm = config["algorithm"]
        selected_key = algorithm["selected_date"].replace("-", "")
        temperature = float(dataset.values[selected_key])
        input_variable = _build_variable(algorithm["input_sets"])
        output_variable = _build_variable(algorithm["output_sets"])
        rules = [MamdaniRule(item["if"], item["then"]) for item in algorithm["rules"]]
        controller = MamdaniController(input_variable, output_variable, rules, float(algorithm["centroid_step"]))
        trace = controller.infer(temperature)
        effective_seed = int(config["seed"] if seed is None else seed)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("Configuración difusa inválida") from exc
    artifacts = _write_plots(output_dir, controller, trace)
    points = algorithm["empirical_points_c"]
    stages = [
        {"stage_id": "preprocess", "title_es": "Preparación de temperatura", "inputs": {"observations": len(dataset.values)}, "operation": "Validar T2M 2024 y excluir el centinela -999.", "outputs": {"min_q1_median_q3_max_c": points, "selected_date": algorithm["selected_date"], "temperature_c": temperature}, "code_objects": ["PowerDataset"]},
        {"stage_id": "memberships", "title_es": "Construcción de membresías", "inputs": {"empirical_points_c": points}, "operation": "Construir hombros trapezoidales bajo/alto y triángulo medio desde mínimo, cuartiles y máximo.", "outputs": {"input_sets": algorithm["input_sets"], "output_sets": algorithm["output_sets"]}, "code_objects": ["MembershipFunction", "FuzzyVariable"]},
        {"stage_id": "fuzzification", "title_es": "Fusificación", "inputs": {"temperature_c": temperature}, "operation": "Evaluar μ_baja, μ_media y μ_alta en la observación real.", "outputs": {"memberships": trace.memberships}, "code_objects": ["FuzzyVariable", "MembershipFunction"]},
        {"stage_id": "rule-evaluation", "title_es": "Evaluación de reglas", "inputs": {"memberships": trace.memberships, "rules": algorithm["rules"]}, "operation": "Activar bajo→lento, medio→moderado y alto→rápido.", "outputs": {"activations": trace.activations}, "code_objects": ["MamdaniRule", "MamdaniController"]},
        {"stage_id": "implication", "title_es": "Implicación por mínimo", "inputs": {"activations": trace.activations}, "operation": "Recortar cada conjunto consecuente con min(activación, μ_salida).", "outputs": {"peak_memberships": {name: float(curve.max()) for name, curve in trace.implicated.items()}}, "code_objects": ["MamdaniController"]},
        {"stage_id": "aggregation", "title_es": "Agregación por máximo", "inputs": {"implicated_sets": list(trace.implicated)}, "operation": "Combinar los consecuentes punto a punto mediante máximo.", "outputs": {"maximum_membership": float(trace.aggregated.max()), "nonzero_points": int(np.count_nonzero(trace.aggregated))}, "code_objects": ["MamdaniController", "InferenceTrace"]},
        {"stage_id": "centroid", "title_es": "Desdifusificación por centroide", "inputs": {"step": controller.step}, "operation": "Calcular Σ(y·μ_agregada(y))/Σμ_agregada(y) sobre 0–100.", "outputs": {"numerator": trace.centroid_numerator, "denominator": trace.centroid_denominator, "crisp_percent": trace.crisp_output}, "code_objects": ["MamdaniController", "InferenceTrace"]},
        {"stage_id": "interpretation", "title_es": "Interpretación", "inputs": {"crisp_percent": trace.crisp_output}, "operation": "Interpretar el centroide como comando porcentual diseñado para el ventilador.", "outputs": {"command": "ventilación media-alta", "temperature_context": "media diaria exterior"}, "code_objects": ["InferenceTrace"]},
    ]
    manifest = {
        "schema_version": "1.0", "technique": "fuzzy",
        "dataset": {"dataset_id": dataset.metadata["dataset_id"], "metadata_path": config["dataset_metadata"], "snapshot_sha256": _sha256(data_path)},
        "execution": {"seed": effective_seed, "input_sha256": _sha256(data_path), "config_sha256": _sha256(config_path), "command": "run_exercise(data, metadata, config, output_dir, seed)"},
        "pdf_mapping": config["pdf_mapping"], "stages": stages,
        "summary": {"primary_value": trace.crisp_output, "units": "% de comando del ventilador", "conclusion_es": f"Para {temperature:.2f} °C, el centroide Mamdani ordena {trace.crisp_output:.6f}% de ventilación."},
        "artifacts": artifacts,
        "citations": [{"label": "NASA POWER Daily API", "url": dataset.metadata["source_url"], "license": dataset.metadata["license"]}, {"label": "NASA POWER Referencing", "url": "https://power.larc.nasa.gov/docs/referencing/", "license": "NASA Open Data"}],
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
    parser.add_argument("--data", type=Path, required=True); parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_exercise(args.data, args.metadata, args.config, args.output_dir, args.seed)
    except ConfigurationError as exc: print(f"Error de configuración: {exc}", file=sys.stderr); return 2
    except DataValidationError as exc: print(f"Error de datos: {exc}", file=sys.stderr); return 3
    except ComputationError as exc: print(f"Error numérico: {exc}", file=sys.stderr); return 4
    except OutputError as exc: print(f"Error de salida: {exc}", file=sys.stderr); return 5
    except ConsistencyError as exc: print(f"Error de consistencia: {exc}", file=sys.stderr); return 6
    for stage in result["stages"]: print(f"[{stage['stage_id']}] {stage['title_es']}")
    print(result["summary"]["conclusion_es"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
