"""Una actualización MLP 2-2-1 explicable sobre notas reales de UCI."""

from __future__ import annotations

import argparse
import csv
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
import pandas as pd


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
class StudentDataset:
    frame: pd.DataFrame
    labels: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, data_path: Path, metadata_path: Path) -> "StudentDataset":
        if not data_path.is_file() or not metadata_path.is_file():
            raise DataValidationError("Falta la instantánea estudiantil o su metadata")
        metadata = _read_json(metadata_path, DataValidationError)
        if metadata.get("technique") != "mlp" or _sha256(data_path) != metadata.get("snapshot_sha256"):
            raise DataValidationError("Metadata o hash Student Performance inválido")
        try:
            frame = pd.read_csv(data_path, sep=";")
        except Exception as exc:
            raise DataValidationError("No se pudo leer Student Performance") from exc
        if len(frame) != 395 or not {"G1", "G2", "G3"}.issubset(frame.columns):
            raise DataValidationError("El esquema estudiantil no cumple el contrato")
        grades = frame[["G1", "G2", "G3"]]
        if grades.isna().any().any() or ((grades < 0) | (grades > 20)).any().any():
            raise DataValidationError("Las notas contienen valores inválidos")
        labels = (frame["G3"].to_numpy(dtype=float) >= 10).astype(int)
        return cls(frame, labels, metadata)

    def stratified_split(self, test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
        if not 0 < test_ratio < 1:
            raise ConfigurationError("La proporción de prueba debe estar entre cero y uno")
        rng = np.random.default_rng(seed)
        test: list[int] = []
        for label in sorted(np.unique(self.labels)):
            indices = np.flatnonzero(self.labels == label)
            rng.shuffle(indices)
            count = round(len(indices) * test_ratio)
            test.extend(int(value) for value in indices[:count])
        test_set = set(test)
        train = [index for index in range(len(self.labels)) if index not in test_set]
        return train, sorted(test)


@dataclass(frozen=True)
class GradeNormalizer:
    minimum: float = 0.0
    maximum: float = 20.0

    def transform(self, g1: float, g2: float) -> np.ndarray:
        values = np.asarray([g1, g2], dtype=float)
        if (values < self.minimum).any() or (values > self.maximum).any():
            raise DataValidationError("Nota fuera del rango de normalización")
        return (values - self.minimum) / (self.maximum - self.minimum)


@dataclass
class DenseLayer:
    weights: np.ndarray
    biases: np.ndarray

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.weights @ inputs + self.biases


class Sigmoid:
    @staticmethod
    def forward(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def derivative_from_output(outputs: np.ndarray) -> np.ndarray:
        return outputs * (1.0 - outputs)


class BinaryCrossEntropy:
    @staticmethod
    def value(prediction: float, target: float) -> float:
        clipped = float(np.clip(prediction, 1e-12, 1 - 1e-12))
        return float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))


@dataclass(frozen=True)
class TrainingTrace:
    inputs: list[float]
    target: float
    hidden_pre_activation: list[float]
    hidden_activation_before: list[float]
    output_pre_activation: float
    prediction_before: float
    loss_before: float
    output_delta: float
    hidden_delta: list[float]
    weight_gradients_hidden: list[list[float]]
    bias_gradients_hidden: list[float]
    weight_gradients_output: list[float]
    bias_gradient_output: float
    parameters_before: dict[str, Any]
    parameters_after: dict[str, Any]
    prediction_after: float
    loss_after: float


class MLP221:
    def __init__(self, hidden: DenseLayer, output: DenseLayer) -> None:
        if hidden.weights.shape != (2, 2) or hidden.biases.shape != (2,):
            raise ConfigurationError("La capa oculta debe ser 2x2")
        if output.weights.shape != (1, 2) or output.biases.shape != (1,):
            raise ConfigurationError("La salida debe ser 1x2")
        self.hidden = hidden
        self.output = output

    @classmethod
    def from_parameters(cls, hidden_weights: list[list[float]], hidden_biases: list[float], output_weights: list[float], output_bias: float) -> "MLP221":
        return cls(
            DenseLayer(np.asarray(hidden_weights, dtype=float), np.asarray(hidden_biases, dtype=float)),
            DenseLayer(np.asarray([output_weights], dtype=float), np.asarray([output_bias], dtype=float)),
        )

    def predict_with_trace(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        hidden_z = self.hidden.forward(inputs)
        hidden_a = Sigmoid.forward(hidden_z)
        output_z = float(self.output.forward(hidden_a)[0])
        prediction = float(Sigmoid.forward(np.asarray(output_z)))
        return hidden_z, hidden_a, output_z, prediction

    def parameters(self) -> dict[str, Any]:
        return {
            "input_hidden_weights": self.hidden.weights.tolist(),
            "hidden_biases": self.hidden.biases.tolist(),
            "hidden_output_weights": self.output.weights[0].tolist(),
            "output_bias": float(self.output.biases[0]),
        }

    def train_one(self, inputs: np.ndarray, target: float, learning_rate: float) -> TrainingTrace:
        if inputs.shape != (2,) or target not in (0.0, 1.0) or learning_rate <= 0:
            raise ConfigurationError("Entrada, clase o tasa de aprendizaje inválida")
        before = self.parameters()
        hidden_z, hidden_a, output_z, prediction = self.predict_with_trace(inputs)
        loss = BinaryCrossEntropy.value(prediction, target)
        output_delta = prediction - target
        gradient_output_w = output_delta * hidden_a
        gradient_output_b = output_delta
        hidden_delta = (self.output.weights[0] * output_delta) * Sigmoid.derivative_from_output(hidden_a)
        gradient_hidden_w = np.outer(hidden_delta, inputs)
        gradient_hidden_b = hidden_delta
        self.output.weights -= learning_rate * gradient_output_w.reshape(1, 2)
        self.output.biases -= learning_rate * np.asarray([gradient_output_b])
        self.hidden.weights -= learning_rate * gradient_hidden_w
        self.hidden.biases -= learning_rate * gradient_hidden_b
        _, _, _, prediction_after = self.predict_with_trace(inputs)
        loss_after = BinaryCrossEntropy.value(prediction_after, target)
        return TrainingTrace(
            inputs.tolist(), target, hidden_z.tolist(), hidden_a.tolist(), output_z, prediction, loss,
            float(output_delta), hidden_delta.tolist(), gradient_hidden_w.tolist(), gradient_hidden_b.tolist(),
            gradient_output_w.tolist(), float(gradient_output_b), before, self.parameters(), prediction_after, loss_after,
        )


def _artifact(path: Path, media_type: str, purpose: str) -> dict[str, str]:
    return {"path": path.name, "media_type": media_type, "sha256": _sha256(path), "purpose": purpose}


def _write_outputs(output_dir: Path, trace: TrainingTrace) -> list[dict[str, str]]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        table = output_dir / "parameter_update.csv"
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parameter", "before", "after"])
            for name in trace.parameters_before:
                writer.writerow([name, json.dumps(trace.parameters_before[name]), json.dumps(trace.parameters_after[name])])
            writer.writerow(["prediction", trace.prediction_before, trace.prediction_after])
            writer.writerow(["binary_cross_entropy", trace.loss_before, trace.loss_after])
        figure = output_dir / "network_update.png"
        fig, ax = plt.subplots(figsize=(10, 5.6))
        input_pos = [(0.1, 0.7), (0.1, 0.3)]
        hidden_pos = [(0.5, 0.7), (0.5, 0.3)]
        output_pos = [(0.9, 0.5)]
        for source in input_pos:
            for target_pos in hidden_pos:
                ax.plot([source[0], target_pos[0]], [source[1], target_pos[1]], color="#94a3b8", linewidth=1.5)
        for source in hidden_pos:
            ax.plot([source[0], output_pos[0][0]], [source[1], output_pos[0][1]], color="#94a3b8", linewidth=1.5)
        for label, positions, color in (("Entrada", input_pos, "#38bdf8"), ("Oculta", hidden_pos, "#8b5cf6"), ("Salida", output_pos, "#f59e0b")):
            for index, (x, y) in enumerate(positions, 1):
                ax.scatter(x, y, s=1000, color=color, edgecolor="#0f172a", zorder=3)
                ax.text(x, y, f"{label[0]}{index}", ha="center", va="center", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.05, f"ŷ: {trace.prediction_before:.6f} → {trace.prediction_after:.6f}    BCE: {trace.loss_before:.6f} → {trace.loss_after:.6f}", ha="center", fontsize=18)
        ax.set_title("MLP 2-2-1: efecto de una actualización", fontsize=20)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.tight_layout()
        fig.savefig(figure, dpi=150, metadata={"Software": "si-midterm"})
        plt.close(fig)
        return [_artifact(figure, "image/png", "red y cambio de predicción"), _artifact(table, "text/csv", "parámetros antes y después")]
    except OSError as exc:
        raise OutputError("No se pudieron escribir los artefactos MLP") from exc


def run_exercise(data_path: Path, metadata_path: Path, config_path: Path, output_dir: Path, seed: int | None = None) -> dict[str, Any]:
    data_path, metadata_path, config_path, output_dir = map(Path, (data_path, metadata_path, config_path, output_dir))
    dataset = StudentDataset.load(data_path, metadata_path)
    if not config_path.is_file():
        raise ConfigurationError("No existe la configuración MLP")
    config = _read_json(config_path, ConfigurationError)
    if config.get("technique") != "mlp":
        raise ConfigurationError("La configuración no corresponde a mlp")
    try:
        algorithm = config["algorithm"]
        effective_seed = int(config["seed"] if seed is None else seed)
        train, test = dataset.stratified_split(float(algorithm["test_ratio"]), effective_seed)
        row_index = int(algorithm["source_row"]) - 1
        if row_index not in train:
            raise ConfigurationError("La fila de presentación debe permanecer en entrenamiento")
        row = dataset.frame.iloc[row_index]
        normalizer = GradeNormalizer(float(algorithm["normalization"]["minimum"]), float(algorithm["normalization"]["maximum"]))
        inputs = normalizer.transform(float(row["G1"]), float(row["G2"]))
        target = float(row["G3"] >= 10)
        model = MLP221.from_parameters(algorithm["input_hidden_weights"], algorithm["hidden_biases"], algorithm["hidden_output_weights"], float(algorithm["output_bias"]))
        trace = model.train_one(inputs, target, float(algorithm["learning_rate"]))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ConfigurationError("Configuración MLP inválida") from exc
    artifacts = _write_outputs(output_dir, trace)
    stages = [
        {"stage_id": "preprocess-split", "title_es": "Preparación y partición", "inputs": {"rows": len(dataset.frame), "features": ["G1", "G2"], "target": "G3 >= 10", "seed": effective_seed}, "operation": "Derivar la clase sin usar G3 como entrada y estratificar 80/20.", "outputs": {"train_rows": len(train), "test_rows": len(test), "presentation_row_in_train": row_index in train}, "code_objects": ["StudentDataset"]},
        {"stage_id": "sample-normalization", "title_es": "Normalización de la muestra", "inputs": {"source_row": int(algorithm["source_row"]), "G1": int(row["G1"]), "G2": int(row["G2"]), "G3": int(row["G3"])}, "operation": "Aplicar x1=G1/20, x2=G2/20 y y=1 si G3>=10.", "outputs": {"x": trace.inputs, "y": trace.target}, "code_objects": ["GradeNormalizer", "StudentDataset"]},
        {"stage_id": "initialization", "title_es": "Inicialización 2-2-1", "inputs": {"architecture": algorithm["architecture"]}, "operation": "Cargar pesos y sesgos deterministas para dos entradas, dos neuronas ocultas y una salida.", "outputs": trace.parameters_before, "code_objects": ["DenseLayer", "MLP221"]},
        {"stage_id": "forward", "title_es": "Propagación hacia adelante", "inputs": {"x": trace.inputs}, "operation": "Calcular z_h=W_hx+b_h, h=σ(z_h), z_o=W_oh+b_o y ŷ=σ(z_o).", "outputs": {"hidden_z": trace.hidden_pre_activation, "hidden_activation": trace.hidden_activation_before, "output_z": trace.output_pre_activation, "prediction": trace.prediction_before}, "code_objects": ["DenseLayer", "Sigmoid", "MLP221"]},
        {"stage_id": "loss", "title_es": "Pérdida", "inputs": {"target": trace.target, "prediction": trace.prediction_before}, "operation": "Calcular entropía cruzada binaria −[y ln ŷ +(1−y) ln(1−ŷ)].", "outputs": {"binary_cross_entropy": trace.loss_before}, "code_objects": ["BinaryCrossEntropy"]},
        {"stage_id": "backward", "title_es": "Retropropagación", "inputs": {"prediction": trace.prediction_before, "target": trace.target}, "operation": "Propagar δ_o=ŷ−y y δ_h=(W_o δ_o)⊙h(1−h); formar gradientes externos.", "outputs": {"output_delta": trace.output_delta, "hidden_delta": trace.hidden_delta, "hidden_weight_gradients": trace.weight_gradients_hidden, "output_weight_gradients": trace.weight_gradients_output}, "code_objects": ["Sigmoid", "MLP221"]},
        {"stage_id": "update", "title_es": "Actualización de parámetros", "inputs": {"learning_rate": float(algorithm["learning_rate"]), "parameters_before": trace.parameters_before}, "operation": "Aplicar θ_nuevo=θ−η∇θ a pesos y sesgos.", "outputs": {"parameters_after": trace.parameters_after}, "code_objects": ["DenseLayer", "MLP221", "TrainingTrace"]},
        {"stage_id": "post-update-forward", "title_es": "Verificación posterior", "inputs": {"parameters_after": trace.parameters_after}, "operation": "Repetir la propagación y comprobar el cambio de predicción y pérdida.", "outputs": {"prediction_before": trace.prediction_before, "prediction_after": trace.prediction_after, "loss_before": trace.loss_before, "loss_after": trace.loss_after}, "code_objects": ["MLP221", "BinaryCrossEntropy", "TrainingTrace"]},
    ]
    manifest = {
        "schema_version": "1.0", "technique": "mlp",
        "dataset": {"dataset_id": dataset.metadata["dataset_id"], "metadata_path": config["dataset_metadata"], "snapshot_sha256": _sha256(data_path)},
        "execution": {"seed": effective_seed, "input_sha256": _sha256(data_path), "config_sha256": _sha256(config_path), "command": "run_exercise(data, metadata, config, output_dir, seed)"},
        "pdf_mapping": config["pdf_mapping"], "stages": stages,
        "summary": {"primary_value": trace.prediction_after, "units": "probabilidad estimada de aprobar", "conclusion_es": f"Una actualización eleva ŷ de {trace.prediction_before:.6f} a {trace.prediction_after:.6f} y reduce BCE de {trace.loss_before:.6f} a {trace.loss_after:.6f}."},
        "artifacts": artifacts,
        "citations": [{"label": "UCI Student Performance", "url": dataset.metadata["source_url"], "license": dataset.metadata["license"]}, {"label": "DOI 10.24432/C5TG7T", "url": "https://doi.org/10.24432/C5TG7T", "license": "CC BY 4.0"}],
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
