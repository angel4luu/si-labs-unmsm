# Examen parcial — Software Inteligente

Proyecto académico reproducible para explicar, con datos reales, un algoritmo genético, inferencia
Mamdani y una red neuronal multicapa 2-2-1. El PDF original se conserva como `Examen parcial.pdf`.

## Entorno

uv es el único administrador del entorno Python. No se deben mezclar paquetes de Conda con `.venv`.
LibreOffice, Poppler y Liberation Sans son herramientas externas usadas únicamente para validar la
presentación; sus versiones reales se registran en `artifacts/submission-checklist.md`.

```bash
uv sync --project midterm --locked
```

## Ejecución

Desde la raíz del repositorio:

```bash
uv run --project midterm python midterm/src/genetic_algorithm.py --data midterm/data/derived/online_retail_85123a_daily.csv --metadata midterm/data/metadata/genetic.json --config midterm/configs/genetic.json --output-dir midterm/results/genetic
uv run --project midterm python midterm/src/fuzzy_logic.py --data midterm/data/raw/nasa_power_lima_2024.json --metadata midterm/data/metadata/fuzzy.json --config midterm/configs/fuzzy.json --output-dir midterm/results/fuzzy
uv run --project midterm python midterm/src/mlp.py --data midterm/data/raw/student-mat.csv --metadata midterm/data/metadata/mlp.json --config midterm/configs/mlp.json --output-dir midterm/results/mlp
uv run --project midterm pytest midterm/tests -q
```

Los notebooks explicativos son `notebooks/genetic_algorithm.ipynb`, `notebooks/fuzzy_logic.ipynb` y
`notebooks/mlp.ipynb`. Se ejecutan en kernel limpio con el comando documentado en la guía completa:
[`../specs/001-midterm-presentation/quickstart.md`](../specs/001-midterm-presentation/quickstart.md).
Las únicas implementaciones de los ejercicios son los tres archivos en `src/`; los notebooks llaman
a `run_exercise(...)` y no duplican algoritmos.

## Presentación y artefactos

```bash
uv run --project midterm python midterm/presentation/generate_presentation.py \
  --results-dir midterm/results --content midterm/presentation/content.json \
  --output midterm/artifacts/midterm.pptx \
  --script-output midterm/artifacts/guion.md --strict
```

- `results/*/result.json`: fuente canónica de valores, etapas, citas y limitaciones.
- `artifacts/midterm.pptx` y `artifacts/guion.md`: presentación y narración sincronizadas.
- `artifacts/midterm.pdf` y `artifacts/rendered/`: evidencia renderizada.
- `artifacts/visual-review.md`, `academic-review.md`, `rehearsal.md` y
  `submission-checklist.md`: puertas finales de calidad.

Las fuentes son UCI Online Retail (DOI 10.24432/C5BW33, CC BY 4.0), NASA POWER (datos científicos
abiertos) y UCI Student Performance (DOI 10.24432/C5TG7T, CC BY 4.0). Consulte `data/README.md` para
procedencia, transformaciones y redistribución. Entre las limitaciones: el objetivo GA es cercanía a
demanda y no beneficio; T2M no es un sensor de aula; y una sola actualización MLP no evalúa un modelo
completo ni generaliza a UNMSM.

## Integridad académica

Las fuentes, licencias, transformaciones y limitaciones se documentan en `data/README.md` y
`data/metadata/`. Los resultados se derivan de instantáneas locales verificadas. La estudiante debe
revisar, comprender y poder defender todo el contenido. Este repositorio declara asistencia de IA en
la planificación, programación, pruebas y redacción; la autoría y responsabilidad final corresponden
a la estudiante y esta declaración debe ajustarse si la política del curso exige otro formato.

No se considera limpio un resultado si cambió el código, dato o configuración sin regenerar sus
manifiestos, notebooks, presentación y evidencia de revisión.
