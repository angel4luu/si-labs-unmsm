# Laboratorio 10 – Comparación de modelos de clasificación

Comparación de cuatro clasificadores (Regresión Logística, KNN, Bosque Aleatorio y Gradient
Boosting) para predecir diabetes con el conjunto *Pima Indians Diabetes*. Usa validación cruzada
estratificada, curvas ROC/AUC y diagnóstico de subajuste y sobreajuste. El mejor modelo se reentrena
sin validación cruzada, se evalúa con el 20 % reservado y se guarda para usarlo en un software
inteligente.

## Contenido

| Archivo | Descripción |
|---|---|
| `laboratorio10_clasificacion_diabetes.ipynb` | Cuaderno principal (entregable), con las salidas de una ejecución limpia |
| `data/diabetes.csv` | Conjunto de datos (768 pacientes, 8 variables, objetivo `Outcome`) |
| `models/modelo_diabetes.joblib` | Modelo final generado por la sección 14 del cuaderno |
| `pyproject.toml`, `uv.lock` | Entorno reproducible (Python 3.14) |

## Ejecución

```bash
cd lab10
uv sync
uv run python -m ipykernel install --user --name lab10 --display-name "Lab 10 (uv)"
uv run jupyter execute --inplace --kernel_name=lab10 laboratorio10_clasificacion_diabetes.ipynb
```

También puede abrirse en JupyterLab o VS Code con el kernel **Lab 10 (uv)** y ejecutar *Restart & Run
All*. La ejecución completa tarda unos 15 s y no requiere conexión a internet.

## Uso del modelo guardado

```python
import joblib, pandas as pd

artefacto = joblib.load("models/modelo_diabetes.joblib")
paciente = pd.DataFrame([{"Pregnancies": 2, "Glucose": 150, "BloodPressure": 80, "SkinThickness": 30,
                          "Insulin": 0, "BMI": 34.5, "DiabetesPedigreeFunction": 0.6, "Age": 45}])
probabilidad = artefacto["pipeline"].predict_proba(paciente[artefacto["features"]])[0, 1]
riesgo_alto = probabilidad >= artefacto["umbral"]
```

Es un modelo educativo de apoyo al tamizaje, **no un dispositivo de diagnóstico**.

## Datos

*Pima Indians Diabetes Database*: NIDDK; Smith et al. (1988). Se distribuye en UCI y Kaggle
(`uciml/pima-indians-diabetes-database`, licencia CC0). Se descargó de
`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
(SHA-256 `6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af`) y se le agregó la
cabecera de Kaggle. El SHA-256 del archivo con cabecera (`27939f6c…ee66`) se verifica en el cuaderno.
