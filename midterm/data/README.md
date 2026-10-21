# Datos del examen parcial

Los tres ejercicios se ejecutan sin red a partir de instantáneas versionadas. Ningún script descarga
datos durante su ejecución. Antes de calcular, cada implementación verifica el SHA-256 de su
instantánea contra el registro JSON de `metadata/`.

## Fuentes y redistribución

| Técnica | Fuente | Instantánea | Uso |
|---|---|---|---|
| Algoritmo genético | UCI Online Retail, DOI 10.24432/C5BW33, CC BY 4.0 | `derived/online_retail_85123a_daily.csv` | 305 demandas diarias agregadas del producto 85123A. |
| Lógica difusa | NASA POWER/MERRA-2, datos científicos abiertos de NASA | `raw/nasa_power_lima_2024.json` | 366 temperaturas medias diarias T2M cerca de UNMSM. |
| MLP | UCI Student Performance, DOI 10.24432/C5TG7T, CC BY 4.0 | `raw/student-mat.csv` | 395 registros académicos del curso de Matemática. |

Los datos UCI se redistribuyen bajo CC BY 4.0 con atribución. Los datos NASA se usan conforme a su
política de datos científicos abiertos y se conserva la consulta exacta. Las fechas de recuperación,
licencias, esquemas, transformaciones, limitaciones y hashes están en `metadata/*.json`.

## Preparación y privacidad

- Retail: se eliminan cancelaciones, cantidades/precios no positivos, se filtra el producto 85123A,
  se agrega por fecha, se limita al percentil 95 (331 unidades) y se convierte a docenas con techo.
- POWER: se conserva la serie diaria T2M de 2024 para latitud -12.06 y longitud -77.08.
- Student Performance: se conserva el CSV original; el código usa G1 y G2 como entradas y deriva la
  etiqueta `G3 >= 10` sin usar G3 como característica.
- La instantánea derivada de Retail excluye `CustomerID`; ninguna instantánea añade nombres ni otros
  identificadores personales.

Los archivos fuente completos usados para derivar Retail no se requieren en tiempo de ejecución. Si
una fuente cambia, debe regenerarse la instantánea, actualizar su hash y volver a ejecutar todos los
resultados, notebooks y artefactos de presentación.
