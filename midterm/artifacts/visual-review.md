# Revisión visual de la presentación

**Fecha:** 2026-10-15 (revisión 2)  
**Visor de referencia:** PDF generado por LibreOffice 26.2.6.3 y PNG a 150 DPI con Poppler 26.01.0.

Se inspeccionaron las doce páginas renderizadas de `rendered/slide-01.png` a `slide-12.png`.

| Diapositivas | Revisión | Resultado |
|---|---|---|
| 1–4 | GA: panorama, componentes, primera generación con cálculos, código | PASS |
| 5–8 | Mamdani: panorama, conjuntos y reglas, inferencia con cálculos, código | PASS |
| 9–12 | MLP: panorama, inicialización, iteración con cálculos, código y declaración de IA | PASS |

No se observaron recortes, desbordes, imágenes rotas ni sustitución visible de fuentes. Los títulos
usan al menos 28 pt; el cuerpo, las tablas y las etiquetas al menos 18 pt; el código 14 pt en
Liberation Mono; pies, citas y declaración de IA 10 pt. Cada técnica sigue la misma secuencia:

1. **Panorama:** tipo de técnica, flujo del algoritmo, dataset y tabla «examen vs. esta versión».
2. **Componentes:** representación, parámetros y valores iniciales concretos.
3. **Paso a paso:** tabla de cinco pasos con la fórmula sustituida por los valores del manifiesto.
4. **Implementación:** fragmento real del código fuente con líneas numeradas y notas que las conectan
   con la fórmula teórica. Una prueba verifica que cada fragmento coincide con `src/`.

Correcciones de esta revisión: el pie del fragmento GA se dividía en dos líneas y quedaba recortado,
la tarjeta «Tipo de técnica» del MLP desbordaba ligeramente, y las notas del código se centraban y
tenían poco ancho. Se regeneraron PPTX, PDF y las doce imágenes y se volvió a inspeccionar el conjunto.

**Decisión visual:** PASS.
