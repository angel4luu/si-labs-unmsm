# Guion — Examen parcial de Software Inteligente

Duración planificada: 9:00 (3:00 por técnica, sin portada ni cierre).

## 1. P1 · Algoritmo genético — Panorama [ga-overview]

**Tiempo:** 30 s

La primera pregunta usa un algoritmo genético, una metaheurística evolutiva: una población binaria mejora mediante aptitud, torneo, cruce, mutación y elitismo. En lugar de la función sintética del examen, uso ventas diarias reales del producto 85123A de UCI Online Retail, en docenas. Por eso el objetivo pasa a ser uno entre uno más el error absoluto medio, y el rango sube a cero–treinta y uno con cinco bits.

## 2. GA — Componentes y configuración [ga-data]

**Tiempo:** 35 s

Cada individuo es un lote entre cero y treinta y uno docenas, codificado con cinco bits; la población inicial es cuatro, once, catorce y veinticinco. La aptitud compara cada lote con los trescientos cinco días observados: menor error absoluto medio significa mayor aptitud. Uso torneo por pares, cruce de un punto tras el tercer bit y mutación por bit con probabilidad cero coma cero cinco y semilla fija.

## 3. GA — Primera generación paso a paso [ga-process]

**Tiempo:** 55 s

Primero evalúo la población: cuatro docenas tiene la mayor aptitud, cero coma ciento cincuenta. En los torneos gana cuatro frente a veinticinco y once frente a catorce. Los padres se cortan después del tercer bit y producen siete y ocho. Con la semilla fija, la mutación cambia solo el último bit del segundo hijo, que pasa a nueve. Reemplazo conservando los dos padres élite y agrego los hijos. Respondiendo al examen: sí, la nueva generación mejora, porque siete docenas alcanza aptitud cero coma ciento sesenta y cuatro, un nueve coma dos por ciento más que la mejor inicial.

## 4. GA — Implementación en Python [ga-code]

**Tiempo:** 60 s

Así se ve la teoría en el código. DemandProblem calcula el error absoluto medio con NumPy sobre todas las demandas y lo convierte en aptitud con la fórmula anterior. En GeneticAlgorithm, la selección aplica el torneo a cada par y conserva al ganador. Para cruzar, cada padre se convierte a su cadena de cinco bits y se une la cabeza de uno con la cola del otro en el punto de corte. La mutación genera un número aleatorio por bit y lo invierte cuando es menor que cero coma cero cinco. Finalmente, la nueva población une los dos élites con los hijos mutados. Todo queda en una traza, de donde salen los números de esta presentación.

## 5. P2 · Lógica difusa Mamdani — Panorama [fuzzy-overview]

**Tiempo:** 30 s

La segunda pregunta es un sistema de inferencia difusa Mamdani: reglas si–entonces sobre conjuntos donde un valor pertenece parcialmente a varias categorías. El flujo es fusificar, evaluar reglas, implicar con mínimo, agregar con máximo y defusificar con el centroide. En vez de veintiocho grados, uso la temperatura media real del catorce de mayo de 2024 en Lima, de NASA POWER: diecinueve coma cuatro grados.

## 6. Mamdani — Conjuntos y reglas [fuzzy-data]

**Tiempo:** 35 s

Como la temperatura de Lima en 2024 solo varió entre dieciséis y veintidós grados, los conjuntos fijos del examen no tendrían sentido. Construyo los de entrada con el mínimo, los cuartiles y el máximo anuales: baja y alta son trapecios en los extremos y media es un triángulo centrado en la mediana. La salida conserva los tres triángulos del examen, lenta, moderada y rápida, y sus tres reglas.

## 7. Mamdani — Inferencia paso a paso [fuzzy-process]

**Tiempo:** 55 s

Fusifico diecinueve coma cuatro grados. La pertenencia a baja es cero. Para media uso la pendiente derecha del triángulo: veinte coma cero nueve menos diecinueve coma cuatro, entre veinte coma cero nueve menos dieciocho coma sesenta y ocho, que da cero coma cuarenta y nueve; alta recibe cero coma cincuenta y uno. Esos grados son la fuerza de las reglas media–moderada y alta–rápida. Con el mínimo recorto cada triángulo de salida y con el máximo los uno en una sola región. Su centroide da sesenta y cinco coma dos por ciento: ventilación media-alta.

## 8. Mamdani — Implementación en Python [fuzzy-code]

**Tiempo:** 60 s

Todo el procedimiento Mamdani está en el método infer de MamdaniController. Primero, la variable de entrada calcula la pertenencia de la temperatura a cada conjunto, con la clase MembershipFunction que evalúa triángulos y trapecios. Luego defino la malla de salida, de cero a cien por ciento con paso de cero coma cero uno. Para cada regla, la fuerza es la pertenencia de su antecedente, y con np.minimum recorto toda la función del consecuente a esa altura: es la implicación por mínimo. np.maximum.reduce combina los conjuntos recortados, que es la agregación por máximo. Finalmente, el centroide discreto divide la suma de v por mu entre la suma de mu, después de validar que el denominador no sea cero. Cada valor intermedio queda registrado en una traza.

## 9. P3 · Red neuronal multicapa — Panorama [mlp-overview]

**Tiempo:** 30 s

La tercera pregunta es una red neuronal feedforward dos-dos-uno: dos entradas, dos neuronas ocultas y una salida, todas con sigmoide. Es aprendizaje supervisado para clasificación binaria. En lugar de asistencia y actividades sintéticas, uso Student Performance de UCI: las notas G1 y G2 entre veinte son las entradas, y la clase vale uno si la nota final G3 aprueba. G3 nunca entra a la red.

## 10. MLP — Componentes e inicialización [mlp-data]

**Tiempo:** 35 s

Tras una partición estratificada con semilla fija, tomo un estudiante con G1 igual a quince y G2 igual a catorce; como G3 es quince, la clase esperada es uno. Al dividir entre veinte obtengo cero coma setenta y cinco y cero coma setenta. Como el examen no fija los pesos, declaro pesos y sesgos pequeños y deterministas, y conservo la tasa de cero coma dos, la sigmoide y la entropía cruzada binaria.

## 11. MLP — Una iteración paso a paso [mlp-process]

**Tiempo:** 55 s

En la capa oculta calculo z uno igual a W por x más b: cero coma trescientos quince y menos cero coma ciento cinco; la sigmoide da cero coma cincuenta y ocho y cero coma cuarenta y siete. La salida produce una probabilidad de cero coma quinientos treinta y dos, con pérdida de cero coma sesenta y tres. Al combinar sigmoide y entropía cruzada, el delta de salida es predicción menos objetivo; ese error llega a la capa oculta a través de los pesos. Resto tasa por gradiente a cada parámetro y repito el forward: la probabilidad sube a cero coma quinientos sesenta y nueve y la pérdida baja.

## 12. MLP — Implementación en Python [mlp-code]

**Tiempo:** 60 s

En el código, cada capa es un objeto DenseLayer que calcula W por x más b, y la clase Sigmoid aplica la activación. predict_with_trace encadena capa oculta, sigmoide, capa de salida y sigmoide, exactamente como el forward anterior. En train_one, BinaryCrossEntropy mide la pérdida; el delta de salida es predicción menos objetivo, y el gradiente de salida es ese delta por las activaciones ocultas. El delta oculto multiplica el error por los pesos y por la derivada de la sigmoide, a por uno menos a, y np.outer forma los gradientes de la primera capa. Por último, cada matriz de pesos resta la tasa de aprendizaje por su gradiente. Un segundo forward con los pesos nuevos confirma que la pérdida baja.

## Declaración de asistencia de IA

Se utilizó asistencia de IA para planificación, programación, pruebas y redacción. La estudiante revisó los resultados y conserva la responsabilidad académica final.
