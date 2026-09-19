# Arquitectura del Proyecto

Documento técnico. Si lo que buscas es prender el proyecto, ve al **[README](README.md)**.

---

## 1. Qué hace y qué logra

Este proyecto toma la base transaccional de órdenes de trabajo de un taller automotriz (34.935 registros), la convierte en dos datasets entrenables y ajusta **dos modelos que responden dos preguntas distintas**:

| Modelo | Pregunta | Para qué sirve |
|---|---|---|
| Regresión | ¿En cuántos días vuelve este vehículo? | Agendar, dimensionar bahías, planear repuestos |
| Clasificación de fuga | ¿Este cliente va a dejar de volver? | Priorizar a quién llamar antes de perderlo |

Ambos se congelan en disco y se exponen como una **API REST**, todo respaldado por pruebas automáticas e integración continua.

**La unidad de predicción es el chasis (VIN), que en esta base equivale al cliente** — no la gama ni el modelo del vehículo. Es una decisión de planteamiento con consecuencias en todo el pipeline, justificada en la [sección 4](#el-sujeto-el-chasis-no-la-gama).

Lo que logra, concretamente:

- **Un ciclo de vida de ML completo y reproducible.** Cualquiera clona el repo, ejecuta un comando y obtiene exactamente los mismos modelos. Sin notebooks, sin pasos manuales, sin "en mi máquina sí funciona".
- **Separación real entre entrenar y servir.** La API no reimplementa ni una línea del preprocesamiento; consume los pipelines congelados.
- **Un contrato de datos explícito en las dos fronteras.** Pandera valida el archivo del ERP y cada etapa del pipeline antes de entrenar; Pydantic rechaza las peticiones inválidas de la API antes de que toquen el modelo.
- **Decisiones de modelado respaldadas por evidencia.** Cada entrenamiento recalcula el diagnóstico, la comparación de tres algoritmos y el contraste global contra segmentado, y deja todo en `models/metricas_entrenamiento.json`.
- **Verificación automática en cada push.** GitHub Actions reconstruye el entorno desde cero, reentrena y corre las pruebas en una máquina limpia.

Lo que **no** logra: decir el día exacto del regreso. La regresión explica el 11,5 % de la variabilidad. La clasificación de fuga sí llega a un nivel utilizable (AUC 0,724), y es la que se recomienda para operar. Ver [sección 6](#6-qué-le-falta-y-qué-mejoraría).

---

## 2. Cómo fluyen los datos

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                    FASE 1 — ENTRENAMIENTO                        │
   │                  (offline, se corre a mano)                      │
   └──────────────────────────────────────────────────────────────────┘

   data/Base de datos Limpia..csv
   34.935 filas de sub-trabajos
            │
            ▼
   ┌─────────────────────┐  cargar_datos()
   │  src/data.py        │  · detecta separador y encoding
   │  CARGA              │
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐  src/esquema.py — CONTRATO DE DATOS
   │  VALIDACIÓN         │  · columnas presentes · fechas dd/mm/aaaa
   │  (se detiene aquí   │  · tipos coercibles   · volumen mínimo
   │   si algo cambió)   │  Vuelve a validar tras cada etapa de abajo
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐  limpiar_calidad()
   │  CALIDAD            │  · fuera facturas anuladas (13,5 %)
   │                     │  · fuera líneas de BODEGAJE
   └─────────────────────┘  · año centinela 1997 → nulo
            │              34.935 → 29.319 líneas
            ▼
   ┌─────────────────────┐  consolidar_visitas_por_ot()
   │  CONSOLIDACIÓN      │  agrupa por VIN + OT
   └─────────────────────┘  29.319 líneas → 23.430 visitas (9.289 VIN)
            │
            ▼
   ┌─────────────────────┐  agregar_features_historia()
   │  HISTORIA POR VIN   │  ritmo previo, km/día, regla 365 d / 10.000 km
   └─────────────────────┘
            │
            ├───────────────┬───────────────┬──────────────┐
            ▼               ▼               ▼              ▼
   ┌────────────────┐ ┌──────────────┐ ┌──────────┐ ┌─────────────┐
   │ DATASET        │ │ DATASET FUGA │ │ src/     │ │ CATÁLOGO    │
   │ REGRESIÓN      │ │ ¿vuelve en   │ │ diagnos- │ │ POR VIN     │
   │ 8.938 filas    │ │  365 días?   │ │ tico.py  │ │ 9.289 fichas│
   │ 2.450 veh.     │ │ 11.189 filas │ │ 3 medidas│ │ + historia  │
   │                │ │ 3.309 veh.   │ │ de ritmo │ │ ya calculada│
   └────────────────┘ └──────────────┘ └──────────┘ └─────────────┘
            │               │                              │
            ▼               ▼                              │
   ┌─────────────────────────────────────┐                 │
   │  src/features.py  ColumnTransformer │                 │
   │  imputar → escalar → codificar      │                 │
   │  19 columnas → 63 columnas          │                 │
   └─────────────────────────────────────┘                 │
            │               │                              │
            ▼               ▼                              │
   ┌────────────────┐ ┌──────────────────────────────────┐ │
   │ RandomForest   │ │ src/modelos.py  3 CLASIFICADORES │ │
   │ Regressor      │ │ RandomForest / HistGB / ExtraTr. │ │
   │                │ │ global + uno por origen          │ │
   └────────────────┘ └──────────────────────────────────┘ │
            │               │                              │
            ▼               ▼                              │
   ┌─────────────────────────────────────┐                 │
   │  src/evaluate.py                    │                 │
   │  MAE/RMSE/R²  ·  AUC/PR-AUC/FP/FN   │                 │
   │  validación GroupKFold(5) por VIN   │                 │
   └─────────────────────────────────────┘                 │
            │               │                              ▼
            ▼               ▼               models/catalogo_vehiculos.joblib
   modelo_retencion    modelo_fuga.joblib
      .joblib          modelos_fuga_segmentados.joblib
                       metricas_entrenamiento.json


   ┌──────────────────────────────────────────────────────────────────┐
   │                      FASE 2 — INFERENCIA                         │
   │                 (online, mientras la API vive)                   │
   └──────────────────────────────────────────────────────────────────┘

   Cliente (asesor / CRM / Swagger)
            │  HTTP
            ▼
   ┌─────────────────────┐
   │  main.py            │  carga los .joblib UNA vez al arrancar
   │  API FastAPI        │
   └─────────────────────┘
            │
            ├── POST /predict ──────► Pydantic valida (422 si no cumple)
            │                              │
            │                              ▼
            │                    construir_fila_prediccion()
            │                    mismas fórmulas del entrenamiento
            │                              │
            ├── GET /predict/vin/{vin} ──► busca en catálogo (404 si no está)
            │                              │
            │                              ▼
            │                    ¿tiene 2+ ingresos?
            │                      sí → copia su vector del catálogo
            │                      no → regla 365 d / 10.000 km + advertencia
            │                              │
            ├── GET /fuga/vin/{vin} ─────► probabilidad de que NO vuelva
            │                              (409 si el chasis no tiene historia)
            │                              │
            ▼                              ▼
   ┌──────────────────────────────────────────────┐
   │  pipeline.predict(DataFrame de 1 fila)       │
   │  el MISMO preprocesamiento del entrenamiento │
   └──────────────────────────────────────────────┘
            │
            ▼
   JSON: { "prediccion_dias_retorno": 28.2, "origen_estimacion": "modelo",
           "regla_preventiva_dias": 62.2,
           "riesgo_de_fuga": { "probabilidad_fuga": 0.51, "alerta": true } }
```

**La clave del diagrama:** el bloque de preprocesamiento aparece una sola vez, dentro del `.joblib`. Las dos fases lo comparten físicamente. Eso es lo que elimina el *training/serving skew*.

---

## 3. Qué hace cada archivo

```
Trabajo de Dataops/
│
├── data/
│   └── Base de datos Limpia..csv   Materia prima: 34.935 sub-trabajos históricos
│
├── src/                            Capa de entrenamiento (offline)
│   ├── config.py                   Constantes: rutas, semilla, nombres de columnas
│   ├── data.py                     Carga, calidad, consolidación, historia, etiquetas
│   ├── esquema.py                  Contrato de datos: valida forma y volumen
│   ├── diagnostico.py              Las 3 medidas de ritmo de ingreso
│   ├── features.py                 ColumnTransformer: imputación, escalado, codificación
│   ├── modelos.py                  Catálogo de estimadores a comparar
│   ├── evaluate.py                 Métricas de regresión y de clasificación
│   └── train.py                    Orquestador: diagnostica, compara, entrena y serializa
│
├── models/                         Artefactos congelados (salida de train.py)
│   ├── modelo_retencion.joblib     Regresión: ¿en cuántos días vuelve?
│   ├── modelo_fuga.joblib          Clasificación: ¿va a dejar de volver?
│   ├── modelos_fuga_segmentados.joblib  Un clasificador por origen del vehículo
│   ├── catalogo_vehiculos.joblib   Ficha + historia por VIN, para búsqueda por chasis
│   └── metricas_entrenamiento.json Evidencia citable: diagnóstico, CV y comparaciones
│
├── main.py                         API REST con FastAPI (capa online)
│
├── tests/
│   └── test_api.py                 37 pruebas de API, datos, contrato y métricas
│
├── .github/workflows/ci.yml        CI: ruff → train → pytest
├── pyproject.toml                  Configuración de pytest y ruff
└── requirements.txt                Dependencias del entorno
```

### Cómo se conectan

**`config.py` es la fuente única de verdad.** No importa a nadie; todos lo importan a él. Si mañana el taller renombra una columna, se corrige en un solo lugar y el resto queda corregido.

```
                    ┌──────────────┐
                    │  config.py   │  ← no importa a nadie
                    └──────────────┘
                           ▲
     ┌─────────┬───────────┼───────────┬─────────┬─────────┐
     │         │           │           │         │         │
  data.py  features.py  evaluate.py  modelos.py  diagnostico.py  main.py
     ▲         ▲           ▲           ▲         ▲
     └─────────┴───────────┴───────────┴────┬────┘
                                            │
                                        train.py    ← el único que los llama a todos
```

- **`data.py`** solo sabe de pandas. No conoce scikit-learn. Su trabajo termina cuando entrega un DataFrame limpio.
- **`esquema.py`** es el portero. No transforma nada: solo deja pasar o detiene. Está separado de `data.py` a propósito, para que las reglas del contrato se puedan leer sin mezclarse con la lógica de limpieza.
- **`diagnostico.py`** no entrena nada. Solo mide, y devuelve un diccionario con las tres medidas de ritmo. Está aislado porque responde una pregunta de negocio que existe con o sin modelo.
- **`features.py`** solo sabe de scikit-learn. No lee archivos ni conoce rutas. Devuelve un `ColumnTransformer` sin ajustar.
- **`modelos.py`** es el catálogo de estimadores. Agregar o quitar un candidato a comparar es cambiar una entrada de un diccionario, sin tocar `train.py` ni la lógica de evaluación.
- **`evaluate.py`** no conoce el modelo. Recibe dos vectores de números y devuelve un diccionario. Se puede reutilizar con cualquier otro modelo.
- **`train.py`** es el director de orquesta: llama a los demás en orden y congela el resultado. No define lógica propia.
- **`main.py`** no importa `train.py`. Solo `config.py`, `data.py` (para las fórmulas de servicio) y `features.py`. **Esa es la frontera entre entrenar y servir**, y está dibujada a propósito.

---

## 4. Por qué se tomó cada decisión

### El sujeto: el chasis, no la gama

La pregunta del negocio es *cuándo vuelve este cliente*, no *cómo se comporta un Tucson*. La unidad de predicción es el **chasis (VIN)**, que en esta base equivale al cliente: la relación vehículo–cliente es casi 1:1, y `Cliente Cargo` es el *pagador de la línea*, no el dueño (en el 7,5 % de las órdenes conviven el cliente y una aseguradora). El VIN es llave de búsqueda y clave de agrupación, pero **nunca entra al modelo**: codificar 9.289 chasis produciría un modelo que memoriza cada carro en vez de aprender el patrón.

La **gama tampoco entra**, y esa sí es una renuncia medible. En esta base está confundida con la antigüedad del parque — Kona promedia 1,4 años y Grand I10 HB 9,8 — así que funcionaría como atajo hacia la edad del carro en lugar de describir la conducta de quien lo trae:

| Variables | AUC | PR-AUC |
|---|---:|---:|
| Con gama y antigüedad | 0,7366 | 0,4957 |
| **Sin gama** (configuración actual) | **0,7239** | **0,4773** |
| Sin gama ni antigüedad | 0,7201 | 0,4713 |
| Solo historia del chasis | 0,7036 | 0,4421 |

Poco más de un punto de AUC a cambio de robustez: una gama nunca vista deja de ser un problema. La antigüedad se conserva porque su costo adicional es marginal y describe una realidad mecánica, no una etiqueta comercial. La gama sigue viajando en el catálogo **solo como dato de pantalla**.

Tampoco existe un "cada cuánto vuelve la gente" global: `evaluate.diagnostico_intervalos()` separa tres poblaciones porque describen tres momentos del ciclo de vida y promediarlas daría una cifra que no le sirve a ninguna.

| | Vehículos | Intervalos | Días (mediana) | Días (p25–p75) | Km (mediana) | Vuelve ≤ 365 d |
|---|---:|---:|---:|---:|---:|---:|
| **1. Nuevo → primer ingreso** | 1.581 | 1.581 | **165** | 52 – 305 | **5.072** | 82,2 % |
| **2. Vendido por nosotros, ya con historia** | 759 | 1.677 | **139** | 58 – 233 | **5.086** | 90,4 % |
| **3. Externo, 2ª visita en adelante** | 1.886 | 7.759 | **79** | 31 – 184 | **5.577** | 92,4 % |

El externo vuelve más seguido que el propio (79 días contra 139): llega con un parque más viejo y más exigente en correctivo, mientras que el vehículo nuevo solo necesita el preventivo. Son dos negocios dentro del mismo taller, y es la evidencia que sostiene la [segmentación por origen](#la-elección-del-algoritmo-y-del-punto-de-operación).

### El contrato de datos: qué se valida antes de entrenar

Pandera valida el CSV del ERP y cada etapa posterior. Tres comprobaciones no son obvias y cubren fallos que de otro modo no avisarían:

**1. La forma del texto de la fecha, no solo que parsee.** La intuición dice que si el ERP cambia el formato, el parseo fallará y se notará. No es así: `pd.to_datetime` resuelve `2017-06-21`, `06/21/2017`, `21-Jun-2017` y `20170621` **sin generar un solo nulo**. Un cambio de formato no produce datos faltantes sino datos **equivocados**: `06/07/2017` en formato estadounidense significa 7 de junio y el código lo lee como 6 de julio. Como el proyecto entero mide **intervalos entre fechas**, todos quedarían mal con el CI en verde. Por eso se verifica la forma (`dd/mm/aaaa`, que descarta el ISO y el mes en letras) **y** la semántica: el segundo número debe ser un mes válido. `06/21/2017` cumple la forma pero su mes es 21, y sobre decenas de miles de filas siempre hay días mayores que 12, así que el cambio salta de inmediato.

**2. El volumen, no solo los tipos.** Un DataFrame de 40 filas cumple exactamente el mismo esquema que uno de 30.000. Es lo que produce la limpieza silenciosa: `errors="coerce"` convierte en nulo lo que no entiende, `dropna` bota la fila, y el pipeline sigue con lo que quede. `verificar_volumen()` corre tras cada etapa con mínimos declarados en `config.py`, holgados respecto de lo que produce la base actual (34.935 / 23.430 / 8.938 / 9.289 contra 20.000 / 12.000 / 3.000 / 4.000): la idea es detectar un derrumbe, no una variación normal.

**3. El catálogo.** Cubre el fallo más difícil de ver: agregar una variable al entrenamiento y olvidar producirla en la ficha. El modelo entrenaría bien, las métricas saldrían bien, el CI verde — y `/predict/vin` reventaría en producción. `validar_catalogo()` lo convierte en un fallo de build, que es donde debe doler.

### De 34.935 líneas a dos datasets

**Primero limpiar, después consolidar**, porque tres defectos viven a nivel de línea y hay que quitarlos mientras aún se distinguen:

| Defecto | Registros | Por qué importa |
|---|---:|---|
| Facturas **anuladas** | 4.715 (13,5 %) | La OT no fue un ingreso real: contarla inventa una visita y acorta el intervalo entre las verdaderas |
| Líneas de **BODEGAJE** | 922 | Es cobro de parqueo, no mantenimiento |
| Año **1997** (centinela) | 667 | Relleno del ERP cuando el campo viene vacío. Se anula solo el año, no la fila: la visita sí ocurrió |

**Consolidar por OT** convierte el CSV de sub-trabajos en visitas reales. Sin eso, un vehículo que entró una vez pero necesitó cinco reparaciones pesaría cinco veces más, y el modelo aprendería a predecir *complejidad de reparación*, no *retorno*. Cuando las líneas de una misma OT discrepan en `Tipo Cargo` o `Tipo de Trabajo` (1.971 y 1.510 órdenes), se resuelve por **moda** con desempate declarado en `config.py`; tomar el primer valor dejaría que el orden de exportación del ERP decidiera el tipo de la visita.

> 29.319 líneas limpias → **23.430 visitas reales** de **9.289 vehículos únicos**

**El objetivo no existe en el CSV**: se construye con `shift(-1)`, trayendo la fecha de la visita siguiente a la fila actual. Se mide desde el **cierre** y no desde la entrada, porque el reloj de la retención empieza cuando el cliente se lleva el carro.

De ahí sale el embudo de la regresión:

| Paso | Filas | Se pierden | Por qué |
|---|---:|---:|---|
| Visitas consolidadas | 23.430 | — | — |
| Tienen visita siguiente | 14.141 | **9.289** | Es la última visita: no hay respuesta que aprender |
| Retorno ≥ 0 días | 13.751 | 390 | Fechas invertidas (digitación) |
| Antigüedad ≥ 0 años | 12.381 | 1.370 | Año de modelo posterior a la visita (digitación) |
| Kilometraje < 500.000 | 12.100 | 281 | Valores tipo `123.456.789` |
| No es el primer ingreso | **8.938** | **3.162** | En la primera visita no hay historia previa que mirar |

Los filtros de kilometraje no son cosmética: un solo valor absurdo distorsiona la media y la desviación del `StandardScaler`, y con eso toda la escala de la variable.

El último paso define la naturaleza del dataset. **La historia del propio chasis es la variable central**: sin ella, un taxi que entra cada dos semanas y un particular que entra una vez al año le llegan idénticos al estimador, y lo mejor que puede hacer es predecir el promedio. Por eso se descartan los vehículos de un solo ingreso — no tienen ningún intervalo observado — aunque en producción sigan siendo consultables con la regla de fábrica.

Las importancias confirman el planteamiento: las cinco variables que más pesan son de historia o de uso, y ninguna domina sola.

| Variable | Importancia |
|---|---:|
| Días desde la visita anterior | 0,093 |
| Ausencia más larga conocida del vehículo | 0,092 |
| Ritmo mediano previo del propio chasis | 0,088 |
| Kilometraje | 0,071 |
| Ciclos de 10.000 km acumulados | 0,070 |

La **regla de 365 días / 10.000 km** entra también como variable (`Regla_Preventiva_Dias`, `Dias_Para_10k_Km`, `Ciclos_10k_Acumulados`): un vehículo de 160 km diarios alcanza los 10.000 km en 62 días, no en un año, y el modelo necesita saber cuál de los dos umbrales manda en cada caso. Y entra `Dias_En_Taller` (`F. Cierre − F. Entrada`), el único dato de la **experiencia** del cliente que registra el ERP — la mitad de las visitas se resuelven el mismo día y el 10 % pasa de 34. No hay fuga de información: el target se cuenta desde `F. Cierre`, así que la demora ya ocurrió cuando arranca el reloj.

### Por qué dos modelos y no uno

"En cuántos días vuelve" y "va a volver o no" parecen la misma pregunta. La segunda es mucho más contestable con estos datos (R² = +0,115 contra AUC = 0,724): predecir el día exacto es una meta mal planteada cuando el intervalo tiene un coeficiente de variación del 143 %. Y la clasificación se puede medir con AUC y con falsos positivos, que es lo que permite elegir el punto de operación con criterio de negocio. Un MAE de 100 días no le dice a nadie a quién llamar.

La clasificación además **entrena con más casos**. La regresión solo aprende de visitas con regreso observado, así que descarta la última visita de cada vehículo — exactamente los clientes que interesa detectar. La etiqueta binaria sí los alcanza:

| Caso | Cómo se etiqueta |
|---|---|
| Hay visita siguiente | Fuga = el intervalo superó los 365 días |
| Última visita y ya pasaron más de 365 días | **Fuga = sí.** Es un dato duro, no una suposición |
| Última visita y aún no pasa el plazo | **Censurado.** Se descarta: sigue dentro de plazo |

| | Filas | Vehículos |
|---|---:|---:|
| Regresión | 8.938 | 2.450 |
| **Fuga** | **11.189** | **3.309** |
| … de ellas, últimas visitas ya vencidas | **5.270** | — |
| Censuradas (se descartan) | 4.019 | — |

Contar las censuradas como fuga sería inventar un abandono que no ha ocurrido: en el lenguaje del análisis de supervivencia son observaciones **censuradas por la derecha**, y el tratamiento correcto es excluirlas.

### La elección del algoritmo y del punto de operación

Se comparan tres ensambles elegidos para que difieran de verdad — RandomForest (bagging con cortes óptimos), HistGradientBoosting (boosting secuencial) y ExtraTrees (bagging con cortes aleatorios, que difiere de RandomForest en exactamente una decisión de diseño). Los lineales quedan fuera por decisión del negocio; además el árbol no asume linealidad, captura interacciones sin que se las declaren y tolera los valores extremos que abundan en una base de taller. `random_state=42` fijo.

**Los tres quedaron prácticamente empatados**: AUC 0,7239 / 0,7235 / 0,7164, con 0,0075 entre el primero y el tercero — menos que la variación entre particiones. La conclusión es que **el algoritmo no es el cuello de botella**; lo que limita el resultado son las variables disponibles.

Solo el 26,5 % de las visitas son fuga, así que sin balanceo un modelo que responda "nadie se va" acierta el 73,5 % y es inútil. `HistGradientBoosting` no acepta `class_weight` y recibe el balanceo como `sample_weight` en el `.fit()`: sin esa corrección su recall cae a 0,285 contra 0,570, y la comparación mediría el desbalance en lugar del algoritmo.

**El ganador se elige por AUC y no por F1** porque son dos decisiones distintas: qué algoritmo *ordena* mejor el riesgo se responde con AUC, que no depende del umbral; dónde poner el corte de alerta es del negocio. Y es del negocio porque los dos errores no cuestan lo mismo: un **falso positivo** es una llamada innecesaria (barato), un **falso negativo** es un cliente perdido sin que nadie lo intentara (caro). Promediarlos en una cifra oculta justamente la decisión, así que `evaluate.curva_punto_operacion()` imprime en cada entrenamiento qué cuesta cada umbral:

| Umbral | Precisión | Recall | Alertas emitidas | Llamadas de más (FP) | Clientes perdidos sin alerta (FN) |
|---:|---:|---:|---:|---:|---:|
| 0,30 | 0,329 | **0,905** | 8.142 | 5.462 | **282** |
| 0,40 | 0,377 | 0,766 | 6.022 | 3.754 | 694 |
| **0,50** (actual) | 0,449 | 0,570 | 3.761 | 2.074 | 1.275 |
| 0,60 | 0,512 | 0,360 | 2.083 | 1.016 | 1.895 |
| 0,70 | 0,589 | 0,194 | 978 | 402 | 2.386 |

Bajar a 0,30 detecta el 90,5 % de las fugas. Si una llamada cuesta minutos de un asesor y un cliente perdido cuesta todos sus mantenimientos futuros, la aritmética favorece llamar de más. Se deja en `config.UMBRAL_DECISION_FUGA` para que lo decida el negocio.

Por último se contrasta un modelo global contra uno por **origen del vehículo**:

| RandomForest | AUC | Precisión | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Global, sobre **vendidos por nosotros** | 0,7153 | 0,446 | 0,583 | 0,505 | 357 | 205 |
| **Segmentado, sobre vendidos por nosotros** | 0,7132 | **0,462** | **0,596** | **0,520** | **341** | **199** |
| Global, sobre **externos** | 0,7257 | 0,449 | 0,567 | 0,501 | 1.717 | 1.070 |
| Segmentado, sobre externos | 0,7250 | 0,444 | 0,567 | 0,498 | 1.751 | 1.069 |

En vehículos propios el segmentado comete **menos errores de los dos tipos**. El AUC es casi idéntico — ambos ordenan igual de bien — pero el segmentado está **mejor calibrado** para esa población: sus probabilidades caen donde el umbral las aprovecha mejor. En externos es un empate. Se congelan los tres modelos y la API devuelve las dos lecturas; cuando discrepan, el asesor ve una señal real.

> **Cómo leer esa tabla.** Los AUC de segmentos distintos **no son comparables entre sí**, porque cada uno se mide contra la dificultad de su propia población. Lo comparable es global contra segmentado *dentro de la misma fila*, que es como está armada.

### Cómo se mide para que las métricas no mientan

Dos disciplinas, y sin ellas todos los números de arriba serían ficticios.

**Las variables de historia se calculan con `expanding()`**, que acumula solo lo ya ocurrido fila por fila. El intervalo que se predice va de la visita *i* a la *i+1*, y las pistas llegan como máximo hasta la *i*. Si el ritmo previo incluyera el intervalo que se quiere predecir, el modelo estaría leyendo la respuesta. La prueba `test_historia_no_filtra_el_futuro` lo verifica en cada corrida.

**La evaluación se agrupa por VIN.** Un mismo vehículo aporta varias visitas y todas comparten su ritmo; con un `train_test_split` aleatorio la visita 3 de un carro quedaría en entrenamiento y la 4 en prueba, y el modelo habría visto en la práctica la respuesta que se le va a preguntar. `GroupShuffleSplit` y `GroupKFold` dejan cada vehículo **entero** de un lado o del otro, y se valida cinco veces para reportar media y desviación en vez de un número que depende de la suerte de la partición.

### De entrenar a servir

**Se congela el Pipeline completo, no solo el modelo.** Es la decisión de diseño más importante del proyecto. El `.joblib` guarda todo lo que se aprendió de los datos durante el `.fit()` — las medianas de los 15 campos numéricos, la media y desviación de cada uno para el `StandardScaler`, la moda de las categóricas, las 63 columnas tras el OneHot — y esos números **no están escritos en ninguna parte del código fuente**. Si la API recalculara su propia media y desviación usaría los datos que le llegan, no las 8.938 filas del entrenamiento: un `Kms.` de 45.000 se convertiría en un número distinto al que el modelo aprendió a interpretar, y las predicciones saldrían mal **sin arrojar ningún error**. Ese fallo silencioso se llama *training/serving skew*, y congelar el pipeline entero lo elimina por construcción.

Por el mismo motivo, las fórmulas de servicio (`data.construir_fila_prediccion()`) viven en el mismo archivo que las del entrenamiento: que ambos calculen distinto una misma variable es el error más caro y más difícil de detectar de un proyecto de ML.

Tres decisiones menores completan la frontera:

- **La API pide el año del modelo, no la antigüedad**, porque es el dato que el asesor tiene a la vista en la tarjeta de propiedad. La traducción a antigüedad es interna y replica el cálculo del entrenamiento. El contrato se diseña alrededor de lo que el usuario **tiene**, no de lo que el modelo **necesita**.
- **El catálogo se construye al entrenar**, no al arrancar la API: evita reprocesar 34.935 filas en cada reinicio y garantiza que la ficha servida sea la misma que vio el modelo.
- **La regla preventiva viaja en la respuesta** junto a la predicción. Aplicarla tal cual da un MAE de 129,5 días, peor que predecir siempre la mediana (107,9): el plan describe cuándo el vehículo *debería* volver, no cuándo vuelve. La brecha entre ambas es información para el asesor.

**El CI corre `ruff` → `train` → `pytest`**, en ese orden: ruff falla en segundos y es lo más barato; el entrenamiento va antes que las pruebas porque estas cargan los `.joblib` y en una máquina limpia hay que generarlos. Reentrenar en cada push es lo que demuestra que el pipeline es reproducible de verdad.

---

## 5. Qué recibe y devuelve la API

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/` | Health check: confirma que el servicio vive y que los artefactos cargaron |
| `POST` | `/predict` | Predicción con datos manuales |
| `GET` | `/predict/vin/{vin}` | Predicción por chasis, completando los datos desde el catálogo |
| `GET` | `/fuga/vin/{vin}` | Riesgo de que ese cliente deje de volver |
| `GET` | `/docs` | Swagger UI autogenerado, con el contrato siempre al día |

Los JSON de abajo son la respuesta **completa** de cada endpoint. El [README](README.md#cómo-se-usa) muestra los mismos ejemplos recortados a lo esencial.

### `GET /` — Health check

```json
{
  "mensaje": "API de Retención Predictiva de Automotor.co S.A.S. activa 🚗",
  "modelo_cargado": true,
  "modelo_fuga_cargado": true,
  "algoritmo_fuga": "RandomForest",
  "modelos_fuga_por_origen": ["externos", "vendidos"],
  "catalogo_cargado": true,
  "vehiculos_en_catalogo": 9289,
  "vehiculos_predecibles": 4496,
  "regla_preventiva": "365 días o 10000 km",
  "umbral_fuga_dias": 365
}
```

`vehiculos_predecibles` es el subconjunto con al menos dos ingresos; el resto se puede consultar pero solo recibe la regla de fábrica. Y si los `.joblib` no existen el servidor **no se cae**: arranca con `modelo_cargado: false`, para que el health check siga respondiendo y el problema se pueda diagnosticar en vez de quedarse a oscuras.

### `POST /predict`

**Entrada**, validada por Pydantic con la clase `SolicitudVehiculo`:

```json
{
  "Tipo Cargo": "Cliente",
  "Tipo de Trabajo": "MECANICA",
  "Kms.": 45000.0,
  "Año Modelo": 2022,
  "Es_Vehiculo_Vendido": true,
  "Visitas_Previas": 3,
  "Dias_Desde_Visita_Anterior": 180.0,
  "Kms_Visita_Anterior": 32000.0
}
```

| Campo | Tipo | Obligatorio | Restricción |
|---|---|---|---|
| `Tipo Cargo` | string | Sí | — |
| `Tipo de Trabajo` | string | No (`"MECANICA"` por defecto) | — |
| `Kms.` | float | Sí | — |
| `Año Modelo` | int | Sí | entre 1990 y el año entrante |
| `Es_Vehiculo_Vendido` | bool | Sí | — |
| `Visitas_Previas` | int | **Sí** | ≥ 1 |
| `Dias_Desde_Visita_Anterior` | float | **Sí** | ≥ 0 |
| `Kms_Visita_Anterior` | float | No | ≥ 0 |
| `Ritmo_Mediano_Previo` | float | No | ≥ 0 — si falta, se usa el último intervalo |

Los cuatro últimos son la **historia del vehículo**, y los dos primeros de ellos son obligatorios a propósito: aceptar la petición sin ellos devolvería la mediana de la población disfrazada de predicción individual. El contrato pide exactamente lo que el modelo consume y solo eso; un campo de más se ignora, y `test_modelo_solo_consume_columnas_declaradas` lo deja fijado. Las variables derivadas (km por día, días para los 10.000 km, regla preventiva) las calcula `data.construir_fila_prediccion()`, que vive en el mismo archivo que las fórmulas del entrenamiento.

**Salida:**

```json
{
  "prediccion_dias_retorno": 196.6,
  "unidad": "días",
  "regla_preventiva_dias": 138.5,
  "vehiculo_atrasado_frente_a_regla": true,
  "riesgo_de_fuga": {
    "algoritmo": "RandomForest",
    "probabilidad_fuga": 0.5762,
    "alerta": true,
    "segmento": "vendidos",
    "probabilidad_fuga_modelo_del_segmento": 0.5524
  }
}
```

### `GET /predict/vin/{vin}`

Sin cuerpo. El VIN viaja en la URL y se normaliza (mayúsculas, sin espacios) antes de buscar.

```json
{
  "vin": "ADM130850",
  "datos_encontrados": {
    "gama": "I10",
    "anio_modelo": 2013,
    "antiguedad_calculada": 7.0,
    "ultimo_kilometraje": 487636.0,
    "ultima_visita": "2020-02-18",
    "ultima_entrega": "2020-02-18",
    "vendido_por_nosotros": false,
    "total_visitas": 55,
    "ritmo_habitual_dias": 14.5,
    "km_por_dia": 160.7
  },
  "prediccion_dias_retorno": 28.2,
  "unidad": "días",
  "origen_estimacion": "modelo",
  "regla_preventiva_dias": 62.2,
  "riesgo_de_fuga": { "probabilidad_fuga": 0.5127, "alerta": true, "segmento": "externos" }
}
```

Si el chasis tiene **un solo ingreso** la respuesta cambia de naturaleza y lo declara, en vez de devolver un número con la misma cara que una predicción real:

```json
{
  "prediccion_dias_retorno": 365.0,
  "origen_estimacion": "regla_preventiva",
  "regla_preventiva_dias": 365.0,
  "advertencia": "El VIN '0F6S00876' tiene un solo ingreso registrado: no hay intervalo observado del que estimar su ritmo. Se devuelve la regla preventiva (365 días o 10000 km), no una predicción del modelo."
}
```

El historial está incompleto para algunos chasis (hay VIN sin año de modelo o sin kilometraje). En esos casos el campo viaja como `null` y el imputador del pipeline rellena con la mediana, igual que en el entrenamiento. No se inventa un `0`, porque eso le diría al modelo que es un carro nuevo.

### `GET /fuga/vin/{vin}`

Está separado de `/predict/vin` porque se usa distinto: aquel agenda a un cliente concreto, este alimenta la **lista de a quiénes hay que llamar**.

```json
{
  "vin": "ADM130850",
  "riesgo_de_fuga": {
    "algoritmo": "RandomForest",
    "umbral_dias": 365,
    "umbral_decision": 0.5,
    "segmento": "externos",
    "probabilidad_fuga": 0.5127,
    "alerta": true,
    "probabilidad_fuga_modelo_del_segmento": 0.525,
    "alerta_modelo_del_segmento": true
  },
  "historia_que_lo_sustenta": {
    "total_visitas": 55,
    "ritmo_habitual_dias": 14.5,
    "dias_desde_visita_anterior": 24.0,
    "km_por_dia": 160.7,
    "ultima_visita": "2020-02-18",
    "vendido_por_nosotros": false
  }
}
```

Un chasis **sin historia** responde **`409 Conflict`**, no `200` con un número inventado: sin un segundo ingreso no hay ritmo contra el cual comparar la ausencia, y el riesgo simplemente no es calculable.

### Qué significa cada campo de la respuesta

| Campo | Qué es |
|---|---|
| `prediccion_dias_retorno` | Días estimados hasta el próximo ingreso |
| `origen_estimacion` | `modelo` o `regla_preventiva`. Permite a un CRM distinguir programáticamente una predicción de un valor por defecto, sin interpretar texto |
| `regla_preventiva_dias` | Lo que dice el plan de fábrica para ese vehículo. Viaja siempre al lado de la predicción, como referencia contra la cual juzgar el número en vez de pedirle fe ciega al modelo |
| `vehiculo_atrasado_frente_a_regla` | Si ya se pasó el plazo de la regla |
| `riesgo_de_fuga.probabilidad_fuga` · `alerta` | Probabilidad del modelo global y si supera `UMBRAL_DECISION_FUGA` |
| `riesgo_de_fuga.probabilidad_fuga_modelo_del_segmento` | La misma lectura según el modelo entrenado solo con vehículos del mismo origen. Cuando las dos discrepan, el asesor está viendo una señal real |
| `datos_encontrados` | Solo en `/predict/vin`: ficha del vehículo, para que el asesor verifique que se buscó el carro correcto antes de confiar en la cifra |
| `historia_que_lo_sustenta` | Solo en `/fuga/vin`: visitas, ritmo habitual, días de ausencia. Un asesor no puede llamar con una probabilidad sin contexto — necesita poder decir *"lleva 24 días sin venir y normalmente viene cada 14"* |
| `advertencia` | Aparece cuando el chasis tiene un solo ingreso y se devolvió la regla de fábrica |

### Códigos de error

| Código | Cuándo ocurre |
|---|---|
| `422` | El JSON no cumple el contrato: falta un campo, llega texto donde va número, o el año está fuera de rango |
| `404` | El VIN no tiene historial en el taller |
| `409` | El VIN existe pero tiene un solo ingreso: su riesgo de fuga no es calculable |
| `500` | Falta algún `.joblib` del modelo o del catálogo |
| `400` | Error inesperado durante la inferencia |

---

## 6. Qué le falta y qué mejoraría

### Dónde está hoy

| | Métrica | Resultado | Lectura |
|---|---|---:|---|
| **Regresión** | MAE | 100,4 días | Desfase promedio entre lo predicho y lo real |
| | RMSE | 167,5 días | Muy por encima del MAE → hay casos con errores extremos |
| | R² | **+0,115** | Supera al promedio, pero por poco |
| **Clasificación** | AUC | **0,724** | Ordena bien a los clientes por riesgo de fuga |
| | PR-AUC | 0,477 | Contra una tasa base de 0,265 |
| | Recall @0,50 | 0,570 | Detecta algo más de la mitad de las fugas |
| | Recall @0,30 | **0,905** | Bajando el umbral detecta 9 de cada 10 |

**La conclusión principal es que la clasificación funciona y la regresión apenas.** Un AUC de 0,724 es utilizable para priorizar llamadas; un R² de 0,115 no sirve para decidir sobre un cliente concreto.

Contra las referencias sin aprendizaje, la regresión gana en las tres métricas — pero el margen es estrecho:

| Referencia | MAE | RMSE | R² |
|---|---:|---:|---:|
| Regla preventiva 365 d / 10.000 km | 129,52 | 196,43 | −0,216 |
| Ritmo mediano del propio vehículo | 105,02 | 196,32 | −0,214 |
| Predecir siempre la mediana | 107,88 | 186,21 | −0,093 |
| **Random Forest** | **100,40** | **167,51** | **+0,115** |

Y una observación que vale para el negocio: **aplicar la regla de fábrica tal cual es la peor de las referencias** (MAE de 129,5 días). El plan describe cuándo el vehículo *debería* volver, no cuándo vuelve.

### El techo que queda

El límite no es el algoritmo — los tres clasificadores quedaron dentro de 0,008 de AUC entre sí. Es la información disponible:

1. **Lo que hace volver a un cliente no está en los datos.** Si quedó conforme, qué pagó, si le cumplieron la fecha de entrega. El ERP registra la transacción, no la experiencia. La demora en taller es lo único que se pudo rescatar de esa dimensión.
2. **El objetivo es muy disperso.** Coeficiente de variación del 143 %: media de 144 días contra mediana de 77, con un máximo de 2.936. Hay al menos dos poblaciones mezcladas — quien vuelve al mantenimiento de rutina y quien desaparece por años.
3. **Falta `edad_mora`**, el atraso contra el plan de mantenimiento por gama. Es reconstruible si la empresa entrega ese plan, y sería la variable más directa del problema.
4. **No se observa lo que pasa fuera de este concesionario.** Un vehículo que se mantiene al día en otro taller de la red Hyundai es, para esta base, indistinguible de uno abandonado.
5. **Los 2.218 vehículos entregados que nunca volvieron quedan fuera del alcance.** No tienen un segundo ingreso del cual estimar ritmo, así que ningún modelo basado en historia de taller puede alcanzarlos. Necesitan un enfoque distinto, alimentado con datos de la venta.

### Mejora 1 — Investigar la causa real del abandono

Esta es la mejora de mayor impacto, y es de **negocio antes que de modelado**.

Hoy predecimos *cuándo* vuelve un cliente y *si* se va, pero nunca preguntamos *por qué* se fue. Los datos que faltan son justamente los que explican el abandono:

- **Satisfacción con el servicio** — ¿quedó conforme con la reparación? ¿se la tuvieron que repetir?
- **Precio** — ¿cuánto facturó? ¿le pareció caro frente al taller de la esquina?
- **Cumplimiento de la fecha de entrega** — la demora está, pero no si se cumplió lo prometido.
- **Trato recibido** — ¿qué asesor lo atendió? ¿hubo reclamos?
- **Garantía** — ¿se le venció? Porque muchos clientes solo vuelven mientras la tienen.

Incorporar estas variables atacaría el techo por la raíz, en vez de seguir exprimiendo las que ya hay.

### Mejora 2 — Análisis de supervivencia

Las 4.019 visitas **censuradas** hoy se descartan. El análisis de supervivencia (Kaplan-Meier, Cox) está diseñado exactamente para ese caso: datos donde el evento aún no ocurrió pero la observación sigue siendo informativa. Aprovecharlas en vez de botarlas es la vía natural de mejora sobre el clasificador actual.

### Mejora 3 — Investigar los errores de digitación en el origen

Los 1.760 registros descartados por fechas y años invertidos no son ruido aleatorio: son errores sistemáticos de digitación en el sistema del taller. Hoy el pipeline los bota y sigue, que es lo correcto para entrenar, pero vale la pena investigarlos porque señalan un problema en el proceso de captura que se puede corregir en el origen — y ahí el arreglo vale para todo el concesionario, no solo para este modelo.

Una segunda capa útil sería **reportar** lo descartado en vez de solo descartarlo: un resumen por causa y por período, guardado junto a las métricas, permitiría ver si la digitación está mejorando o empeorando con el tiempo.

### Mejora 4 — Modelado

- **Búsqueda de hiperparámetros** con `GridSearchCV` anidado dentro de la validación por grupos. Los tres modelos usan valores fijos elegidos a mano; dado lo parejos que quedaron, es poco probable que la ganancia sea grande, pero no está medido.
- **XGBoost / LightGBM.** Agregarlos implica sumar dependencias al CI para un contraste que, visto el empate entre los tres algoritmos actuales, probablemente tampoco mueva la aguja.
- **Discretizar el objetivo de la regresión** en rangos (0-3 meses, 3-6, 6+). Con un RMSE de 167 días, predecir el valor exacto es una meta mal planteada; predecir el rango es útil y alcanzable.
- **Intervalo de predicción en vez de un número.** Entregar "vuelve en 196 días" transmite una precisión que el modelo no tiene. Un rango con su nivel de confianza sería más honesto y más accionable.
- **Separar por tipo de trabajo**: el retorno tras una `COLISION` y tras un mantenimiento de `MECANICA` son fenómenos distintos. Se puede contrastar con la misma maquinaria de `comparar_clasificadores()` cambiando la columna de corte.
