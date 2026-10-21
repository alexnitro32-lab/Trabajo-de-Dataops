# Arquitectura del Proyecto

Documento técnico. Si lo que buscas es prender el proyecto, ve al **[README](README.md)**.

---

## 1. Qué hace y qué logra

Este proyecto toma la base transaccional de órdenes de trabajo de un taller automotriz (34.935 registros), la convierte en dos datasets entrenables y ajusta **dos modelos que responden dos preguntas distintas**:

| Modelo | Pregunta | Para qué sirve |
|---|---|---|
| Regresión | ¿En cuántos días vuelve este vehículo? | Agendar, dimensionar bahías, planear repuestos |
| Clasificación de fuga | ¿Este cliente va a dejar de volver? | Priorizar a quién llamar antes de perderlo |

Ambos se congelan en disco y se exponen como una **API REST** con un solo endpoint de predicción: el asesor registra el ingreso del vehículo y recibe cuándo volverá, si se va a ir y **entre qué fechas hay que llamarlo**. La API está **instrumentada** (logs JSON, métricas de tráfico y latencia, `/health`), se empaqueta en una **imagen Docker** y todo está respaldado por pruebas automáticas e integración continua.

**La unidad de predicción es el chasis (VIN), que en esta base equivale al cliente** — no la gama ni el modelo del vehículo. Es una decisión de planteamiento con consecuencias en todo el pipeline, justificada en la [sección 4](#el-sujeto-el-chasis-no-la-gama).

Lo que logra, concretamente:

- **Un ciclo de vida de ML completo y reproducible.** Cualquiera clona el repo, ejecuta un comando y obtiene exactamente los mismos modelos. Sin notebooks, sin pasos manuales, sin "en mi máquina sí funciona".
- **Separación real entre entrenar y servir.** La API no reimplementa ni una línea del preprocesamiento; consume los pipelines congelados.
- **Un contrato de datos explícito en las dos fronteras.** Pandera valida el archivo del ERP y cada etapa del pipeline antes de entrenar; Pydantic rechaza las peticiones inválidas de la API antes de que toquen el modelo.
- **Decisiones de modelado respaldadas por evidencia.** Cada entrenamiento recalcula el diagnóstico, la comparación de tres algoritmos y el contraste global contra segmentado, y deja todo en `models/metricas_entrenamiento.json`.
- **Verificación automática en cada push.** GitHub Actions reconstruye el entorno desde cero, reentrena y corre las pruebas en una máquina limpia.
- **Observable en producción.** Cada evento sale como una línea JSON, cada petición se mide, y `/health` y `/metrics` permiten saber en segundos si el servicio vive, cuánto falla y qué tan rápido responde.

Lo que **no** logra: decir el día exacto del regreso. La regresión explica el 11,5 % de la variabilidad; por eso la API entrega una ventana del ancho de su error medio en vez de un día. La clasificación de fuga sí llega a un nivel utilizable (AUC 0,727), y es la que se recomienda para operar. Ver [sección 6](#6-qué-le-falta-y-qué-mejoraría).

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
   │  HISTORIA POR VIN   │  ritmo previo, km/día, regla del plan por año
   └─────────────────────┘  (2022+: 365 d / 10.000 km · ≤2021: 180 d / 5.000 km)
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
   │                │ │ global (contrastado por origen)  │ │
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
   modelo_retencion    modelo_fuga.joblib   models/historial_visitas.joblib
      .joblib          metricas_entrenamiento.json

   src/actualizar_catalogo.py regenera SOLO el catálogo y el historial desde
   un CSV nuevo, sin reentrenar: lo que cambia a diario son las visitas.


   ┌──────────────────────────────────────────────────────────────────┐
   │                      FASE 2 — INFERENCIA                         │
   │                 (online, mientras la API vive)                   │
   └──────────────────────────────────────────────────────────────────┘

   Asesor (Swagger / CRM): VIN + kilometraje + para qué vino
            │  HTTP
            ▼
   ┌─────────────────────┐
   │  main.py            │  carga los .joblib UNA vez al arrancar
   │  API FastAPI        │  (dentro de la imagen Docker o con uvicorn)
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐  middleware de observabilidad (TODA petición)
   │  src/metrics.py     │  · mide la latencia y cuenta éxito / error
   │  src/logger.py      │  · log JSON de advertencia si pasa de 200 ms
   └─────────────────────┘  · lo acumulado se lee en GET /metrics
            │
            ├── GET /health    ¿vive y cargaron los modelos?
            ├── GET /metrics   tráfico, tasa de error, latencia p50 / p95
            │
            └── POST /predict/vin/{vin}/ingreso
                       │
                       ▼
             Pydantic valida el cuerpo (422 si no cumple)
             busca las visitas del VIN en el historial (404 si no tiene)
             valida fecha y odómetro contra su último ingreso (409 / 422)
                       │
                       ▼
             data.ficha_con_ingreso_nuevo()
             agrega la visita de hoy y recalcula la historia con
             construir_catalogo_vehiculos(): las MISMAS fórmulas del entrenamiento
                       │
                       ▼
   ┌──────────────────────────────────────────────┐
   │  pipeline.predict(DataFrame de 1 fila)       │
   │  modelo_fuga.predict_proba(misma fila)       │
   │  el MISMO preprocesamiento del entrenamiento │
   └──────────────────────────────────────────────┘
                       │
                       ▼
             fecha = lo que ocurra primero entre el modelo y el plan
                     de mantenimiento de su año de modelo
             ventana de llamada = fecha ± MAE del modelo
             alerta de fuga → no esperar a la ventana
                       │
                       ▼
   JSON: { "prediccion_dias_retorno": 180.0,
           "riesgo_de_fuga": { "probabilidad_fuga": 0.78, "alerta": true },
           "plan_de_llamada": { "estado": "llamar_ahora", "mensaje": "…" } }
   + log JSON "prediccion_exitosa" con VIN, días, probabilidad y estado
```

**La clave del diagrama:** el bloque de preprocesamiento aparece una sola vez, dentro del `.joblib`, y la historia del vehículo se calcula con una sola función, `construir_catalogo_vehiculos()`, tanto al entrenar como al servir. Las dos fases las comparten físicamente. Eso es lo que elimina el *training/serving skew*.

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
│   ├── train.py                    Orquestador: diagnostica, compara, entrena y serializa
│   ├── actualizar_catalogo.py      Regenera catálogo e historial desde un CSV nuevo, sin reentrenar
│   ├── logger.py                   Logger JSON de una línea por evento (capa online)
│   └── metrics.py                  Acumulador thread-safe de peticiones, errores y latencias (capa online)
│
├── models/                         Artefactos congelados (salida de train.py)
│   ├── modelo_retencion.joblib     Regresión: ¿en cuántos días vuelve?
│   ├── modelo_fuga.joblib          Clasificación: ¿va a dejar de volver?
│   ├── historial_visitas.joblib    Todas las visitas por VIN: la API recalcula la historia con ellas
│   ├── catalogo_vehiculos.joblib   Ficha validada de la última visita por VIN (la usan las pruebas)
│   └── metricas_entrenamiento.json Evidencia citable: diagnóstico, CV y comparaciones
│
├── main.py                         API REST con FastAPI (capa online)
│
├── tests/
│   └── test_api.py                 48 pruebas de API, observabilidad, datos, contrato y métricas
│
├── .github/workflows/ci.yml        CI: ruff → train → pytest
├── Dockerfile                      Imagen de la API: solo src/, models/ y main.py
├── .dockerignore                   Deja fuera de la imagen datos, pruebas y entorno virtual
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
- **`actualizar_catalogo.py`** es la mitad de `train.py` que cambia a diario: carga, valida y consolida el CSV y congela el catálogo y el historial, sin tocar los modelos.
- **`logger.py`** y **`metrics.py`** solo existen para la capa online. No saben nada del modelo: uno formatea eventos como JSON, el otro cuenta peticiones y calcula percentiles de latencia. `metrics.py` usa un `Lock` porque FastAPI atiende peticiones en varios hilos a la vez.
- **`main.py`** no importa `train.py`. Solo `config.py`, `data.py` (para recalcular la historia con las mismas fórmulas y aplicar el plan de mantenimiento), `features.py` y las dos piezas de observabilidad. **Esa es la frontera entre entrenar y servir**, y está dibujada a propósito.

---

## 4. Por qué se tomó cada decisión

### El sujeto: el chasis, no la gama

La pregunta del negocio es *cuándo vuelve este cliente*, no *cómo se comporta un Tucson*. La unidad de predicción es el **chasis (VIN)**, que en esta base equivale al cliente: la relación vehículo–cliente es casi 1:1, y `Cliente Cargo` es el *pagador de la línea*, no el dueño (en el 7,5 % de las órdenes conviven el cliente y una aseguradora). El VIN es llave de búsqueda y clave de agrupación, pero **nunca entra al modelo**: codificar 9.289 chasis produciría un modelo que memoriza cada carro en vez de aprender el patrón.

La **gama tampoco entra**, y esa sí es una renuncia medible. En esta base está confundida con la antigüedad del parque — Kona promedia 1,4 años y Grand I10 HB 9,8 — así que funcionaría como atajo hacia la edad del carro en lugar de describir la conducta de quien lo trae:

| Variables | AUC | PR-AUC |
|---|---:|---:|
| Con gama y antigüedad | 0,7366 | 0,4957 |
| **Sin gama** (configuración elegida) | **0,7239** | **0,4773** |
| Sin gama ni antigüedad | 0,7201 | 0,4713 |
| Solo historia del chasis | 0,7036 | 0,4421 |

*Comparación hecha antes de introducir el plan de mantenimiento por año; con él, la configuración elegida subió a AUC 0,7274 · PR-AUC 0,4808.*

Poco más de un punto de AUC a cambio de robustez: una gama nunca vista deja de ser un problema. La antigüedad se conserva porque su costo adicional es marginal y describe una realidad mecánica, no una etiqueta comercial. La gama sigue viajando en el catálogo **solo como dato de pantalla**.

Tampoco existe un "cada cuánto vuelve la gente" global: `diagnostico.diagnosticar_ritmos()` separa tres poblaciones porque describen tres momentos del ciclo de vida y promediarlas daría una cifra que no le sirve a ninguna.

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

**3. El catálogo.** Cubre el fallo más difícil de ver: agregar una variable al entrenamiento y olvidar producirla en la ficha. El modelo entrenaría bien, las métricas saldrían bien, el CI verde — y el endpoint de ingreso reventaría en producción. `validar_catalogo()` lo convierte en un fallo de build, que es donde debe doler.

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

El último paso define la naturaleza del dataset. **La historia del propio chasis es la variable central**: sin ella, un taxi que entra cada dos semanas y un particular que entra una vez al año le llegan idénticos al estimador, y lo mejor que puede hacer es predecir el promedio. Por eso se descartan los vehículos de un solo ingreso — no tienen ningún intervalo observado. En producción se vuelven predecibles en cuanto ingresan por segunda vez: la API agrega esa visita y ya tienen su primer intervalo.

Las importancias confirman el planteamiento: las cinco variables que más pesan son de historia o de uso, y ninguna domina sola.

| Variable | Importancia |
|---|---:|
| Ausencia más larga conocida del vehículo | 0,093 |
| Días desde la visita anterior | 0,092 |
| Ritmo mediano previo del propio chasis | 0,087 |
| Kilometraje | 0,069 |
| Ciclos de 10.000 km acumulados | 0,069 |

El **plan de mantenimiento de fábrica** entra también como variable (`Regla_Preventiva_Dias`), y depende del año del modelo: los vehículos **2022 en adelante** siguen el plan de **365 días o 10.000 km**; los **2021 y anteriores**, o de año desconocido, el de **6 meses (180 días) o 5.000 km**. En ambos manda lo que ocurra primero: un vehículo 2015 de 100 km diarios cumple los 5.000 km en 50 días, no en seis meses, y el modelo necesita saber cuál de los dos umbrales manda en cada caso. La regla vive en `config.py` y la aplica una sola función de `data.py`, la misma para entrenar y para servir. Junto a ella entran `Dias_Para_10k_Km` y `Ciclos_10k_Acumulados`, que miden la intensidad de uso con una vara común a todos los vehículos. Y entra `Dias_En_Taller` (`F. Cierre − F. Entrada`), el único dato de la **experiencia** del cliente que registra el ERP — la mitad de las visitas se resuelven el mismo día y el 10 % pasa de 34. No hay fuga de información: el target se cuenta desde `F. Cierre`, así que la demora ya ocurrió cuando arranca el reloj.

### Por qué dos modelos y no uno

"En cuántos días vuelve" y "va a volver o no" parecen la misma pregunta. La segunda es mucho más contestable con estos datos (R² = +0,115 contra AUC = 0,727): predecir el día exacto es una meta mal planteada cuando el intervalo tiene un coeficiente de variación del 143 %. Y la clasificación se puede medir con AUC y con falsos positivos, que es lo que permite elegir el punto de operación con criterio de negocio. Un MAE de 100 días no le dice a nadie a quién llamar.

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

**Los tres quedaron prácticamente empatados**: AUC 0,7274 (RandomForest) / 0,7270 (ExtraTrees) / 0,7184 (HistGradientBoosting), con 0,009 entre el primero y el tercero — menos que la variación entre particiones. La conclusión es que **el algoritmo no es el cuello de botella**; lo que limita el resultado son las variables disponibles.

Solo el 26,5 % de las visitas son fuga, así que sin balanceo un modelo que responda "nadie se va" acierta el 73,5 % y es inútil. `HistGradientBoosting` no acepta `class_weight` y recibe el balanceo como `sample_weight` en el `.fit()`: sin esa corrección su recall cae a 0,285 contra 0,570, y la comparación mediría el desbalance en lugar del algoritmo.

**El ganador se elige por AUC y no por F1** porque son dos decisiones distintas: qué algoritmo *ordena* mejor el riesgo se responde con AUC, que no depende del umbral; dónde poner el corte de alerta es del negocio. Y es del negocio porque los dos errores no cuestan lo mismo: un **falso positivo** es una llamada innecesaria (barato), un **falso negativo** es un cliente perdido sin que nadie lo intentara (caro). Promediarlos en una cifra oculta justamente la decisión, así que `evaluate.curva_punto_operacion()` imprime en cada entrenamiento qué cuesta cada umbral:

| Umbral | Precisión | Recall | Alertas emitidas | Llamadas de más (FP) | Clientes perdidos sin alerta (FN) |
|---:|---:|---:|---:|---:|---:|
| 0,30 | 0,331 | **0,905** | 8.101 | 5.421 | **282** |
| 0,40 | 0,379 | 0,766 | 5.988 | 3.718 | 692 |
| **0,50** (actual) | 0,449 | 0,573 | 3.783 | 2.086 | 1.265 |
| 0,60 | 0,519 | 0,361 | 2.061 | 992 | 1.893 |
| 0,70 | 0,594 | 0,191 | 955 | 388 | 2.395 |

Bajar a 0,30 detecta el 90,5 % de las fugas. Si una llamada cuesta minutos de un asesor y un cliente perdido cuesta todos sus mantenimientos futuros, la aritmética favorece llamar de más. Se deja en `config.UMBRAL_DECISION_FUGA` para que lo decida el negocio.

Por último se contrasta un modelo global contra uno por **origen del vehículo**:

| RandomForest | AUC | Precisión | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Global, sobre **vendidos por nosotros** | 0,7225 | 0,450 | 0,587 | 0,510 | 353 | 203 |
| **Segmentado, sobre vendidos por nosotros** | 0,7199 | **0,457** | **0,593** | **0,516** | **347** | **200** |
| Global, sobre **externos** | 0,7284 | 0,448 | 0,570 | 0,502 | 1.733 | 1.062 |
| Segmentado, sobre externos | 0,7266 | 0,449 | 0,576 | 0,505 | 1.743 | 1.047 |

En vehículos propios el segmentado comete **menos errores de los dos tipos**. El AUC es casi idéntico — ambos ordenan igual de bien — pero el segmentado está **mejor calibrado** para esa población: sus probabilidades caen donde el umbral las aprovecha mejor. En externos es prácticamente un empate. La ganancia es pequeña (6 falsos positivos y 3 falsos negativos menos), así que **la API usa solo el modelo global**, con una sola probabilidad y una sola alerta, que es lo que el asesor necesita para decidir. El contraste se recalcula en cada entrenamiento y queda en `metricas_entrenamiento.json`, así que si el negocio decide actuar distinto sobre cada origen la evidencia ya está; los modelos segmentados en sí no se congelan, porque nada los consume.

> **Cómo leer esa tabla.** Los AUC de segmentos distintos **no son comparables entre sí**, porque cada uno se mide contra la dificultad de su propia población. Lo comparable es global contra segmentado *dentro de la misma fila*, que es como está armada.

### Cómo se mide para que las métricas no mientan

Dos disciplinas, y sin ellas todos los números de arriba serían ficticios.

**Las variables de historia se calculan con `expanding()`**, que acumula solo lo ya ocurrido fila por fila. El intervalo que se predice va de la visita *i* a la *i+1*, y las pistas llegan como máximo hasta la *i*. Si el ritmo previo incluyera el intervalo que se quiere predecir, el modelo estaría leyendo la respuesta. La prueba `test_historia_no_filtra_el_futuro` lo verifica en cada corrida.

**La evaluación se agrupa por VIN.** Un mismo vehículo aporta varias visitas y todas comparten su ritmo; con un `train_test_split` aleatorio la visita 3 de un carro quedaría en entrenamiento y la 4 en prueba, y el modelo habría visto en la práctica la respuesta que se le va a preguntar. `GroupShuffleSplit` y `GroupKFold` dejan cada vehículo **entero** de un lado o del otro, y se valida cinco veces para reportar media y desviación en vez de un número que depende de la suerte de la partición.

### De entrenar a servir

**Se congela el Pipeline completo, no solo el modelo.** Es la decisión de diseño más importante del proyecto. El `.joblib` guarda todo lo que se aprendió de los datos durante el `.fit()` — las medianas de los 15 campos numéricos, la media y desviación de cada uno para el `StandardScaler`, la moda de las categóricas, las 63 columnas tras el OneHot — y esos números **no están escritos en ninguna parte del código fuente**. Si la API recalculara su propia media y desviación usaría los datos que le llegan, no las 8.938 filas del entrenamiento: un `Kms.` de 45.000 se convertiría en un número distinto al que el modelo aprendió a interpretar, y las predicciones saldrían mal **sin arrojar ningún error**. Ese fallo silencioso se llama *training/serving skew*, y congelar el pipeline entero lo elimina por construcción.

Por el mismo motivo, **la API no tiene fórmulas propias para la historia del vehículo.** Cuando un carro ingresa, `data.ficha_con_ingreso_nuevo()` agrega esa visita a las que ya tenía en el historial y pasa todo por `construir_catalogo_vehiculos()`, la misma función que arma el catálogo del entrenamiento. Que ambos calculen distinto una misma variable es el error más caro y más difícil de detectar de un proyecto de ML; la prueba `test_historial_reproduce_el_catalogo` verifica en cada corrida que recalcular desde el historial da exactamente la ficha del entrenamiento.

Cuatro decisiones más completan la frontera:

- **La API pide lo que el asesor tiene en la recepción**: el VIN, el kilometraje y para qué vino el carro (`tipo_cargo`, `tipo_trabajo`). Todo lo demás — año del modelo, visitas anteriores, ritmo, km por día — lo saca del historial. El contrato se diseña alrededor de lo que el usuario **tiene**, no de lo que el modelo **necesita**.
- **Registrar un ingreso no exige reentrenar.** El modelo aprendió la relación *historia → regreso*, y esa relación no cambia de un día para otro; lo que cambia es la historia de cada carro. Por eso la API recalcula la historia con la visita de hoy y le vuelve a preguntar al mismo modelo. El ingreso no se guarda: quedará en la base cuando el ERP lo exporte y se corra `python -m src.actualizar_catalogo`.
- **El historial se construye fuera de la API**, al entrenar o con `actualizar_catalogo`, no al arrancar: evita reprocesar 34.935 filas en cada reinicio y garantiza que la historia servida sea la misma que vio el modelo.
- **El plan de mantenimiento actúa como tope.** Aplicado solo, el plan por año da un MAE de 111,8 días, peor que predecir siempre la mediana (107,9): describe cuándo el vehículo *debería* volver, no cuándo vuelve. Pero agendar a un cliente **después** de que ya le tocaba el mantenimiento no tiene sentido, así que la fecha final es lo que ocurra primero entre el modelo y el plan. El campo `base_de_la_fecha` le dice al asesor cuál de los dos mandó.

### El plan de llamada

La regresión se equivoca en promedio 98 días (MAE del holdout por VIN). Entregar "vuelve el 17 de abril" transmitiría una precisión que el modelo no tiene, así que la API entrega una **ventana**:

```
fecha estimada = fecha de entrega + días estimados (tope: plan de mantenimiento)
ventana        = [fecha estimada − MAE, fecha estimada + MAE]   (nunca antes de la entrega)
```

El MAE se lee de `metricas_entrenamiento.json` al arrancar, así que la ventana se ajusta sola cada vez que se reentrena. Contra el día de la consulta, el vehículo queda en uno de tres estados: `programar` (la ventana no ha abierto), `llamar_ahora` (está dentro) o `llamar_ya` (ya cerró sin que volviera).

**La alerta de fuga adelanta la llamada.** La ventana responde *"si vuelve, ¿cuándo?"*; la alerta responde *"¿va a volver?"*. Son preguntas distintas y pueden no coincidir: la regresión solo aprendió de carros que sí volvieron. Si el clasificador dice que lo más probable es que el cliente no vuelva, esperar a que abra la ventana es esperar a perderlo, así que un vehículo en `programar` con alerta pasa a `llamar_ahora`. Si el carro todavía está en el concesionario, el mensaje pide dejarle agendada la próxima cita al entregarlo: es el momento en que el cliente está presente.

**El CI corre `ruff` → `train` → `pytest`**, en ese orden: ruff falla en segundos y es lo más barato; el entrenamiento va antes que las pruebas porque estas cargan los `.joblib` y en una máquina limpia hay que generarlos. Reentrenar en cada push es lo que demuestra que el pipeline es reproducible de verdad.

### Observabilidad: saber qué pasa cuando nadie está mirando

Una API en producción que solo devuelve respuestas es una caja negra: si se pone lenta o empieza a fallar, nadie se entera hasta que se queja un asesor. Tres piezas lo resuelven:

- **Logs estructurados (`src/logger.py`).** Cada evento sale como **una línea JSON** con `timestamp`, `level`, `mensaje` y los campos propios del evento (`vin`, `latencia_ms`, `probabilidad_fuga`…). Una plataforma como Render o cualquier agregador de logs puede filtrar por campo — "todas las predicciones con alerta", "todos los 404" — sin parsear texto libre. Se registran: predicción exitosa, VIN sin historial, error de validación (422), petición lenta (> 200 ms) y las consultas a `/health` y `/metrics`.
- **Middleware de métricas (`src/metrics.py`).** Envuelve **todas** las peticiones: mide la latencia con `perf_counter`, cuenta como error cualquier respuesta 4xx/5xx o excepción, y guarda las últimas 1.000 latencias para calcular p50 y p95. Se reporta el **p95** y no solo el promedio porque el promedio esconde a los usuarios que esperan mucho.
- **`GET /health` y `GET /metrics`.** El primero responde en milisegundos si el servicio vive y si los tres `.joblib` cargaron, que es lo que un balanceador o un monitor necesita preguntar cada pocos segundos. El segundo expone el acumulado de tráfico, tasa de error y latencias.

Las métricas viven **en memoria**: se reinician con el servidor y cada réplica cuenta lo suyo. Es suficiente para un servicio de una sola instancia; con varias réplicas se reemplazaría por Prometheus o el sistema de métricas de la plataforma.

### Despliegue con Docker

El `Dockerfile` parte de `python:3.12-slim`, instala las dependencias en una capa propia (para que cambiar el código no obligue a reinstalar pandas y scikit-learn) y copia **solo lo que la inferencia necesita**: `src/`, `models/` y `main.py`. Los datos crudos, las pruebas y el entorno virtual quedan fuera por `.dockerignore`: la imagen sirve el modelo, no lo entrena. Uvicorn escucha en `0.0.0.0:8000` para ser alcanzable desde fuera del contenedor.

```bash
docker build -t retencion-api .
docker run -p 8000:8000 retencion-api
```

---

## 5. Qué recibe y devuelve la API

| Método | Ruta | Función |
|---|---|---|
| `POST` | `/predict/vin/{vin}/ingreso` | Registra el ingreso del vehículo y devuelve regreso estimado, riesgo de fuga y plan de llamada |
| `GET` | `/` | Estado detallado: qué artefactos cargaron, algoritmo de fuga, vehículos en historial, margen de error y fecha de corte |
| `GET` | `/health` | Health check corto para monitoreo: ¿vive y cargaron los modelos? |
| `GET` | `/metrics` | Peticiones totales, errores, tasa de error y latencia promedio / p50 / p95 |
| `GET` | `/docs` | Swagger UI autogenerado, con el contrato siempre al día |

Hay un solo endpoint de predicción a propósito: toda predicción parte de una visita registrada, así que la historia del vehículo siempre está al día con lo que el asesor tiene enfrente, aunque la base de datos todavía no lo sepa.

### `GET /` — Estado detallado

```json
{
  "mensaje": "API de Retención Predictiva del concesionario activa 🚗",
  "modelo_cargado": true,
  "modelo_fuga_cargado": true,
  "algoritmo_fuga": "RandomForest",
  "historial_cargado": true,
  "vehiculos_en_historial": 9289,
  "margen_error_dias": 98.4,
  "fecha_corte_datos": "2026-06-30"
}
```

Si los `.joblib` no existen el servidor **no se cae**: arranca con `modelo_cargado: false` o `historial_cargado: false`, para que el health check siga respondiendo y el problema se pueda diagnosticar en vez de quedarse a oscuras.

### `GET /health` y `GET /metrics` — Monitoreo

```json
{ "estado": "activo", "servicio": "API Retención Automotor.co", "modelos_cargados": true, "version_api": "6.0.0" }
```

```json
{
  "peticiones_totales": 120,
  "errores_totales": 6,
  "tasa_error_pct": 5.0,
  "latencia_promedio_ms": 38.2,
  "latencia_p50_ms": 31.5,
  "latencia_p95_ms": 92.7
}
```

`modelos_cargados` es `true` solo si cargaron los tres artefactos que la predicción necesita: regresor, clasificador de fuga e historial. Los errores incluyen los 4xx: un 404 por un VIN mal digitado cuenta, porque es una petición que no le sirvió al asesor.

### `POST /predict/vin/{vin}/ingreso`

El VIN viaja en la URL y se normaliza (mayúsculas, sin espacios) antes de buscar. **Entrada**, validada por Pydantic con la clase `IngresoHoy`:

```json
{
  "kilometraje": 20400,
  "tipo_cargo": "Cliente",
  "tipo_trabajo": "MECANICA",
  "fecha_ingreso": "2026-10-19",
  "fecha_entrega": "2026-10-19"
}
```

| Campo | Tipo | Obligatorio | Restricción |
|---|---|---|---|
| `kilometraje` | float | **Sí** | ≥ 0 y < 500.000; no menor que el de la visita anterior |
| `tipo_cargo` | string | No (`"Cliente"`) | `Cliente`, `Garantia`, `Seguros` o `Interno` |
| `tipo_trabajo` | string | No (`"MECANICA"`) | `MECANICA`, `COLISION`, `MECANICA Y COLISION` u `OTRO` |
| `fecha_ingreso` | fecha | No (hoy) | No futura; posterior al último ingreso registrado del VIN |
| `fecha_entrega` | fecha | No (= ingreso) | No anterior al ingreso |

`tipo_cargo` y `tipo_trabajo` son las dos variables categóricas con las que se entrenó el modelo, y sus valores son exactamente los que existen en la base. Un mantenimiento normal es `Cliente` + `MECANICA`; un siniestro, `Seguros` + `COLISION`.

**Salida:**

```json
{
  "vin": "DEU822806",
  "datos_encontrados": {
    "gama": "Tucson",
    "anio_modelo": 2014,
    "antiguedad_calculada": 12.0,
    "ultimo_kilometraje_visita_concesionario": 20400.0,
    "ultima_visita_concesionario": "2026-10-19",
    "vendido_por_nosotros": false,
    "total_visitas": 3,
    "ritmo_habitual_dias": 576.0,
    "km_por_dia": 6.6
  },
  "prediccion_dias_retorno": 180.0,
  "riesgo_de_fuga": {
    "probabilidad_fuga": 0.7758,
    "alerta": true
  },
  "plan_de_llamada": {
    "estado": "llamar_ahora",
    "mensaje": "Riesgo alto de fuga (78%): no esperar a la ventana. Dejarle agendada la próxima cita al entregarle el carro (su ventana normal sería del 2027-01-09 al 2027-07-24).",
    "fecha_referencia": "2026-10-19",
    "fecha_estimada_retorno": "2027-04-17",
    "margen_error_dias": 98.4,
    "fecha_corte_datos": "2026-06-30",
    "base_de_la_fecha": "Plan de mantenimiento: cumple 180 días (plan 6 meses o 5.000 km, modelos 2021 o anteriores), antes que los 319 días que estima el modelo por su historia."
  }
}
```

Cómo se lee este ejemplo: el Tucson había venido en agosto de 2023 y en junio de 2026. Con el ingreso de hoy suma 3 visitas; su ritmo habitual queda en 576 días (la mediana de los intervalos de 1.041 y 111 días) y sus km por día en 6,6. El modelo estima 319 días, pero es un 2014 y su plan es de 6 meses: la fecha estimada queda a los 180 días. Con 78 % de probabilidad de fuga, la instrucción es no esperar a la ventana.

El historial está incompleto para algunos chasis (hay VIN sin año de modelo). En esos casos el campo viaja como `null`, el imputador del pipeline rellena con la mediana igual que en el entrenamiento, y se aplica el plan de mantenimiento corto. No se inventa un `0`, porque eso le diría al modelo que es un carro nuevo.

### Qué significa cada campo de la respuesta

| Campo | Qué es |
|---|---|
| `datos_encontrados` | Ficha del vehículo **ya con la visita de hoy**, para que el asesor verifique que se buscó el carro correcto antes de confiar en la cifra |
| `ultimo_kilometraje_visita_concesionario` · `ultima_visita_concesionario` | Odómetro y fecha de la visita que se acaba de registrar |
| `ritmo_habitual_dias` | Mediana de los días entre visitas del propio chasis |
| `km_por_dia` | Mediana de los km por día entre visitas |
| `prediccion_dias_retorno` | Días hasta el próximo ingreso, contados desde la entrega: lo que ocurra primero entre el modelo y el plan de mantenimiento |
| `riesgo_de_fuga.probabilidad_fuga` · `alerta` | Probabilidad de que no vuelva en 365 días según el modelo global, y si supera `UMBRAL_DECISION_FUGA` (0,50) |
| `plan_de_llamada.estado` | `programar`, `llamar_ahora` o `llamar_ya` |
| `plan_de_llamada.mensaje` | La instrucción para el asesor, con las fechas de la ventana |
| `plan_de_llamada.fecha_referencia` | Día de la consulta contra el cual se decide el estado |
| `plan_de_llamada.fecha_estimada_retorno` | Fecha de entrega + días estimados |
| `plan_de_llamada.margen_error_dias` | MAE del modelo: el ancho de la ventana hacia cada lado |
| `plan_de_llamada.fecha_corte_datos` | Último ingreso que trae la base. Lo posterior solo lo conoce la API si se registra en la petición |
| `plan_de_llamada.base_de_la_fecha` | Si mandó el modelo o el plan de mantenimiento, y por qué |

### Códigos de error

| Código | Cuándo ocurre |
|---|---|
| `422` | El cuerpo no cumple el contrato (falta el kilometraje, llega texto donde va número, un tipo de trabajo que no existe), el odómetro es menor que el de la visita anterior, o las fechas son incoherentes. Si el fallo es de Pydantic, la respuesta trae `detalle` y la lista `errores` con el campo exacto, y queda registrado en el log |
| `404` | El VIN no tiene historial en el concesionario: sería su primer ingreso y no hay ritmo del que estimar su regreso |
| `409` | La fecha de ingreso es igual o anterior al último ingreso registrado del VIN |
| `500` | Falta algún `.joblib` del modelo o del historial |
| `400` | Error inesperado durante la inferencia |

---

## 6. Qué le falta y qué mejoraría

### Dónde está hoy

| | Métrica | Resultado | Lectura |
|---|---|---:|---|
| **Regresión** | MAE | 100,5 días | Desfase promedio entre lo predicho y lo real (98,4 en el holdout por VIN) |
| | RMSE | 167,5 días | Muy por encima del MAE → hay casos con errores extremos |
| | R² | **+0,115** | Supera al promedio, pero por poco |
| **Clasificación** | AUC | **0,727** | Ordena bien a los clientes por riesgo de fuga |
| | PR-AUC | 0,481 | Contra una tasa base de 0,265 |
| | Recall @0,50 | 0,573 | Detecta algo más de la mitad de las fugas |
| | Recall @0,30 | **0,905** | Bajando el umbral detecta 9 de cada 10 |

**La conclusión principal es que la clasificación funciona y la regresión apenas.** Un AUC de 0,727 es utilizable para priorizar llamadas; un R² de 0,115 no sirve para decidir sobre un cliente concreto. Por eso la API convierte la regresión en una ventana del ancho de su error y deja que la alerta de fuga mande sobre ella.

Contra las referencias sin aprendizaje, la regresión gana en las tres métricas — pero el margen es estrecho:

| Referencia | MAE | RMSE | R² |
|---|---:|---:|---:|
| Plan de mantenimiento por año (2022+: 365 d / 10.000 km · ≤2021: 180 d / 5.000 km) | 111,77 | 189,51 | −0,132 |
| Ritmo mediano del propio vehículo | 105,02 | 196,32 | −0,214 |
| Predecir siempre la mediana | 107,88 | 186,21 | −0,093 |
| **Random Forest** | **100,53** | **167,54** | **+0,115** |

Y una observación que vale para el negocio: **aplicar el plan de fábrica tal cual sigue siendo la peor de las referencias**, aunque diferenciarlo por año la acercó bastante a la realidad (MAE de 111,8 días contra 129,5 con el plan único de 365 días o 10.000 km). El plan describe cuándo el vehículo *debería* volver, no cuándo vuelve; por eso en la API funciona como tope y no como predicción.

### El techo que queda

El límite no es el algoritmo — los tres clasificadores quedaron dentro de 0,009 de AUC entre sí. Es la información disponible:

1. **Lo que hace volver a un cliente no está en los datos.** Si quedó conforme, qué pagó, si le cumplieron la fecha de entrega. El ERP registra la transacción, no la experiencia. La demora en taller es lo único que se pudo rescatar de esa dimensión.
2. **El objetivo es muy disperso.** Coeficiente de variación del 143 %: media de 144 días contra mediana de 77, con un máximo de 2.936. Hay al menos dos poblaciones mezcladas — quien vuelve al mantenimiento de rutina y quien desaparece por años.
3. **El plan de mantenimiento es todavía grueso.** Hoy se distingue por año del modelo (2022 en adelante contra 2021 y anteriores). Un plan por gama y versión permitiría calcular `edad_mora`, el atraso exacto contra lo que le toca a cada vehículo, que sería la variable más directa del problema.
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
- **Una ventana a la medida de cada vehículo.** Hoy la ventana de llamada usa el mismo margen (el MAE global, 98 días) para todos. Un intervalo por cuantiles — o la dispersión entre los árboles del bosque — daría ventanas angostas para los clientes predecibles y anchas para los erráticos.
- **Una fuga a la medida del plan.** La etiqueta de fuga usa 365 días para todos los vehículos. Para los 2021 y anteriores, cuyo plan es de 6 meses, un umbral más corto podría detectar el abandono antes; requiere reentrenar y comparar.
- **Separar por tipo de trabajo**: el retorno tras una `COLISION` y tras un mantenimiento de `MECANICA` son fenómenos distintos. Se puede contrastar con la misma maquinaria de `comparar_clasificadores()` cambiando la columna de corte.
