# Arquitectura del Proyecto

Documento técnico. Si lo que buscas es prender el proyecto, ve al **[README](README.md)**.

---

## 1. Qué hace y qué logra

Este proyecto toma la base transaccional de órdenes de trabajo de un taller automotriz (34.935 registros), la convierte en un dataset entrenable, ajusta un modelo de regresión que estima **cuántos días pasarán hasta que un vehículo regrese**, congela ese modelo en disco y lo expone como una **API REST**, todo respaldado por pruebas automáticas e integración continua.

Lo que logra, concretamente:

- **Un ciclo de vida de ML completo y reproducible.** Cualquiera clona el repo, ejecuta un comando y obtiene exactamente el mismo modelo. Sin notebooks, sin pasos manuales, sin "en mi máquina sí funciona".
- **Separación real entre entrenar y servir.** La API no reimplementa ni una línea del preprocesamiento; consume el pipeline congelado.
- **Un contrato de datos explícito en la frontera.** Pydantic rechaza entradas inválidas antes de que toquen el modelo.
- **Verificación automática en cada push.** GitHub Actions reconstruye el entorno desde cero, reentrena y corre las pruebas en una máquina limpia.

Lo que **no** logra todavía: predecir con precisión útil. Ver [sección 6](#6-qué-le-falta-y-qué-mejoraría).

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
   │  CARGA              │  · limpia BOM y espacios de los encabezados
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐  consolidar_visitas_por_ot()
   │  CONSOLIDACIÓN      │  agrupa por VIN + OT
   └─────────────────────┘  34.935 sub-trabajos → 27.697 visitas reales
            │
            ├──────────────────────────────┐
            ▼                              ▼
   ┌─────────────────────┐      ┌─────────────────────────┐
   │  TARGET + FEATURES  │      │  CATÁLOGO POR VIN       │
   │  shift(-1) y        │      │  última visita de cada  │
   │  filtros de sanidad │      │  uno de los 9.729 autos │
   │  → 14.903 filas     │      └─────────────────────────┘
   └─────────────────────┘                  │
            │                               │
            ▼                               │
   ┌─────────────────────┐                  │
   │  src/features.py    │  ColumnTransformer
   │  PREPROCESAMIENTO   │  imputar → escalar → codificar
   └─────────────────────┘  6 columnas → 51 columnas
            │                               │
            ▼                               │
   ┌─────────────────────┐                  │
   │  src/train.py       │  split 80/20 + RandomForest.fit()
   │  ENTRENAMIENTO      │                  │
   └─────────────────────┘                  │
            │                               │
            ▼                               │
   ┌─────────────────────┐                  │
   │  src/evaluate.py    │  MAE / RMSE / R² sobre el 20% de test
   └─────────────────────┘                  │
            │                               │
            ▼                               ▼
   models/modelo_retencion.joblib   models/catalogo_vehiculos.joblib
   (pipeline completo, 18 MB)       (9.729 fichas, 104 KB)


   ┌──────────────────────────────────────────────────────────────────┐
   │                      FASE 2 — INFERENCIA                         │
   │                 (online, mientras la API vive)                   │
   └──────────────────────────────────────────────────────────────────┘

   Cliente (asesor / CRM / Swagger)
            │  HTTP
            ▼
   ┌─────────────────────┐
   │  main.py            │  carga ambos .joblib UNA vez al arrancar
   │  API FastAPI        │
   └─────────────────────┘
            │
            ├── POST /predict ──────► Pydantic valida (422 si no cumple)
            │                              │
            │                              ▼
            │                        traduce Año Modelo → Antigüedad
            │                              │
            ├── GET /predict/vin/{vin} ──► busca en catálogo (404 si no está)
            │                              │
            │                              ▼
            │                        autocompleta las 6 características
            │                              │
            ▼                              ▼
   ┌──────────────────────────────────────────────┐
   │  pipeline.predict(DataFrame de 1 fila)       │
   │  el MISMO preprocesamiento del entrenamiento │
   └──────────────────────────────────────────────┘
            │
            ▼
   JSON: { "prediccion_dias_retorno": 267.4, "unidad": "días" }
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
│   ├── data.py                     Carga, consolidación, target y catálogo
│   ├── features.py                 ColumnTransformer: imputación, escalado, codificación
│   ├── evaluate.py                 Métricas de regresión
│   └── train.py                    Orquestador: entrena, evalúa y serializa
│
├── models/                         Artefactos congelados (salida de train.py)
│   ├── modelo_retencion.joblib     Pipeline COMPLETO: preprocesador + RandomForest
│   └── catalogo_vehiculos.joblib   Ficha por VIN para la búsqueda por chasis
│
├── main.py                         API REST con FastAPI (capa online)
│
├── tests/
│   └── test_api.py                 7 pruebas de integración con TestClient
│
├── .github/workflows/ci.yml        CI: ruff → train → pytest
├── pyproject.toml                  Configuración de pytest y ruff
└── requirements.txt                Dependencias del entorno
```

### Cómo se conectan

**`config.py` es la fuente única de verdad.** No importa a nadie; todos lo importan a él. Si mañana el taller renombra una columna, se corrige en un solo lugar y `data.py`, `features.py`, `train.py` y `main.py` quedan corregidos.

```
                    ┌──────────────┐
                    │  config.py   │  ← no importa a nadie
                    └──────────────┘
                           ▲
        ┌──────────┬───────┴───────┬──────────┐
        │          │               │          │
    data.py   features.py    evaluate.py   main.py
        ▲          ▲               ▲
        └──────────┴───────┬───────┘
                           │
                       train.py       ← el único que llama a los tres
```

- **`data.py`** solo sabe de pandas. No conoce scikit-learn. Su trabajo termina cuando entrega un DataFrame limpio.
- **`features.py`** solo sabe de scikit-learn. No lee archivos ni conoce rutas. Devuelve un `ColumnTransformer` sin ajustar.
- **`evaluate.py`** no conoce el modelo. Recibe dos vectores de números y devuelve un diccionario. Se puede reutilizar con cualquier otro modelo.
- **`train.py`** es el director de orquesta: llama a los tres en orden y congela el resultado. No define lógica propia.
- **`main.py`** no importa `data.py`, `features.py` ni `train.py`. Solo importa `config.py` y carga los `.joblib`. **Esa es la frontera entre entrenar y servir**, y está dibujada a propósito: la API puede desplegarse en un servidor que no tenga acceso al CSV.

---

## 4. Por qué se tomó cada decisión

### Por qué consolidar por OT antes de cualquier cosa

El CSV viene a nivel de *sub-trabajo*: una sola visita al taller genera varias filas (`Núm. Trabajo` 1, 2, 3...). Si se entrenara así, un vehículo que entró una vez pero necesitó cinco reparaciones pesaría cinco veces más que uno que entró una vez y necesitó una. El modelo aprendería a predecir *complejidad de reparación*, no *retorno*.

Agrupando por `VIN` + `OT` se obtiene una fila por **ingreso físico real**: fecha de entrada mínima, cierre máximo, kilometraje más alto.

> 34.935 sub-trabajos → **27.697 visitas reales** de **9.729 vehículos únicos**

### Por qué la variable objetivo se construye con `shift(-1)`

La columna a predecir **no existe en el CSV**. Hay que construirla mirando al futuro: ordenando las visitas de cada VIN por fecha y trayendo con `shift(-1)` la fecha de la siguiente visita a la fila actual.

```
Dias_Hasta_Retorno = (F. Entrada de la visita SIGUIENTE) − (F. Cierre de la visita ACTUAL)
```

Se mide desde el **cierre** y no desde la entrada, porque el reloj de la retención empieza cuando el cliente se lleva el carro, no cuando lo deja.

### Por qué se descartan tantas filas (y cuáles)

Este es el embudo completo, con números reales:

| Paso | Filas | Se pierden | Por qué |
|---|---:|---:|---|
| Visitas consolidadas | 27.697 | — | — |
| Tienen visita siguiente | 17.968 | **9.729** | Es la última visita del vehículo: no hay respuesta que aprender |
| Retorno ≥ 0 días | 17.201 | 767 | Fechas invertidas (error de digitación) |
| Antigüedad ≥ 0 años | 15.203 | 1.998 | Año de modelo posterior a la visita (error de digitación) |
| Kilometraje < 500.000 | **14.903** | 293 | Valores tipo `123.456.789` |

Los filtros de kilometraje no son cosmética: un solo valor absurdo distorsiona la media y la desviación que calcula el `StandardScaler`, y con eso **toda** la escala de la variable.

> ⚠️ **Ojo con la primera fila del embudo.** Esos 9.729 registros que se descartan son exactamente uno por vehículo: su última visita. Y ahí adentro están los clientes que **nunca volvieron** — es decir, la señal de abandono. Ver [sección 6](#6-qué-le-falta-y-qué-mejoraría).

### Por qué RandomForest y no una regresión lineal

Tres razones:

1. **No asume linealidad.** La relación entre kilometraje y retorno no es una recta: un carro de 10.000 km y uno de 300.000 km no se comportan proporcionalmente.
2. **Captura interacciones sin que se las declaren.** Un `Tipo de Trabajo = COLISION` en un vehículo nuevo significa algo distinto que en uno de 15 años. Un árbol descubre eso solo; una regresión lineal necesitaría que se lo escribieran a mano.
3. **Tolera el desorden.** No exige normalidad ni se rompe con los valores extremos que abundan en una base de taller.

Con `random_state=42` fijo, para que el resultado sea reproducible en cualquier máquina.

### Por qué se congela el Pipeline COMPLETO y no solo el modelo

Es la decisión de diseño más importante del proyecto.

El `.joblib` no guarda "el modelo". Guarda todo lo que se **aprendió de los datos** durante el `.fit()`:

```
Mediana para rellenar Kms. faltantes : 45.428
Media que resta el StandardScaler    : 85.456,76 (Kms.) / 3,36 (Antigüedad)
Desviación por la que divide         : 97.703,94 (Kms.) / 4,19 (Antigüedad)
Moda para categóricas faltantes      : Accent / Cliente / MECANICA
Categorías conocidas de 'Gama'       : 40
Columnas tras el OneHot              : 51
```

Esos números **no están escritos en ninguna parte del código fuente**. Si la API recalculara su propia media y desviación, usaría los datos que le llegan — no los 14.903 del entrenamiento. Un `Kms.` de 45.000 se convertiría en un número distinto al que el modelo aprendió a interpretar, y las predicciones saldrían mal **sin arrojar ningún error**.

Ese fallo silencioso se llama *training/serving skew*, y congelar el pipeline entero lo elimina por construcción.

### Por qué el VIN nunca entra al modelo

El VIN es la **llave de búsqueda**, no un predictor. Codificar 9.729 chasis produciría un modelo que memoriza cada carro individual en vez de aprender el patrón, y que no sabría absolutamente nada frente a un vehículo nuevo.

### Por qué la API pide el año del modelo y no la antigüedad

Porque es el dato que el asesor tiene a la vista en la tarjeta de propiedad. La API traduce año → antigüedad internamente, replicando el mismo cálculo del entrenamiento. La idea es diseñar el contrato alrededor de lo que el usuario **tiene**, no de lo que el modelo **necesita**.

El campo está acotado entre 1990 y el año entrante: fuera de ese rango responde `422`, porque el modelo nunca vio antigüedades mayores.

### Por qué el catálogo se construye al entrenar y no al arrancar la API

Si se construyera al arrancar, cada reinicio del servidor obligaría a reprocesar 34.935 filas del CSV. Además, el catálogo generado al entrenar es **el mismo** que vio el modelo: coherencia garantizada entre lo que se aprendió y lo que se sirve.

### Por qué el orden del CI es ruff → train → pytest

- **Ruff primero** porque falla en segundos y es lo más barato. No tiene sentido entrenar un modelo para descubrir después que hay un import sin usar.
- **Train antes que pytest** porque las pruebas cargan `models/modelo_retencion.joblib` y necesitan que exista. En una máquina limpia de GitHub, ese archivo se genera en ese paso.
- **Reentrenar en cada push** no es un capricho: es lo que demuestra que el pipeline es reproducible de verdad y que el CSV versionado alcanza para reconstruir todo desde cero.

---

## 5. Qué recibe y devuelve la API

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predicción con datos manuales |
| `GET` | `/predict/vin/{vin}` | Predicción por chasis |
| `GET` | `/docs` | Swagger UI autogenerado |

### `GET /` — Health check

Sirve para que soporte verifique en un segundo si el servicio vive **y** si los artefactos cargaron.

```json
{
  "mensaje": "API de Retención Predictiva de Automotor.co S.A.S. activa 🚗",
  "modelo_cargado": true,
  "catalogo_cargado": true,
  "vehiculos_en_catalogo": 9729
}
```

Si el `.joblib` no existe, el servidor **no se cae**: arranca igual con `modelo_cargado: false`. Así el health check sigue respondiendo y el problema se puede diagnosticar en vez de quedarse a oscuras.

### `POST /predict`

**Entrada** (validada por Pydantic con la clase `SolicitudVehiculo`):

```json
{
  "Gama": "Tucson",
  "Tipo Cargo": "Cliente",
  "Tipo de Trabajo": "MECANICA",
  "Kms.": 45000.0,
  "Año Modelo": 2022,
  "Es_Vehiculo_Vendido": true
}
```

| Campo | Tipo | Obligatorio | Restricción |
|---|---|---|---|
| `Gama` | string | Sí | — |
| `Tipo Cargo` | string | Sí | — |
| `Tipo de Trabajo` | string | No (`"MECANICA"` por defecto) | — |
| `Kms.` | float | Sí | — |
| `Año Modelo` | int | Sí | entre 1990 y el año entrante |
| `Es_Vehiculo_Vendido` | bool | Sí | — |

**Salida:**

```json
{
  "prediccion_dias_retorno": 484.9,
  "unidad": "días"
}
```

### `GET /predict/vin/{vin}`

Sin cuerpo. El VIN viaja en la URL y se normaliza (mayúsculas, sin espacios) antes de buscar.

**Salida** para `GET /predict/vin/ADM130850`:

```json
{
  "vin": "ADM130850",
  "datos_encontrados": {
    "gama": "I10",
    "anio_modelo": 2013,
    "antiguedad_calculada": 13.0,
    "ultimo_kilometraje": 487636.0,
    "ultima_visita": "2020-02-18",
    "vendido_por_nosotros": false
  },
  "prediccion_dias_retorno": 267.4,
  "unidad": "días"
}
```

Devuelve la ficha encontrada a propósito: el asesor debe poder **verificar** que el sistema buscó el vehículo correcto antes de confiar en la cifra.

El historial del taller está incompleto para algunos chasis (hay VIN sin año de modelo y sin kilometraje). En esos casos el campo viaja como `null` y el imputador del pipeline rellena con la mediana — igual que en el entrenamiento. No se inventa un `0`, porque eso le diría al modelo que es un carro nuevo.

### Códigos de error

| Código | Cuándo ocurre |
|---|---|
| `422` | El JSON no cumple el contrato: falta un campo, llega texto donde va número, o el año está fuera de rango |
| `404` | El VIN no tiene historial en el taller |
| `500` | Falta el `.joblib` del modelo o del catálogo |
| `400` | Error inesperado durante la inferencia |

---

## 6. Qué le falta y qué mejoraría

### El problema de fondo: el modelo no le gana al promedio

| Métrica | Valor | Lectura |
|---|---|---|
| MAE | 122,7 días | Desfase promedio entre lo predicho y lo real |
| RMSE | 214,7 días | Muy superior al MAE → hay casos con errores extremos |
| R² | **−0,096** | El modelo **no supera** a predecir siempre el promedio |

Un R² negativo no es un error de programación. Es un resultado: **con las seis variables disponibles, el fenómeno no se explica.** Y tiene sentido si uno se detiene a pensarlo — las variables actuales describen el *carro* (gama, kilómetros, antigüedad), pero el que decide volver o no volver es el *cliente*.

La dispersión del objetivo lo confirma: mediana de 69 días contra media de 136, con desviación de 200 y un máximo de 2.936 días. Hay dos poblaciones mezcladas — quien vuelve al mantenimiento de rutina y quien desaparece por años — y el modelo está tratando de ajustar una sola curva sobre ambas.

### Mejora 1 — Investigar la causa real del abandono

Esta es la mejora de mayor impacto, y es de **negocio antes que de modelado**.

Hoy predecimos *cuándo* vuelve un cliente, pero nunca preguntamos *por qué* se fue. Los datos que faltan son justamente los que explican el abandono:

- **Satisfacción con el servicio** — ¿quedó conforme con la reparación? ¿se la tuvieron que repetir?
- **Precio** — ¿cuánto facturó? ¿le pareció caro frente al taller de la esquina?
- **Tiempo de respuesta** — ¿cuántos días estuvo el carro retenido? ¿le cumplieron la fecha de entrega?
- **Trato recibido** — ¿qué asesor lo atendió? ¿hubo reclamos?
- **Garantía** — ¿se le venció? Porque muchos clientes solo vuelven mientras la tienen.

Un cliente que no vuelve en 400 días probablemente no está "tardando": **se fue a otro taller**. Hoy el modelo trata ambos casos como lo mismo, y por eso no aprende ninguno de los dos.

Incorporar estas variables atacaría el R² por la raíz, en vez de seguir exprimiendo las seis que ya tenemos.

### Mejora 2 — Rescatar la señal de abandono que hoy se bota

Relacionado con lo anterior, y visible en el embudo de la [sección 4](#por-qué-se-descartan-tantas-filas-y-cuáles): se descartan **9.729 visitas** — una por vehículo — porque no tienen visita siguiente.

Pero esos no son datos inservibles. Son precisamente **los clientes que no volvieron**: la población más valiosa para estudiar la retención, y la única que el modelo tiene prohibido ver por construcción.

Dos caminos para aprovecharlos:

- **Reformular como clasificación:** en vez de "¿en cuántos días vuelve?", preguntar "¿volverá en los próximos 6 meses, sí o no?". Así las últimas visitas entran al dataset como ejemplos negativos.
- **Usar análisis de supervivencia** (Kaplan-Meier, Cox), que está diseñado exactamente para este caso: datos censurados, donde el evento aún no ocurrió pero la observación sigue siendo informativa.

### Mejora 3 — Limpieza de datos más rigurosa

La limpieza actual funciona, pero es **silenciosa**: `errors="coerce"` convierte basura en nulo, `dropna` bota filas, y nadie se entera de cuánto se perdió ni por qué.

Los 2.765 registros descartados por fechas y años invertidos no son ruido aleatorio: son errores sistemáticos de digitación en el sistema del taller. Vale la pena investigarlos antes de botarlos, porque pueden indicar un problema en el proceso de captura que se puede corregir en el origen.

Falta además un **contrato de datos explícito**. Hoy, si el taller re-exporta la base con las fechas en otro formato, el código no se cae: convierte todo a nulo, bota casi todo, entrena con un puñado de filas y **el CI sale verde**. Degradación silenciosa con chulito de aprobado.

`pandera` ya está declarado en `requirements.txt` pero **no se usa en ninguna parte** — quedó como dependencia pendiente. Implementarlo permitiría:

- Validar el esquema del CSV antes de entrenar y fallar fuerte si cambió.
- Fijar un mínimo de filas supervivientes que dispare alarma.
- Validar también la **salida** de la transformación y el catálogo, no solo la entrada.

### Mejora 4 — Modelado

Una vez resuelto lo anterior:

- **Transformar el objetivo con logaritmo** o discretizarlo en rangos (0-3 meses, 3-6, 6+). Con una desviación de 200 días, predecir el valor exacto es una meta mal planteada; predecir el rango es útil y alcanzable.
- **Variables de historial**: número de visitas previas, kilómetros recorridos por mes, valor facturado acumulado, días desde la compra.
- **Gradient boosting** (XGBoost, LightGBM) con validación cruzada y `GridSearchCV`.
- **Separar por tipo de trabajo**: el retorno tras una `COLISION` y tras un mantenimiento de `MECANICA` son fenómenos distintos y quizá merezcan modelos distintos.

### Mejora 5 — Ingeniería

| Pendiente | Por qué importa |
|---|---|
| Fijar `scikit-learn==1.9.0` en vez de `>=1.4.0` | Los `.joblib` son pickles atados a la versión que los creó; sin fijarla, el CI puede instalar otra y generar advertencias o resultados inválidos |
| Registrar las métricas históricas | Hoy el MAE se imprime en consola y se pierde. Sin registro no se sabe si un cambio mejoró o empeoró el modelo |
| Versionar los artefactos del modelo | Los `.joblib` se sobrescriben en cada entrenamiento; no hay forma de volver a una versión anterior |
| Umbral mínimo de calidad en el CI | El CI valida que el código corra, no que el modelo sirva. Podría fallar el build si el MAE supera cierto límite |
| Autenticación en la API | Hoy cualquiera con la URL puede consultar el historial de cualquier chasis |
| Dockerizar | Elimina el "en mi máquina sí funciona" en el despliegue, no solo en el desarrollo |

---

## 7. Integración continua

El workflow `.github/workflows/ci.yml` se dispara en cada `push` y `pull_request` a `main`, levanta una máquina Ubuntu limpia con Python 3.12 y ejecuta:

1. `pip install -r requirements.txt` — reconstruye el entorno desde cero
2. `python -m ruff check .` — inspección estática
3. `python -m src.train` — reentrena (valida la reproducibilidad de punta a punta)
4. `python -m pytest` — ejecuta las 7 pruebas

Si cualquier paso falla, el commit queda en rojo.

### Qué cubren las pruebas

| Prueba | Qué verifica |
|---|---|
| `test_health_check` | El servicio responde y ambos artefactos cargaron |
| `test_prediccion_exitosa` | Camino feliz de `POST /predict` |
| `test_validacion_pydantic_error_422` | Rechaza tipos incorrectos |
| `test_anio_modelo_fuera_de_rango_422` | Respeta los límites del contrato |
| `test_prediccion_por_vin_exitosa` | Autocompletado por chasis |
| `test_vin_inexistente_404` | Un VIN sin historial no tumba el servicio |
| `test_vin_con_datos_incompletos_no_rompe` | Un chasis sin año ni kilometraje responde igual, con `null` |
