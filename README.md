# API de Retención Predictiva — Automotor.co S.A.S.


Proyecto de MLOps que toma la base transaccional de órdenes de trabajo de un taller
automotriz, entrena un modelo de regresión para estimar **cuántos días pasarán hasta
que un vehículo regrese al taller**, congela ese modelo en disco y lo expone como
una **API REST con FastAPI**, todo respaldado por pruebas automáticas e integración
continua en GitHub Actions.

---

## El problema de negocio

El taller no sabe cuándo va a volver cada cliente. Eso hace imposible planear
la capacidad de bahías, la compra de repuestos y las campañas de retención.

La pregunta que responde este proyecto es concreta:

> Dado un vehículo que acaba de salir del taller, ¿en cuántos días volverá?

La respuesta permite anticipar la carga de trabajo y contactar al cliente **antes**
de que se vaya a la competencia.

---

## Arquitectura del proyecto

```
Trabajo de Dataops/
│
├── data/
│   └── Base de datos Limpia..csv      # Materia prima: 34.935 sub-trabajos históricos
│
├── src/                               # Capa de entrenamiento (modular, por responsabilidad)
│   ├── config.py                      # Rutas, semilla, nombres de columnas (fuente única de verdad)
│   ├── data.py                        # Carga, consolidación por OT/VIN y creación del target
│   ├── features.py                    # ColumnTransformer: escalado, imputación y codificación
│   ├── evaluate.py                    # Métricas de regresión (MAE, RMSE, R²)
│   └── train.py                       # Orquestador: entrena, evalúa y serializa
│
├── models/
│   ├── modelo_retencion.joblib        # Pipeline COMPLETO congelado (preprocesador + modelo)
│   └── catalogo_vehiculos.joblib      # Ficha por VIN para autocompletar la búsqueda por chasis
│
├── main.py                            # API REST con FastAPI (health check + /predict + /predict/vin)
│
├── tests/
│   └── test_api.py                    # Pruebas de integración con TestClient
│
├── .github/workflows/ci.yml           # Integración continua: ruff → train → pytest
├── requirements.txt                   # Dependencias exactas del entorno            
└── README.md
```
---

## Del CSV crudo al dataset de entrenamiento

El dataset original **no sirve tal cual** para entrenar. Estas son las tres
transformaciones críticas que hace `src/data.py`:

### Consolidación transaccional por OT y VIN
El CSV está a nivel de *sub-trabajo*: una misma visita al taller genera varias filas
(`Núm. Trabajo` 1, 2, 3...). Se agrupa por `VIN` + `OT` para obtener **una fila por
ingreso físico real** del vehículo, tomando la fecha de entrada mínima, la de cierre
máxima y el kilometraje más alto.

> 34.935 sub-trabajos → **27.697 visitas reales** de **9.729 vehículos únicos**

### Generación de la variable objetivo con `shift(-1)`
La columna a predecir no existe en el CSV: se construye. Ordenando las visitas de
cada `VIN` por fecha y usando `shift(-1)` se trae la fecha de la **siguiente** visita
a la fila actual:

```
Dias_Hasta_Retorno = (F. Entrada de la siguiente visita) - (F. Cierre de la visita actual)
```

Las visitas que no tienen una siguiente (el último ingreso de cada vehículo) se
descartan, porque no tienen respuesta que aprender.

> 27.697 visitas → **15.196 registros entrenables** (mediana: 67 días, media: 135,5 días)

### características
- `Antiguedad_Vehiculo` = año de la visita − año del modelo.
- `Es_Vehiculo_Vendido` = bandera 0/1 derivada de si la **primera** OT del chasis fue
  un alistamiento de vehículo nuevo (indicio de que el concesionario lo vendió).

También se aplican tres filtros de sanidad, todos contra errores de digitación del
sistema del taller: se eliminan retornos negativos, antigüedades negativas y
kilometrajes imposibles (la base traía valores como `123.456.789`, que distorsionaban
la media y la desviación calculadas por el `StandardScaler` y con ello toda la escala
de la variable `Kms.`).

---

## El pipeline de Machine Learning

| Tipo de variable | Columnas | Tratamiento |
|---|---|---|
| Numéricas | `Kms.`, `Antiguedad_Vehiculo` | `SimpleImputer(median)` → `StandardScaler` |
| Categóricas | `Gama`, `Tipo Cargo`, `Tipo de Trabajo` | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")` |
| Binarias | `Es_Vehiculo_Vendido` | `SimpleImputer(most_frequent)` |

- **Estimador:** `RandomForestRegressor(n_estimators=100, random_state=42)`
- **División:** 80 % entrenamiento / 20 % test, con semilla fija (reproducibilidad)
- **Serialización:** se guarda el **Pipeline completo**, no solo el estimador

Este último punto es la buena práctica central de la Clase 6: el `.joblib` incluye el
`ColumnTransformer`. Gracias a eso, la API **no reimplementa ni una línea** del
preprocesamiento, lo que elimina el riesgo de *training/serving skew* (que los datos
se preparen distinto al entrenar y al predecir).

---

## Resultados obtenidos

Métricas reales sobre el 20 % de test (datos nunca vistos por el modelo):

| Métrica | Valor | Interpretación |
|---|---|---|
| MAE | **119,23 días** | Desfase promedio entre lo predicho y el retorno real |
| RMSE | **209,86 días** | Muy superior al MAE: hay casos con errores extremos |
| R² | **−0,0580** | El modelo aún **no supera** a predecir siempre el promedio |

Sobre 15.190 visitas con retorno registrado, tras aplicar los filtros de sanidad.

### Lectura honesta de estos números

El R² negativo indica que, con las variables disponibles, el modelo todavía no captura
el fenómeno. Esto **no es un fallo del pipeline** sino un hallazgo del negocio: el
momento en que un cliente regresa depende de factores que hoy no están en el dataset
(hábitos de uso, campañas de mercadeo, siniestros, garantías vencidas), y la variable
objetivo tiene una dispersión enorme (mediana 67 días frente a media 135,5).

El objetivo declarado del Hito 1 es tener el **ciclo de vida completo funcionando y
reproducible**, y ese objetivo se cumple. La mejora del desempeño predictivo queda
como trabajo del siguiente hito:

- Incorporar variables de historial (número de visitas previas, kilómetros/mes, valor facturado).
- Transformar el objetivo con logaritmo o discretizarlo en rangos (0-3 meses, 3-6, 6+).
- Probar modelos de gradient boosting y ajustar hiperparámetros con validación cruzada.

---

## La API REST

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/` | Health check: confirma que el servicio vive y que el modelo y el catálogo cargaron |
| `POST` | `/predict` | Recibe los datos del vehículo a mano y devuelve los días estimados |
| `GET` | `/predict/vin/{vin}` | Busca el vehículo por chasis, autocompleta sus datos y predice |
| `GET` | `/docs` | Documentación interactiva Swagger UI (autogenerada) |

### Predicción manual — `POST /predict`

**Contrato de entrada** (validado por Pydantic con la clase `SolicitudVehiculo`):

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

Se pide el **año del modelo** (el de la tarjeta de propiedad) y no la antigüedad,
porque es el dato que el asesor tiene a la vista. La API traduce año → antigüedad
internamente, replicando el mismo cálculo con que se entrenó el modelo. El campo
está acotado entre 1990 y el año entrante: fuera de ese rango la API responde `422`,
porque el modelo nunca vio antigüedades mayores a 29 años.

**Respuesta:**

```json
{
  "prediccion_dias_retorno": 128.4,
  "unidad": "días"
}
```

### Predicción por chasis — `GET /predict/vin/{vin}`

Cuando el vehículo ya tiene historial en el taller, el asesor no necesita escribir
nada: basta el VIN. La API busca la ficha en el catálogo congelado
(`models/catalogo_vehiculos.joblib`, 9.729 vehículos), autocompleta las seis
características y llama al **mismo modelo**.

> **El VIN nunca entra al modelo.** Es solo la llave de búsqueda. Un identificador
> único no es un predictor: codificar 9.729 chasis produciría un modelo que memoriza
> cada carro en vez de aprender el patrón, y que no sabría nada de un vehículo nuevo.

Ejemplo — `GET /predict/vin/ADM130850`:

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
  "prediccion_dias_retorno": 102.4,
  "unidad": "días"
}
```

La respuesta devuelve también los datos encontrados, para que el asesor pueda
verificar que el sistema buscó el vehículo correcto antes de confiar en la cifra.

**Manejo de errores:**

| Código | Cuándo ocurre |
|---|---|
| `422` | Pydantic rechaza el JSON (falta un campo, llega texto donde va un número, o el año del modelo está fuera de rango) |
| `404` | El VIN consultado no tiene historial en el taller |
| `500` | Falta el `.joblib` del modelo o del catálogo: hay que ejecutar `python -m src.train` primero |
| `400` | Error inesperado durante la inferencia |

---

## Reproducción del proyecto paso a paso

```powershell
# 1. Clonar el repositorio y entrar a la carpeta
git clone <url-del-repositorio>
cd "Trabajo de Dataops"

# 2. Crear y activar el entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar el modelo y congelarlo en models/
python -m src.train

# 5. Auditar el código y ejecutar las pruebas
python -m ruff check .
python -m pytest

# 6. Levantar la API en local
uvicorn main:app --reload
```

---

## Calidad e integración continua

El workflow `.github/workflows/ci.yml` se dispara en cada `push` y `pull_request`
a `main`, levanta una máquina Ubuntu limpia con Python 3.12 y ejecuta:

1. `pip install -r requirements.txt` — reconstruye el entorno desde cero
2. `python -m ruff check .` — inspección estática de código
3. `python -m src.train` — reentrena el modelo (valida la reproducibilidad end-to-end)
4. `python -m pytest` — ejecuta las 6 pruebas de la API

Si cualquier paso falla, el commit queda marcado en rojo. Esto garantiza que el
proyecto funciona en una máquina distinta a la del autor.

**Estado actual verificado en local:** `ruff` sin hallazgos, **6/6 pruebas pasando**.

---

