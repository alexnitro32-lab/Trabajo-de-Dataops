# Predicción de Retención en Concesionario Automotriz

Una API que, en el momento en que un vehículo ingresa al concesionario, predice **cuándo va a volver**, **si el cliente va a dejar de volver** y **entre qué fechas hay que llamarlo**, a partir del historial de órdenes de trabajo.

## ¿Qué problema resuelve?

El concesionario no sabe cuándo va a regresar cada cliente. Sin eso no se puede planear cuántas bahías se van a necesitar, qué repuestos comprar, ni a quién llamar antes de que se lo lleve la competencia.

El proyecto responde tres preguntas:

> **1.** Este vehículo acaba de ingresar. ¿En cuántos días volverá?
> **2.** ¿Este cliente va a dejar de volver?
> **3.** ¿Cuándo hay que llamarlo?

La primera sirve para **agendar y dimensionar**. La segunda dispara la **acción comercial**: a quién retener antes de perderlo. La tercera convierte las dos anteriores en una **instrucción concreta para el asesor**. Las dos primeras son dos modelos separados porque son dos preguntas distintas; la tercera combina sus resultados con el plan de mantenimiento de fábrica.

La unidad de predicción es el **chasis (VIN)**, que en esta base equivale al cliente. Lo que se usa para predecir es su conducta observada — cada cuánto viene volviendo, cuántos kilómetros hace por día, cuánto lleva sin aparecer — y no la gama del vehículo. El porqué está en [ARQUITECTURA.md](ARQUITECTURA.md#el-sujeto-el-chasis-no-la-gama).

---

## Qué tan bien funciona

| Modelo | Pregunta | Métrica | Resultado |
|---|---|---|---|
| Regresión | ¿En cuántos días vuelve? | MAE · R² | 100,5 días · R² = +0,115 |
| **Clasificación** | ¿Va a dejar de volver? | **AUC** · PR-AUC | **0,727** · 0,481 |

**La clasificación de fuga es la más confiable para operar.** Un AUC de 0,727 ordena bien a los clientes por riesgo y sirve para priorizar llamadas. La regresión le gana a las referencias sin modelo, pero por poco margen: su error medio es de unos 98 días. Por eso la API no entrega un solo día sino una **ventana de llamada** del ancho de ese error, y nunca agenda al cliente después de la fecha en que le toca el mantenimiento según su plan de fábrica.

La evidencia completa está en [ARQUITECTURA.md § 6](ARQUITECTURA.md#6-qué-le-falta-y-qué-mejoraría) y se recalcula en cada entrenamiento dentro de `models/metricas_entrenamiento.json`.

Dos hallazgos del diagnóstico que cambian cómo se lee el negocio:

- **De 3.799 vehículos entregados, solo 1.581 volvieron alguna vez (41,6 %).** Los otros 2.218 nunca regresaron después del alistamiento: son clientes que ya compraron y que el concesionario nunca capturó.
- **El vehículo nuevo no vuelve al año: vuelve a los 165 días de mediana.** La mitad aparece antes de los seis meses, así que programar la primera llamada para el mes 12 llega tarde para medio parque.

---

## Requisitos

- **Python 3.12 o superior**

---

## Instalación

Copia y pega el bloque de tu sistema operativo. No hace falta nada más: los modelos entrenados vienen en el repositorio.

### Windows (PowerShell)

```powershell
git clone https://github.com/alexnitro32-lab/Trabajo-de-Dataops.git
cd Trabajo-de-Dataops

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone https://github.com/alexnitro32-lab/Trabajo-de-Dataops.git
cd Trabajo-de-Dataops

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Prenderlo

```bash
uvicorn main:app --reload
```

Abre **<http://127.0.0.1:8000/docs>** en el navegador. Vas a ver la documentación interactiva: puedes probar la API desde ahí sin escribir código.

Para confirmar que todo cargó bien, abre **<http://127.0.0.1:8000/>** — la dirección raíz, sin `/docs` al final. El navegador muestra directamente este JSON:

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

También se puede ver desde `/docs`: despliega **`GET /` — Estado del servicio**, da clic en **Try it out** y luego en **Execute**.

Si `modelo_cargado` o `historial_cargado` salen en `false`, ve a [Problemas comunes](#problemas-comunes).

### Con Docker (alternativa)

Si tienes Docker instalado, no necesitas Python ni el entorno virtual:

```bash
docker build -t retencion-api .
docker run -p 8000:8000 retencion-api
```

La API queda igual en **<http://127.0.0.1:8000/docs>**. La imagen solo trae lo necesario para predecir (`src/`, `models/` y `main.py`); no trae los datos crudos ni las pruebas.

---

## Cómo se usa

La API tiene un solo endpoint de predicción: **`POST /predict/vin/{vin}/ingreso`**. Se usa cuando el vehículo ingresa al concesionario. El asesor envía el chasis, el kilometraje y para qué vino; la API agrega esa visita a las que el vehículo ya tenía, recalcula su historia y vuelve a predecir. **No hace falta reentrenar ni esperar a que el ingreso llegue a la base de datos.**

1. Abre <http://127.0.0.1:8000/docs>.
2. Despliega **`POST /predict/vin/{vin}/ingreso`** y da clic en **Try it out**.
3. Escribe el VIN, por ejemplo `DEU822806`.
4. En *Request body* pega:
   ```json
   {
     "kilometraje": 20400,
     "tipo_cargo": "Cliente",
     "tipo_trabajo": "MECANICA"
   }
   ```
5. Da clic en **Execute**.

### Qué se envía

| Campo | Obligatorio | Valores | Para qué sirve |
|---|---|---|---|
| `kilometraje` | **Sí** | Número entre 0 y 500.000 | Odómetro con el que entra. No puede ser menor que el de la visita anterior |
| `tipo_cargo` | No (`Cliente`) | `Cliente`, `Garantia`, `Seguros`, `Interno` | Quién paga la visita |
| `tipo_trabajo` | No (`MECANICA`) | `MECANICA`, `COLISION`, `MECANICA Y COLISION`, `OTRO` | Qué se le hace. Un mantenimiento normal es `Cliente` + `MECANICA` |
| `fecha_ingreso` | No (hoy) | `AAAA-MM-DD` | Día en que entró. No puede ser futura ni anterior a su último ingreso registrado |
| `fecha_entrega` | No (= ingreso) | `AAAA-MM-DD` | Día en que se entrega o se espera entregar. El regreso se cuenta desde aquí |

### Qué se recibe

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

| Campo | Qué es |
|---|---|
| `datos_encontrados` | Ficha del vehículo **ya con la visita de hoy**, para que el asesor confirme que la API encontró el carro correcto |
| `ritmo_habitual_dias` | Cada cuánto suele volver: la mediana de los días entre sus visitas |
| `km_por_dia` | Cuánto lo usa: la mediana de los km por día entre sus visitas |
| `prediccion_dias_retorno` | Días hasta el próximo ingreso, contados desde la entrega |
| `riesgo_de_fuga` | Probabilidad de que no vuelva en 365 días, y si supera el umbral de alerta (0,50) |
| `plan_de_llamada.estado` | `programar`, `llamar_ahora` o `llamar_ya` (ver abajo) |
| `plan_de_llamada.mensaje` | La instrucción para el asesor, con las fechas |
| `fecha_estimada_retorno` | Entrega + días predichos |
| `margen_error_dias` | Error medio del modelo; define el ancho de la ventana de llamada |
| `fecha_corte_datos` | Último día que trae la base de datos. Lo que pasó después solo lo conoce la API si se registra en la petición |
| `base_de_la_fecha` | Si la fecha salió del modelo o del plan de mantenimiento, y por qué |

### Cómo se calcula la fecha estimada

Se toma **lo que ocurra primero** entre dos cosas:

1. **Lo que estima el modelo** a partir de la historia del vehículo.
2. **Lo que exige el plan de mantenimiento de fábrica**, según el año del modelo:

| Año del modelo | Plan |
|---|---|
| 2022 en adelante | 365 días o 10.000 km, lo que ocurra primero |
| 2021 o anteriores (o año desconocido) | 6 meses (180 días) o 5.000 km, lo que ocurra primero |

Los kilómetros se convierten en días con los km/día del vehículo. Un carro 2015 que hace 100 km/día cumple 5.000 km en 50 días, y ese es su tope aunque el modelo diga más.

### Cómo se decide cuándo llamar

La **ventana de llamada** va de `fecha estimada − margen de error` a `fecha estimada + margen de error`, y nunca empieza antes de la entrega.

| Estado | Significa |
|---|---|
| `programar` | La ventana todavía no abre. El mensaje dice entre qué fechas llamar |
| `llamar_ahora` | Hoy está dentro de la ventana, **o** el cliente tiene alerta de fuga |
| `llamar_ya` | La ventana ya cerró y no hay registro de que volviera: está atrasado |

**Regla de fuga:** si el modelo de fuga enciende la alerta, no se espera a que abra la ventana. Si el carro todavía está en el concesionario, el mensaje pide dejarle agendada la próxima cita al entregárselo.

### Errores que puede devolver

| Código | Cuándo |
|---|---|
| `404` | El VIN no tiene historial: sería su primer ingreso y no hay ritmo del que estimar su regreso |
| `409` | La fecha de ingreso es igual o anterior al último ingreso registrado |
| `422` | Un dato no es válido: kilometraje menor que el de la visita anterior, tipo de trabajo inexistente, texto donde va un número o fechas incoherentes |
| `500` | Faltan los archivos del modelo o del historial |

---

## Endpoints disponibles

| Método | Ruta | Para qué sirve |
|---|---|---|
| `POST` | `/predict/vin/{vin}/ingreso` | Registra el ingreso y predice regreso, fuga y fechas de llamada |
| `GET` | `/` | Estado detallado: qué cargó, cuántos vehículos hay en el historial, margen de error |
| `GET` | `/health` | Health check corto para el monitoreo: ¿vive y cargaron los modelos? |
| `GET` | `/metrics` | Peticiones, errores, tasa de error y latencias (promedio, p50, p95) desde que arrancó |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

El contrato completo está en [ARQUITECTURA.md § 5](ARQUITECTURA.md#5-qué-recibe-y-devuelve-la-api).

---

## Observabilidad

La API deja rastro de lo que hace para que se pueda vigilar en producción:

- **Logs en JSON**, una línea por evento, en la salida estándar. Se registran las predicciones (VIN, días estimados, probabilidad de fuga, estado de la llamada), los VIN sin historial, los errores de validación y las peticiones que tardan más de 200 ms.
- **Métricas en `/metrics`**: un middleware mide todas las peticiones. Las respuestas 4xx y 5xx cuentan como error.

Ejemplo de una línea de log:

```json
{"timestamp": "2026-10-20T15:02:11Z", "level": "INFO", "logger": "automotor_api", "mensaje": "Predicción de retención ejecutada exitosamente", "evento": "prediccion_exitosa", "vin": "DEU822806", "dias_retorno_predichos": 180.0, "probabilidad_fuga": 0.7758, "alerta_fuga": true, "estado_llamada": "llamar_ahora"}
```

Las métricas viven en memoria: se reinician cada vez que se reinicia la API.

---

## Correr las pruebas

```bash
python -m pytest
```

Esperado: **48 passed**.

Para revisar el estilo del código:

```bash
python -m ruff check .
```

Esperado: **All checks passed!**

GitHub Actions corre en cada push esa misma secuencia en una máquina limpia: `ruff` → `python -m src.train` → `pytest`. Es decir, reentrena desde el CSV versionado antes de probar.

---

## Actualizar los datos sin reentrenar

Cuando llega una base nueva del ERP, reemplaza el CSV en `data/` y corre:

```bash
python -m src.actualizar_catalogo
```

Tarda segundos. Reconstruye el historial de visitas y el catálogo de vehículos que consulta la API, **sin tocar los modelos**: lo que cambia de un día para otro son las visitas, no la relación que aprendió el modelo. Reinicia la API después para que tome los datos nuevos.

---

## Reentrenar el modelo

El repositorio ya trae los modelos entrenados en `models/`, así que **no necesitas entrenar para usar la API**. Si cambias el código de `src/` o quieres que el modelo aprenda de una base mucho más grande:

```bash
python -m src.train
```

Tarda un par de minutos (valida de forma cruzada, así que entrena varias veces) y escribe cinco archivos en `models/`: los dos modelos, el catálogo, el historial de visitas y `metricas_entrenamiento.json` con toda la evidencia numérica.

---

## Problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `modelo_cargado: false` | Los `.joblib` no existen | `python -m src.train` |
| `historial_cargado: false` | Falta `models/historial_visitas.joblib` | `python -m src.actualizar_catalogo` |
| `/docs` muestra una versión vieja | Hay otro `uvicorn` corriendo en el puerto 8000, o el navegador guardó la página | Cierra todas las terminales con `uvicorn`, abre una sola y recarga con **Ctrl + F5** |
| `404` con un VIN que sí vino | Su ingreso aún no está en la base | Actualiza los datos con `python -m src.actualizar_catalogo` |
| `422` por kilometraje | El odómetro enviado es menor que el de la visita anterior | Verifica el kilometraje con el asesor |
| `Activate.ps1` no ejecuta | Política de ejecución de Windows | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Documentación técnica

¿Cómo funciona por dentro, qué hace cada archivo y por qué se tomó cada decisión?

👉 **[ARQUITECTURA.md](ARQUITECTURA.md)**
