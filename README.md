# Predicción de Retención en Concesionario Automotriz

Una API que predice **cuándo vuelve un cliente al taller** y **si va a dejar de volver**, a partir del historial de órdenes de trabajo del concesionario.

## ¿Qué problema resuelve?

El taller no sabe cuándo va a regresar cada cliente. Sin eso no se puede planear cuántas bahías se van a necesitar, qué repuestos comprar, ni a quién llamar antes de que se lo lleve la competencia.

El proyecto responde dos preguntas, y son distintas:

> **1.** Un vehículo acaba de salir del taller. ¿En cuántos días volverá?
> **2.** ¿Este cliente va a dejar de volver?

La primera sirve para **agendar y dimensionar**. La segunda dispara la **acción comercial**: a quién llamar antes de perderlo. Son dos modelos separados porque son dos preguntas separadas.

La unidad de predicción es el **chasis (VIN)**, que en esta base equivale al cliente. Lo que se usa para predecir es su conducta observada — cada cuánto viene volviendo, cuántos kilómetros hace por día, cuánto lleva sin aparecer — y no la gama del vehículo. El porqué está en [ARQUITECTURA.md](ARQUITECTURA.md#el-sujeto-el-chasis-no-la-gama).

---

## Qué tan bien funciona

| Modelo | Pregunta | Métrica | Resultado |
|---|---|---|---|
| Regresión | ¿En cuántos días vuelve? | MAE · R² | 100,4 días · R² = +0,115 |
| **Clasificación** | ¿Va a dejar de volver? | **AUC** · PR-AUC | **0,724** · 0,477 |

**La clasificación de fuga es la que se recomienda para operar.** Un AUC de 0,724 ordena bien a los clientes por riesgo y sirve para priorizar llamadas; la regresión le gana a las referencias sin modelo, pero por poco margen, y no alcanza para decidir sobre un cliente concreto. La evidencia completa está en [ARQUITECTURA.md § 6](ARQUITECTURA.md#6-qué-le-falta-y-qué-mejoraría), y se recalcula en cada entrenamiento dentro de `models/metricas_entrenamiento.json`.

Dos hallazgos del diagnóstico que cambian cómo se lee el negocio:

- **De 3.799 vehículos entregados, solo 1.581 volvieron alguna vez (41,6 %).** Los otros 2.218 nunca regresaron después del alistamiento: son clientes que ya compraron y que el taller nunca capturó.
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

Abre **<http://127.0.0.1:8000/docs>** en el navegador. Vas a ver la documentación interactiva: puedes probar cada endpoint desde ahí sin escribir código.

Para confirmar que todo cargó bien, entra a <http://127.0.0.1:8000/> y deberías ver:

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

Si `modelo_cargado` sale en `false`, ve a [Problemas comunes](#problemas-comunes).

---

## Cómo se usa

### Opción A — consultar por chasis (VIN)

Si el vehículo ya estuvo en el taller, basta el número de chasis. La API completa el resto sola.

```bash
curl http://127.0.0.1:8000/predict/vin/ADM130850
```

```json
{
  "vin": "ADM130850",
  "datos_encontrados": {
    "gama": "I10",
    "anio_modelo": 2013,
    "antiguedad_calculada": 7.0,
    "ultimo_kilometraje": 487636.0,
    "ultima_visita": "2020-02-18",
    "vendido_por_nosotros": false,
    "total_visitas": 55,
    "ritmo_habitual_dias": 14.5,
    "km_por_dia": 160.7
  },
  "prediccion_dias_retorno": 28.2,
  "unidad": "días",
  "origen_estimacion": "modelo",
  "regla_preventiva_dias": 62.2,
  "riesgo_de_fuga": {
    "probabilidad_fuga": 0.5127,
    "alerta": true,
    "segmento": "externos",
    "probabilidad_fuga_modelo_del_segmento": 0.525
  }
}
```

La respuesta trae `datos_encontrados` para que el asesor confirme que la API buscó el carro correcto antes de creerle a la cifra, y `regla_preventiva_dias` al lado de la predicción: la brecha entre lo que dice el plan de fábrica y lo que dice el modelo es información.

> **Vehículos de un solo ingreso.** Si el chasis solo ha venido una vez, no existe ningún intervalo observado del cual estimar su ritmo. La API lo declara (`"origen_estimacion": "regla_preventiva"` más una `advertencia`) y devuelve la regla de fábrica, en lugar de disfrazar de predicción personalizada lo que sería la mediana de la población.

### Opción B — ingresar los datos a mano

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Tipo Cargo": "Cliente",
    "Tipo de Trabajo": "MECANICA",
    "Kms.": 45000.0,
    "Año Modelo": 2022,
    "Es_Vehiculo_Vendido": true,
    "Visitas_Previas": 3,
    "Dias_Desde_Visita_Anterior": 180.0,
    "Kms_Visita_Anterior": 32000.0
  }'
```

```json
{
  "prediccion_dias_retorno": 196.6,
  "unidad": "días",
  "regla_preventiva_dias": 138.5,
  "vehiculo_atrasado_frente_a_regla": true,
  "riesgo_de_fuga": {
    "probabilidad_fuga": 0.5762,
    "alerta": true,
    "segmento": "vendidos",
    "probabilidad_fuga_modelo_del_segmento": 0.5524
  }
}
```

Los tres últimos campos del envío son la **historia del vehículo** y son obligatorios: son los datos que más pesan en la predicción. Sin ellos la API responde `422` en lugar de devolver un número sin fundamento.

En **PowerShell**, `curl` es otra cosa. Usa esto:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{
  "Tipo Cargo": "Cliente",
  "Tipo de Trabajo": "MECANICA",
  "Kms.": 45000.0,
  "Año Modelo": 2022,
  "Es_Vehiculo_Vendido": true,
  "Visitas_Previas": 3,
  "Dias_Desde_Visita_Anterior": 180.0,
  "Kms_Visita_Anterior": 32000.0
}'
```

### Opción C — ¿este cliente se va a ir?

```bash
curl http://127.0.0.1:8000/fuga/vin/ADM130850
```

```json
{
  "vin": "ADM130850",
  "riesgo_de_fuga": {
    "algoritmo": "RandomForest",
    "probabilidad_fuga": 0.5127,
    "alerta": true,
    "segmento": "externos",
    "probabilidad_fuga_modelo_del_segmento": 0.525
  },
  "historia_que_lo_sustenta": {
    "total_visitas": 55,
    "ritmo_habitual_dias": 14.5,
    "dias_desde_visita_anterior": 24.0,
    "km_por_dia": 160.7,
    "vendido_por_nosotros": false
  }
}
```

Devuelve dos probabilidades — la del modelo global y la del modelo del segmento al que pertenece el vehículo — y la **historia que sustenta la cifra**, porque un asesor no puede llamar a un cliente con un número sin contexto. Un chasis de un solo ingreso responde `409`: sin un segundo ingreso no hay ritmo contra el cual comparar.

---

## Endpoints disponibles

| Método | Ruta | Para qué sirve |
|---|---|---|
| `GET` | `/` | Confirma que el servicio vive y que los modelos cargaron |
| `POST` | `/predict` | Predice con datos ingresados a mano |
| `GET` | `/predict/vin/{vin}` | Predice buscando el vehículo por chasis |
| `GET` | `/fuga/vin/{vin}` | Probabilidad de que ese cliente deje de volver |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

El contrato completo de cada uno — campos, rangos válidos y códigos de error — está en [ARQUITECTURA.md § 5](ARQUITECTURA.md#5-qué-recibe-y-devuelve-la-api).

---

## Correr las pruebas

```bash
python -m pytest
```

Esperado: **37 passed**.

Para revisar el estilo del código:

```bash
python -m ruff check .
```

Esperado: **All checks passed!**

GitHub Actions corre en cada push esa misma secuencia en una máquina limpia: `ruff` → `python -m src.train` → `pytest`. Es decir, reentrena desde el CSV versionado antes de probar.

---

## Reentrenar el modelo

El repositorio ya trae los modelos entrenados en `models/`, así que **no necesitas entrenar para usar la API**. Si cambias el CSV o el código de `src/`:

```bash
python -m src.train
```

Tarda unos minutos (valida de forma cruzada, así que entrena varias veces) y escribe cinco archivos en `models/`: los tres modelos, el catálogo de vehículos y `metricas_entrenamiento.json` con toda la evidencia numérica.

---

## Problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `modelo_cargado: false` | Los `.joblib` no existen | `python -m src.train` |
| `409` al consultar fuga | El chasis tiene un solo ingreso | No es calculable; usa `POST /predict` con la historia a mano |
| `422` en `POST /predict` | Falta la historia del vehículo | Envía `Visitas_Previas` y `Dias_Desde_Visita_Anterior` |
| `Activate.ps1` no ejecuta | Política de ejecución de Windows | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Documentación técnica

¿Cómo funciona por dentro, qué hace cada archivo y por qué se tomó cada decisión?

👉 **[ARQUITECTURA.md](ARQUITECTURA.md)**
