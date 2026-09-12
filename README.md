# API de Retención Predictiva — Taller Automotor.co S.A.S.

Una API que predice **en cuántos días volverá un vehículo al taller**, a partir del historial de órdenes de trabajo del concesionario.

## ¿Qué problema resuelve?

Hoy el taller no sabe cuándo va a regresar cada cliente. Sin eso no se puede planear cuántas bahías se van a necesitar, qué repuestos comprar, ni a quién llamar antes de que se lo lleve la competencia.

Este proyecto responde una pregunta concreta:

> Un vehículo acaba de salir del taller. ¿En cuántos días volverá?

Con esa cifra, el asesor puede contactar al cliente **antes** de que se le pase el turno — y el taller puede anticipar su carga de trabajo.

> ⚠️ **Antes de usarlo en producción, lee [Estado actual del modelo](#estado-actual-del-modelo).** El pipeline funciona de punta a punta, pero la precisión todavía no es de nivel productivo.

---

## Requisitos

- **Python 3.12 o superior** — verifica con `python --version`
- **Git**

Nada más. No hace falta base de datos ni Docker.

---

## Instalación

Copia y pega. Toma unos 5 minutos, casi todo descargando librerías.

### Windows (PowerShell)

```powershell
git clone https://github.com/alexnitro32-lab/Trabajo-de-Dataops.git
cd "Trabajo-de-Dataops"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone https://github.com/alexnitro32-lab/Trabajo-de-Dataops.git
cd "Trabajo-de-Dataops"

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **¿`Activate.ps1` te dio error de permisos en Windows?** Ejecuta esto una sola vez y vuelve a intentar:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

---

## Prenderlo

```bash
uvicorn main:app --reload
```

Abre **<http://127.0.0.1:8000/docs>** en el navegador. Vas a ver la documentación interactiva: puedes probar cada endpoint desde ahí sin escribir código.

Para confirmar que el modelo cargó bien, entra a <http://127.0.0.1:8000/> y deberías ver:

```json
{
  "mensaje": "API de Retención Predictiva de Automotor.co S.A.S. activa 🚗",
  "modelo_cargado": true,
  "catalogo_cargado": true,
  "vehiculos_en_catalogo": 9729
}
```

Si `modelo_cargado` sale en `false`, ve a [Problemas comunes](#problemas-comunes).

---

## Ejemplo de uso real

### Opción A — consultar por chasis (VIN)

Si el vehículo ya estuvo en el taller, basta el número de chasis. La API completa el resto sola.

```bash
curl http://127.0.0.1:8000/predict/vin/ADM130850
```

**Respuesta real:**

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

Devuelve también los datos que encontró, para que el asesor confirme que buscó el carro correcto antes de creerle a la cifra.

### Opción B — ingresar los datos a mano

Para un vehículo sin historial en el taller:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gama": "Tucson",
    "Tipo Cargo": "Cliente",
    "Tipo de Trabajo": "MECANICA",
    "Kms.": 45000.0,
    "Año Modelo": 2022,
    "Es_Vehiculo_Vendido": true
  }'
```

**Respuesta real:**

```json
{
  "prediccion_dias_retorno": 484.9,
  "unidad": "días"
}
```

En **PowerShell**, `curl` es otra cosa. Usa esto:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{
  "Gama": "Tucson",
  "Tipo Cargo": "Cliente",
  "Tipo de Trabajo": "MECANICA",
  "Kms.": 45000.0,
  "Año Modelo": 2022,
  "Es_Vehiculo_Vendido": true
}'
```

---

## Correr las pruebas

```bash
python -m pytest
```

Esperado: **7 passed**.

Para revisar el estilo del código:

```bash
python -m ruff check .
```

Esperado: **All checks passed!**

---

## Reentrenar el modelo

El repositorio ya trae el modelo entrenado en `models/`, así que **no necesitas entrenar para usar la API**. Pero si cambias el CSV o el código de `src/`, reentrena con:

```bash
python -m src.train
```

Tarda unos segundos y sobrescribe los dos archivos de `models/`. Al final imprime las métricas de evaluación.

---

## Problemas comunes

| Síntoma | Causa | Solución |
|---|---|---|
| `modelo_cargado: false` en `/` | Faltan los `.joblib` de `models/` | `python -m src.train` |
| `ModuleNotFoundError: No module named 'fastapi'` | El entorno virtual no está activo | Actívalo y repite `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Estás parado en otra carpeta | `cd` a la raíz del proyecto (donde está `main.py`) |
| `Activate.ps1 cannot be loaded` | Política de ejecución de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `404` al consultar un VIN | Ese chasis no tiene historial | Usa `POST /predict` e ingresa los datos a mano |
| `InconsistentVersionWarning` de scikit-learn | Tu versión difiere de la que creó el `.joblib` | `python -m src.train` para regenerarlo |

---

## Endpoints disponibles

| Método | Ruta | Para qué sirve |
|---|---|---|
| `GET` | `/` | Confirma que el servicio vive y que el modelo cargó |
| `POST` | `/predict` | Predice con datos ingresados a mano |
| `GET` | `/predict/vin/{vin}` | Predice buscando el vehículo por chasis |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

El detalle de cada contrato está en **[ARQUITECTURA.md](ARQUITECTURA.md)**.

---

## Estado actual del modelo

Seamos claros sobre qué tan bien predice, porque importa para saber en qué confiar:

| Métrica | Valor |
|---|---|
| MAE (error promedio) | **122,7 días** |
| R² | **−0,096** |

Un R² negativo significa que el modelo **todavía no le gana a simplemente decir el promedio**. En la práctica: la cifra sirve como referencia gruesa, no para tomar decisiones comerciales sobre un cliente específico.

Esto no es un bug del pipeline — es un hallazgo del negocio. El momento en que un cliente vuelve depende de cosas que hoy no están en los datos (si quedó conforme con el servicio, el precio que pagó, si lo atendieron rápido). El razonamiento completo y el plan de mejora están en **[ARQUITECTURA.md](ARQUITECTURA.md#6-qué-le-falta-y-qué-mejoraría)**.

---

## Documentación técnica

¿Quieres entender cómo funciona por dentro, qué hace cada archivo y por qué se tomó cada decisión?

👉 **[ARQUITECTURA.md](ARQUITECTURA.md)**
