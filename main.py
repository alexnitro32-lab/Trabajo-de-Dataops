# Este archivo convierte el modelo congelado en un SERVICIO WEB. Sin el, el modelo solo serviria dentro de un cuaderno de Python; con el, cualquier
# sistema del concesionario puede registrar el ingreso de un chasis y saber cuando vuelve, si se va a ir y cuando hay que llamarlo.

import json
import time
from datetime import date, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from src import config, data, features
from src.logger import obtener_logger
from src.metrics import METRICAS_API

# Logger estructurado en JSON (una linea por evento, lista para Render o cualquier monitoreo)
logger = obtener_logger("automotor_api")

# INICIALIZACION DE LA APLICACION FASTAPI
app = FastAPI(
    title="API de Retención Predictiva - Concesionario",
    description=(
        "Servicio de inferencia sobre el retorno de clientes al concesionario. El asesor registra "
        "el ingreso del vehículo (VIN, kilometraje y para qué vino) y la API recalcula su "
        "historia con esa visita y responde tres cosas: **cuándo** vuelve (regresión, "
        "limitada por el plan de mantenimiento de su año de modelo), **si va a dejar de "
        "volver** (clasificación de fuga) y **entre qué fechas hay que llamarlo** (la fecha "
        "estimada de retorno ± el error medio del modelo). Todo se apoya en la HISTORIA del "
        "chasis — cada cuánto venía volviendo, cuántos km hace por día — y no en la gama."
    ),
    version="6.0.0"
)


# MIDDLEWARE DE OBSERVABILIDAD: mide cuanto tarda CADA peticion, la suma al
# acumulador de metricas (exito/fallo) y avisa en el log si pasa de 200 ms.
@app.middleware("http")
async def middleware_observabilidad(request: Request, call_next):
    inicio_tiempo = time.perf_counter()
    exitoso = True
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            exitoso = False
        return response
    except Exception:
        exitoso = False
        raise
    finally:
        latencia_ms = round((time.perf_counter() - inicio_tiempo) * 1000, 2)
        METRICAS_API.registrar_peticion(latencia_ms=latencia_ms, exitoso=exitoso)
        if latencia_ms > 200.0:
            logger.warning(
                "Petición lenta detectada",
                extra={"ruta": request.url.path, "metodo": request.method, "latencia_ms": latencia_ms},
            )


# MANEJADOR DE ERRORES DE VALIDACION: el 422 de Pydantic sigue igual, pero
# ahora queda registrado en el log. jsonable_encoder evita que un error con
# objetos no serializables (p. ej. un ValueError dentro de "ctx") rompa el JSON.
@app.exception_handler(RequestValidationError)
async def manejador_error_validacion(request: Request, exc: RequestValidationError):
    errores = jsonable_encoder(exc.errors())
    logger.error(
        "Error de validación en la petición recibida",
        extra={"ruta": request.url.path, "metodo": request.method, "errores": errores},
    )
    return JSONResponse(
        status_code=422,
        content={"detalle": "Datos de entrada inválidos", "errores": errores},
    )

# CARGA DEL PIPELINE COMPLETO CONGELADO (.joblib) MUY IMPORTANTE: esto esta FUERA de las funciones, por lo tanto se ejecuta UNA SOLA VEZ cuando arranca el servidor, no en cada peticion. Si estuviera dentro del endpoint, cada usuario obligaria a leer el modelo del disco otra vez.
try:
    pipeline = joblib.load(config.RUTA_MODELO_SERIALIZADO)
except Exception: # Si el .joblib no existe (por ejemplo, nunca se corrio el entrenamiento), NO tumbamos el servidor: lo dejamos en None y avisamos en los endpoints. Asi el health check sigue respondiendo y se puede diagnosticar el problema.
    pipeline = None

# Todas las visitas de cada VIN (no solo la ultima). Hacen falta para recalcular
# la historia cuando el carro entra: la mediana de sus intervalos o sus km/dia
# no se pueden actualizar sabiendo solo el resultado anterior.
try:
    historial = joblib.load(config.RUTA_HISTORIAL_VISITAS)
except Exception:
    historial = None

# Clasificador de fuga: "este cliente va a dejar de volver". Se carga aparte del
# regresor porque responde otra pregunta y puede evolucionar por su cuenta.
try:
    _paquete_fuga = joblib.load(config.RUTA_MODELO_FUGA)
    modelo_fuga = _paquete_fuga["modelo"]
    algoritmo_fuga = _paquete_fuga["algoritmo"]
except Exception:
    modelo_fuga, algoritmo_fuga = None, None

# ERROR MEDIO DEL MODELO (MAE). Es el ancho de la ventana de llamada: en promedio
# el retorno real cae a +/- este numero de dias de la prediccion. Se lee del
# holdout agrupado por VIN que deja el entrenamiento, para que la ventana se
# actualice sola cada vez que se reentrena. Si el archivo no esta, se usa la
# media de la validacion cruzada como respaldo y, en ultima instancia, None.
try:
    _regresion = json.loads(config.RUTA_METRICAS.read_text(encoding="utf-8"))["regresion"]
    MAE_DIAS = float(
        _regresion.get("holdout_agrupado_por_vin", {}).get("MAE_dias")
        or _regresion["validacion_cruzada_por_vin"]["media_MAE_dias"]
    )
except Exception:
    MAE_DIAS = None

# FECHA DE CORTE: el ultimo ingreso que la base alcanza a ver. Lo que paso
# despues solo lo conoce la API si el asesor lo registra en la peticion.
FECHA_CORTE_DATOS = (
    None if historial is None
    else pd.Timestamp(historial[config.COLUMNA_FECHA_ENTRADA].max()).date()
)


# AYUDANTES PARA DATOS INCOMPLETOS DEL HISTORIAL

def _a_float(valor):
    """Convierte un valor de la ficha a float, o a None si viene vacío/nulo.

    Se usa para que un dato faltante viaje como null en el JSON de respuesta.
    """
    if valor is None or pd.isna(valor):
        return None
    return float(valor)


def _a_fecha(valor):
    """Convierte una marca de tiempo de la ficha a texto 'AAAA-MM-DD', o None."""
    if valor is None or pd.isna(valor):
        return None
    return str(pd.Timestamp(valor).date())


def _predecir(df_entrada: pd.DataFrame) -> float:
    """Ejecuta el pipeline sobre una fila y devuelve los días estimados.

    El reindex garantiza que las columnas lleguen con el MISMO nombre y en el
    mismo orden con que se entrenó: el ColumnTransformer busca por nombre exacto
    y una columna de menos rompe la inferencia. El clip a 0 evita devolver días
    negativos, que no tienen sentido físico aunque un árbol pueda promediarlos.
    """
    df_entrada = df_entrada.reindex(columns=features.columnas_modelo())
    return float(np.clip(pipeline.predict(df_entrada)[0], 0, None))


def _fila_desde_ficha(ficha: pd.Series) -> pd.DataFrame:
    """Reconstruye el vector de características de un chasis desde su ficha.

    La ficha ya trae las mismas columnas con las que se entrenó, calculadas con
    las mismas fórmulas. Solo se fuerza el tipo numérico, porque al pasar por
    una Serie de pandas todo llega como objeto.
    """
    fila = ficha.to_frame().T.reindex(columns=features.columnas_modelo())
    for columna in config.COLUMNAS_NUMERICAS + config.COLUMNAS_BINARIAS:
        fila[columna] = pd.to_numeric(fila[columna], errors="coerce")
    return fila


def _clasificar_fuga(fila: pd.DataFrame) -> dict:
    """Probabilidad de que el cliente NO vuelva dentro del plazo de la regla."""
    probabilidad = float(modelo_fuga.predict_proba(fila)[0, 1])
    return {
        "probabilidad_fuga": round(probabilidad, 4),
        "alerta": bool(probabilidad >= config.UMBRAL_DECISION_FUGA),
    }


def _plan_mantenimiento(anio_modelo: float | None) -> dict:
    """Plan de fábrica que le corresponde al vehículo según su año de modelo.

    Es la misma regla con la que se entrenó (src/data.py): si cambia, cambia
    en los dos lados a la vez.
    """
    return data.plan_mantenimiento(anio_modelo)


def _dias_hasta_mantenimiento(anio_modelo: float | None, km_por_dia: float | None) -> tuple[float, str]:
    """Días hasta que al vehículo le toca mantenimiento, y por qué.

    Es "lo que ocurra primero" entre el plazo en días del plan y los días que
    tarda en recorrer los km del plan a su ritmo de uso. Si no se conoce cuántos
    km hace por día, solo cuenta el plazo en días.
    """
    plan = _plan_mantenimiento(anio_modelo)
    if km_por_dia is not None and km_por_dia > 0:
        dias_por_km = plan["kms"] / km_por_dia
        if dias_por_km < plan["dias"]:
            kms_plan = f"{plan['kms']:,}".replace(",", ".")  # 5000 -> "5.000"
            return dias_por_km, (f"a {km_por_dia:.1f} km/día cumple {kms_plan} km en "
                                 f"{dias_por_km:.0f} días (plan {plan['nombre']})")
    return float(plan["dias"]), f"cumple {plan['dias']} días (plan {plan['nombre']})"


def _ventana_llamada(fecha_base: date, dias_retorno: float) -> tuple[date, date, date]:
    """Devuelve (retorno estimado, llamar desde, llamar hasta).

    La predicción no es exacta: en promedio el retorno real cae a +/- MAE días
    de ella. Por eso no se devuelve un solo día sino una VENTANA:

        retorno estimado = fecha de entrega + días predichos
        ventana          = [retorno - MAE, retorno + MAE]

    El inicio nunca puede quedar antes de la entrega (no se llama a un cliente
    que todavía tiene el carro en el concesionario).
    """
    margen = MAE_DIAS if MAE_DIAS is not None else 0.0
    retorno = fecha_base + timedelta(days=round(dias_retorno))
    desde = max(fecha_base + timedelta(days=1), retorno - timedelta(days=round(margen)))
    hasta = retorno + timedelta(days=round(margen))
    return retorno, desde, hasta


def _plan_de_llamada(fecha_base: date, dias_retorno: float, fecha_referencia: date,
                     riesgo: dict | None = None) -> dict:
    """Convierte la predicción en una instrucción concreta para el call center.

    Contra la fecha de referencia (hoy, por defecto) el vehículo queda en uno
    de tres estados:

      - "programar":    la ventana todavía no abre -> llamar entre desde/hasta.
      - "llamar_ahora": hoy está dentro de la ventana -> es el momento.
      - "llamar_ya":    la ventana ya cerró y no hay registro de que volviera ->
                        el cliente está atrasado frente a su propio ritmo.

    REGLA DE FUGA: la ventana responde "si vuelve, cuándo"; la alerta responde
    "¿va a volver?". Si el clasificador dice que lo más probable es que NO
    vuelva, esperar a que abra la ventana es esperar a perderlo, así que un
    vehículo en "programar" con alerta pasa a "llamar_ahora".
    """
    retorno, desde, hasta = _ventana_llamada(fecha_base, dias_retorno)
    alerta = bool(riesgo and riesgo["alerta"])

    if fecha_referencia < desde and alerta:
        estado = "llamar_ahora"
        pct = round(100 * riesgo["probabilidad_fuga"])
        if fecha_referencia <= fecha_base:
            # El carro todavía está en el concesionario (o se entrega hoy): el mejor
            # momento para retenerlo es mientras el cliente está presente.
            mensaje = (f"Riesgo alto de fuga ({pct}%): no esperar a la ventana. Dejarle agendada la "
                       f"próxima cita al entregarle el carro (su ventana normal sería del {desde} al {hasta}).")
        else:
            mensaje = (f"Riesgo alto de fuga ({pct}%): llamar ahora, no esperar a que abra la ventana "
                       f"el {desde} (ventana normal: del {desde} al {hasta}).")
    elif fecha_referencia < desde:
        estado = "programar"
        mensaje = f"Llamar entre el {desde} y el {hasta}. Faltan {(desde - fecha_referencia).days} días para abrir la ventana."
    elif fecha_referencia <= hasta:
        estado = "llamar_ahora"
        mensaje = f"Está dentro de la ventana de llamada: llamar antes del {hasta} ({(hasta - fecha_referencia).days} días restantes)."
    else:
        estado = "llamar_ya"
        mensaje = f"Debió volver hacia el {retorno} y la ventana cerró el {hasta}: lleva {(fecha_referencia - hasta).days} días de atraso. Llamar ya."

    return {
        "estado": estado,
        "mensaje": mensaje,
        "fecha_referencia": str(fecha_referencia),
        "fecha_estimada_retorno": str(retorno),
        "margen_error_dias": None if MAE_DIAS is None else round(MAE_DIAS, 1),
        "fecha_corte_datos": None if FECHA_CORTE_DATOS is None else str(FECHA_CORTE_DATOS),
    }


def _respuesta_desde_ficha(vin: str, ficha: pd.Series, fecha_referencia: date) -> dict:
    """Arma la respuesta completa a partir de la ficha recalculada del vehículo."""
    try:
        anio_modelo = _a_float(ficha["Anio_Modelo"])
        km_por_dia = _a_float(ficha["Km_Por_Dia_Medio"])

        df_entrada = _fila_desde_ficha(ficha)
        dias_modelo = _predecir(df_entrada)
        riesgo = _clasificar_fuga(df_entrada) if modelo_fuga is not None else None

        # LO QUE OCURRA PRIMERO: el modelo dice cuándo suele volver según su
        # historia; el plan de mantenimiento dice cuándo le TOCA volver. Agendar
        # al cliente después de que ya le tocaba no tiene sentido.
        dias_plan, motivo_plan = _dias_hasta_mantenimiento(anio_modelo, km_por_dia)
        if dias_plan < dias_modelo:
            dias = dias_plan
            base = (f"Plan de mantenimiento: {motivo_plan}, antes que los "
                    f"{dias_modelo:.0f} días que estima el modelo por su historia.")
        else:
            dias = dias_modelo
            base = (f"Modelo predictivo: según su historia vuelve en {dias_modelo:.0f} días, "
                    f"antes de que le toque el mantenimiento ({motivo_plan}).")

        # El target se entrenó contando desde la ENTREGA (F. Cierre), no desde la
        # entrada: las fechas de llamada tienen que partir del mismo punto.
        fecha_base = ficha["Ultima_Entrega"] if pd.notna(ficha["Ultima_Entrega"]) else ficha["Ultima_Visita"]
        plan = _plan_de_llamada(pd.Timestamp(fecha_base).date(), dias, fecha_referencia, riesgo)
        plan["base_de_la_fecha"] = base
    except Exception as e:
        # Red de seguridad: un dato corrupto en el historial no debe devolverle
        # al asesor un stack trace.
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la predicción del VIN '{vin}': {str(e)}"
        )

    # Devolvemos tambien los datos del vehiculo: el asesor debe poder VERIFICAR
    # que el sistema busco el carro correcto antes de creerle a la prediccion.
    # Los faltantes viajan como null (JSON valido); NaN NO es JSON valido y
    # rompe a cualquier cliente que intente leer la respuesta.
    respuesta = {
        "vin": vin,
        "datos_encontrados": {
            "gama": ficha["Gama"],
            "anio_modelo": None if anio_modelo is None else int(anio_modelo),
            "antiguedad_calculada": _a_float(ficha["Antiguedad_Vehiculo"]),
            "ultimo_kilometraje_visita_concesionario": _a_float(ficha[config.COLUMNA_KMS]),
            "ultima_visita_concesionario": _a_fecha(ficha["Ultima_Visita"]),
            "vendido_por_nosotros": bool(ficha["Es_Vehiculo_Vendido"]),
            # La historia es lo que explica la predicción: se muestra para que
            # el asesor pueda contrastar el número con lo que ya sabe del carro.
            "total_visitas": int(ficha["Total_Visitas"]),
            "ritmo_habitual_dias": _a_float(ficha["Ritmo_Mediano_Previo"]),
            "km_por_dia": None if km_por_dia is None else round(km_por_dia, 1),
        },
        "prediccion_dias_retorno": round(dias, 1),
    }
    if riesgo:
        respuesta["riesgo_de_fuga"] = riesgo
    respuesta["plan_de_llamada"] = plan
    return respuesta


# ENDPOINTS DE LA API

# El decorador @app.get("/") le dice a FastAPI: "cuando alguien haga una peticion GET a la raiz del sitio, ejecuta la funcion de abajo".
@app.get("/", summary="Estado del servicio")
def home(): #Endpoint de comprobación de estado (Health Check). Un health check sirve para que el monitoreo o el equipo de soporte verifiquen en 1 segundo si el servicio esta vivo Y si ademas el modelo quedo cargado correctamente.
    visitas_por_vin = None if historial is None else historial[config.COLUMNA_VIN].value_counts()
    return {
        "mensaje": "API de Retención Predictiva del concesionario activa 🚗",
        "modelo_cargado": pipeline is not None,
        "modelo_fuga_cargado": modelo_fuga is not None,
        "algoritmo_fuga": algoritmo_fuga,
        "historial_cargado": historial is not None,
        "vehiculos_en_historial": 0 if visitas_por_vin is None else len(visitas_por_vin),
        "margen_error_dias": None if MAE_DIAS is None else round(MAE_DIAS, 1),
        "fecha_corte_datos": None if FECHA_CORTE_DATOS is None else str(FECHA_CORTE_DATOS),
    }


@app.get("/health", tags=["Monitoreo"], summary="Health check para el monitoreo")
def health_check():
    """Verifica que el servicio esté vivo y que los artefactos .joblib hayan cargado."""
    modelos_cargados = pipeline is not None and historial is not None and modelo_fuga is not None
    logger.info("Health check consultado", extra={"modelos_cargados": modelos_cargados})
    return {
        "estado": "activo",
        "servicio": "API Retención Automotor.co",
        "modelos_cargados": modelos_cargados,
        "version_api": app.version,
    }


@app.get("/metrics", tags=["Monitoreo"], summary="Métricas agregadas del servicio")
def obtener_metricas():
    """Tráfico, tasa de error y latencias (promedio, p50 y p95) desde que arrancó el servidor."""
    metricas = METRICAS_API.obtener_metricas()
    logger.info("Métricas de la API consultadas", extra={"metricas": metricas})
    return metricas


# CONTRATO DE ENTRADA DEL INGRESO. Pydantic valida el JSON ANTES de que el
# codigo toque el modelo: un tipo de trabajo que no existe o un kilometraje
# negativo se rechazan con 422 automaticamente.
class IngresoHoy(BaseModel):
    kilometraje: float = Field(
        ..., ge=0, lt=config.KMS_MAXIMO_VALIDO,
        description="Odómetro con el que entra el vehículo.",
        json_schema_extra={"example": 23500},
    )
    # Para qué vino: quién paga y qué se le hace. Son las mismas dos variables
    # categóricas con las que se entrenó el modelo. Un mantenimiento normal es
    # Cliente + MECANICA; una garantía es Garantia; un siniestro, Seguros + COLISION.
    tipo_cargo: Literal["Cliente", "Garantia", "Seguros", "Interno"] = Field(
        "Cliente", json_schema_extra={"example": "Cliente"},
    )
    tipo_trabajo: Literal["MECANICA", "COLISION", "MECANICA Y COLISION", "OTRO"] = Field(
        "MECANICA", json_schema_extra={"example": "MECANICA"},
    )
    fecha_ingreso: date | None = Field(
        None, description="Día en que entró el vehículo. Si no se envía, se usa hoy.",
    )
    fecha_entrega: date | None = Field(
        None, description="Día en que se entrega (o se espera entregar). Si no se envía, el mismo día del ingreso.",
    )


@app.post("/predict/vin/{vin}/ingreso")
def predecir_con_ingreso_de_hoy(vin: str, ingreso: IngresoHoy):
    """Registra el ingreso del vehículo y predice su próximo regreso.

    No hace falta reentrenar ni esperar a que se entregue el carro: el modelo ya
    sabe convertir una historia en una predicción. Lo que se hace es agregar
    esta visita a las que el VIN ya tenía, recalcular su historia (una visita
    más, un intervalo nuevo, su ritmo y sus km/día) con las mismas fórmulas del
    entrenamiento, y volver a preguntarle a los dos modelos.

    El ingreso NO se guarda: la API solo lee. Quedará en la base cuando el ERP
    lo exporte y se corra `python -m src.actualizar_catalogo`.
    """
    if pipeline is None or historial is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo o historial no disponibles. Ejecute 'python -m src.actualizar_catalogo' primero."
        )

    # Normalizamos la entrada: los VIN del historial estan en mayusculas y sin
    # espacios. Sin esto, " adm130850 " no encontraria nada.
    vin = vin.strip().upper()
    visitas_vin = historial[historial[config.COLUMNA_VIN] == vin]
    if visitas_vin.empty:
        logger.warning(
            "Consulta de VIN sin historial",
            extra={"vin": vin, "evento": "vin_no_encontrado"},
        )
        raise HTTPException(
            status_code=404,
            detail=(f"El VIN '{vin}' no tiene historial en el concesionario: este sería su primer ingreso, "
                    f"así que no hay ritmo del que estimar su regreso.")
        )

    hoy = date.today()
    fecha_ingreso = ingreso.fecha_ingreso or hoy
    fecha_entrega = ingreso.fecha_entrega or fecha_ingreso
    ultima = visitas_vin.sort_values(config.COLUMNA_FECHA_ENTRADA).iloc[-1]
    ultima_fecha = pd.Timestamp(ultima[config.COLUMNA_FECHA_ENTRADA]).date()

    # Validaciones que dependen de la historia del carro (Pydantic no la conoce).
    # 409 = el dato es válido en sí mismo pero choca con lo que ya está registrado.
    if fecha_ingreso > hoy:
        raise HTTPException(status_code=422, detail="La fecha de ingreso no puede estar en el futuro.")
    if fecha_entrega < fecha_ingreso:
        raise HTTPException(status_code=422, detail="La fecha de entrega no puede ser anterior a la de ingreso.")
    if fecha_ingreso <= ultima_fecha:
        raise HTTPException(
            status_code=409,
            detail=(f"El VIN '{vin}' ya tiene un ingreso registrado el {ultima_fecha}. "
                    f"El ingreso nuevo debe ser posterior.")
        )
    km_anterior = ultima[config.COLUMNA_KMS]
    if pd.notna(km_anterior) and km_anterior < config.KMS_MAXIMO_VALIDO and ingreso.kilometraje < km_anterior:
        # El odómetro no baja: casi siempre es un error de digitación, y aceptarlo
        # dejaría los km/día del carro sin calcular.
        raise HTTPException(
            status_code=422,
            detail=(f"El kilometraje ({ingreso.kilometraje:,.0f}) es menor que el de la visita anterior "
                    f"({km_anterior:,.0f} el {ultima_fecha}). Verifique el odómetro.")
        )

    try:
        ficha = data.ficha_con_ingreso_nuevo(
            visitas_vin,
            fecha_entrada=fecha_ingreso,
            fecha_cierre=fecha_entrega,
            kms=ingreso.kilometraje,
            tipo_cargo=ingreso.tipo_cargo,
            tipo_trabajo=ingreso.tipo_trabajo,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al recalcular la historia del VIN '{vin}': {str(e)}")

    # Si llamar o no se decide contra el día REAL de la consulta: las fechas del
    # ingreso y de la entrega ya viajan en el cuerpo de la petición.
    respuesta = _respuesta_desde_ficha(vin, ficha, hoy)

    riesgo = respuesta.get("riesgo_de_fuga") or {}
    logger.info(
        "Predicción de retención ejecutada exitosamente",
        extra={
            "evento": "prediccion_exitosa",
            "vin": vin,
            "kilometraje": ingreso.kilometraje,
            "dias_retorno_predichos": respuesta["prediccion_dias_retorno"],
            "probabilidad_fuga": riesgo.get("probabilidad_fuga"),
            "alerta_fuga": riesgo.get("alerta"),
            "estado_llamada": respuesta["plan_de_llamada"]["estado"],
        },
    )
    return respuesta
