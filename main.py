# Este archivo convierte el modelo congelado en un SERVICIO WEB. Sin el, el modelo solo serviria dentro de un cuaderno de Python; con el, cualquier

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import pandas as pd
from src import config, data, features
from datetime import date

# INICIALIZACION DE LA APLICACION FASTAPI
app = FastAPI(
    title="API de Retención Predictiva - Automotor.co S.A.S.",
    description=(
        "Servicio de inferencia sobre el retorno de clientes al taller. Responde dos "
        "preguntas distintas con dos modelos distintos: **cuándo** vuelve un vehículo "
        "(regresión, para agendar y dimensionar bahías) y **si va a dejar de volver** "
        "(clasificación de fuga, para actuar antes de perder al cliente). "
        "Ambos se apoyan en la HISTORIA del chasis — cada cuánto venía volviendo, cuántos "
        "km hace por día — y no en la gama ni en el modelo del vehículo. Un chasis con un "
        "solo ingreso registrado no es predecible: para esos casos se devuelve la regla "
        "preventiva de fábrica (365 días o 10.000 km, lo que ocurra primero)."
    ),
    version="3.0.0"
)

# CARGA DEL PIPELINE COMPLETO CONGELADO (.joblib) MUY IMPORTANTE: esto esta FUERA de las funciones, por lo tanto se ejecuta UNA SOLA VEZ cuando arranca el servidor, no en cada peticion. Si estuviera dentro del endpoint, cada usuario obligaria a leer el modelo del disco otra vez.
try:
    pipeline = joblib.load(config.RUTA_MODELO_SERIALIZADO)
except Exception: # Si el .joblib no existe (por ejemplo, nunca se corrio el entrenamiento), NO tumbamos el servidor: lo dejamos en None y avisamos en los endpoints. Asi el health check sigue respondiendo y se puede diagnosticar el problema.
    pipeline = None

# El catalogo de vehiculos se carga igual que el modelo: UNA sola vez al arrancar.
# Es la "ficha tecnica + historia" por VIN que permite predecir sin escribir los
# datos a mano y sin reprocesar el CSV completo en cada peticion.
try:
    catalogo = joblib.load(config.RUTA_CATALOGO_VEHICULOS)
except Exception:
    catalogo = None

# Clasificador de fuga: "este cliente va a dejar de volver". Se carga aparte del
# regresor porque responde otra pregunta y puede evolucionar por su cuenta.
try:
    _paquete_fuga = joblib.load(config.RUTA_MODELO_FUGA)
    modelo_fuga = _paquete_fuga["modelo"]
    algoritmo_fuga = _paquete_fuga["algoritmo"]
except Exception:
    modelo_fuga, algoritmo_fuga = None, None

# Clasificadores por origen. El negocio pidió poder actuar distinto sobre el
# cliente al que le vendimos el carro y sobre el que llegó de otro concesionario.
try:
    modelos_fuga_segmentados = joblib.load(config.RUTA_MODELOS_FUGA_SEGMENTADOS)["modelos"]
except Exception:
    modelos_fuga_segmentados = {}

# AYUDANTES PARA DATOS INCOMPLETOS DEL HISTORIAL

# NaN es el "no se sabe" que entienden pandas y scikit-learn: el SimpleImputer
# lo reemplaza por la mediana. Lo declaramos una sola vez con nombre propio para
# que quede claro que es intencional y no un calculo que salio mal.
FALTANTE = float("nan")


def _a_float(valor):
    """Convierte un valor del catálogo a float, o a None si viene vacío/nulo.

    Se usa para que un dato faltante viaje como null en el JSON de respuesta.
    """
    if valor is None or pd.isna(valor):
        return None
    return float(valor)


def _a_fecha(valor):
    """Convierte una marca de tiempo del catálogo a texto 'AAAA-MM-DD', o None."""
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


def _fila_desde_catalogo(ficha: pd.Series) -> pd.DataFrame:
    """Reconstruye el vector de características de un chasis desde su ficha.

    El catálogo ya guarda las mismas columnas con las que se entrenó, tomadas de
    la última visita del vehículo. No se recalcula nada: se copia y se fuerza el
    tipo numérico, porque al pasar por una Serie de pandas todo llega como objeto.
    """
    fila = ficha.to_frame().T.reindex(columns=features.columnas_modelo())
    for columna in config.COLUMNAS_NUMERICAS + config.COLUMNAS_BINARIAS:
        fila[columna] = pd.to_numeric(fila[columna], errors="coerce")
    return fila


def _clasificar_fuga(fila: pd.DataFrame, es_vendido: bool) -> dict:
    """Probabilidad de que el cliente NO vuelva dentro del plazo de la regla.

    Se devuelven las dos lecturas a propósito: la del modelo global y la del
    modelo entrenado solo con vehículos del mismo origen. Cuando discrepan, el
    asesor está viendo una señal real — el comportamiento de un cliente al que le
    vendimos el carro no es el mismo que el de uno que llegó de afuera — y no un
    número que haya que tomar como verdad única.
    """
    segmento = "vendidos" if es_vendido else "externos"
    resultado = {
        "algoritmo": algoritmo_fuga,
        "umbral_dias": config.UMBRAL_FUGA_DIAS,
        "umbral_decision": config.UMBRAL_DECISION_FUGA,
        "segmento": segmento,
    }

    probabilidad = float(modelo_fuga.predict_proba(fila)[0, 1])
    resultado["probabilidad_fuga"] = round(probabilidad, 4)
    resultado["alerta"] = bool(probabilidad >= config.UMBRAL_DECISION_FUGA)

    modelo_segmento = modelos_fuga_segmentados.get(segmento)
    if modelo_segmento is not None:
        prob_seg = float(modelo_segmento.predict_proba(fila)[0, 1])
        resultado["probabilidad_fuga_modelo_del_segmento"] = round(prob_seg, 4)
        resultado["alerta_modelo_del_segmento"] = bool(prob_seg >= config.UMBRAL_DECISION_FUGA)

    return resultado


# CONTRATO DE ENTRADA CON PYDANTIC Esta clase es el "formulario obligatorio" de la API: define que campos deben llegar y de que tipo. FastAPI valida el JSON contra esta clase ANTES de que el codigo toque el modelo. Si algo no cuadra, responde 422 automaticamente.
class SolicitudVehiculo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # El formulario describe al CLIENTE y a su visita: quién paga, qué se le hizo,
    # cuánto lleva recorrido y cada cuánto viene volviendo. Con eso basta para
    # predecir, porque lo que determina el retorno es la conducta del chasis.
    Tipo_Cargo: str = Field(..., alias="Tipo Cargo", json_schema_extra={"example": "Cliente"}) # alias="Tipo Cargo": el JSON entrante usa el nombre con espacio.
    Tipo_Trabajo: str = Field("MECANICA", alias="Tipo de Trabajo", json_schema_extra={"example": "MECANICA"}) # Aqui el primer argumento NO son los tres puntos sino "MECANICA": eso lo convierte en un campo OPCIONAL con valor por defecto.
    Kms: float = Field(..., alias="Kms.", json_schema_extra={"example": 45000.0}) # float: si llega el texto "cinco_mil" en lugar de un numero, Pydantic rechaza la peticion con 422 y el modelo nunca recibe basura.
    Anio_Modelo: int = Field(..., ge=1990, le=date.today().year + 1, #acotacion del año y modelo
                             alias="Año Modelo",
                             json_schema_extra={"example": 2022})
    Es_Vehiculo_Vendido: bool = Field(..., json_schema_extra={"example": True}) # bool: acepta true/false en el JSON. Se convierte a 0/1 mas abajo.

    # --- HISTORIA DEL VEHICULO (lo que realmente mueve la predicción) ---
    # El modelo aprendió que la mejor pista de cuándo vuelve un carro es cada
    # cuánto venía volviendo. Sin estos tres datos la predicción degenera en la
    # mediana de la población, así que se piden explícitamente. Son datos que el
    # asesor tiene a la vista: el ingreso anterior del mismo chasis.
    Visitas_Previas: int = Field(..., ge=1, json_schema_extra={"example": 3},
                                 description="Cuántas veces ha ingresado este VIN, contando la visita actual.")
    Dias_Desde_Visita_Anterior: float = Field(..., ge=0, json_schema_extra={"example": 180.0},
                                              description="Días entre el ingreso anterior y el actual.")
    Kms_Visita_Anterior: float | None = Field(None, ge=0, json_schema_extra={"example": 32000.0},
                                              description="Odómetro registrado en el ingreso anterior.")
    Ritmo_Mediano_Previo: float | None = Field(None, ge=0, json_schema_extra={"example": 165.0},
                                               description="Mediana de los intervalos anteriores del vehículo. Si no se envía, se usa el último intervalo.")


# ENDPOINTS DE LA API

# El decorador @app.get("/") le dice a FastAPI: "cuando alguien haga una peticion GET a la raiz del sitio, ejecuta la funcion de abajo".
@app.get("/")
def home(): #Endpoint de comprobación de estado (Health Check). Un health check sirve para que el monitoreo o el equipo de soporte verifiquen en 1 segundo si el servicio esta vivo Y si ademas el modelo quedo cargado correctamente.
    return {
        "mensaje": "API de Retención Predictiva de Automotor.co S.A.S. activa 🚗",
        "modelo_cargado": pipeline is not None,
        "modelo_fuga_cargado": modelo_fuga is not None,
        "algoritmo_fuga": algoritmo_fuga,
        "modelos_fuga_por_origen": sorted(modelos_fuga_segmentados),
        "catalogo_cargado": catalogo is not None,
        "vehiculos_en_catalogo": 0 if catalogo is None else len(catalogo),
        # Cuántos de esos chasis tienen al menos dos ingresos y por tanto son
        # predecibles por el modelo. El resto solo recibe la regla de fábrica.
        "vehiculos_predecibles": 0 if catalogo is None else int(catalogo["Apto_Para_Modelo"].sum()),
        "regla_preventiva": f"{config.REGLA_MANTENIMIENTO_DIAS} días o {config.REGLA_MANTENIMIENTO_KMS} km",
        "umbral_fuga_dias": config.UMBRAL_FUGA_DIAS
    }


# POST (y no GET) porque el cliente ENVIA datos en el cuerpo de la peticion.
# El parametro "datos: SolicitudVehiculo" es la magia de FastAPI: gracias a esa
# anotacion de tipo, el framework lee el JSON, lo valida y lo entrega ya
# convertido en objeto de Python.
@app.post("/predict")
def predecir_retorno(datos: SolicitudVehiculo):
    """Endpoint POST para recibir los datos de un vehículo y retornar los días estimados hasta su próximo ingreso."""

    # Primera defensa: si el modelo no se pudo cargar al arrancar, respondemos
    # 500 (error del servidor) con una instruccion clara de como solucionarlo.
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo serializado no está disponible en 'models/modelo_retencion.joblib'. Ejecute 'python -m src.train' primero."
        )

    try:
        # construir_fila_prediccion() vive en src/data.py, al lado de las
        # fórmulas del entrenamiento, para que ambas no se separen nunca.
        df_entrada = data.construir_fila_prediccion(
            tipo_cargo=datos.Tipo_Cargo,
            tipo_trabajo=datos.Tipo_Trabajo,
            kms_actual=datos.Kms,
            anio_modelo=datos.Anio_Modelo,
            es_vehiculo_vendido=int(datos.Es_Vehiculo_Vendido),
            visitas_previas=datos.Visitas_Previas,
            dias_desde_visita_ant=datos.Dias_Desde_Visita_Anterior,
            kms_visita_ant=datos.Kms_Visita_Anterior,
            ritmo_mediano_previo=datos.Ritmo_Mediano_Previo,
        )

        dias = _predecir(df_entrada)
        regla = float(df_entrada["Regla_Preventiva_Dias"].iloc[0])

        respuesta = {
            "prediccion_dias_retorno": round(dias, 1),
            "unidad": "días",
            # La regla de fábrica viaja siempre al lado de la predicción: le da
            # al asesor una referencia contra la cual juzgar si el número es
            # razonable, en vez de pedirle fe ciega en el modelo.
            "regla_preventiva_dias": round(regla, 1),
            "vehiculo_atrasado_frente_a_regla": bool(dias > regla)
        }

        # La segunda pregunta, la que dispara la acción comercial: ¿se va a ir?
        if modelo_fuga is not None:
            respuesta["riesgo_de_fuga"] = _clasificar_fuga(
                df_entrada.reindex(columns=features.columnas_modelo()),
                bool(datos.Es_Vehiculo_Vendido),
            )

        return respuesta
    except Exception as e:
        # Red de seguridad: cualquier fallo inesperado se traduce en un 400 con
        # el detalle del error, en lugar de mostrarle al cliente un stack trace.
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


# GET y no POST: aqui el cliente no ENVIA datos, solo CONSULTA por un
# identificador que viaja en la propia URL. {vin} es un "parametro de ruta":
# FastAPI lo extrae de la direccion y lo entrega como argumento de la funcion.
@app.get("/predict/vin/{vin}")
def predecir_por_vin(vin: str):
    """Busca el vehículo por su número de chasis (VIN), autocompleta su ficha y
    su historia con la última visita registrada, y devuelve la predicción.

    El VIN NO es una variable del modelo: es la llave de búsqueda en el catálogo.
    Lo que sí entra al modelo es la historia de ese chasis, ya calculada durante
    el entrenamiento con las mismas fórmulas.
    """
    if pipeline is None or catalogo is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo o catálogo no disponibles. Ejecute 'python -m src.train' primero."
        )

    # Normalizamos la entrada: los VIN del historial estan en mayusculas y sin
    # espacios. Sin esto, " adm130850 " no encontraria nada.
    vin = vin.strip().upper()

    # 404 significa "ese recurso no existe", que es el codigo HTTP correcto para
    # un chasis sin historial. No es un error del servidor ni del formato.
    if vin not in catalogo.index:
        raise HTTPException(
            status_code=404,
            detail=f"El VIN '{vin}' no tiene historial en el taller. Use POST /predict e ingrese los datos manualmente."
        )

    ficha = catalogo.loc[vin]  # .loc[vin] trae la fila del catalogo como una Serie de pandas.

    try:
        anio_modelo = _a_float(ficha["Anio_Modelo"])
        kms = _a_float(ficha[config.COLUMNA_KMS])
        total_visitas = int(ficha["Total_Visitas"])
        apto = bool(ficha["Apto_Para_Modelo"])
        regla = _a_float(ficha["Regla_Preventiva_Dias"]) or float(config.REGLA_MANTENIMIENTO_DIAS)

        riesgo = None
        if apto:
            # El catálogo ya guarda el mismo vector con que se entrenó, tomado
            # de la última visita del vehículo. No se recalcula nada: se copia.
            df_entrada = _fila_desde_catalogo(ficha)
            dias = _predecir(df_entrada)
            origen = "modelo"
            advertencia = None
            if modelo_fuga is not None:
                riesgo = _clasificar_fuga(df_entrada, bool(ficha["Es_Vehiculo_Vendido"]))
        else:
            # UN SOLO INGRESO: no hay ningún intervalo observado, así que no se
            # puede saber cada cuánto vuelve este carro. Predecir igual sería
            # devolver la mediana de la población disfrazada de predicción
            # personalizada. Se devuelve la regla de fábrica, dicha como tal.
            dias = regla
            origen = "regla_preventiva"
            advertencia = (
                f"El VIN '{vin}' tiene un solo ingreso registrado: no hay intervalo "
                f"observado del que estimar su ritmo. Se devuelve la regla preventiva "
                f"({config.REGLA_MANTENIMIENTO_DIAS} días o {config.REGLA_MANTENIMIENTO_KMS} km), "
                f"no una predicción del modelo."
            )
    except HTTPException:
        raise
    except Exception as e:
        # Misma red de seguridad que en POST /predict: un dato corrupto en el
        # catalogo no debe devolverle al asesor un stack trace.
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la predicción del VIN '{vin}': {str(e)}"
        )

    # Devolvemos tambien los datos encontrados: el asesor debe poder VERIFICAR
    # que el sistema busco el carro correcto antes de creerle a la prediccion.
    # Los faltantes viajan como null (JSON valido); NaN NO es JSON valido y
    # rompe a cualquier cliente que intente leer la respuesta.
    respuesta = {
        "vin": vin,
        "datos_encontrados": {
            "gama": ficha["Gama"],
            "anio_modelo": None if anio_modelo is None else int(anio_modelo),
            "antiguedad_calculada": _a_float(ficha["Antiguedad_Vehiculo"]),
            "ultimo_kilometraje": kms,
            "ultima_visita": _a_fecha(ficha["Ultima_Visita"]),
            "ultima_entrega": _a_fecha(ficha["Ultima_Entrega"]),
            "vendido_por_nosotros": bool(ficha["Es_Vehiculo_Vendido"]),
            # La historia es lo que explica la predicción: se muestra para que
            # el asesor pueda contrastar el número con lo que ya sabe del carro.
            "total_visitas": total_visitas,
            "ritmo_habitual_dias": _a_float(ficha["Ritmo_Mediano_Previo"]),
            "km_por_dia": _a_float(ficha["Km_Por_Dia_Medio"]),
        },
        "prediccion_dias_retorno": round(dias, 1),
        "unidad": "días",
        "origen_estimacion": origen,
        "regla_preventiva_dias": round(regla, 1),
    }
    if riesgo:
        respuesta["riesgo_de_fuga"] = riesgo
    if advertencia:
        respuesta["advertencia"] = advertencia
    return respuesta


@app.get("/fuga/vin/{vin}")
def riesgo_de_fuga(vin: str):
    """Responde la pregunta comercial: ¿este cliente va a dejar de volver?

    Es un endpoint aparte del de predicción porque responde algo distinto y se
    usa distinto: `/predict/vin` sirve para agendar a un cliente concreto,
    mientras que este alimenta la lista de clientes a los que hay que llamar.

    Devuelve la probabilidad de que el vehículo NO vuelva dentro del plazo de la
    regla de fábrica, junto con la historia que sustenta esa cifra.
    """
    if modelo_fuga is None or catalogo is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo de fuga o catálogo no disponibles. Ejecute 'python -m src.train' primero."
        )

    vin = vin.strip().upper()
    if vin not in catalogo.index:
        raise HTTPException(status_code=404, detail=f"El VIN '{vin}' no tiene historial en el taller.")

    ficha = catalogo.loc[vin]
    if not bool(ficha["Apto_Para_Modelo"]):
        # Sin un segundo ingreso no hay ritmo que comparar, y el riesgo de fuga
        # no se puede estimar. Decirlo es más útil que devolver un número falso.
        raise HTTPException(
            status_code=409,
            detail=(f"El VIN '{vin}' tiene un solo ingreso registrado: no hay intervalo observado "
                    f"del que estimar su ritmo, así que su riesgo de fuga no es calculable.")
        )

    try:
        riesgo = _clasificar_fuga(_fila_desde_catalogo(ficha), bool(ficha["Es_Vehiculo_Vendido"]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al calcular el riesgo del VIN '{vin}': {str(e)}")

    return {
        "vin": vin,
        "riesgo_de_fuga": riesgo,
        # El asesor necesita ver POR QUÉ el modelo cree que este cliente se va.
        "historia_que_lo_sustenta": {
            "total_visitas": int(ficha["Total_Visitas"]),
            "ritmo_habitual_dias": _a_float(ficha["Ritmo_Mediano_Previo"]),
            "dias_desde_visita_anterior": _a_float(ficha["Dias_Desde_Visita_Ant"]),
            "km_por_dia": _a_float(ficha["Km_Por_Dia_Medio"]),
            "ultima_visita": _a_fecha(ficha["Ultima_Visita"]),
            "vendido_por_nosotros": bool(ficha["Es_Vehiculo_Vendido"]),
        },
    }
