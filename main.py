# Este archivo convierte el modelo congelado en un SERVICIO WEB. Sin el, el modelo solo serviria dentro de un cuaderno de Python; con el, cualquier

from fastapi import FastAPI, HTTPException      
from pydantic import BaseModel, Field, ConfigDict  
import joblib                                  
import pandas as pd                             
from src import config                         
from datetime import date                 

# INICIALIZACION DE LA APLICACION FASTAPI
app = FastAPI(
    title="API de Retención Predictiva - Automotor.co S.A.S.",
    description="Servicio de inferencia para predecir los días transcurridos hasta el próximo ingreso del vehículo al taller.",
    version="1.0.0"
)

# CARGA DEL PIPELINE COMPLETO CONGELADO (.joblib) MUY IMPORTANTE: esto esta FUERA de las funciones, por lo tanto se ejecuta UNA SOLA VEZ cuando arranca el servidor, no en cada peticion. Si estuviera dentro del endpoint, cada usuario obligaria a leer el modelo del disco otra vez.
try:
    pipeline = joblib.load(config.RUTA_MODELO_SERIALIZADO)
except Exception: # Si el .joblib no existe (por ejemplo, nunca se corrio el entrenamiento), NO tumbamos el servidor: lo dejamos en None y avisamos en los endpoints. Asi el health check sigue respondiendo y se puede diagnosticar el problema. 
    pipeline = None

# El catalogo de vehiculos se carga igual que el modelo: UNA sola vez al arrancar.
# Es la "ficha tecnica" por VIN que permite predecir sin escribir los datos a mano.
try:
    catalogo = joblib.load(config.RUTA_CATALOGO_VEHICULOS)
except Exception: # Si el .joblib no existe (por ejemplo, nunca se corrio el entrenamiento), NO tumbamos el servidor: lo dejamos en None y avisamos en los endpoints. Asi el health check sigue respondiendo y se puede diagnosticar el problema.
    catalogo = None

# AYUDANTES PARA DATOS INCOMPLETOS DEL HISTORIAL

# NaN es el "no se sabe" que entienden pandas y scikit-learn: el SimpleImputer
# lo reemplaza por la mediana. Lo declaramos una sola vez con nombre propio para
# que quede claro que es intencional y no un calculo que salio mal.
FALTANTE = float("nan")


def _a_float(valor):
    """Convierte un valor del catálogo a float, o a None si viene vacío/nulo.

    Se usa para que un dato faltante viaje como null en el JSON de respuesta.
    """
    if pd.isna(valor):
        return None
    return float(valor)


def _a_fecha(valor):
    """Convierte una marca de tiempo del catálogo a texto 'AAAA-MM-DD', o None."""
    if pd.isna(valor):
        return None
    return str(pd.Timestamp(valor).date())


# CONTRATO DE ENTRADA CON PYDANTIC Esta clase es el "formulario obligatorio" de la API: define que campos deben llegar y de que tipo. FastAPI valida el JSON contra esta clase ANTES de que l codigo toque el modelo. Si algo no cuadra, responde 422 automaticamente.
class SolicitudVehiculo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Gama: str = Field(..., json_schema_extra={"example": "Tucson"}) #esto es un campo obligatorio, y el ejemplo que aparece en la documentacion es "Tucson".
    Tipo_Cargo: str = Field(..., alias="Tipo Cargo", json_schema_extra={"example": "Cliente"}) # alias="Tipo Cargo": el JSON entrante usa el nombre con espacio.
    Tipo_Trabajo: str = Field("MECANICA", alias="Tipo de Trabajo", json_schema_extra={"example": "MECANICA"}) # Aqui el primer argumento NO son los tres puntos sino "MECANICA": eso lo convierte en un campo OPCIONAL con valor por defecto.
    Kms: float = Field(..., alias="Kms.", json_schema_extra={"example": 45000.0}) # float: si llega el texto "cinco_mil" en lugar de un numero, Pydantic rechaza la peticion con 422 y el modelo nunca recibe basura.
    Anio_Modelo: int = Field(..., ge=1990, le=date.today().year + 1, #acotacion del año y modelo
                             alias="Año Modelo",
                             json_schema_extra={"example": 2022})
    Es_Vehiculo_Vendido: bool = Field(..., json_schema_extra={"example": True}) # bool: acepta true/false en el JSON. Se convierte a 0/1 mas abajo.

# ENDPOINTS DE LA API-

# El decorador @app.get("/") le dice a FastAPI: "cuando alguien haga una peticion GET a la raiz del sitio, ejecuta la funcion de abajo".
@app.get("/")
def home(): #Endpoint de comprobación de estado (Health Check). Un health check sirve para que el monitoreo o el equipo de soporte verifiquen en 1 segundo si el servicio esta vivo Y si ademas el modelo quedo cargado correctamente.
    return {
        "mensaje": "API de Retención Predictiva de Automotor.co S.A.S. activa 🚗",
        "modelo_cargado": pipeline is not None,
        "catalogo_cargado": catalogo is not None,
        "vehiculos_en_catalogo": 0 if catalogo is None else len(catalogo)
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
        # Convertimos los datos de Pydantic en un DataFrame de UNA fila.
        # Las llaves deben llamarse EXACTAMENTE igual que las columnas con las
        # que se entreno el pipeline, o el ColumnTransformer no las encontrara.
        df_entrada = pd.DataFrame([{
            "Gama": datos.Gama,
            "Tipo Cargo": datos.Tipo_Cargo,
            "Tipo de Trabajo": datos.Tipo_Trabajo,
            "Kms.": datos.Kms,
            # TRADUCCION: el modelo NO conoce "Anio_Modelo". Aqui replicamos el
            # mismo calculo de src/data.py (anio de la visita - anio del modelo).
            # max(0, ...) protege el caso de un modelo del anio entrante, que daria
            # antiguedad negativa y nunca aparecio en el entrenamiento.
            "Antiguedad_Vehiculo": max(0.0, float(date.today().year - datos.Anio_Modelo)),
            # int() convierte True/False a 1/0 porque asi se entreno la variable
            # en data.py (SimpleImputer no admite el tipo bool).
            "Es_Vehiculo_Vendido": int(datos.Es_Vehiculo_Vendido)
        }])

        # Inferencia: el pipeline aplica por si solo imputacion, escalado y
        # codificacion, y luego pasa el resultado al RandomForest.
        # Devuelve un array de numpy, por eso tomamos la posicion [0].
        prediccion = pipeline.predict(df_entrada)

        return {
            "prediccion_dias_retorno": round(float(prediccion[0]), 1),
            "unidad": "días"
        }
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
    """Busca el vehículo por su número de chasis (VIN), autocompleta sus datos
    con la última visita registrada y devuelve la predicción de retorno.

    El VIN NO es una variable del modelo: es solo la llave de búsqueda en el
    catálogo. El modelo sigue recibiendo exactamente las mismas 6 características.
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

    # .loc[vin] trae la fila del catalogo como una Serie de pandas.
    ficha = catalogo.loc[vin]

    try:
        # El historial del taller no siempre esta completo: hay chasis sin anio
        # de modelo o sin kilometraje registrado. Los dejamos como NaN a
        # proposito para que el SimpleImputer del pipeline los rellene con la
        # mediana, exactamente igual que hace durante el entrenamiento.
        # Inventar un 0 seria peor: le diria al modelo que es un carro nuevo.
        anio_modelo = _a_float(ficha["Anio_Modelo"])
        kms = _a_float(ficha["Kms."])

        # Misma conversion que en POST /predict: el modelo solo entiende
        # antiguedad. Si no hay anio de modelo, la antiguedad tambien es NaN.
        antiguedad = FALTANTE if anio_modelo is None else max(0.0, float(date.today().year - anio_modelo))

        # Mismas 6 llaves y mismos nombres que en POST /predict: el ColumnTransformer
        # busca las columnas por nombre exacto y falla si alguna cambia.
        df_entrada = pd.DataFrame([{
            "Gama": ficha["Gama"],
            "Tipo Cargo": ficha["Tipo Cargo"],
            "Tipo de Trabajo": ficha["Tipo de Trabajo"],
            "Kms.": FALTANTE if kms is None else kms,
            "Antiguedad_Vehiculo": antiguedad,
            "Es_Vehiculo_Vendido": int(ficha["Es_Vehiculo_Vendido"])
        }])

        prediccion = pipeline.predict(df_entrada)
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
    return {
        "vin": vin,
        "datos_encontrados": {
            "gama": ficha["Gama"],
            "anio_modelo": None if anio_modelo is None else int(anio_modelo),
            "antiguedad_calculada": None if anio_modelo is None else antiguedad,
            "ultimo_kilometraje": kms,
            "ultima_visita": _a_fecha(ficha["Ultima_Visita"]),
            "vendido_por_nosotros": bool(ficha["Es_Vehiculo_Vendido"])
        },
        "prediccion_dias_retorno": round(float(prediccion[0]), 1),
        "unidad": "días"
    }
