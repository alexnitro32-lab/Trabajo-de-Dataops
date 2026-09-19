# CONTRATO DE DATOS
#
# Este archivo es el portero del pipeline. Su trabajo es UNO: que el proyecto
# falle FUERTE y TEMPRANO cuando los datos no son los que se esperaban, en vez
# de seguir corriendo con basura y terminar en verde.
#
# Cubre dos peligros concretos, y ninguno de los dos levanta una alarma por si
# solo.
#
# PRIMERO: datos EQUIVOCADOS que parecen correctos. Si el taller re-exporta la
# base en formato estadounidense, "06/07/2017" pasa a significar 7 de junio, pero
# el codigo lo lee con dayfirst=True y entiende 6 de julio. pd.to_datetime no se
# queja: resuelve casi cualquier formato sin generar un solo nulo. El resultado
# es un mes de error en todas las fechas ambiguas, y como el proyecto entero mide
# INTERVALOS entre fechas, todos los intervalos quedan mal sin que nada lo
# delate. Por eso el contrato valida la forma del texto y no solo que parsee.
#
# SEGUNDO: datos que DESAPARECEN. La limpieza de data.py es silenciosa por
# diseno: to_datetime(errors="coerce") convierte lo que no entiende en nulo y
# dropna bota la fila. Con un punaido de registros sucios esta bien; si algo
# cambia de raiz, el DataFrame se vacia, el modelo entrena con lo que quede y el
# CI sale VERDE sobrescribiendo los .joblib con un modelo inservible.
#
# La defensa tiene dos capas, una para cada peligro:
#   ESQUEMA   valida forma y tipo: que las columnas esten, que las fechas vengan
#             escritas como se espera, que los numeros sean numeros.
#   VOLUMEN   valida cantidad: que despues de cada etapa sobrevivan suficientes
#             filas. Un DataFrame casi vacio sigue cumpliendo cualquier esquema,
#             asi que solo contar filas atrapa ese caso.

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from src import config


class ContratoDeDatosError(Exception):
    """Se incumplio el contrato de datos.

    Es una excepcion propia y no un SchemaError de pandera para que quien la
    reciba (el CI, un log, un orquestador) sepa que el problema es del DATO y no
    del codigo, sin tener que conocer la libreria que hace la validacion.
    """


# Formato en el que el ERP entrega las fechas: dia/mes/anio separados por barra.
# El dia y el mes pueden venir con uno o dos digitos ("9/06/2017" y "21/06/2017"
# conviven en la base), pero el anio siempre trae cuatro.
PATRON_FECHA_DIA_MES_ANIO = r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$"

# El mes nunca pasa de 12. Es la propiedad que distingue dd/mm/aaaa de mm/dd/aaaa
# y la unica forma de separarlos mirando los datos.
MES_MAXIMO = 12


def _fechas_en_el_formato_esperado(serie: pd.Series) -> bool:
    """Comprueba que las fechas vengan escritas como dia/mes/anio.

    Es LA verificacion critica del contrato, y valida la FORMA DEL TEXTO en vez
    de limitarse a intentar parsearlo. La razon es que pd.to_datetime es
    demasiado tolerante para servir de control: le pasas "2017-06-21",
    "06/21/2017" o "21-Jun-2017" y los tres los resuelve sin un solo nulo.

    Eso hace que un cambio de formato en el ERP no se manifieste como datos
    faltantes sino como datos EQUIVOCADOS, que es peor. El caso concreto: si el
    export pasara a formato estadounidense, "06/07/2017" significaria 7 de junio,
    pero al leerlo con dayfirst=True se interpreta como 6 de julio. Un mes de
    diferencia, en todas las fechas ambiguas, sin ningun error visible. Y como el
    proyecto entero mide intervalos entre fechas, esos intervalos quedarian mal
    sin que nada lo delate.

    La comprobacion tiene dos partes, porque una sola no alcanza:

    1. FORMA. El texto debe verse como dia/mes/anio. Esto descarta el ISO
       ("2017-06-21"), el compacto ("20170621"), el mes en letras ("21-Jun-2017")
       y los separadores distintos ("21-06-2017").

    2. SEMANTICA. El segundo componente debe ser un MES, es decir, no pasar de 12.
       Esta es la que atrapa el formato estadounidense, que en forma es idéntico
       al nuestro: "06/21/2017" cumple el patron pero su segundo numero es 21, y
       un mes 21 no existe. Sobre decenas de miles de filas siempre hay fechas con
       dia mayor que 12, asi que el cambio salta de inmediato.

    Se pide una proporcion y no el 100% porque la base real trae un punaido de
    celdas vacias o corruptas (302 cierres nulos sobre 34.935 filas) que la
    limpieza sabe manejar. Lo que no puede pasar es que cambie el formato general.
    """
    texto = serie.astype(str).str.strip()
    tiene_la_forma = texto.str.match(PATRON_FECHA_DIA_MES_ANIO, na=False)
    if float(tiene_la_forma.mean()) < config.PROPORCION_MINIMA_FECHAS_VALIDAS:
        return False

    # Solo sobre las que tienen la forma correcta: extraemos el segundo numero y
    # comprobamos que sea un mes valido. Si aparecen "meses" por encima de 12, el
    # archivo viene en formato mes/dia/anio y hay que detenerlo.
    segundo_componente = pd.to_numeric(
        texto[tiene_la_forma].str.split("/").str[1], errors="coerce"
    )
    son_meses_validos = segundo_componente.between(1, MES_MAXIMO)
    return bool(son_meses_validos.all())


# --------------------------------------------------------------------------
# ESQUEMA 1 — EL CSV TAL COMO LO ENTREGA EL ERP
# --------------------------------------------------------------------------
# Valida la ENTRADA del pipeline. No rechaza filas sucias: de eso se encarga
# limpiar_calidad(). Aqui solo se responde una pregunta: "este archivo es el que
# esperabamos?". Por eso casi todo es nullable y coerce.
ESQUEMA_CSV_CRUDO = pa.DataFrameSchema(
    columns={
        # La clave primaria del ERP. Sin OT no hay visita que reconstruir.
        config.COLUMNA_OT: pa.Column(str, nullable=True),
        config.COLUMNA_NUM_TRABAJO: pa.Column(nullable=True, coerce=True),
        config.COLUMNA_DESCRIPCION: pa.Column(nullable=True, coerce=True),
        # El VIN es la unidad de analisis de todo el proyecto.
        config.COLUMNA_VIN: pa.Column(str, nullable=True),
        config.COLUMNA_MARCA: pa.Column(nullable=True, coerce=True),
        config.COLUMNA_GAMA: pa.Column(nullable=True, coerce=True),
        # Anio del modelo: se acota por abajo y por arriba porque un valor fuera
        # de rango produce antiguedades imposibles que arrastran toda la escala.
        config.COLUMNA_ANO: pa.Column(
            float,
            checks=pa.Check.in_range(1950, pd.Timestamp.today().year + 1, include_min=True, include_max=True),
            nullable=True,
            coerce=True,
        ),
        config.COLUMNA_TIPO_CARGO: pa.Column(nullable=True, coerce=True),
        config.COLUMNA_TIPO_TRABAJO: pa.Column(nullable=True, coerce=True),
        # El odometro no puede ser negativo. El tope de arriba NO se valida aqui
        # a proposito: la base trae valores absurdos (123.456.789) que son reales
        # como dato y que generar_target_y_features() ya filtra.
        config.COLUMNA_KMS: pa.Column(
            float, checks=pa.Check.greater_than_or_equal_to(0), nullable=True, coerce=True
        ),
        # LAS DOS COLUMNAS QUE MAS IMPORTAN. El Check de aqui es el que atrapa un
        # cambio de formato de fecha antes de que vacie el dataset.
        config.COLUMNA_FECHA_ENTRADA: pa.Column(
            nullable=True,
            coerce=True,
            checks=pa.Check(
                _fechas_en_el_formato_esperado,
                element_wise=False,
                error=(f"Menos del {100*config.PROPORCION_MINIMA_FECHAS_VALIDAS:.0f}% de las fechas de ENTRADA "
                       f"vienen escritas como dd/mm/aaaa. El ERP cambio el formato: si se siguiera adelante, "
                       f"las fechas ambiguas se leerian con el dia y el mes intercambiados."),
            ),
        ),
        config.COLUMNA_FECHA_CIERRE: pa.Column(
            nullable=True,
            coerce=True,
            checks=pa.Check(
                _fechas_en_el_formato_esperado,
                element_wise=False,
                error=(f"Menos del {100*config.PROPORCION_MINIMA_FECHAS_VALIDAS:.0f}% de las fechas de CIERRE "
                       f"vienen escritas como dd/mm/aaaa. El ERP cambio el formato: si se siguiera adelante, "
                       f"las fechas ambiguas se leerian con el dia y el mes intercambiados."),
            ),
        ),
        # El estado distingue una factura real de una anulada. Si desapareciera,
        # el 13,5% de registros anulados entraria al modelo como visitas reales.
        config.COLUMNA_ESTADO: pa.Column(nullable=True, coerce=True),
    },
    # strict=False: el ERP puede agregar columnas nuevas sin romper el pipeline.
    # Lo que no puede es QUITAR una de las de arriba.
    strict=False,
    # Sin ordenar: no dependemos del orden en que vengan las columnas.
    ordered=False,
    name="CSV crudo del ERP",
)


# --------------------------------------------------------------------------
# ESQUEMA 2 — LAS VISITAS YA CONSOLIDADAS
# --------------------------------------------------------------------------
# Valida la SALIDA de la consolidacion. Aqui si se exige limpieza, porque este
# DataFrame ya paso por limpiar_calidad() y por el groupby.
ESQUEMA_VISITAS = pa.DataFrameSchema(
    columns={
        config.COLUMNA_VIN: pa.Column(str, nullable=False),
        # unique=True es la comprobacion de que una OT quedo como UNA visita. Si
        # esto falla, el groupby no colapso los sub-trabajos y toda la frecuencia
        # de ingreso del proyecto queda inflada.
        config.COLUMNA_OT: pa.Column(str, nullable=False, unique=True),
        config.COLUMNA_FECHA_ENTRADA: pa.Column("datetime64[ns]", nullable=False),
        config.COLUMNA_FECHA_CIERRE: pa.Column("datetime64[ns]", nullable=False),
        config.COLUMNA_KMS: pa.Column(float, nullable=True, coerce=True),
        config.COLUMNA_ANO: pa.Column(float, nullable=True, coerce=True),
        "Es_Alistamiento_OT": pa.Column(bool, nullable=False, coerce=True),
    },
    strict=False,
    name="Visitas consolidadas por OT",
)


def _esquema_dataset(nombre: str, columna_objetivo: str, checks_objetivo: list) -> pa.DataFrameSchema:
    """Arma el esquema de un dataset de entrenamiento.

    Los dos datasets (regresion y fuga) comparten el mismo vector de variables y
    solo se diferencian en la columna objetivo, asi que el esquema se construye
    una vez y se parametriza. Si manana se agrega una variable a config, este
    esquema la exige automaticamente: es la red que atrapa el caso de agregar la
    variable al entrenamiento y olvidar producirla en el catalogo o en la API.
    """
    columnas = {
        config.COLUMNA_GRUPO: pa.Column(str, nullable=False),
        columna_objetivo: pa.Column(float, checks=checks_objetivo, nullable=False, coerce=True),
    }
    # Las numericas pueden venir nulas: el SimpleImputer del pipeline las cubre
    # con la mediana, igual que en produccion. Lo que NO puede pasar es que la
    # columna no exista.
    for columna in config.COLUMNAS_NUMERICAS:
        columnas[columna] = pa.Column(float, nullable=True, coerce=True)
    for columna in config.COLUMNAS_CATEGORICAS:
        columnas[columna] = pa.Column(nullable=True, coerce=True)
    for columna in config.COLUMNAS_BINARIAS:
        columnas[columna] = pa.Column(
            int, checks=pa.Check.isin([0, 1]), nullable=False, coerce=True
        )
    return pa.DataFrameSchema(columns=columnas, strict=False, name=nombre)


# El objetivo de la regresion son dias: nunca negativo, y con un tope que
# corresponde a la ventana real de la base (algo mas de 8 anios).
ESQUEMA_DATASET_REGRESION = _esquema_dataset(
    "Dataset de regresion",
    config.TARGET,
    [pa.Check.in_range(0, 3650, include_min=True, include_max=True)],
)

# El objetivo de la fuga es binario y no admite nulos: una visita censurada no
# llega hasta aqui, se descarta en preparar_dataset_fuga().
ESQUEMA_DATASET_FUGA = _esquema_dataset(
    "Dataset de fuga",
    config.TARGET_FUGA,
    [pa.Check.isin([0.0, 1.0])],
)


# --------------------------------------------------------------------------
# VALIDACION
# --------------------------------------------------------------------------

def _validar(df: pd.DataFrame, esquema: pa.DataFrameSchema, etapa: str) -> pd.DataFrame:
    """Aplica un esquema y traduce el fallo a un mensaje que se pueda accionar.

    lazy=True hace que pandera recoja TODOS los incumplimientos antes de fallar,
    en vez de detenerse en el primero. Cuando el export del ERP cambia, casi
    nunca cambia una sola cosa: conviene ver la lista completa de una vez.
    """
    try:
        return esquema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as error:
        detalle = getattr(error, "failure_cases", None)
        resumen = ""
        if detalle is not None and not detalle.empty:
            # Solo las primeras filas: un fallo masivo generaria miles y el
            # mensaje dejaria de ser legible justo cuando mas se necesita.
            resumen = "\n" + detalle.head(15).to_string(index=False)
        raise ContratoDeDatosError(
            f"Los datos no cumplen el contrato en la etapa '{etapa}'.\n"
            f"El pipeline se detiene aqui a proposito: seguir produciria un modelo "
            f"entrenado sobre datos que no son los esperados.{resumen}"
        ) from error


def verificar_volumen(df: pd.DataFrame, minimo: int, etapa: str) -> pd.DataFrame:
    """Comprueba que sobrevivan suficientes filas despues de una etapa.

    Es la mitad mas importante del contrato. Un cambio de formato no rompe el
    TIPO de una columna: la vacia. El DataFrame resultante sigue siendo valido
    para cualquier esquema, solo que tiene 40 filas en vez de 30.000, y el modelo
    entrena igual sin protestar. Solo contar filas atrapa ese caso.
    """
    if len(df) < minimo:
        raise ContratoDeDatosError(
            f"La etapa '{etapa}' dejo solo {len(df)} filas y se esperaban al menos {minimo}.\n"
            f"Una caida asi casi siempre significa que el formato del archivo de entrada "
            f"cambio y la limpieza esta descartando registros validos."
        )
    return df


def validar_csv_crudo(df: pd.DataFrame) -> pd.DataFrame:
    """Puerta de entrada: el archivo del ERP es el que esperabamos."""
    _validar(df, ESQUEMA_CSV_CRUDO, "CSV crudo")
    return verificar_volumen(df, config.MINIMO_FILAS_CSV, "CSV crudo")


def validar_visitas(df: pd.DataFrame) -> pd.DataFrame:
    """La consolidacion produjo una fila por ingreso real, sin OT repetidas."""
    _validar(df, ESQUEMA_VISITAS, "visitas consolidadas")
    return verificar_volumen(df, config.MINIMO_VISITAS_CONSOLIDADAS, "visitas consolidadas")


def validar_dataset_regresion(df: pd.DataFrame) -> pd.DataFrame:
    """El dataset de regresion trae todas las variables declaradas en config."""
    _validar(df, ESQUEMA_DATASET_REGRESION, "dataset de regresion")
    return verificar_volumen(df, config.MINIMO_FILAS_ENTRENAMIENTO, "dataset de regresion")


def validar_dataset_fuga(df: pd.DataFrame) -> pd.DataFrame:
    """El dataset de fuga trae la etiqueta binaria y no quedo degenerado.

    Ademas del esquema se comprueba el BALANCE. Un dataset donde el 99% es de una
    sola clase pasa cualquier validacion de tipos y produce un clasificador que
    responde siempre lo mismo con una exactitud aparente altisima.
    """
    _validar(df, ESQUEMA_DATASET_FUGA, "dataset de fuga")
    verificar_volumen(df, config.MINIMO_FILAS_ENTRENAMIENTO, "dataset de fuga")

    tasa = float(df[config.TARGET_FUGA].mean())
    if not (config.TASA_FUGA_MINIMA <= tasa <= config.TASA_FUGA_MAXIMA):
        raise ContratoDeDatosError(
            f"La tasa de fuga es {tasa:.1%} y se esperaba entre "
            f"{config.TASA_FUGA_MINIMA:.0%} y {config.TASA_FUGA_MAXIMA:.0%}.\n"
            f"Con una clase tan desbalanceada el clasificador aprenderia a responder "
            f"siempre lo mismo, con una exactitud alta y ningun valor real."
        )
    return df


def validar_catalogo(catalogo: pd.DataFrame) -> pd.DataFrame:
    """El catalogo trae exactamente el vector que el modelo espera recibir.

    Esta es la validacion que evita el fallo mas dificil de detectar del
    proyecto: agregar una variable al entrenamiento y olvidar producirla en el
    catalogo. El modelo entrena bien, el CI sale verde, y el endpoint por VIN
    falla en produccion. Comprobarlo aqui lo convierte en un fallo de build.
    """
    faltantes = [c for c in config.COLUMNAS_NUMERICAS + config.COLUMNAS_CATEGORICAS
                 + config.COLUMNAS_BINARIAS if c not in catalogo.columns]
    if faltantes:
        raise ContratoDeDatosError(
            f"Al catalogo le faltan columnas que el modelo si consume: {faltantes}.\n"
            f"El entrenamiento funcionaria, pero el endpoint /predict/vin fallaria en "
            f"produccion. Revise construir_catalogo_vehiculos() en src/data.py."
        )
    return verificar_volumen(catalogo, config.MINIMO_VEHICULOS_CATALOGO, "catalogo de vehiculos")
