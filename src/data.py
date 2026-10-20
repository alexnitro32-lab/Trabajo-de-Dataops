# En este archivo leemos el CSV original, aplicamos las reglas de calidad de la
# base, consolidamos las lineas de trabajo en visitas reales, construimos las
# dos variables objetivo (dias hasta el retorno y fuga si/no) junto con la
# HISTORIA DE CADA VIN, y devolvemos DataFrames listos para entrenar. Tambien
# construimos un catalogo por vehiculo para que la API pueda predecir a partir
# del chasis sin recalcular nada.

import numpy as np
import pandas as pd
from src import config


def cargar_datos(ruta_csv: str) -> pd.DataFrame: #esto es una funcion que recibe la ruta del CSV y devuelve un DataFrame de pandas con los datos cargados.
    try: #este try/except es un plan A/B defensivo: si el archivo fue guardado con codificacion UTF-8, lo leemos asi; si no, reintentamos con latin1.
        df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="latin1")

    df.columns = df.columns.str.replace('﻿', '', regex=False).str.strip() #esto es para limpiar los encabezados de columnas: a veces el CSV trae un caracter invisible al inicio (BOM) y espacios en blanco al final, y eso rompe el pipeline. Lo que hacemos es reemplazar el BOM por nada y quitar los espacios en blanco al inicio y al final de cada nombre de columna.
    return df


def _moda_con_prioridad(serie: pd.Series, prioridad: dict) -> object:
    """Devuelve el valor MAS FRECUENTE de la serie; si hay empate, el de mayor
    prioridad de negocio.

    1.971 ordenes mezclan valores de Tipo Cargo y 1.510 mezclan Tipo de Trabajo:
    una linea MECANICA y otra COLISION dentro del mismo ingreso. Tomar el primer
    valor dejaria que el orden en que el ERP exporto las lineas decidiera el tipo
    de la visita, que es un sorteo y no un criterio. Con moda + prioridad el
    resultado es el mismo sin importar como venga ordenado el archivo.
    """
    conteo = serie.dropna().value_counts()
    if conteo.empty:
        return np.nan
    maximo = conteo.max()
    empatados = [valor for valor, veces in conteo.items() if veces == maximo]
    return sorted(empatados, key=lambda v: prioridad.get(v, 99))[0]


def limpiar_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las tres reglas de calidad que la base necesita.

    Se hace ANTES de consolidar por OT, porque los tres defectos viven a nivel
    de linea y hay que quitarlos mientras todavia se pueden distinguir.
    """
    df = df.copy()
    df.columns = df.columns.str.replace('﻿', '', regex=False).str.strip()

    # 1) Facturas anuladas (13,5% de la base). Una OT anulada no es un ingreso
    #    real del vehiculo; contarla inventa visitas que nunca ocurrieron y
    #    acorta artificialmente el intervalo entre las visitas verdaderas.
    if config.COLUMNA_ESTADO in df.columns:
        df = df[~df[config.COLUMNA_ESTADO].isin(config.ESTADOS_EXCLUIDOS)]

    # 2) Lineas de BODEGAJE (cobro de parqueo). No son mantenimiento ni
    #    reparacion, asi que no representan un ingreso a taller.
    col_desc = config.COLUMNA_DESCRIPCION if config.COLUMNA_DESCRIPCION in df.columns else "Descripcion Trabajo"
    if col_desc in df.columns:
        descripcion = df[col_desc].astype(str).str.upper()
        df = df[~descripcion.str.contains(config.PATRON_BODEGAJE, na=False)]

    # 3) Anio centinela 1997. No se borra la fila (la visita SI ocurrio): se
    #    anula solo el anio, para que la antiguedad quede como faltante y el
    #    imputador la rellene con la mediana. Borrar la fila perderia una visita
    #    real; dejar el 1997 le diria al modelo que es un carro de 29 anios.
    col_ano = config.COLUMNA_ANO if config.COLUMNA_ANO in df.columns else "Anio"
    if col_ano in df.columns:
        df[col_ano] = pd.to_numeric(df[col_ano], errors="coerce")
        df.loc[df[col_ano] == config.ANIO_CENTINELA, col_ano] = np.nan

    return df


def consolidar_visitas_por_ot(df: pd.DataFrame) -> pd.DataFrame: #esto sirve para consolidar multiples filas de sub-trabajos de una misma OT en 1 solo registro de ingreso fisico del vehiculo al taller. Recibe un DataFrame y devuelve otro DataFrame con las visitas consolidadas.
    df = limpiar_calidad(df)

    df[config.COLUMNA_FECHA_ENTRADA] = pd.to_datetime(
        df[config.COLUMNA_FECHA_ENTRADA], dayfirst=True, errors="coerce"
    )
    df[config.COLUMNA_FECHA_CIERRE] = pd.to_datetime(
        df[config.COLUMNA_FECHA_CIERRE], dayfirst=True, errors="coerce"
    )

    col_kms = config.COLUMNA_KMS if config.COLUMNA_KMS in df.columns else "Kms."
    col_ano = config.COLUMNA_ANO if config.COLUMNA_ANO in df.columns else "Anio"

    df[col_kms] = pd.to_numeric(df[col_kms], errors="coerce") # to_numeric convierte texto a numero; lo que no se pueda convertir queda NaN.
    df[col_ano] = pd.to_numeric(df[col_ano], errors="coerce")

    posibles_cols_cliente = [
        config.COLUMNA_CLIENTE_CARGO,
        "Cod. Cliente Cargo",
        "Cliente Cargo",
        "Cliente",
    ]

    col_cliente = next((c for c in posibles_cols_cliente if c in df.columns), None) #esto busca la primera columna que exista en el DataFrame y que coincida con alguna de las posibles columnas de cliente. Si no encuentra ninguna, devuelve None.
    if col_cliente is None:
        # Ultimo recurso: buscamos cualquier columna que contenga "cliente".
        col_cliente = next((c for c in df.columns if "cliente" in c.lower()), "Cliente_Temp")
        if col_cliente == "Cliente_Temp":
            # Si de plano no existe, creamos una columna dummy para no romper el groupby.
            df["Cliente_Temp"] = "Desconocido"

    # OJO: esta columna es el PAGADOR de la linea, no el dueno del vehiculo (en
    # el 7,5% de las OT conviven el cliente y una aseguradora o la marca). Por eso
    # NO se usa como identificador de cliente en el modelo: la unidad de analisis
    # es el VIN. Se conserva solo como dato de auditoria.
    df[col_cliente] = df[col_cliente].astype(str)  # Forzamos a texto: los codigos de cliente son identificadores, no cantidades.

    col_ot = config.COLUMNA_OT if config.COLUMNA_OT in df.columns else "OT"
    col_vin = config.COLUMNA_VIN if config.COLUMNA_VIN in df.columns else "VIN"

    df = df.dropna(subset=[col_vin, col_ot, config.COLUMNA_FECHA_ENTRADA, config.COLUMNA_FECHA_CIERRE]) #Eliminamos registros sin VIN o fechas validas: sin chasis no sabemos de que carro hablamos, y sin fechas no podemos calcular dias de retorno.

    col_desc = config.COLUMNA_DESCRIPCION if config.COLUMNA_DESCRIPCION in df.columns else "Descripcion Trabajo"

    # Marcamos a nivel de LINEA si el trabajo fue un alistamiento (entrega de
    # vehiculo nuevo). Basta con que UNA linea de la OT lo sea.
    df = df.assign(_es_alistamiento=df[col_desc].astype(str).str.upper().str.contains(
        config.PATRON_ALISTAMIENTO, regex=True, na=False
    )) #esto revisa la descripcion de cada linea y devuelve True si contiene "ALISTAMIENTO" o "VEHICULO NUEVO", y False en caso contrario. Eso nos permite identificar si la OT fue un alistamiento (venta) o no. El .upper() es para comparar sin importar mayusculas.


    # Agrupacion por VIN y OT: aqui se colapsan los sub-trabajos.
    # Una fila del ERP NO es una visita: 34.935 lineas son 27.935 ordenes.
    dict_agg = {
        config.COLUMNA_FECHA_ENTRADA: (config.COLUMNA_FECHA_ENTRADA, "min"),
        config.COLUMNA_FECHA_CIERRE: (config.COLUMNA_FECHA_CIERRE, "max"),
        col_kms: (col_kms, "max"),
        config.COLUMNA_MARCA: (config.COLUMNA_MARCA, "first"),
        config.COLUMNA_GAMA: (config.COLUMNA_GAMA, "first"),
        col_ano: (col_ano, "first"),
        # Moda + prioridad en vez de "first": ver _moda_con_prioridad().
        config.COLUMNA_TIPO_CARGO: (
            config.COLUMNA_TIPO_CARGO,
            lambda s: _moda_con_prioridad(s, config.PRIORIDAD_TIPO_CARGO),
        ),
        col_cliente: (col_cliente, "first"),
        "Es_Alistamiento_OT": ("_es_alistamiento", "any"),
    }

    # Solo agregamos Tipo de Trabajo si realmente existe en el archivo.
    if config.COLUMNA_TIPO_TRABAJO in df.columns:
        dict_agg[config.COLUMNA_TIPO_TRABAJO] = (
            config.COLUMNA_TIPO_TRABAJO,
            lambda s: _moda_con_prioridad(s, config.PRIORIDAD_TIPO_TRABAJO),
        )

    # groupby + agg: de ~35.000 filas de sub-trabajos pasamos a las visitas reales.
    visitas = df.groupby([col_vin, col_ot]).agg(**dict_agg).reset_index()
    return visitas


def agregar_features_historia(df_visitas: pd.DataFrame) -> pd.DataFrame:
    """Construye la variable objetivo y la HISTORIA de cada vehiculo.

    Es el nucleo del modelo. Sin historia, dos vehiculos con la misma ficha
    estatica (kilometraje, antiguedad) pero con ritmos de ingreso completamente
    distintos le llegarian identicos al estimador, y lo mejor que podria hacer
    seria predecir el promedio. La historia es lo que los separa.

    TODAS las variables de historia miran SOLO hacia atras respecto de la visita
    actual. El intervalo que se predice va de la visita i a la i+1; lo que se
    usa como pista llega, como maximo, hasta la visita i. Sin esa disciplina el
    modelo estaria leyendo el futuro y las metricas serian mentira.

    Devuelve TODAS las visitas (incluida la ultima de cada VIN, que no tiene
    target): el filtrado para entrenar se hace despues, en
    preparar_dataset_entrenamiento(). Asi el catalogo de la API puede usar la
    ultima visita de cada carro, que es justamente la que hay que proyectar.
    """
    col_vin = config.COLUMNA_VIN if config.COLUMNA_VIN in df_visitas.columns else "VIN"
    col_ano = config.COLUMNA_ANO if config.COLUMNA_ANO in df_visitas.columns else "Anio"

    # Ordenar por VIN y fecha es OBLIGATORIO: todo el calculo de historia y de
    # "proxima visita" depende de que las visitas de cada carro esten en orden.
    df = df_visitas.sort_values(by=[col_vin, config.COLUMNA_FECHA_ENTRADA]).reset_index(drop=True)
    g = df.groupby(col_vin)

    # ORIGEN DEL VEHICULO: si el PRIMER ingreso del chasis fue un alistamiento,
    # lo vendimos nosotros. No es una variable cualquiera: marca dos poblaciones
    # que entran a la base en momentos distintos de su vida (el vendido entra con
    # el odometro en cero; el externo, con anios que el ERP nunca vio). En
    # estadistica eso se llama truncamiento por la izquierda.
    df["Es_Vehiculo_Vendido"] = g["Es_Alistamiento_OT"].transform("first").fillna(False).astype(int)

    # CUANTAS VECES HA VENIDO ESTE VIN. cumcount() numera las visitas de cada
    # carro empezando en 0, por eso el +1: la primera visita es la numero 1.
    df["Num_Visita"] = g.cumcount() + 1
    df["Total_Visitas_VIN"] = g[config.COLUMNA_OT].transform("size")

    # VARIABLE OBJETIVO: dias entre que le entregamos el carro (F. Cierre) y la
    # fecha en que volvio (F. Entrada de la siguiente visita).
    df["F. Entrada_Siguiente"] = g[config.COLUMNA_FECHA_ENTRADA].shift(-1)
    df[config.TARGET] = (df["F. Entrada_Siguiente"] - df[config.COLUMNA_FECHA_CIERRE]).dt.days

    # --- HISTORIA PASADA DEL VEHICULO ---
    # Cuanto tardo en volver la VEZ ANTERIOR (shift(1) = mirar hacia atras).
    df["Dias_Desde_Visita_Ant"] = (
        df[config.COLUMNA_FECHA_ENTRADA] - g[config.COLUMNA_FECHA_ENTRADA].shift(1)
    ).dt.days

    # Cuantos km recorrio entre la visita anterior y esta. El odometro no baja:
    # un valor negativo es un error de digitacion y se anula.
    km_recorridos = df[config.COLUMNA_KMS] - g[config.COLUMNA_KMS].shift(1)
    df["Km_Desde_Visita_Ant"] = km_recorridos.where(km_recorridos >= 0)

    # SU COSTUMBRE. expanding() acumula solo lo que ya paso, fila por fila, asi
    # que en la visita 4 la mediana se calcula con los intervalos 1-2, 2-3 y
    # 3-4, nunca con el que se quiere predecir. Es la variable mas importante
    # del modelo: la mejor pista de cuando vuelve un carro es cada cuanto venia.
    intervalos = df.groupby(col_vin)["Dias_Desde_Visita_Ant"]
    df["Ritmo_Mediano_Previo"] = intervalos.transform(lambda s: s.expanding().median())
    df["Ritmo_Max_Previo"] = intervalos.transform(lambda s: s.expanding().max())

    # INTENSIDAD DE USO. Un taxi y un carro de fin de semana pueden tener el
    # mismo odometro y volver con ritmos totalmente distintos; lo que los separa
    # es cuantos km hacen por dia.
    df["Km_Por_Dia"] = df["Km_Desde_Visita_Ant"] / df["Dias_Desde_Visita_Ant"].replace(0, np.nan)
    df["Km_Por_Dia_Medio"] = df.groupby(col_vin)["Km_Por_Dia"].transform(
        lambda s: s.expanding().median()
    )

    df["Antiguedad_Vehiculo"] = df[config.COLUMNA_FECHA_ENTRADA].dt.year - df[col_ano]

    # Cuanto duro el carro adentro. Se conoce al momento de la entrega, y el
    # target se cuenta desde F. Cierre, asi que no hay fuga de informacion.
    df["Dias_En_Taller"] = (
        df[config.COLUMNA_FECHA_CIERRE] - df[config.COLUMNA_FECHA_ENTRADA]
    ).dt.days.clip(lower=0, upper=120)
    df["Mes_Entrada"] = df[config.COLUMNA_FECHA_ENTRADA].dt.month

    # --- REGLA DE NEGOCIO: PLAZO O KILOMETRAJE, LO QUE OCURRA PRIMERO ---
    # A su ritmo de uso, cuantos dias tarda este vehiculo en hacer 10.000 km.
    # Es una medida de intensidad de uso, comun a todos los vehiculos.
    # El tope de 3.000 dias evita que un carro casi parado (0,5 km/dia) genere
    # un numero astronomico que descuadre la escala de la variable.
    df["Dias_Para_10k_Km"] = (
        config.REGLA_MANTENIMIENTO_KMS / df["Km_Por_Dia_Medio"].replace(0, np.nan)
    ).clip(upper=3000)

    # La regla en si, SEGUN EL AÑO DEL MODELO: los 2022 en adelante siguen el
    # plan de 365 dias o 10.000 km; los 2021 y anteriores (o de año desconocido),
    # el de 6 meses o 5.000 km. En ambos casos manda el que se cumpla primero.
    # Es la referencia contra la cual se mide si el modelo aporta algo por
    # encima del plan de fabrica, y la misma que la API usa como tope.
    es_plan_nuevo = df[col_ano] >= config.ANIO_DESDE_PLAN_NUEVO  # NaN -> False -> plan corto
    plan_dias = np.where(es_plan_nuevo, config.PLAN_MANTENIMIENTO_NUEVO["dias"],
                         config.PLAN_MANTENIMIENTO_ANTERIOR["dias"]).astype(float)
    plan_kms = np.where(es_plan_nuevo, config.PLAN_MANTENIMIENTO_NUEVO["kms"],
                        config.PLAN_MANTENIMIENTO_ANTERIOR["kms"]).astype(float)
    dias_por_km = (plan_kms / df["Km_Por_Dia_Medio"].replace(0, np.nan)).clip(upper=3000)
    df["Regla_Preventiva_Dias"] = np.fmin(plan_dias, dias_por_km.fillna(pd.Series(plan_dias, index=df.index)))

    # El recorrido expresado en ciclos de mantenimiento en vez de en kilometros
    # sueltos: "hizo 1,4 mantenimientos de distancia" dice mas que "14.000 km".
    df["Km_Sobre_Umbral_10k"] = df["Km_Desde_Visita_Ant"] / config.REGLA_MANTENIMIENTO_KMS
    df["Ciclos_10k_Acumulados"] = df[config.COLUMNA_KMS] / config.REGLA_MANTENIMIENTO_KMS

    return df


def plan_mantenimiento(anio_modelo: float | None) -> dict:
    """Plan de fabrica que le corresponde a UN vehiculo segun su año de modelo.

    Misma regla que agregar_features_historia() aplica a toda la base: 2022 en
    adelante -> 365 dias o 10.000 km; 2021, anteriores o desconocido -> 6 meses
    o 5.000 km.
    """
    if anio_modelo is not None and not pd.isna(anio_modelo) and anio_modelo >= config.ANIO_DESDE_PLAN_NUEVO:
        return config.PLAN_MANTENIMIENTO_NUEVO
    return config.PLAN_MANTENIMIENTO_ANTERIOR


def preparar_dataset_entrenamiento(df_con_historia: pd.DataFrame) -> pd.DataFrame:
    """Filtra las visitas que SI pueden entrenar el modelo y deja solo las columnas utiles.

    Tres descartes, en este orden:
      1. Visitas sin retorno observado (la ultima de cada carro): no hay respuesta
         que aprender. Ademas se quitan los valores imposibles (target negativo,
         antiguedad negativa, kilometrajes de digitacion).
      2. Vehiculos con UNA SOLA visita: no tienen ningun intervalo observado, asi
         que no se puede saber cada cuanto vuelven. Aportan ruido, no senial.
      3. El PRIMER ingreso de cada vehiculo: en ese momento no existe historia
         previa, y todas las variables de ritmo llegarian vacias.
    """
    df = df_con_historia

    df = df[df[config.TARGET].notna()]
    df = df[df[config.TARGET] >= 0]
    df = df[(df["Antiguedad_Vehiculo"].isna()) | (df["Antiguedad_Vehiculo"] >= 0)]

    # La base traia kilometrajes imposibles como 123.456.789 (alguien tecleando
    # el teclado). Un solo valor asi distorsiona la media y la desviacion del
    # StandardScaler y con ello TODA la escala de la variable Kms.
    df = df[df[config.COLUMNA_KMS] < config.KMS_MAXIMO_VALIDO]

    df = df[df["Total_Visitas_VIN"] >= config.MINIMO_VISITAS_POR_VIN]
    if config.EXCLUIR_PRIMER_INGRESO:
        df = df[df["Num_Visita"] >= 2]

    cols_categoricas_presentes = [c for c in config.COLUMNAS_CATEGORICAS if c in df.columns]
    columnas_finales = (
        [c for c in config.COLUMNAS_AUXILIARES if c in df.columns]
        + cols_categoricas_presentes
        + config.COLUMNAS_NUMERICAS
        + config.COLUMNAS_BINARIAS
        + [config.TARGET]
    )
    return df[columnas_finales].copy()


def generar_target_y_features(df_visitas: pd.DataFrame) -> pd.DataFrame: #esto hace que el DataFrame de visitas tenga la variable objetivo (dias hasta retorno) y las features de ingenieria (historia del VIN, antiguedad y bandera de si fue vendido por nosotros). Recibe un DataFrame de visitas y devuelve otro DataFrame listo para entrenar el modelo.
    """Atajo que encadena historia + filtrado en una sola llamada."""
    return preparar_dataset_entrenamiento(agregar_features_historia(df_visitas))


def etiquetar_fuga(df_con_historia: pd.DataFrame, umbral_dias: int = None) -> pd.DataFrame:
    """Marca cada visita como FUGA (1) o RETENCION (0), y rescata la senial que
    el modelo de regresion tenia prohibido ver.

    Aqui esta el aporte de reformular el problema como clasificacion. La
    regresion solo puede aprender de visitas que TUVIERON un regreso observado,
    asi que descarta la ultima visita de cada vehiculo: exactamente los clientes
    que no volvieron, que son los que interesa detectar. La etiqueta binaria si
    los recupera, distinguiendo dos casos:

      A) Hay visita siguiente  -> fuga = el intervalo supero el umbral.
      B) Es la ultima visita   -> si desde la entrega ya pasaron mas dias que el
         umbral y el vehiculo no aparecio, SABEMOS que se fugo. Es un dato duro,
         no una suposicion.

    Lo que queda fuera es solo lo genuinamente desconocido: la ultima visita de
    un vehiculo entregado hace menos del umbral. Todavia esta dentro de plazo, y
    contarla como fuga seria inventar un abandono que no ha ocurrido. En el
    lenguaje de supervivencia, esos casos estan CENSURADOS por la derecha.
    """
    umbral_dias = umbral_dias if umbral_dias is not None else config.UMBRAL_FUGA_DIAS
    df = df_con_historia.copy()

    # La fecha de corte es el ultimo dia que la base alcanza a observar. Todo lo
    # que se afirme sobre el futuro se mide contra ella.
    fecha_corte = df[config.COLUMNA_FECHA_ENTRADA].max()
    tiene_siguiente = df[config.TARGET].notna()
    dias_desde_entrega = (fecha_corte - df[config.COLUMNA_FECHA_CIERRE]).dt.days

    etiqueta = pd.Series(np.nan, index=df.index, dtype=float)
    etiqueta[tiene_siguiente] = (df.loc[tiene_siguiente, config.TARGET] > umbral_dias).astype(float)
    etiqueta[(~tiene_siguiente) & (dias_desde_entrega > umbral_dias)] = 1.0

    df[config.TARGET_FUGA] = etiqueta
    df["Censurado"] = (~tiene_siguiente) & (dias_desde_entrega <= umbral_dias)
    return df


def preparar_dataset_fuga(df_con_historia: pd.DataFrame, umbral_dias: int = None) -> pd.DataFrame:
    """Dataset para el clasificador: mismos filtros de calidad, pero conservando
    las ultimas visitas ya vencidas, que son la evidencia de abandono."""
    df = etiquetar_fuga(df_con_historia, umbral_dias)

    df = df[df[config.TARGET_FUGA].notna()]
    # El target de regresion puede ser nulo (es una ultima visita) pero nunca negativo.
    df = df[df[config.TARGET].isna() | (df[config.TARGET] >= 0)]
    df = df[(df["Antiguedad_Vehiculo"].isna()) | (df["Antiguedad_Vehiculo"] >= 0)]
    df = df[df[config.COLUMNA_KMS] < config.KMS_MAXIMO_VALIDO]
    df = df[df["Total_Visitas_VIN"] >= config.MINIMO_VISITAS_POR_VIN]
    if config.EXCLUIR_PRIMER_INGRESO:
        df = df[df["Num_Visita"] >= 2]

    cols_categoricas = [c for c in config.COLUMNAS_CATEGORICAS if c in df.columns]
    columnas = (
        [c for c in config.COLUMNAS_AUXILIARES if c in df.columns]
        + cols_categoricas
        + config.COLUMNAS_NUMERICAS
        + config.COLUMNAS_BINARIAS
        + [config.TARGET_FUGA]
    )
    return df[columnas].copy()


def construir_catalogo_vehiculos(df_visitas: pd.DataFrame) -> pd.DataFrame:
    """Ficha por chasis con su ULTIMA visita y su historia ya calculada.

    La API no puede recalcular la historia de un VIN en cada peticion (tendria
    que reprocesar 35.000 filas), y tampoco puede pedirsela al asesor: son once
    variables derivadas. Por eso el catalogo guarda exactamente el mismo vector
    de caracteristicas con el que se entreno, tomado de la ultima visita del
    vehiculo, que es justamente la visita cuyo retorno hay que proyectar.
    """
    col_vin = config.COLUMNA_VIN if config.COLUMNA_VIN in df_visitas.columns else "VIN"
    col_ano = config.COLUMNA_ANO if config.COLUMNA_ANO in df_visitas.columns else "Anio"

    df = agregar_features_historia(df_visitas)
    ultima = df.groupby(col_vin).tail(1).set_index(col_vin) #esto hace que para cada chasis nos quedemos con la ultima visita (la mas reciente) y la pongamos en un DataFrame indexado por VIN. De esa manera, para cada chasis tenemos su ultimo tipo de cargo, kilometraje, anio del modelo, fecha de ingreso y toda su historia ya calculada.

    # COHERENCIA ENTRENAMIENTO/SERVICIO: el entrenamiento descarta los
    # kilometrajes imposibles, asi que el modelo NUNCA vio esos valores. No
    # borramos el VIN del catalogo (el asesor debe poder consultarlo): solo
    # anulamos el dato para que el imputador use la mediana, igual que con
    # cualquier faltante.
    kms_catalogo = pd.to_numeric(ultima[config.COLUMNA_KMS], errors="coerce")
    kms_catalogo = kms_catalogo.where(kms_catalogo < config.KMS_MAXIMO_VALIDO)

    catalogo = pd.DataFrame(index=ultima.index)
    for columna in config.COLUMNAS_CATEGORICAS:
        catalogo[columna] = ultima[columna] if columna in ultima.columns else "MECANICA"
    for columna in config.COLUMNAS_NUMERICAS:
        catalogo[columna] = ultima[columna]
    catalogo[config.COLUMNA_KMS] = kms_catalogo
    catalogo["Es_Vehiculo_Vendido"] = ultima["Es_Vehiculo_Vendido"].astype(int)

    # Datos de contexto para que el asesor verifique que se busco el carro
    # correcto, y bandera de si el vehiculo tiene historia suficiente.
    # OJO: la Gama viaja aqui solo como dato de PANTALLA. El modelo no la consume
    # (ver config.COLUMNAS_CATEGORICAS), pero el asesor necesita ver que el
    # sistema encontro el carro correcto antes de creerle a la prediccion.
    catalogo["Gama"] = ultima[config.COLUMNA_GAMA]
    catalogo["Anio_Modelo"] = ultima[col_ano]
    catalogo["Ultima_Visita"] = ultima[config.COLUMNA_FECHA_ENTRADA]
    catalogo["Ultima_Entrega"] = ultima[config.COLUMNA_FECHA_CIERRE]
    catalogo["Total_Visitas"] = ultima["Total_Visitas_VIN"].astype(int)
    catalogo["Apto_Para_Modelo"] = (
        ultima["Total_Visitas_VIN"] >= config.MINIMO_VISITAS_POR_VIN
    ).astype(bool)

    return catalogo


# Columnas de cada visita que hacen falta para volver a calcular la historia de
# un VIN. Es lo minimo: el historial se carga entero en memoria en la API.
COLUMNAS_HISTORIAL = [
    config.COLUMNA_VIN, config.COLUMNA_OT, config.COLUMNA_FECHA_ENTRADA, config.COLUMNA_FECHA_CIERRE,
    config.COLUMNA_KMS, config.COLUMNA_GAMA, config.COLUMNA_ANO,
    config.COLUMNA_TIPO_CARGO, config.COLUMNA_TIPO_TRABAJO, "Es_Alistamiento_OT",
]


def historial_para_servicio(df_visitas: pd.DataFrame) -> pd.DataFrame:
    """Visitas consolidadas, reducidas a lo que la API necesita para recalcular."""
    return df_visitas[COLUMNAS_HISTORIAL].sort_values(
        [config.COLUMNA_VIN, config.COLUMNA_FECHA_ENTRADA]
    ).reset_index(drop=True)


def ficha_con_ingreso_nuevo(
    visitas_vin: pd.DataFrame,
    fecha_entrada,
    fecha_cierre,
    kms: float,
    tipo_cargo: str,
    tipo_trabajo: str,
) -> pd.Series:
    """Ficha del vehiculo como si el ingreso de HOY ya estuviera en la base.

    El modelo no necesita reentrenarse para atender un carro que acaba de
    entrar: lo que cambia es su HISTORIA (una visita mas, un intervalo nuevo, un
    ritmo y unos km/dia distintos). Se agrega la visita al final de las que ya
    tenia y se pasa todo por construir_catalogo_vehiculos(), la misma funcion
    que arma el catalogo del entrenamiento. Asi no hay una segunda version de
    las formulas que se pueda desalinear (training/serving skew).
    """
    previas = visitas_vin.sort_values(config.COLUMNA_FECHA_ENTRADA)
    anios = previas[config.COLUMNA_ANO].dropna()
    nueva = {
        config.COLUMNA_VIN: previas[config.COLUMNA_VIN].iloc[-1],
        config.COLUMNA_OT: "INGRESO_NUEVO",
        config.COLUMNA_FECHA_ENTRADA: pd.Timestamp(fecha_entrada),
        config.COLUMNA_FECHA_CIERRE: pd.Timestamp(fecha_cierre),
        config.COLUMNA_KMS: float(kms),
        config.COLUMNA_GAMA: previas[config.COLUMNA_GAMA].iloc[-1],
        config.COLUMNA_ANO: anios.iloc[-1] if len(anios) else np.nan,
        config.COLUMNA_TIPO_CARGO: tipo_cargo,
        config.COLUMNA_TIPO_TRABAJO: tipo_trabajo,
        # Un carro que ya tenia historia no puede estar entrando a alistamiento.
        "Es_Alistamiento_OT": False,
    }
    visitas = pd.concat([previas, pd.DataFrame([nueva])], ignore_index=True)
    return construir_catalogo_vehiculos(visitas).iloc[0]
