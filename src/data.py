# En este archivo lo que hacemos es leer el CSV original, limpiar y consolidar los datos, crear la variable objetivo y las features de ingenieria, y devolver un DataFrame listo para entrenar el modelo. Tambien construimos un catalogo de vehiculos que se guarda en disco para que la API pueda autocompletar caracteristicas de un carro a partir del VIN.

import pandas as pd      
from src import config   

def cargar_datos(ruta_csv: str) -> pd.DataFrame: #esto es una funcion que recibe la ruta del CSV y devuelve un DataFrame de pandas con los datos cargados.
    try: #este try/except es un plan A/B defensivo: si el archivo fue guardado con codificacion UTF-8, lo leemos asi; si no, reintentamos con latin1.
        df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="latin1")

    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip() #esto es para limpiar los encabezados de columnas: a veces el CSV trae un caracter invisible al inicio (BOM) y espacios en blanco al final, y eso rompe el pipeline. Lo que hacemos es reemplazar el BOM por nada y quitar los espacios en blanco al inicio y al final de cada nombre de columna.
    return df


def consolidar_visitas_por_ot(df: pd.DataFrame) -> pd.DataFrame: #esto sirve para consolidar multiples filas de sub-trabajos de una misma OT en 1 solo registro de ingreso fisico del vehiculo al taller. Recibe un DataFrame y devuelve otro DataFrame con las visitas consolidadas.
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    df[config.COLUMNA_FECHA_ENTRADA] = pd.to_datetime( 
        df[config.COLUMNA_FECHA_ENTRADA], dayfirst=True, errors="coerce"
    )
    df[config.COLUMNA_FECHA_CIERRE] = pd.to_datetime(
        df[config.COLUMNA_FECHA_CIERRE], dayfirst=True, errors="coerce"
    )

    col_kms = config.COLUMNA_KMS if config.COLUMNA_KMS in df.columns else "Kms."
    col_ano = config.COLUMNA_ANO if config.COLUMNA_ANO in df.columns else "Año"

    df[col_kms] = pd.to_numeric(df[col_kms], errors="coerce") # to_numeric convierte texto a numero; lo que no se pueda convertir queda NaN.
    df[col_ano] = pd.to_numeric(df[col_ano], errors="coerce")

    posibles_cols_cliente = [ 
        config.COLUMNA_CLIENTE_CARGO,
        "Cód. Cliente Cargo",
        "Cliente Cargo",
        "Cod. Cliente Cargo",
        "Cliente"
    ]

    col_cliente = next((c for c in posibles_cols_cliente if c in df.columns), None) #esto busca la primera columna que exista en el DataFrame y que coincida con alguna de las posibles columnas de cliente. Si no encuentra ninguna, devuelve None.
    if col_cliente is None:
        # Ultimo recurso: buscamos cualquier columna que contenga "cliente".
        col_cliente = next((c for c in df.columns if "cliente" in c.lower()), "Cliente_Temp")
        if col_cliente == "Cliente_Temp":
            # Si de plano no existe, creamos una columna dummy para no romper el groupby.
            df["Cliente_Temp"] = "Desconocido"

   
    df[col_cliente] = df[col_cliente].astype(str)  # Forzamos a texto: los codigos de cliente son identificadores, no cantidades.

    col_ot = config.COLUMNA_OT if config.COLUMNA_OT in df.columns else "OT"
    col_vin = config.COLUMNA_VIN if config.COLUMNA_VIN in df.columns else "VIN"

    df = df.dropna(subset=[col_vin, col_ot, config.COLUMNA_FECHA_ENTRADA, config.COLUMNA_FECHA_CIERRE]) #Eliminamos registros sin VIN o fechas validas: sin chasis no sabemos de que carro hablamos, y sin fechas no podemos calcular dias de retorno.

    col_desc = config.COLUMNA_DESCRIPCION if config.COLUMNA_DESCRIPCION in df.columns else "Descripción Trabajo"

    def es_alistamiento(descripciones): #definimos una funcion que recibe una serie de descripciones de trabajos y devuelve True si alguna de ellas contiene "ALISTAMIENTO" o "VEHICULO NUEVO", y False en caso contrario. Esto nos permite identificar si la OT fue un alistamiento (venta) o no.
        textos = [str(x) for x in descripciones if pd.notna(x)]
        texto_unido = " ".join(textos).upper()  # upper() para comparar sin importar mayusculas
        return "ALISTAMIENTO" in texto_unido or "VEHICULO NUEVO" in texto_unido

    # Agrupacion por VIN y OT: aqui se colapsan los sub-trabajos.
    dict_agg = {
        config.COLUMNA_FECHA_ENTRADA: "min",   
        config.COLUMNA_FECHA_CIERRE: "max",    
        col_kms: "max",                        
        config.COLUMNA_MARCA: "first",         
        config.COLUMNA_GAMA: "first",
        col_ano: "first",
        config.COLUMNA_TIPO_CARGO: "first",
        col_cliente: "first",
        col_desc: es_alistamiento   
    }

    # Solo agregamos Tipo de Trabajo si realmente existe en el archivo.
    if config.COLUMNA_TIPO_TRABAJO in df.columns:
        dict_agg[config.COLUMNA_TIPO_TRABAJO] = "first"

    # groupby + agg: de ~39.000 filas de sub-trabajos pasamos a las visitas reales reset_index() devuelve VIN y OT de indice a columnas normales.
    visitas = df.groupby([col_vin, col_ot]).agg(dict_agg).reset_index()

    # La columna de descripcion ya no contiene texto sino el True/False quedevolvio es_alistamiento(), por eso la renombramos.
    visitas.rename(columns={col_desc: "Es_Alistamiento_OT"}, inplace=True)
    return visitas


def generar_target_y_features(df_visitas: pd.DataFrame) -> pd.DataFrame: #esto hace que el DataFrame de visitas tenga la variable objetivo (dias hasta retorno) y las features de ingenieria (antiguedad del vehiculo y bandera de si fue vendido por nosotros). Recibe un DataFrame de visitas y devuelve otro DataFrame listo para entrenar el modelo.
    col_vin = config.COLUMNA_VIN if config.COLUMNA_VIN in df_visitas.columns else "VIN" # Ordenar por VIN y fecha es OBLIGATORIO: el calculo de "proxima visita" depende de que las visitas de cada carro esten en orden cronologico.
    df_visitas = df_visitas.sort_values(by=[col_vin, config.COLUMNA_FECHA_ENTRADA])

    # La bandera se deriva del PRIMER ingreso de cada VIN: si esa OT fue un alistamiento, el vehiculo fue vendido por nosotros.
    primeros_ingresos = df_visitas.groupby(col_vin)["Es_Alistamiento_OT"].first().reset_index()
    primeros_ingresos.rename(columns={"Es_Alistamiento_OT": "Es_Vehiculo_Vendido"}, inplace=True)

    # merge = pegar esa bandera a TODAS las visitas del mismo VIN 
    df_visitas = df_visitas.merge(primeros_ingresos, on=col_vin, how="left")

    # DataOps: SimpleImputer no admite dtype bool. Convertimos la bandera a entero 0/1.
    df_visitas["Es_Vehiculo_Vendido"] = (
        df_visitas["Es_Vehiculo_Vendido"].fillna(False).astype(int)
    )

    # shift(-1) dentro de cada VIN trae la fecha de entrada de la SIGUIENTE visita a la fila actual. Este es el truco que permite mirar "hacia el futuro" y construir la etiqueta que el modelo debe aprender.
    df_visitas["F. Entrada_Siguiente"] = df_visitas.groupby(col_vin)[config.COLUMNA_FECHA_ENTRADA].shift(-1)

    # VARIABLE OBJETIVO: dias entre que le entregamos el carro (F. Cierre) y la fecha en que volvio (F. Entrada de la siguiente visita). El .dt.days pasa la diferencia de fechas a un numero entero de dias.
    df_visitas[config.TARGET] = (df_visitas["F. Entrada_Siguiente"] - df_visitas[config.COLUMNA_FECHA_CIERRE]).dt.days

    # La ultima visita de cada carro no tiene "siguiente", asi que su target es nulo. Esas filas no sirven para entrenar y se descartan.
    df_visitas = df_visitas.dropna(subset=[config.TARGET])

    col_ano = config.COLUMNA_ANO if config.COLUMNA_ANO in df_visitas.columns else "Año"

    # antiguedad del vehiculo al momento de la visita.Es mas informativa que el anio/modelo suelto, porque relaciona el carro con el momento en que ingreso al taller.
    df_visitas["Antiguedad_Vehiculo"] = df_visitas[config.COLUMNA_FECHA_ENTRADA].dt.year - df_visitas[col_ano]

    # No existen retornos negativos (volver antes de que sem lo entregaramos) ni carros con antiguedad negativa. Son errores de digitacion.
    df_visitas = df_visitas[df_visitas[config.TARGET] >= 0]
    df_visitas = df_visitas[df_visitas["Antiguedad_Vehiculo"] >= 0]

    # Tercer filtro de sanidad: la base traia kilometrajes imposibles como 123.456.789 (alguien tecleando el teclado en el sistema del taller). Un solo valor asi distorsiona la media y la desviacion que calcula el StandardScaler, y con ello TODA la escala de la variable Kms.
    df_visitas = df_visitas[df_visitas[config.COLUMNA_KMS] < config.KMS_MAXIMO_VALIDO]

    # Nos quedamos SOLO con las columnas que el modelo va a usar. Todo lo demas (VIN, OT, cliente, fechas) se descarta: son identificadores, no predictores,e incluirlos causaria fuga de informacion o sobreajuste.
    cols_categoricas_presentes = [c for c in config.COLUMNAS_CATEGORICAS if c in df_visitas.columns]
    columnas_finales = (
        cols_categoricas_presentes +
        config.COLUMNAS_NUMERICAS +
        config.COLUMNAS_BINARIAS +
        [config.TARGET]
    )

    # El .copy() evita el SettingWithCopyWarning de pandas al devolver un sub-DataFrame.
    return df_visitas[columnas_finales].copy()


def construir_catalogo_vehiculos(df_visitas: pd.DataFrame) -> pd.DataFrame: # lo que se hizo aca es crear un catalogo de vehiculos con las caracteristicas mas recientes de cada chasis, para que la API pueda autocompletar datos a partir del VIN. Recibe un DataFrame de visitas y devuelve otro DataFrame con el catalogo de vehiculos.
    col_vin = config.COLUMNA_VIN if config.COLUMNA_VIN in df_visitas.columns else "VIN"
    col_ano = config.COLUMNA_ANO if config.COLUMNA_ANO in df_visitas.columns else "Año"

    # Ordenar cronologicamente por chasis: de eso depende cual es la "ultima" visita.
    df = df_visitas.sort_values(by=[col_vin, config.COLUMNA_FECHA_ENTRADA])

    vendido = df.groupby(col_vin)["Es_Alistamiento_OT"].first().fillna(False).astype(int) # Misma logica que generar_target_y_features(): la bandera de "lo vendimos nosotros" se deriva del PRIMER ingreso del chasis (si fue un alistamiento).

    ultima = df.groupby(col_vin).tail(1).set_index(col_vin) #esto hace que para cada chasis nos quedemos con la ultima visita (la mas reciente) y la pongamos en un DataFrame indexado por VIN. De esa manera, para cada chasis tenemos su ultima marca, gama, tipo de cargo, kilometraje, anio del modelo y fecha de ingreso al taller.

    # COHERENCIA ENTRENAMIENTO/SERVICIO: generar_target_y_features() descarta los
    # kilometrajes imposibles (errores de digitacion tipo 123.456.789), asi que el
    # modelo NUNCA vio esos valores. Si el catalogo se los entregara en produccion,
    # el modelo estaria extrapolando fuera de su rango conocido. No borramos el VIN
    # del catalogo (el asesor debe poder consultarlo): solo anulamos el dato para
    # que el imputador use la mediana, igual que con cualquier valor faltante.
    kms_catalogo = pd.to_numeric(ultima[config.COLUMNA_KMS], errors="coerce")
    kms_catalogo = kms_catalogo.where(kms_catalogo < config.KMS_MAXIMO_VALIDO)

    # Armamos la ficha con los mismos nombres que espera el pipeline, mas el anio del modelo y la fecha de la ultima visita (para mostrarlos al asesor).

    catalogo = pd.DataFrame({
        "Gama": ultima[config.COLUMNA_GAMA],
        "Tipo Cargo": ultima[config.COLUMNA_TIPO_CARGO],
        "Tipo de Trabajo": ultima.get(config.COLUMNA_TIPO_TRABAJO, "MECANICA"),
        "Kms.": kms_catalogo,
        "Anio_Modelo": ultima[col_ano],
        "Es_Vehiculo_Vendido": vendido,
        "Ultima_Visita": ultima[config.COLUMNA_FECHA_ENTRADA],
    })

    return catalogo
