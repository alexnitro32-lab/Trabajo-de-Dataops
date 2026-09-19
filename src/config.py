# CENTRO DE CONTROL DEL PROYECTO esto solo guarda constantes (rutas, parametros nombres de columnas). si manana cambia el nombre de una columna o la ruta del CSV, se corrige AQUI y todo el proyecto (data.py, features.py, train.py, main.py) queda corregido.

from pathlib import Path  # Libreria estandar para manejar rutas de forma segura en Windows/Linux/Mac

RUTA_RAIZ = Path(__file__).resolve().parent.parent # es la carpeta raiz del proyecto
RUTA_DATA_RAW = RUTA_RAIZ / "data" / "Base de datos Limpia..csv" #esto lo que hace es concatenar la ruta raiz con la carpeta data y el nombre del archivo csv. el operador / de Path hace que funcione en cualquier sistema operativo.
RUTA_MODELO_SERIALIZADO = RUTA_RAIZ / "models" / "modelo_retencion.joblib" #esto es la ruta donde se guarda el modelo entrenado, para que main.py lo pueda cargar y usar en el endpoint /predict.
RUTA_CATALOGO_VEHICULOS = RUTA_RAIZ / "models" / "catalogo_vehiculos.joblib" #esto es la ruta donde se guarda el catalogo de vehiculos (marca, gama, antiguedad) que se construye en data.py y se usa en features.py para crear variables de ingenieria.
RUTA_MODELO_FUGA = RUTA_RAIZ / "models" / "modelo_fuga.joblib" #clasificador de fuga: responde "este cliente va a dejar de volver?", que es la pregunta que mueve la accion comercial.
RUTA_MODELOS_FUGA_SEGMENTADOS = RUTA_RAIZ / "models" / "modelos_fuga_segmentados.joblib" #un clasificador por origen (vendidos / externos), pedidos por el negocio para poder actuar distinto sobre cada poblacion.
RUTA_METRICAS = RUTA_RAIZ / "models" / "metricas_entrenamiento.json" #evidencia reproducible de la comparacion global vs segmentado; la tesis cita este archivo, no la consola.

SEMILLA_ALEATORIA = 42 #esto es la semilla que se usa para que los resultados sean reproducibles. Si se cambia, el modelo puede dar resultados distintos cada vez que se entrena.
TAMANO_TEST = 0.2  # 20% para evaluacion, 80% para entrenamiento
FOLDS_VALIDACION = 5  # Particiones de la validacion cruzada por VIN (GroupKFold).

KMS_MAXIMO_VALIDO = 5_00_000 #esto es un filtro de calidad de datos: si el kilometraje es mayor a este valor, se considera que es un error de digitacion y se descarta el registro.

COLUMNA_OT = "OT"                              
COLUMNA_NUM_TRABAJO = "Núm. Trabajo"           
COLUMNA_DESCRIPCION = "Descripción Trabajo"    
COLUMNA_VIN = "VIN"                            
COLUMNA_MARCA = "Marca"                        
COLUMNA_GAMA = "Gama"                        
COLUMNA_ANO = "Año"                          
COLUMNA_TIPO_CARGO = "Tipo Cargo"            
COLUMNA_TIPO_TRABAJO = "Tipo de Trabajo"       
COLUMNA_CLIENTE_CARGO = "Cód. Cliente Cargo"   
COLUMNA_KMS = "Kms."                         
COLUMNA_FECHA_ENTRADA = "F. Entrada"          
COLUMNA_FECHA_CIERRE = "F. Cierre"             
COLUMNA_ESTADO = "Estado"                     

# -------------------------------------------------------------------------
# CONTRATO DE DATOS (src/esquema.py)
# -------------------------------------------------------------------------
# Umbrales que separan "la base de siempre" de "algo cambio y hay que mirar".
# Estan aqui y no dentro del esquema para que se puedan ajustar sin tocar la
# logica de validacion, igual que cualquier otra constante de negocio.

# Proporcion minima de fechas que deben poder leerse como dia/mes/anio. Si el
# ERP exporta en otro formato, esta es la comprobacion que lo detiene: por
# debajo de este valor el pipeline falla en vez de convertir todo a nulo.
PROPORCION_MINIMA_FECHAS_VALIDAS = 0.90

# Filas minimas que deben sobrevivir a cada etapa. Son la defensa contra la
# degradacion silenciosa: un cambio de formato no rompe el tipo de dato, vacia
# el DataFrame, y solo contar filas atrapa ese caso. Los valores estan holgados
# respecto de lo que produce la base actual (34.935 / 23.430 / 8.938 / 9.289):
# la idea es detectar un derrumbe, no sobresaltarse por una variacion normal.
MINIMO_FILAS_CSV = 20_000
MINIMO_VISITAS_CONSOLIDADAS = 12_000
MINIMO_FILAS_ENTRENAMIENTO = 3_000
MINIMO_VEHICULOS_CATALOGO = 4_000

# Rango admisible de la tasa de fuga. Fuera de el, la etiqueta quedo degenerada
# y el clasificador aprenderia a responder siempre lo mismo.
TASA_FUGA_MINIMA = 0.05
TASA_FUGA_MAXIMA = 0.60

# -------------------------------------------------------------------------
# REGLAS DE CALIDAD DE LA BASE
# -------------------------------------------------------------------------
# Tres defectos verificados sobre esta base. Cada uno mete ruido directo en la
# variable objetivo, asi que se neutralizan antes de consolidar las visitas.

# 4.715 filas (13,5%) son facturas ANULADAS: el vehiculo no ingreso realmente,
# asi que contarlas inventa visitas y acorta artificialmente los intervalos.
ESTADOS_EXCLUIDOS = ("Anulado",)

# 667 filas traen Año = 1997. No es un año real: es el relleno que pone el ERP
# cuando el campo viene vacio (las gamas dominantes de ese "año" son Kona y
# Palisade, modelos que no existian). Sin neutralizarlo, esos carros entran al
# modelo con 29 años de antiguedad.
ANIO_CENTINELA = 1997

# 922 lineas son cobro de parqueo, no mantenimiento. Una OT de bodegaje no es
# un ingreso a taller y no debe contar como visita de posventa.
PATRON_BODEGAJE = "BODEGAJE"

# Marca el primer ingreso de un vehiculo que vendimos nosotros (alistamiento).
PATRON_ALISTAMIENTO = "ALISTAMIENTO|VEHICULO NUEVO"

# 1.510 OT mezclan Tipo de Trabajo y 1.971 mezclan Tipo Cargo. Resolverlo con
# "first" dejaba que el ORDEN DEL EXPORT decidiera el valor (no reproducible).
# Se resuelve por moda y, si hay empate, por esta prioridad de negocio.
PRIORIDAD_TIPO_TRABAJO = {"MECANICA Y COLISION": 0, "COLISION": 1, "MECANICA": 2, "OTRO": 3}
PRIORIDAD_TIPO_CARGO = {"Cliente": 0, "Garantia": 1, "Seguros": 2, "Interno": 3}

# -------------------------------------------------------------------------
# REGLA DE NEGOCIO DEL MANTENIMIENTO PREVENTIVO
# -------------------------------------------------------------------------
# El plan de fabrica obliga a entrar cada 365 dias O cada 10.000 km, lo que
# ocurra primero. Es la referencia contra la cual el modelo debe compararse:
# un vehiculo que recorre 100 km/dia alcanza los 10.000 km en 100 dias, mucho
# antes del año, y por eso el ritmo real depende del uso, no del calendario.
REGLA_MANTENIMIENTO_DIAS = 365
REGLA_MANTENIMIENTO_KMS = 10_000

# -------------------------------------------------------------------------
# UNIDAD DE ANALISIS: EL VEHICULO, NO LA VISITA SUELTA
# -------------------------------------------------------------------------
# Un VIN con UN SOLO ingreso no tiene intervalo observado: no hay forma de
# saber cada cuanto vuelve, asi que no aporta nada al modelo y se descarta.
# Ademas se descarta el PRIMER ingreso de cada vehiculo, porque en ese momento
# todavia no existe historia previa que mirar.
MINIMO_VISITAS_POR_VIN = 2
EXCLUIR_PRIMER_INGRESO = True

TARGET = "Dias_Hasta_Retorno"
COLUMNA_GRUPO = "VIN"  # Clave de agrupacion del split: ningun VIN puede estar en train y test a la vez.

# -------------------------------------------------------------------------
# UNIDAD DE PREDICCION: EL CLIENTE (VIN), NO EL MODELO DEL VEHICULO
# -------------------------------------------------------------------------
# La pregunta del negocio es "cuando vuelve ESTE cliente", no "como se comporta
# un Tucson". La gama NO entra al modelo: en esta base esta confundida con la
# antiguedad del parque (Kona promedia 1,4 anios y Grand I10 HB 9,8), de modo
# que funciona como un atajo hacia la edad del carro en lugar de describir el
# comportamiento del cliente. Dejarla fuera cuesta 0,013 de AUC y a cambio el
# modelo se sostiene sobre la conducta observada del chasis y no se cae cuando
# entra una gama que nunca se habia visto.
COLUMNAS_CATEGORICAS = [
    "Tipo Cargo",       # quien paga: Cliente / Garantia / Seguros / Interno
    "Tipo de Trabajo"   # que se le hizo: MECANICA / COLISION / OTRO
]

# Variables del estado del vehiculo en la visita actual (lo que ya se sabia antes).
COLUMNAS_NUMERICAS_BASE = [
    "Kms.",
    "Antiguedad_Vehiculo",
    # DEMORA DEL TALLER = F. Cierre - F. Entrada. Es el unico dato de la
    # experiencia del cliente que el ERP si registra: cuantos dias se le retuvo
    # el carro. La mitad de las visitas se resuelven el mismo dia (0 dias), y el
    # 5% pasa de 56 dias. No se usa la columna "Dif. F. Alta-F.Cierre" del ERP
    # porque trae valores negativos y la mitad en cero.
    "Dias_En_Taller",
    "Mes_Entrada",
]

# HISTORIA DEL PROPIO VIN. Este bloque es el que le da sentido al modelo: la
# mejor pista de cuando vuelve un carro es cada cuanto venia volviendo. Todas
# se calculan con informacion ESTRICTAMENTE PASADA respecto de la visita, para
# no filtrar el futuro dentro de la prediccion.
COLUMNAS_NUMERICAS_HISTORIA = [
    "Num_Visita",              # Cuantas veces ha entrado este VIN hasta hoy (1a, 2a, 3a...)
    "Dias_Desde_Visita_Ant",   # Cuanto tardo en volver la ultima vez
    "Km_Desde_Visita_Ant",     # Cuantos km recorrio entre la visita anterior y esta
    "Ritmo_Mediano_Previo",    # Su costumbre: mediana de TODOS sus intervalos anteriores
    "Ritmo_Max_Previo",        # La ausencia mas larga que ya se le conoce
    "Km_Por_Dia",              # Intensidad de uso en el ultimo tramo
    "Km_Por_Dia_Medio",        # Intensidad de uso tipica del vehiculo
    "Dias_Para_10k_Km",        # A su ritmo de uso, cuantos dias tarda en hacer 10.000 km
    "Regla_Preventiva_Dias",   # min(365 dias, dias para 10.000 km) = la regla de negocio
    "Km_Sobre_Umbral_10k",     # Cuantos ciclos de 10.000 km cubrio desde la visita anterior
    "Ciclos_10k_Acumulados",   # Odometro expresado en ciclos de mantenimiento
]

COLUMNAS_NUMERICAS = COLUMNAS_NUMERICAS_BASE + COLUMNAS_NUMERICAS_HISTORIA

COLUMNAS_BINARIAS = [
    "Es_Vehiculo_Vendido"  # Variable de ingenieria creada en data.py
]

# Columnas que viajan con el dataset para agrupar y auditar, pero que el modelo
# NO ve (remainder="drop" del ColumnTransformer las descarta).
COLUMNAS_AUXILIARES = [COLUMNA_VIN, COLUMNA_OT, COLUMNA_FECHA_ENTRADA, COLUMNA_FECHA_CIERRE]

# -------------------------------------------------------------------------
# SEGUNDO PROBLEMA: CLASIFICACION DE FUGA
# -------------------------------------------------------------------------
# "En cuantos dias vuelve" (regresion) y "va a volver o no" (clasificacion) son
# dos preguntas distintas, y la segunda es la que dispara la accion comercial.
# La clasificacion ademas permite medir con AUC y con falsos positivos, que es
# lo que deja decidir el punto de operacion con criterio de negocio.
TARGET_FUGA = "Fuga"

# Un vehiculo se considera FUGADO si no volvio dentro de este plazo. Se toman
# 365 dias porque es el plazo del plan de mantenimiento de fabrica: pasado un
# anio sin aparecer, el vehiculo ya incumplio el plan.
UMBRAL_FUGA_DIAS = 365

# Probabilidad a partir de la cual se emite la alerta. 0,50 maximiza el F1;
# bajarlo atrapa mas fugas a costa de mas llamadas innecesarias. La tabla de
# puntos de operacion se imprime en cada entrenamiento para poder elegirlo con
# criterio de negocio y no por defecto.
UMBRAL_DECISION_FUGA = 0.50

# Modelos a comparar. Se excluye la regresion lineal/logistica por decision del
# negocio; los tres son de ensamble pero de familias distintas: bagging con
# cortes optimos (RandomForest), boosting secuencial (HistGradientBoosting) y
# bagging con cortes aleatorios (ExtraTrees).
MODELOS_CLASIFICACION = ("RandomForest", "HistGradientBoosting", "ExtraTrees")

# El negocio pidio poder actuar distinto sobre cada poblacion, asi que ademas
# del modelo global se entrena y se congela uno por origen.
ENTRENAR_SEGMENTADO_POR_ORIGEN = True
