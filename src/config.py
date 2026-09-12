# CENTRO DE CONTROL DEL PROYECTO esto solo guarda constantes (rutas, parametros nombres de columnas). si manana cambia el nombre de una columna o la ruta del CSV, se corrige AQUI y todo el proyecto (data.py, features.py, train.py, main.py) queda corregido.

from pathlib import Path  # Libreria estandar para manejar rutas de forma segura en Windows/Linux/Mac

RUTA_RAIZ = Path(__file__).resolve().parent.parent # es la carpeta raiz del proyecto
RUTA_DATA_RAW = RUTA_RAIZ / "data" / "Base de datos Limpia..csv" #esto lo que hace es concatenar la ruta raiz con la carpeta data y el nombre del archivo csv. el operador / de Path hace que funcione en cualquier sistema operativo.
RUTA_MODELO_SERIALIZADO = RUTA_RAIZ / "models" / "modelo_retencion.joblib" #esto es la ruta donde se guarda el modelo entrenado, para que main.py lo pueda cargar y usar en el endpoint /predict.
RUTA_CATALOGO_VEHICULOS = RUTA_RAIZ / "models" / "catalogo_vehiculos.joblib" #esto es la ruta donde se guarda el catalogo de vehiculos (marca, gama, antiguedad) que se construye en data.py y se usa en features.py para crear variables de ingenieria.

SEMILLA_ALEATORIA = 42 #esto es la semilla que se usa para que los resultados sean reproducibles. Si se cambia, el modelo puede dar resultados distintos cada vez que se entrena.
TAMANO_TEST = 0.2  # 20% para evaluacion, 80% para entrenamiento

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

TARGET = "Dias_Hasta_Retorno"

COLUMNAS_CATEGORICAS = [
    "Gama",
    "Tipo Cargo",
    "Tipo de Trabajo"
]

COLUMNAS_NUMERICAS = [
    "Kms.",
    "Antiguedad_Vehiculo"
]

COLUMNAS_BINARIAS = [
    "Es_Vehiculo_Vendido"  # Variable de ingenieria creada en data.py
]
