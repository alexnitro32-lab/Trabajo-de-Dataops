# TRANSFORMACION DE CARACTERISTICAS DE LA RECETA
# Este archivo contiene funciones que separan la matriz de caracteristicas (X) del vector objetivo (y), y que construyen un preprocesador para transformar las columnas numericas, categoricas y binarias de X antes de entrenar el modelo.

import pandas as pd
from sklearn.compose import ColumnTransformer          
from sklearn.pipeline import Pipeline                 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer               
from src import config


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: #Separa el DataFrame en la matriz de caracteristicas (X) y el vector objetivo (Y)
    # X = todo lo que el modelo puede VER para decidir (las pistas).
    X = df.drop(columns=[config.TARGET]).copy() # drop() quita la columna objetivo; .copy() evita modificar el original.
    y = df[config.TARGET].copy() # y = la respuesta correcta que el modelo debe aprender a estimar.
    return X, y

def crear_preprocesador() -> ColumnTransformer: #Construye un ColumnTransformer que aplica las transformaciones correspondientes a las columnas numericas, categoricas y binarias."""
    pipeline_numerico = Pipeline(steps=[ # Paso 1: si falta un kilometraje, se rellena con la MEDIANA. para que no de promedios con valores extremos
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()) # tandardScaler deja cada variable con media 0 y desviacion 1. Sin esto, Kms (decenas de miles) pesaria muchisimo mas que Antiguedad (unidades) solo por la magnitud del numero, no por su importancia real.
    ])
    
    pipeline_categorico = Pipeline(steps=[ # Pipeline para variables categoricas (Gama, Tipo Cargo, Tipo de Trabajo) si falta la categoria, se rellena con la MAS FRECUENTE (la moda).
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pipeline_binario = Pipeline(steps=[ # Pipeline para variables binarias (Es_Vehiculo_Vendido) 
        ("imputador", SimpleImputer(strategy="most_frequent")) # Ya vale 0 o 1: no hay nada que escalar ni codificar, solo cubrir nulos.
    ])

    preprocesador = ColumnTransformer( # Ensamble de transformaciones con ColumnTransformer Cada tupla es (nombre_del_bloque, receta_a_aplicar, lista_de_columnas). El ColumnTransformer corre los tres bloques en paralelo y pega los resultados en una sola matriz final lista para el modelo.
        transformers=[
            ("num", pipeline_numerico, config.COLUMNAS_NUMERICAS),
            ("cat", pipeline_categorico, config.COLUMNAS_CATEGORICAS),
            ("bin", pipeline_binario, config.COLUMNAS_BINARIAS)
        ],
        
        remainder="drop" # remainder="drop": cualquier columna que llegue y no este declarada en config se descarta. Es una defensa contra que se cuele basura al modelo.
    )

    return preprocesador
