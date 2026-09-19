# TRANSFORMACION DE CARACTERISTICAS DE LA RECETA
# Este archivo contiene funciones que separan la matriz de caracteristicas (X) del vector objetivo (y), y que construyen un preprocesador para transformar las columnas numericas, categoricas y binarias de X antes de entrenar el modelo.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src import config


def columnas_modelo(df: pd.DataFrame | None = None) -> list[str]:
    """Lista exacta de columnas que el modelo consume, en orden fijo.

    Se centraliza aqui para que el entrenamiento y la API construyan SIEMPRE el
    mismo vector. Si una columna se agrega en config y alguien olvida tocar
    main.py, el ColumnTransformer fallaria en produccion pero no en el CI.
    """
    categoricas = config.COLUMNAS_CATEGORICAS
    if df is not None:
        categoricas = [c for c in categoricas if c in df.columns]
    return list(config.COLUMNAS_NUMERICAS) + list(categoricas) + list(config.COLUMNAS_BINARIAS)


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: #Separa el DataFrame en la matriz de caracteristicas (X) y el vector objetivo (Y)
    # X = solo las columnas declaradas en config. El VIN, la OT y las fechas
    # viajan en el DataFrame para poder agrupar y auditar, pero NO son pistas:
    # son identificadores, y meterlos al modelo seria sobreajuste puro.
    X = df[columnas_modelo(df)].copy()
    y = df[config.TARGET].copy() # y = la respuesta correcta que el modelo debe aprender a estimar.
    return X, y


def separar_x_y_grupo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Igual que separar_x_y(), pero devuelve ademas el VIN como vector de grupo.

    Es la pieza que hace honesta la evaluacion. Un mismo vehiculo aporta varias
    visitas al dataset, y todas comparten su ritmo. Con un split aleatorio, la
    visita 3 de un carro queda en entrenamiento y la visita 4 en prueba: el
    modelo ya vio la respuesta y la metrica sale inflada. Agrupando por VIN,
    un vehiculo esta entero en train o entero en test, nunca repartido.
    """
    X, y = separar_x_y(df)
    grupos = df[config.COLUMNA_GRUPO].copy()
    return X, y, grupos


def crear_preprocesador() -> ColumnTransformer: #Construye un ColumnTransformer que aplica las transformaciones correspondientes a las columnas numericas, categoricas y binarias."""
    pipeline_numerico = Pipeline(steps=[ # Paso 1: si falta un kilometraje, se rellena con la MEDIANA. para que no de promedios con valores extremos
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()) # StandardScaler deja cada variable con media 0 y desviacion 1. Sin esto, Kms (decenas de miles) pesaria muchisimo mas que Antiguedad (unidades) solo por la magnitud del numero, no por su importancia real.
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
