# Este es el DIRECTOR DE ORQUESTA del proyecto. No define logica nueva: llama en
# orden a los modulos anteriores (data -> diagnostico -> features -> modelos ->
# evaluate) y termina congelando los resultados en disco.
#
# Responde dos preguntas distintas con dos modelos distintos:
#   REGRESION      "en cuantos dias vuelve este vehiculo"  -> para agendar y
#                  dimensionar bahias.
#   CLASIFICACION  "este cliente va a dejar de volver"     -> para actuar antes
#                  de perderlo. Es la que mueve la accion comercial, y la unica
#                  que se puede medir con AUC y con falsos positivos.

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src import config, data, diagnostico, esquema, evaluate, features, modelos


def _cv_por_vin(df, y, folds=None):
    """Genera los cortes de validacion agrupando por VIN.

    Un mismo vehiculo aporta varias visitas y todas comparten su ritmo. Con un
    split aleatorio, la visita 3 de un carro cae en entrenamiento y la 4 en
    prueba: el modelo ya vio la respuesta y la metrica sale inflada. Agrupando,
    un vehiculo esta entero en train o entero en test.
    """
    folds = folds or config.FOLDS_VALIDACION
    return GroupKFold(n_splits=folds).split(df, y, df[config.COLUMNA_GRUPO])


# ---------------------------------------------------------------------------
# REGRESION: en cuantos dias vuelve
# ---------------------------------------------------------------------------

def validar_regresion(df: pd.DataFrame) -> dict[str, float]:
    """Validacion cruzada de la regresion, agrupada por vehiculo."""
    X, y, _ = features.separar_x_y_grupo(df)
    y = y.astype(float)

    resultados = []
    for idx_train, idx_test in _cv_por_vin(df, y):
        modelo = modelos.crear_regresor().fit(X.iloc[idx_train], y.iloc[idx_train])
        prediccion = np.clip(modelo.predict(X.iloc[idx_test]), 0, None)
        resultados.append(evaluate.evaluar_modelo(y.iloc[idx_test], prediccion))

    return {
        f"{estadistico}_{metrica}": float(funcion([r[metrica] for r in resultados]))
        for metrica in ("MAE_dias", "RMSE_dias", "R2_score")
        for estadistico, funcion in (("media", np.mean), ("desv", np.std))
    }


def calcular_referencias(df: pd.DataFrame) -> dict:
    """Referencias sin aprendizaje contra las que la regresion debe ganar.

    Sin ellas un R2 de 0,12 no dice nada. Estas dicen contra que se compara:
      - La MEDIANA: predecir siempre lo mismo. Es el piso absoluto.
      - La REGLA PREVENTIVA de fabrica (365 dias o 10.000 km). Es lo que el
        taller ya puede calcular sin ningun modelo, y por tanto la barra real.
      - El RITMO del propio vehiculo, sin modelo que lo corrija.
    """
    y = df[config.TARGET].astype(float)
    return {
        "mediana_constante": evaluate.evaluar_modelo(y, np.full(len(y), y.median())),
        "regla_preventiva_365d_10000km": evaluate.evaluar_modelo(
            y, df["Regla_Preventiva_Dias"].fillna(float(config.REGLA_MANTENIMIENTO_DIAS))
        ),
        "ritmo_mediano_del_propio_vin": evaluate.evaluar_modelo(
            y, df["Ritmo_Mediano_Previo"].fillna(y.median())
        ),
    }


# ---------------------------------------------------------------------------
# CLASIFICACION: va a dejar de volver
# ---------------------------------------------------------------------------

def comparar_clasificadores(df: pd.DataFrame) -> dict:
    """Compara los tres modelos, en version global y segmentada por origen.

    Todo se evalua sobre LOS MISMOS cortes de validacion y con predicciones
    fuera de muestra: cada fila recibe su probabilidad del fold en el que quedo
    como prueba. Asi las cifras de los seis modelos son comparables entre si, y
    la matriz de confusion se calcula una sola vez sobre el dataset completo.
    """
    df = df.reset_index(drop=True)
    columnas = features.columnas_modelo(df)
    X = df[columnas]
    y = df[config.TARGET_FUGA].astype(int)
    origen = df["Es_Vehiculo_Vendido"]

    cortes = list(_cv_por_vin(df, y))
    salida = {}

    for nombre in config.MODELOS_CLASIFICACION:
        proba_global = np.zeros(len(df))
        proba_segmentada = np.zeros(len(df))

        for idx_train, idx_test in cortes:
            # Modelo unico: recibe Es_Vehiculo_Vendido como una variable mas, asi
            # que puede separar los dos origenes por si mismo si le conviene.
            ajustado = modelos.ajustar_clasificador(nombre, X.iloc[idx_train], y.iloc[idx_train])
            proba_global[idx_test] = ajustado.predict_proba(X.iloc[idx_test])[:, 1]

            # Modelos separados: uno por origen, cada uno con solo su mitad.
            for valor in (0, 1):
                train_seg = idx_train[origen.iloc[idx_train].to_numpy() == valor]
                test_seg = idx_test[origen.iloc[idx_test].to_numpy() == valor]
                if len(train_seg) == 0 or len(test_seg) == 0:
                    continue
                seg = modelos.ajustar_clasificador(nombre, X.iloc[train_seg], y.iloc[train_seg])
                proba_segmentada[test_seg] = seg.predict_proba(X.iloc[test_seg])[:, 1]

        entrada = {"global": evaluate.evaluar_clasificacion(y, proba_global)}
        for valor, sufijo in ((1, "vendidos"), (0, "externos")):
            mascara = (origen == valor).to_numpy()
            if mascara.sum() == 0:
                continue
            entrada[f"global_sobre_{sufijo}"] = evaluate.evaluar_clasificacion(y[mascara], proba_global[mascara])
            entrada[f"segmentado_sobre_{sufijo}"] = evaluate.evaluar_clasificacion(y[mascara], proba_segmentada[mascara])
        entrada["_proba_global"] = proba_global
        salida[nombre] = entrada

    return salida


def _imprimir_clasificacion(comparacion: dict) -> None:
    cabecera = (f"{'':<44} {'AUC':>7} {'PR_AUC':>7} {'precis':>7} {'recall':>7} {'F1':>7} | "
                f"{'FP':>5} {'FN':>5} | {'%FP':>6} {'%FN':>6}")

    def fila(etiqueta, m):
        print(f"   {etiqueta:<44} {m['AUC']:7.4f} {m['PR_AUC']:7.4f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['F1']:7.3f} | {m['falsos_positivos']:5d} {m['falsos_negativos']:5d} | "
              f"{100*m['tasa_falsos_positivos']:5.1f}% {100*m['tasa_falsos_negativos']:5.1f}%")

    print("   " + cabecera)
    for nombre, m in comparacion.items():
        fila(f"{nombre}  [GLOBAL, todo el test]", m["global"])

    print("\n   GLOBAL vs SEGMENTADO, evaluado sobre cada origen:")
    print("   " + cabecera)
    for nombre, m in comparacion.items():
        print(f"   --- {nombre} ---")
        for sufijo, etiqueta in (("vendidos", "VENDIDOS por nosotros"), ("externos", "EXTERNOS")):
            if f"global_sobre_{sufijo}" not in m:
                continue
            fila(f"  GLOBAL     sobre {etiqueta}", m[f"global_sobre_{sufijo}"])
            fila(f"  SEGMENTADO sobre {etiqueta}", m[f"segmentado_sobre_{sufijo}"])


def elegir_mejor(comparacion: dict) -> str:
    """El ganador se decide por AUC, que no depende del umbral de decision.

    Usar F1 o exactitud para elegir mezclaria dos decisiones distintas: que
    algoritmo ordena mejor el riesgo, y donde poner el corte de alerta. El AUC
    responde solo la primera, que es la que corresponde al modelo; la segunda es
    del negocio y se resuelve con la tabla de puntos de operacion.
    """
    return max(config.MODELOS_CLASIFICACION, key=lambda n: comparacion[n]["global"]["AUC"])


# ---------------------------------------------------------------------------

def ejecutar_entrenamiento(): #Ejecuta el ciclo de vida completo de entrenamiento y congelamiento de los modelos.
    print(" [1/7] Cargando base de datos transaccional...")
    df_raw = data.cargar_datos(config.RUTA_DATA_RAW)

    # CONTRATO DE DATOS. Se valida ANTES de transformar nada: si el archivo no es
    # el que esperabamos, el pipeline se detiene aqui en vez de producir un
    # modelo entrenado sobre datos equivocados. Ver src/esquema.py.
    esquema.validar_csv_crudo(df_raw)
    print(f"   Contrato de entrada verificado: {len(df_raw)} filas")

    print(" [2/7] Limpieza, consolidacion por OT e historia por VIN...")
    df_visitas = esquema.validar_visitas(data.consolidar_visitas_por_ot(df_raw))
    df_historia = data.agregar_features_historia(df_visitas)
    # Cada dataset se valida contra su propio esquema: exige que existan todas
    # las variables declaradas en config y que el objetivo este en rango.
    df_regresion = esquema.validar_dataset_regresion(data.preparar_dataset_entrenamiento(df_historia))
    df_fuga = esquema.validar_dataset_fuga(data.preparar_dataset_fuga(df_historia))

    print(f"   Visitas reales (OT unicas tras limpieza): {len(df_visitas)}")
    print(f"   Dataset REGRESION: {len(df_regresion)} filas / {df_regresion[config.COLUMNA_GRUPO].nunique()} vehiculos")
    print(f"   Dataset FUGA     : {len(df_fuga)} filas / {df_fuga[config.COLUMNA_GRUPO].nunique()} vehiculos "
          f"| tasa de fuga: {100*df_fuga[config.TARGET_FUGA].mean():.1f}%")

    print("\n [3/7] Diagnostico descriptivo del ritmo de ingreso...")
    diag = diagnostico.diagnosticar_ritmos(df_historia)
    diagnostico.imprimir_diagnostico(diag)

    print("\n [4/7] Regresion: referencias sin modelo y validacion cruzada...")
    referencias = calcular_referencias(df_regresion)
    for nombre, m in referencias.items():
        print(f"   {nombre:<32} MAE={m['MAE_dias']:7.2f}  RMSE={m['RMSE_dias']:7.2f}  R2={m['R2_score']:+.4f}")
    cv_reg = validar_regresion(df_regresion)
    print(f"   {'RandomForest (regresion)':<32} MAE={cv_reg['media_MAE_dias']:7.2f}  "
          f"RMSE={cv_reg['media_RMSE_dias']:7.2f}  R2={cv_reg['media_R2_score']:+.4f}")

    print(f"\n [5/7] Clasificacion de fuga (no vuelve en {config.UMBRAL_FUGA_DIAS} dias): tres modelos...")
    comparacion = comparar_clasificadores(df_fuga)
    _imprimir_clasificacion(comparacion)

    ganador = elegir_mejor(comparacion)
    proba_ganador = comparacion[ganador].pop("_proba_global")
    for m in comparacion.values():
        m.pop("_proba_global", None)
    print(f"\n   Mejor modelo por AUC: {ganador} (AUC={comparacion[ganador]['global']['AUC']:.4f})")

    puntos = evaluate.curva_punto_operacion(df_fuga[config.TARGET_FUGA].astype(int), proba_ganador)
    print(f"\n   Punto de operacion ({ganador}) - el umbral es una decision de negocio:")
    print(f"   {'umbral':>7} {'precis':>8} {'recall':>8} {'alertas':>9} {'llamadas de mas':>17} {'clientes perdidos':>19}")
    for p in puntos:
        print(f"   {p['umbral']:7.2f} {p['precision']:8.3f} {p['recall']:8.3f} {p['alertas_emitidas']:9d} "
              f"{p['llamadas_innecesarias']:17d} {p['clientes_perdidos_sin_alerta']:19d}")

    print("\n [6/7] Entrenando modelos finales sobre vehiculos no vistos...")
    # --- Regresion: holdout agrupado por VIN ---
    X_reg, y_reg, grupos_reg = features.separar_x_y_grupo(df_regresion)
    idx_train, idx_test = next(GroupShuffleSplit(
        n_splits=1, test_size=config.TAMANO_TEST, random_state=config.SEMILLA_ALEATORIA
    ).split(X_reg, y_reg, grupos_reg))
    pipeline_regresion = modelos.crear_regresor().fit(X_reg.iloc[idx_train], y_reg.iloc[idx_train])
    metricas_reg = evaluate.evaluar_modelo(
        y_reg.iloc[idx_test], np.clip(pipeline_regresion.predict(X_reg.iloc[idx_test]), 0, None)
    )
    print(f"   REGRESION  MAE={metricas_reg['MAE_dias']:.2f} d  RMSE={metricas_reg['RMSE_dias']:.2f} d  "
          f"R2={metricas_reg['R2_score']:+.4f}")

    # --- Clasificacion: el ganador, entrenado sobre todo el dataset ---
    columnas_fuga = features.columnas_modelo(df_fuga)
    modelo_fuga = modelos.ajustar_clasificador(
        ganador, df_fuga[columnas_fuga], df_fuga[config.TARGET_FUGA].astype(int)
    )

    # --- Clasificadores por origen: el negocio quiere poder actuar distinto ---
    modelos_segmentados = {}
    if config.ENTRENAR_SEGMENTADO_POR_ORIGEN:
        for valor, nombre_seg in ((1, "vendidos"), (0, "externos")):
            sub = df_fuga[df_fuga["Es_Vehiculo_Vendido"] == valor]
            if len(sub) == 0:
                continue
            modelos_segmentados[nombre_seg] = modelos.ajustar_clasificador(
                ganador, sub[columnas_fuga], sub[config.TARGET_FUGA].astype(int)
            )
            print(f"   FUGA [{nombre_seg}] entrenado con {len(sub)} filas / "
                  f"{sub[config.COLUMNA_GRUPO].nunique()} vehiculos")

    importancias = _importancias(modelo_fuga)
    print("\n   Top 10 variables del clasificador de fuga:")
    for nombre, peso in importancias.head(10).items():
        print(f"      {nombre:<34} {peso:.4f}")

    print("\n [7/7] Serializando modelos, catalogo y metricas a disco...")
    config.RUTA_MODELO_SERIALIZADO.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline_regresion, config.RUTA_MODELO_SERIALIZADO, compress=3)
    joblib.dump({"modelo": modelo_fuga, "algoritmo": ganador}, config.RUTA_MODELO_FUGA, compress=3)
    if modelos_segmentados:
        joblib.dump({"modelos": modelos_segmentados, "algoritmo": ganador},
                    config.RUTA_MODELOS_FUGA_SEGMENTADOS, compress=3)

    # El catalogo se valida antes de congelarlo: tiene que traer exactamente las
    # columnas que el modelo consume, o el endpoint por VIN fallaria en
    # produccion con el CI en verde.
    catalogo = esquema.validar_catalogo(data.construir_catalogo_vehiculos(df_visitas))
    joblib.dump(catalogo, config.RUTA_CATALOGO_VEHICULOS, compress=3)
    print(f"   Catalogo de {len(catalogo)} vehiculos ({int(catalogo['Apto_Para_Modelo'].sum())} con historia suficiente)")

    # El JSON es la evidencia citable: la tesis no deberia depender de que
    # alguien haya copiado bien un numero de la consola.
    config.RUTA_METRICAS.write_text(json.dumps({
        "umbral_fuga_dias": config.UMBRAL_FUGA_DIAS,
        "visitas_consolidadas": int(len(df_visitas)),
        "dataset_regresion": {
            "filas": int(len(df_regresion)),
            "vehiculos": int(df_regresion[config.COLUMNA_GRUPO].nunique()),
        },
        "dataset_fuga": {
            "filas": int(len(df_fuga)),
            "vehiculos": int(df_fuga[config.COLUMNA_GRUPO].nunique()),
            "tasa_de_fuga": float(df_fuga[config.TARGET_FUGA].mean()),
        },
        "diagnostico_ritmos": diag,
        "regresion": {
            "referencias_sin_modelo": referencias,
            "validacion_cruzada_por_vin": cv_reg,
            "holdout_agrupado_por_vin": metricas_reg,
        },
        "clasificacion_fuga": {
            "modelo_elegido": ganador,
            "comparacion": comparacion,
            "punto_de_operacion": puntos,
            "importancias_top10": importancias.head(10).round(4).to_dict(),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   Metricas en: {config.RUTA_METRICAS}")


def _importancias(pipeline) -> pd.Series:
    """Importancia por variable, con los nombres ya expandidos del OneHot."""
    nombres = pipeline.named_steps["preprocesador"].get_feature_names_out()
    estimador = pipeline.named_steps["modelo"]
    if hasattr(estimador, "feature_importances_"):
        pesos = estimador.feature_importances_
    else:
        # HistGradientBoosting no expone feature_importances_.
        return pd.Series(dtype=float)
    return pd.Series(pesos, index=nombres).sort_values(ascending=False)


if __name__ == "__main__":
    ejecutar_entrenamiento()
