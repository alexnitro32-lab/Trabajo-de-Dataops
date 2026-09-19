
# CAPA DE EVALUACION DEL MODELO
# Un modelo que entrena sin errores NO es un modelo bueno. Este modulo responde
# la pregunta "que tan equivocado esta" comparando las predicciones contra la
# realidad del conjunto de test (datos que el modelo nunca vio al entrenar).
# Se aisla en su propio archivo para poder cambiar de metricas sin tocar el
# entrenamiento, y para poder reutilizarlo en futuros modelos.
#este archivo lo que hace es calcular las metricas de evaluacion del modelo (MAE, RMSE, R2) y devolverlas en un diccionario para que main.py las pueda imprimir o guardar en un log.

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

from src import config

def evaluar_modelo(y_real, y_pred) -> dict[str, float]: # Calcula las metricas de regresion clave, expresadas en dias. y_real = los dias que el vehiculo REALMENTE tardo en volver.y_pred = los dias que el modelo ESTIMO.
    mae = mean_absolute_error(y_real, y_pred) # MAE (Mean Absolute Error): promedio de la diferencia absoluta entre lo predicho y lo real. Es la metrica mas facil de explicarle al taller: "en promedio nos equivocamos por X dias". No castiga errores grandes.
    rmse = np.sqrt(mean_squared_error(y_real, y_pred)) # RMSE (Root Mean Squared Error): eleva los errores al cuadrado antes de promediar y luego saca la raiz. Al elevar al cuadrado, PENALIZA mucho mas los errores grandes. Si RMSE es muy superior al MAE, significa que hay casos puntuales donde el modelo se equivoca de forma escandalosa.
    r2 = r2_score(y_real, y_pred) # R2 (coeficiente de determinacion): que porcentaje de la variabilidad logra explicar el modelo. 1.0 es perfecto, 0.0 equivale a predecir siempre el promedio, y un valor NEGATIVO significa que el modelo lo hace PEOR que simplemente decir el promedio de todos los dias.

    return { # Devolvemos un diccionario (y no prints) para que quien llame a esta funcion decida si lo imprime, lo guarda en un log o lo compara con otro modelo.
        "MAE_dias": float(mae),
        "RMSE_dias": float(rmse),
        "R2_score": float(r2)
    }


def evaluar_clasificacion(y_real, y_proba, umbral: float = None) -> dict[str, float]:
    """Metricas del clasificador de fuga, con los dos errores separados.

    La clase POSITIVA es la FUGA: el vehiculo no volvio dentro del plazo. Se
    define asi, y no al reves, porque es el evento sobre el que se quiere actuar,
    y porque deja que "recall" signifique lo que el negocio quiere maximizar:
    de todos los clientes que se fueron, a cuantos alcanzamos a detectar.

    Los dos errores NO cuestan lo mismo, por eso se reportan por separado:

      FALSO POSITIVO  alertamos fuga y el cliente si volvio.
                      Costo: una llamada innecesaria. Barato.
      FALSO NEGATIVO  no alertamos y el cliente se fue.
                      Costo: un cliente perdido sin que nadie lo intentara. Caro.

    El AUC se calcula sobre la PROBABILIDAD y no sobre la decision, asi que mide
    la capacidad de ordenar clientes por riesgo sin depender del umbral elegido.
    Es la metrica correcta para comparar modelos entre si; la matriz de confusion
    es la correcta para decidir el punto de operacion.
    """
    umbral = umbral if umbral is not None else config.UMBRAL_DECISION_FUGA
    y_real = np.asarray(y_real).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()

    return {
        # AUC: probabilidad de que un cliente fugado reciba mayor riesgo que uno
        # retenido, tomados al azar. 0,5 es azar puro; 1,0 es separacion perfecta.
        "AUC": float(roc_auc_score(y_real, y_proba)) if len(set(y_real.tolist())) > 1 else float("nan"),
        # PR-AUC: mas informativa que el AUC cuando la clase positiva es minoritaria,
        # porque no se deja inflar por los verdaderos negativos, que aqui abundan.
        "PR_AUC": float(average_precision_score(y_real, y_proba)),
        "precision": float(precision_score(y_real, y_pred, zero_division=0)),
        "recall": float(recall_score(y_real, y_pred, zero_division=0)),
        "F1": float(f1_score(y_real, y_pred, zero_division=0)),
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
        # Tasa de falsa alarma: de los clientes que SI volvieron, a cuantos
        # molestamos sin necesidad.
        "tasa_falsos_positivos": float(fp / (tn + fp)) if (tn + fp) else 0.0,
        # De los clientes que se fueron, a cuantos dejamos ir sin detectarlos.
        "tasa_falsos_negativos": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "umbral_decision": float(umbral),
    }


def curva_punto_operacion(y_real, y_proba, umbrales=(0.30, 0.40, 0.50, 0.60, 0.70)) -> list[dict]:
    """Como cambia el negocio al mover el umbral de alerta.

    Elegir 0,50 no es neutral: es una decision comercial disfrazada de valor por
    defecto. Esta tabla la pone a la vista, para que el taller decida cuantas
    llamadas de mas esta dispuesto a hacer con tal de no perder un cliente.
    """
    filas = []
    for umbral in umbrales:
        m = evaluar_clasificacion(y_real, y_proba, umbral)
        filas.append({
            "umbral": umbral,
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "F1": round(m["F1"], 3),
            "alertas_emitidas": m["verdaderos_positivos"] + m["falsos_positivos"],
            "llamadas_innecesarias": m["falsos_positivos"],
            "clientes_perdidos_sin_alerta": m["falsos_negativos"],
        })
    return filas
