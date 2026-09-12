
# CAPA DE EVALUACION DEL MODELO
# Un modelo que entrena sin errores NO es un modelo bueno. Este modulo responde
# la pregunta "que tan equivocado esta" comparando las predicciones contra la
# realidad del conjunto de test (datos que el modelo nunca vio al entrenar).
# Se aisla en su propio archivo para poder cambiar de metricas sin tocar el
# entrenamiento, y para poder reutilizarlo en futuros modelos.
#este archivo lo que hace es calcular las metricas de evaluacion del modelo (MAE, RMSE, R2) y devolverlas en un diccionario para que main.py las pueda imprimir o guardar en un log.

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar_modelo(y_real, y_pred) -> dict[str, float]: # Calcula las metricas de regresion clave, expresadas en dias. y_real = los dias que el vehiculo REALMENTE tardo en volver.y_pred = los dias que el modelo ESTIMO.
    mae = mean_absolute_error(y_real, y_pred) # MAE (Mean Absolute Error): promedio de la diferencia absoluta entre lo predicho y lo real. Es la metrica mas facil de explicarle al taller: "en promedio nos equivocamos por X dias". No castiga errores grandes.
    rmse = np.sqrt(mean_squared_error(y_real, y_pred)) # RMSE (Root Mean Squared Error): eleva los errores al cuadrado antes de promediar y luego saca la raiz. Al elevar al cuadrado, PENALIZA mucho mas los errores grandes. Si RMSE es muy superior al MAE, significa que hay casos puntuales donde el modelo se equivoca de forma escandalosa.
    r2 = r2_score(y_real, y_pred) # R2 (coeficiente de determinacion): que porcentaje de la variabilidad logra explicar el modelo. 1.0 es perfecto, 0.0 equivale a predecir siempre el promedio, y un valor NEGATIVO significa que el modelo lo hace PEOR que simplemente decir el promedio de todos los dias.

    return { # Devolvemos un diccionario (y no prints) para que quien llame a esta funcion decida si lo imprime, lo guarda en un log o lo compara con otro modelo.
        "MAE_dias": float(mae),
        "RMSE_dias": float(rmse),
        "R2_score": float(r2)
    }
