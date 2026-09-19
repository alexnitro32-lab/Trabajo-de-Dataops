# CATALOGO DE ESTIMADORES A COMPARAR
#
# El negocio pidio contrastar el RandomForest contra otros dos modelos, y medir
# con AUC y falsos positivos cual conviene. Este modulo existe para que agregar
# o quitar un candidato sea cambiar una entrada de un diccionario, sin tocar
# train.py ni la logica de evaluacion.
#
# Los tres son de ensamble, pero de familias distintas a proposito: comparar dos
# variantes de lo mismo no informa nada.
#
#   RandomForest         bagging con cortes optimos: cada arbol ve una muestra
#                        distinta y elige el mejor corte. Robusto, poco sensible
#                        a los hiperparametros.
#   HistGradientBoosting boosting secuencial: cada arbol corrige el error del
#                        anterior. Suele ganar cuando hay senial fina que extraer.
#   ExtraTrees           bagging con cortes AL AZAR: mas varianza por arbol, menos
#                        correlacion entre arboles. Contrasta con RandomForest en
#                        exactamente una decision de diseno.
#
# Por decision del negocio se excluyen los modelos lineales.

from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src import config, features

# Solo el 26,5% de las visitas son fuga. Sin corregir ese desbalance, un modelo
# acierta el 73,5% sin detectar ni una sola fuga: "nadie se va" es una respuesta
# comoda y completamente inutil. class_weight="balanced" hace que equivocarse en
# la clase minoritaria cueste mas.
_CLASIFICADORES = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=5, class_weight="balanced",
        random_state=config.SEMILLA_ALEATORIA, n_jobs=-1,
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, min_samples_leaf=20,
        random_state=config.SEMILLA_ALEATORIA,
    ),
    "ExtraTrees": lambda: ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=5, class_weight="balanced",
        random_state=config.SEMILLA_ALEATORIA, n_jobs=-1,
    ),
}

# HistGradientBoosting no acepta class_weight, asi que el balanceo hay que
# pasarlo como pesos por fila en el .fit(). Sin esto la comparacion seria
# tramposa: se estaria midiendo el desbalance, no el algoritmo.
_NECESITA_SAMPLE_WEIGHT = ("HistGradientBoosting",)


def crear_clasificador(nombre: str) -> Pipeline:
    """Pipeline completo (preprocesador + clasificador) listo para .fit()."""
    if nombre not in _CLASIFICADORES:
        raise ValueError(f"Modelo desconocido: {nombre}. Disponibles: {list(_CLASIFICADORES)}")
    return Pipeline(steps=[
        ("preprocesador", features.crear_preprocesador()),
        ("modelo", _CLASIFICADORES[nombre]()),
    ])


def ajustar_clasificador(nombre: str, X, y) -> Pipeline:
    """Entrena aplicando el balanceo de clases que cada algoritmo necesite."""
    pipeline = crear_clasificador(nombre)
    if nombre in _NECESITA_SAMPLE_WEIGHT:
        pipeline.fit(X, y, modelo__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipeline.fit(X, y)
    return pipeline


def crear_regresor() -> Pipeline:
    """Pipeline de regresion: responde "en cuantos dias vuelve".

    min_samples_leaf=5 es deliberado: la variable objetivo tiene una cola muy
    larga (hay retornos de mas de 2.900 dias) y sin un minimo de muestras por
    hoja el bosque memoriza esos casos aislados en vez de aprender el patron.
    """
    return Pipeline(steps=[
        ("preprocesador", features.crear_preprocesador()),
        ("modelo", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=5,
            random_state=config.SEMILLA_ALEATORIA, n_jobs=-1,
        )),
    ])
