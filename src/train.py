# Este es el DIRECTOR DE ORQUESTA del proyecto. No define logica nueva: llama en orden a los modulos anteriores (data -> features -> modelo -> evaluate) y termina congelando el resultado en disco con joblib.
import joblib                                            
from sklearn.ensemble import RandomForestRegressor       
from sklearn.model_selection import train_test_split     
from sklearn.pipeline import Pipeline                    
from src import config, data, features, evaluate         

def ejecutar_entrenamiento(): #Ejecuta el ciclo de vida completo de entrenamiento y congelamiento del modelo.
    print(" [1/5] Cargando base de datos transaccional...") # EXTRACCION: leer el CSV crudo del taller
    df_raw = data.cargar_datos(config.RUTA_DATA_RAW)

    print(" [2/5] Consolidando sub-trabajos por OT y aplicando DataOps...") # [2/5] TRANSFORMACION: consolidar visitas y construir la variable objetivo
    df_visitas = data.consolidar_visitas_por_ot(df_raw)
    df_limpio = data.generar_target_y_features(df_visitas)

    print(f"   Registros validos para entrenamiento con retorno: {len(df_limpio)}") # Imprimir cuantas filas sobrevivieron es un control de calidad: si este numero cae a cero o baja de golpe, algo se rompio en la limpieza.
    print(" [3/5] Separando X e Y y realizando la division Train/Test...") # [3/5] SEPARACION: pistas (X) contra respuestas (y), y luego train/test
    X, y = features.separar_x_y(df_limpio)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TAMANO_TEST,
        random_state=config.SEMILLA_ALEATORIA
    )

    print(" [4/5] Ensamblando Pipeline completo (Preprocesador + Estimador)...") # [4/5] ENSAMBLE Y ENTRENAMIENTO 
    preprocesador = features.crear_preprocesador()

    estimador = RandomForestRegressor(
        n_estimators=100,                        # Cantidad de arboles del bosque
        random_state=config.SEMILLA_ALEATORIA,   # Reproducibilidad
        n_jobs=-1                                # Usa todos los nucleos del PC para ir mas rapido
    )

    pipeline_completo = Pipeline(steps=[ # encapsulamos preprocesamiento y mode lo en UN SOLO objeto. Asi el .joblib que se guarda mas abajo ya sabe escalar, imputar y codificar por si mismo, y la API no tiene que repetir ni una linea de esa logica (lo que evita el clasico training/serving skew).
        ("preprocesador", preprocesador),
        ("modelo", estimador)
    ])

    print(" Ajustando el Pipeline sobre los datos de entrenamiento (fit)...") # .fit() es donde el modelo realmente APRENDE: calcula medias, desviaciones, categorias y construye los arboles. Solo se hace con datos de entrenamiento.
    pipeline_completo.fit(X_train, y_train)

    print(" Evaluando el modelo en el conjunto de Test (datos no vistos)...") # .predict() sobre el test: el modelo estima sin haber visto estas filas.
    y_pred = pipeline_completo.predict(X_test)
    metricas = evaluate.evaluar_modelo(y_test, y_pred)

    # Reporte legible en consola para dejar constancia del desempeno.
    print("\n --- RESULTADOS DE EVALUACIÓN ---")
    print(f"   MAE (Desfase Promedio): {metricas['MAE_dias']:.2f} días")
    print(f"   RMSE: {metricas['RMSE_dias']:.2f} días")
    print(f"   R² Score: {metricas['R2_score']:.4f}")
    print("-----------------------------------\n")

    # --- [5/5] SERIALIZACION: congelar el modelo entrenado en disco ---
    print(" [5/5] Serializando Pipeline completo a disco...")
    config.RUTA_MODELO_SERIALIZADO.parent.mkdir(parents=True, exist_ok=True) # mkdir con exist_ok=True crea la carpeta models/ si no existe y no falla sin ya existe. Necesario para que el CI de GitHub funcione en una maquina limpia.

    joblib.dump(pipeline_completo, config.RUTA_MODELO_SERIALIZADO, compress=3) # joblib.dump escribe el objeto entrenado a un archivo binario .joblib. Desde este momento el modelo deja de vivir solo en memoria RAM y puede cargarse instantaneamente sin volver a entrenar. compress=3 reduce el archivo de 95 MB a 19 MB sin perder nada de precision (solo comprime los bytes, el modelo predice exactamente igual). Es necesario 
    print(f" ¡Éxito! Modelo congelado y guardado en: {config.RUTA_MODELO_SERIALIZADO}")

    catalogo = data.construir_catalogo_vehiculos(df_visitas) # Ademas del modelo congelamos el catalogo de vehiculos que usara la API para buscar por VIN. Se genera AQUI y no al arrancar la API para que el servicio no tenga que reprocesar 39.000 filas del CSV en cada arranque: es el mismo catalogo que se uso para entrenar el modelo, y no cambia hasta que se vuelva a entrenar.
    joblib.dump(catalogo, config.RUTA_CATALOGO_VEHICULOS, compress=3)
    print(f" Catálogo de {len(catalogo)} vehículos guardado en: {config.RUTA_CATALOGO_VEHICULOS}")

if __name__ == "__main__":
    ejecutar_entrenamiento()
