# Actualiza los datos que consulta la API (catalogo por VIN e historial de
# visitas) a partir del CSV, SIN reentrenar los modelos.
#
# Los modelos aprenden la relacion "historia -> cuando vuelve / si se va", y esa
# relacion no cambia de un dia para otro. Lo que si cambia todos los dias son
# las visitas. Por eso se separan: cuando llega una base nueva del ERP basta con
# correr este script (segundos) y reiniciar la API; el reentrenamiento
# (python -m src.train) queda para cuando se quiera revisar el modelo.
#
# Se ejecuta con:  python -m src.actualizar_catalogo

import joblib

from src import config, data, esquema


def actualizar_catalogo() -> None:
    print(" [1/3] Cargando y validando la base...")
    df_raw = data.cargar_datos(config.RUTA_DATA_RAW)
    esquema.validar_csv_crudo(df_raw)

    print(" [2/3] Consolidando visitas por OT...")
    visitas = esquema.validar_visitas(data.consolidar_visitas_por_ot(df_raw))

    print(" [3/3] Guardando catalogo e historial...")
    catalogo = esquema.validar_catalogo(data.construir_catalogo_vehiculos(visitas))
    joblib.dump(catalogo, config.RUTA_CATALOGO_VEHICULOS, compress=3)
    joblib.dump(data.historial_para_servicio(visitas), config.RUTA_HISTORIAL_VISITAS, compress=3)

    print(f"   Catalogo de {len(catalogo)} vehiculos ({int(catalogo['Apto_Para_Modelo'].sum())} con historia suficiente)")
    print(f"   Historial de {len(visitas)} visitas. Ultimo ingreso en la base: "
          f"{visitas[config.COLUMNA_FECHA_ENTRADA].max().date()}")
    print("   Reinicie la API para que tome los datos nuevos.")


if __name__ == "__main__":
    actualizar_catalogo()
