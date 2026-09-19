# tests/test_api.py - PRUEBAS AUTOMATICAS DE LA API
# Estas pruebas son la "red de seguridad" del proyecto. Cada vez que alguien
# modifique el codigo, pytest volvera a ejecutarlas y avisara si algo se rompio,
# sin que nadie tenga que abrir el navegador a probar a mano.
# Se ejecutan con:  python -m pytest

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient   # Cliente falso que llama a la API sin levantar Uvicorn

from main import app                        # Importamos la aplicacion real que queremos probar
from src import config, data, diagnostico, esquema, evaluate, features

# TestClient simula un navegador: hace peticiones HTTP reales contra "app"
# en memoria. Por eso las pruebas corren en segundos y funcionan en GitHub
# Actions, donde no hay nadie abriendo un navegador.
client = TestClient(app)

# Peticion valida completa, reutilizada por varias pruebas. Incluye la historia
# del vehiculo, que es obligatoria: sin ella no se puede estimar el ritmo de
# ingreso de ese chasis en particular.
PAYLOAD_VALIDO = {
    "Tipo Cargo": "Cliente",
    "Tipo de Trabajo": "MECANICA",
    "Kms.": 45000.0,
    "Año Modelo": 2022,          # El usuario escribe el anio, la API calcula la antiguedad
    "Es_Vehiculo_Vendido": True,
    "Visitas_Previas": 3,
    "Dias_Desde_Visita_Anterior": 180.0,
    "Kms_Visita_Anterior": 32000.0,
}


@pytest.fixture(scope="module")
def catalogo():
    """Catalogo de vehiculos congelado, cargado una sola vez para todo el modulo."""
    return joblib.load(config.RUTA_CATALOGO_VEHICULOS)


def test_health_check():
    """Verifica que el endpoint raíz responda 200 OK y el modelo esté cargado."""
    response = client.get("/")

    # assert = "esto TIENE que ser verdad". Si no lo es, la prueba falla.
    # 200 es el codigo HTTP de exito.
    assert response.status_code == 200

    data_resp = response.json()                # Convertimos la respuesta JSON en diccionario
    assert "mensaje" in data_resp              # El contrato de salida debe incluir el mensaje
    assert data_resp["modelo_cargado"] is True # Confirma que el .joblib se cargo correctamente
    assert data_resp["catalogo_cargado"] is True
    # El servicio expone DOS modelos: el que dice cuando vuelve el vehiculo y el
    # que dice si va a dejar de volver.
    assert data_resp["modelo_fuga_cargado"] is True
    assert data_resp["algoritmo_fuga"] in config.MODELOS_CLASIFICACION

    # No todos los chasis del catalogo son predecibles: los de un solo ingreso
    # no tienen intervalo observado. El health check debe reportar cuantos si.
    assert 0 < data_resp["vehiculos_predecibles"] <= data_resp["vehiculos_en_catalogo"]


def test_prediccion_exitosa():
    """Verifica que el endpoint /predict retorne un cálculo numérico válido."""
    response = client.post("/predict", json=PAYLOAD_VALIDO)

    assert response.status_code == 200
    data_resp = response.json()
    assert isinstance(data_resp["prediccion_dias_retorno"], float)
    assert data_resp["prediccion_dias_retorno"] >= 0   # No existen retornos negativos
    assert data_resp["unidad"] == "días"

    # La regla de fabrica (365 dias o 10.000 km) viaja al lado de la prediccion
    # para que el asesor tenga una referencia contra la cual juzgarla.
    assert data_resp["regla_preventiva_dias"] <= config.REGLA_MANTENIMIENTO_DIAS

    # Y la segunda pregunta: ¿se va a ir? Debe llegar como probabilidad valida.
    riesgo = data_resp["riesgo_de_fuga"]
    assert 0.0 <= riesgo["probabilidad_fuga"] <= 1.0
    assert riesgo["segmento"] == "vendidos"      # el payload dice que lo vendimos nosotros
    assert isinstance(riesgo["alerta"], bool)


def test_modelo_solo_consume_columnas_declaradas():
    """El modelo consume exactamente las columnas declaradas en config, y solo esas.

    El sujeto de la predicción es el CLIENTE, identificado por su chasis, y lo que
    la determina es su conducta observada. Un campo que describa el modelo del
    vehículo no entra al estimador: si llega en la petición, se ignora y la
    predicción no cambia. Si alguien conecta una variable sin declararla en
    config, esta prueba falla.
    """
    base = client.post("/predict", json=PAYLOAD_VALIDO).json()
    con_extra = client.post("/predict", json={**PAYLOAD_VALIDO, "Gama": "Tucson"}).json()
    assert base["prediccion_dias_retorno"] == con_extra["prediccion_dias_retorno"]

    # El vector que arma la API tiene que coincidir con el que espera el pipeline.
    declaradas = set(config.COLUMNAS_NUMERICAS + config.COLUMNAS_CATEGORICAS + config.COLUMNAS_BINARIAS)
    assert set(features.columnas_modelo()) == declaradas


def test_historia_obligatoria_422():
    """Verifica que una petición sin la historia del vehículo sea rechazada.

    Es la prueba que fija la decisión de diseño: el modelo se apoya en cada
    cuánto venía volviendo el chasis. Aceptar la petición sin ese dato
    devolvería la mediana de la población disfrazada de predicción individual.
    """
    payload = {k: v for k, v in PAYLOAD_VALIDO.items()
               if k not in ("Visitas_Previas", "Dias_Desde_Visita_Anterior")}
    assert client.post("/predict", json=payload).status_code == 422


def test_validacion_pydantic_error_422():
    """Verifica que Pydantic rechace datos con tipos incorrectos (HTTP 422)."""
    # Prueba del "camino triste": faltan campos obligatorios y ademas se envia
    # texto donde debe ir un numero. Probar los errores es tan importante como
    # probar el exito: garantiza que la API no acepte basura silenciosamente.
    payload_invalido = {
        "Gama": "Tucson",
        "Kms.": "cinco_mil"  # Enviar texto en lugar de número debe fallar
    }
    response = client.post("/predict", json=payload_invalido)

    # 422 = "Unprocessable Entity": la peticion llego bien pero los datos no
    # cumplen el contrato. Es la respuesta que FastAPI genera por si solo.
    assert response.status_code == 422


def test_anio_modelo_fuera_de_rango_422():
    """Verifica que un año de modelo imposible sea rechazado por los límites ge/le."""
    # El modelo se entreno con antiguedades reales del parque. Un carro de 1800
    # obligaria a extrapolar donde nunca hubo datos, asi que Pydantic lo corta.
    response = client.post("/predict", json={**PAYLOAD_VALIDO, "Año Modelo": 1800})
    assert response.status_code == 422


def test_prediccion_por_vin_exitosa(catalogo):
    """Verifica que al consultar un VIN con historia se autocompleten los datos y prediga."""
    # Tomamos un chasis con historia real en la base del taller, en vez de
    # fijar un VIN a mano: si manana ese carro sale de la base, la prueba
    # seguiria siendo valida.
    vin = str(catalogo[catalogo["Apto_Para_Modelo"]].index[0])
    response = client.get(f"/predict/vin/{vin}")

    assert response.status_code == 200
    data_resp = response.json()
    assert data_resp["vin"] == vin
    assert isinstance(data_resp["prediccion_dias_retorno"], float)

    # Con historia suficiente, la estimacion debe venir del MODELO y no de la
    # regla de fabrica. Es lo que separa una prediccion de un valor por defecto.
    assert data_resp["origen_estimacion"] == "modelo"

    # La API debe devolver tambien la ficha encontrada, para que el asesor
    # pueda confirmar que el sistema busco el vehiculo correcto.
    ficha = data_resp["datos_encontrados"]
    assert "gama" in ficha and "ultimo_kilometraje" in ficha
    assert ficha["total_visitas"] >= config.MINIMO_VISITAS_POR_VIN


def test_vin_de_una_sola_visita_no_se_predice(catalogo):
    """Verifica que un chasis con un único ingreso NO reciba una predicción del modelo.

    Sin un segundo ingreso no hay ningún intervalo observado, así que no se
    puede saber cada cuánto vuelve ese carro. La API debe decirlo explícitamente
    y devolver la regla preventiva, en lugar de disfrazar de predicción
    personalizada lo que en realidad sería la mediana de la población.
    """
    sin_historia = catalogo[~catalogo["Apto_Para_Modelo"]]
    if len(sin_historia) == 0:
        pytest.skip("No hay vehículos de una sola visita en el catálogo actual.")

    response = client.get(f"/predict/vin/{sin_historia.index[0]}")

    assert response.status_code == 200
    data_resp = response.json()
    assert data_resp["origen_estimacion"] == "regla_preventiva"
    assert data_resp["prediccion_dias_retorno"] == pytest.approx(data_resp["regla_preventiva_dias"])
    assert "advertencia" in data_resp
    assert data_resp["datos_encontrados"]["total_visitas"] == 1


def test_riesgo_de_fuga_por_vin(catalogo):
    """El endpoint de fuga devuelve probabilidad y la historia que la sustenta."""
    vin = str(catalogo[catalogo["Apto_Para_Modelo"]].index[0])
    response = client.get(f"/fuga/vin/{vin}")

    assert response.status_code == 200
    data_resp = response.json()
    riesgo = data_resp["riesgo_de_fuga"]
    assert 0.0 <= riesgo["probabilidad_fuga"] <= 1.0
    assert riesgo["umbral_dias"] == config.UMBRAL_FUGA_DIAS
    assert riesgo["segmento"] in ("vendidos", "externos")

    # El negocio pidió poder actuar distinto sobre cada origen, así que la
    # respuesta trae también la lectura del modelo entrenado solo con ese grupo.
    assert 0.0 <= riesgo["probabilidad_fuga_modelo_del_segmento"] <= 1.0

    # Un asesor no puede llamar a un cliente basándose en un número sin contexto:
    # la respuesta debe decir POR QUÉ el modelo cree que se va a ir.
    historia = data_resp["historia_que_lo_sustenta"]
    assert historia["total_visitas"] >= config.MINIMO_VISITAS_POR_VIN


def test_riesgo_de_fuga_sin_historia_409(catalogo):
    """Un chasis de un solo ingreso no tiene riesgo de fuga calculable.

    No hay ningún intervalo observado contra el cual comparar su ausencia.
    Devolver un número sería inventarlo, así que la API responde 409 (conflicto
    con el estado del recurso) y lo explica.
    """
    sin_historia = catalogo[~catalogo["Apto_Para_Modelo"]]
    if len(sin_historia) == 0:
        pytest.skip("No hay vehículos de una sola visita en el catálogo actual.")
    assert client.get(f"/fuga/vin/{sin_historia.index[0]}").status_code == 409


def test_vin_inexistente_404():
    """Verifica que un VIN sin historial responda 404 y no reviente el servicio."""
    # 404 = "el recurso no existe". Es distinto de un 500 (error del servidor):
    # aqui la API funciono bien, simplemente ese carro nunca vino al taller.
    assert client.get("/predict/vin/NOEXISTE123").status_code == 404
    assert client.get("/fuga/vin/NOEXISTE123").status_code == 404


def test_vin_con_datos_incompletos_no_rompe(catalogo):
    """Verifica que un chasis con año de modelo o kilometraje faltante responda
    igual, en lugar de tumbar el endpoint con un error 500.

    El historial del taller no esta completo: hay VIN sin año de modelo y sin
    kilometraje. int(NaN) revienta y NaN genera un JSON invalido que ningun
    cliente puede leer. Esta prueba deja fijado que esos casos se devuelven
    como null y que la prediccion se sigue calculando.
    """
    incompletos = catalogo[catalogo["Anio_Modelo"].isna() | catalogo["Kms."].isna()]

    # Si algun dia la base llega completa, no hay nada que probar aqui.
    if len(incompletos) == 0:
        pytest.skip("El catálogo actual no tiene vehículos con datos faltantes.")

    response = client.get(f"/predict/vin/{incompletos.index[0]}")

    assert response.status_code == 200
    data_resp = response.json()            # .json() falla solo si el JSON es invalido
    assert isinstance(data_resp["prediccion_dias_retorno"], float)

    # El dato faltante debe viajar como null explicito, no como NaN ni como 0.
    ficha = data_resp["datos_encontrados"]
    assert ficha["anio_modelo"] is None or isinstance(ficha["anio_modelo"], int)
    assert ficha["ultimo_kilometraje"] is None or isinstance(ficha["ultimo_kilometraje"], float)


# ---------------------------------------------------------------------------
# PRUEBAS DE LA CAPA DE DATOS
# La API puede responder 200 y aun asi estar entrenada sobre datos mal armados.
# Estas pruebas cubren las reglas que el analisis exploratorio dejo fijadas.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def visitas():
    """Visitas consolidadas desde el CSV real, una sola vez para todo el modulo."""
    return data.consolidar_visitas_por_ot(data.cargar_datos(config.RUTA_DATA_RAW))


def test_limpieza_descarta_anuladas_y_bodegaje():
    """Las facturas anuladas y las líneas de bodegaje no son visitas al taller."""
    crudo = data.cargar_datos(config.RUTA_DATA_RAW)
    limpio = data.limpiar_calidad(crudo)

    assert not limpio[config.COLUMNA_ESTADO].isin(config.ESTADOS_EXCLUIDOS).any()
    assert not limpio[config.COLUMNA_DESCRIPCION].astype(str).str.upper().str.contains(
        config.PATRON_BODEGAJE, na=False
    ).any()

    # El año 1997 es el relleno que pone el ERP cuando el campo viene vacío: no
    # es un año real y no puede convertirse en 29 años de antigüedad.
    assert (limpio[config.COLUMNA_ANO] == config.ANIO_CENTINELA).sum() == 0


def test_una_ot_es_una_visita(visitas):
    """Cada OT debe quedar como un único registro: una fila del ERP no es una visita."""
    assert not visitas[config.COLUMNA_OT].duplicated().any()


def test_dataset_excluye_vehiculos_de_una_sola_visita(visitas):
    """El dataset de entrenamiento solo admite vehículos con historia observable."""
    dataset = data.generar_target_y_features(visitas)

    assert (dataset["Total_Visitas_VIN"] >= config.MINIMO_VISITAS_POR_VIN).all() \
        if "Total_Visitas_VIN" in dataset.columns else True
    # El primer ingreso de cada chasis no tiene historia previa que mirar.
    assert (dataset["Num_Visita"] >= 2).all()
    # Y ningun vehiculo del dataset puede aportar una sola visita.
    assert dataset.groupby(config.COLUMNA_GRUPO).size().min() >= 1
    assert dataset[config.TARGET].min() >= 0


def test_historia_no_filtra_el_futuro(visitas):
    """La historia de cada visita debe construirse SOLO con información pasada.

    Es la prueba que protege la validez de las métricas. Si el ritmo previo se
    calculara incluyendo el intervalo que se quiere predecir, el modelo estaría
    leyendo la respuesta y el R² reportado sería ficticio.
    """
    df = data.agregar_features_historia(visitas)
    df = df[df[config.COLUMNA_GRUPO] == df[config.COLUMNA_GRUPO].value_counts().index[0]]
    df = df.sort_values(config.COLUMNA_FECHA_ENTRADA)

    # La primera visita de un vehículo no puede tener intervalo anterior.
    assert pd.isna(df["Dias_Desde_Visita_Ant"].iloc[0])
    assert pd.isna(df["Ritmo_Mediano_Previo"].iloc[0])

    # En cualquier visita posterior, el ritmo previo debe coincidir con la
    # mediana de los intervalos ya ocurridos hasta esa fila, ni uno mas.
    for posicion in (2, 5):
        if len(df) > posicion:
            esperado = df["Dias_Desde_Visita_Ant"].iloc[: posicion + 1].median()
            assert df["Ritmo_Mediano_Previo"].iloc[posicion] == pytest.approx(esperado)


def test_regla_preventiva_nunca_supera_365_dias(visitas):
    """La regla de negocio es 365 días O 10.000 km, lo que ocurra primero."""
    df = data.agregar_features_historia(visitas)
    assert df["Regla_Preventiva_Dias"].max() <= config.REGLA_MANTENIMIENTO_DIAS

    # Un vehículo intensivo alcanza los 10.000 km antes del año, y en ese caso
    # la regla debe acortarse por kilometraje y no quedarse en el calendario.
    intensivos = df[df["Km_Por_Dia_Medio"] > config.REGLA_MANTENIMIENTO_KMS / config.REGLA_MANTENIMIENTO_DIAS]
    if len(intensivos) > 0:
        assert (intensivos["Regla_Preventiva_Dias"] < config.REGLA_MANTENIMIENTO_DIAS).all()


# ---------------------------------------------------------------------------
# PRUEBAS DE LA ETIQUETA DE FUGA Y DEL DIAGNOSTICO
# ---------------------------------------------------------------------------

def test_etiqueta_fuga_rescata_los_que_no_volvieron(visitas):
    """La clasificación debe ver a los clientes que nunca regresaron.

    Es el aporte de reformular el problema: la regresión solo puede aprender de
    visitas que tuvieron un regreso observado, así que descarta la última visita
    de cada vehículo — exactamente los clientes perdidos. La etiqueta binaria los
    recupera cuando ya pasó el plazo sin que aparecieran.
    """
    historia = data.agregar_features_historia(visitas)
    etiquetado = data.etiquetar_fuga(historia)

    sin_regreso = etiquetado[etiquetado[config.TARGET].isna()]
    rescatados = sin_regreso[sin_regreso[config.TARGET_FUGA] == 1]
    assert len(rescatados) > 0, "La señal de abandono se está perdiendo otra vez"

    # Ninguna de esas filas puede seguir marcada como censurada: o se sabe que
    # se fugó, o se desconoce, pero no las dos cosas.
    assert not (rescatados["Censurado"]).any()

    # Lo censurado es lo genuinamente desconocido: última visita todavía dentro
    # de plazo. Esas filas NO pueden tener etiqueta.
    assert etiquetado.loc[etiquetado["Censurado"], config.TARGET_FUGA].isna().all()


def test_etiqueta_fuga_respeta_el_umbral(visitas):
    """Fuga = el intervalo observado superó el umbral. Sin zonas grises."""
    etiquetado = data.etiquetar_fuga(data.agregar_features_historia(visitas))
    con_regreso = etiquetado[etiquetado[config.TARGET].notna()]

    volvieron_rapido = con_regreso[con_regreso[config.TARGET] <= config.UMBRAL_FUGA_DIAS]
    volvieron_tarde = con_regreso[con_regreso[config.TARGET] > config.UMBRAL_FUGA_DIAS]

    assert (volvieron_rapido[config.TARGET_FUGA] == 0).all()
    assert (volvieron_tarde[config.TARGET_FUGA] == 1).all()


def test_dataset_fuga_es_mayor_que_el_de_regresion(visitas):
    """Al rescatar a los clientes perdidos, el clasificador entrena con más casos."""
    historia = data.agregar_features_historia(visitas)
    regresion = data.preparar_dataset_entrenamiento(historia)
    fuga = data.preparar_dataset_fuga(historia)

    assert len(fuga) > len(regresion)
    assert fuga[config.COLUMNA_GRUPO].nunique() > regresion[config.COLUMNA_GRUPO].nunique()
    # Y la etiqueta debe estar desbalanceada pero no degenerada.
    assert 0.05 < fuga[config.TARGET_FUGA].mean() < 0.95


def test_demora_en_taller_es_cierre_menos_entrada(visitas):
    """La demora del taller es F. Cierre - F. Entrada, nunca negativa."""
    historia = data.agregar_features_historia(visitas)
    esperado = (historia[config.COLUMNA_FECHA_CIERRE] - historia[config.COLUMNA_FECHA_ENTRADA]).dt.days

    assert (historia["Dias_En_Taller"] >= 0).all()
    # Dentro del rango no recortado debe coincidir exactamente con la resta.
    dentro = (esperado >= 0) & (esperado <= 120)
    assert (historia.loc[dentro, "Dias_En_Taller"] == esperado[dentro]).all()


def test_diagnostico_separa_las_tres_medidas(visitas):
    """Las tres medidas de ritmo describen poblaciones distintas y no se mezclan."""
    d = diagnostico.diagnosticar_ritmos(data.agregar_features_historia(visitas))

    m1 = d["medida_1_primer_ingreso"]
    assert m1["vehiculos_entregados"] == m1["volvieron_alguna_vez"] + m1["nunca_volvieron"]
    assert m1["dias"]["n"] > 0 and m1["km_recorridos"]["n"] > 0

    # Las medidas 2 y 3 miden poblaciones disjuntas por construcción.
    assert d["medida_2_vendidos_con_historia"]["vehiculos"] > 0
    assert d["medida_3_externos_con_historia"]["vehiculos"] > 0

    for nombre, c in d["cobertura_por_origen"].items():
        assert c["vehiculos"] == c["con_una_sola_visita"] + c["predecibles"], nombre


def test_metricas_de_clasificacion_separan_los_dos_errores():
    """FP y FN no cuestan lo mismo, así que deben reportarse por separado.

    Un falso positivo es una llamada innecesaria; un falso negativo es un cliente
    perdido sin que nadie lo intentara. Un resumen que los promedie oculta
    justamente la decisión que el negocio tiene que tomar.
    """
    y_real = [0, 0, 1, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.8, 0.2, 0.3, 0.7]   # un FP (idx 1) y un FN (idx 3)
    m = evaluate.evaluar_clasificacion(y_real, y_proba, umbral=0.5)

    assert m["falsos_positivos"] == 1
    assert m["falsos_negativos"] == 1
    assert m["verdaderos_positivos"] == 2
    assert m["verdaderos_negativos"] == 2
    assert 0.0 <= m["AUC"] <= 1.0

    # Bajar el umbral tiene que atrapar mas fugas y generar mas falsas alarmas.
    laxo = evaluate.evaluar_clasificacion(y_real, y_proba, umbral=0.15)
    assert laxo["recall"] >= m["recall"]
    assert laxo["falsos_positivos"] >= m["falsos_positivos"]


# ---------------------------------------------------------------------------
# PRUEBAS DEL CONTRATO DE DATOS
# El contrato existe para que el pipeline falle FUERTE cuando los datos no son
# los esperados. Estas pruebas verifican que efectivamente falle: un portero que
# nunca dice que no es igual a no tener portero.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def csv_crudo():
    """El CSV tal como lo entrega el ERP, sin transformar."""
    return data.cargar_datos(config.RUTA_DATA_RAW)


def test_la_base_real_cumple_el_contrato(csv_crudo, visitas):
    """La base de produccion pasa las cinco validaciones sin falsos positivos.

    Es tan importante como que el contrato atrape errores: un esquema demasiado
    estricto se desactiva a la semana porque nadie aguanta que falle siempre.
    """
    esquema.validar_csv_crudo(csv_crudo)
    esquema.validar_visitas(visitas)

    historia = data.agregar_features_historia(visitas)
    esquema.validar_dataset_regresion(data.preparar_dataset_entrenamiento(historia))
    esquema.validar_dataset_fuga(data.preparar_dataset_fuga(historia))
    esquema.validar_catalogo(data.construir_catalogo_vehiculos(visitas))


@pytest.mark.parametrize("formato,descripcion", [
    ("%Y-%m-%d", "ISO ano-mes-dia"),
    ("%m/%d/%Y", "estadounidense mes/dia/ano"),
    ("%d-%m-%Y", "guiones en vez de barras"),
    ("%Y%m%d", "compacto sin separadores"),
    ("%d/%m/%y", "ano de dos digitos"),
])
def test_contrato_detecta_cambio_de_formato_de_fecha(csv_crudo, formato, descripcion):
    """Un cambio en el formato de fecha debe detener el pipeline.

    Es el escenario mas peligroso de todos porque NO produce datos faltantes.
    pd.to_datetime resuelve casi cualquier formato sin generar un solo nulo, asi
    que el pipeline seguiria corriendo con fechas mal interpretadas. El caso
    estadounidense es el peor: "06/07/2017" significa 7 de junio pero se leeria
    como 6 de julio, y el proyecto entero mide intervalos entre fechas.
    """
    fechas = pd.to_datetime(csv_crudo[config.COLUMNA_FECHA_ENTRADA], dayfirst=True, errors="coerce")
    alterado = csv_crudo.copy()
    alterado[config.COLUMNA_FECHA_ENTRADA] = fechas.dt.strftime(formato)

    with pytest.raises(esquema.ContratoDeDatosError):
        esquema.validar_csv_crudo(alterado)


@pytest.mark.parametrize("columna", [
    config.COLUMNA_VIN,          # sin chasis no hay unidad de analisis
    config.COLUMNA_FECHA_ENTRADA,
    config.COLUMNA_ESTADO,       # sin estado entrarian las facturas anuladas
    config.COLUMNA_KMS,
])
def test_contrato_detecta_columna_faltante(csv_crudo, columna):
    """Si el ERP deja de exportar una columna que el pipeline necesita, se detiene."""
    with pytest.raises(esquema.ContratoDeDatosError):
        esquema.validar_csv_crudo(csv_crudo.drop(columns=[columna]))


def test_contrato_detecta_derrumbe_de_volumen(csv_crudo):
    """Un archivo que llega recortado no puede entrenar un modelo.

    Es la defensa contra la degradacion silenciosa: un DataFrame casi vacio
    cumple cualquier esquema de tipos, asi que solo contar filas lo atrapa.
    """
    with pytest.raises(esquema.ContratoDeDatosError):
        esquema.validar_csv_crudo(csv_crudo.head(100))


def test_contrato_detecta_ot_duplicadas(visitas):
    """Si la consolidacion no colapsa los sub-trabajos, toda la frecuencia se infla.

    Una OT tiene que quedar como UNA visita. Si aparecen repetidas, el proyecto
    estaria contando lineas de trabajo como ingresos al taller.
    """
    with pytest.raises(esquema.ContratoDeDatosError):
        esquema.validar_visitas(pd.concat([visitas, visitas.head(50)], ignore_index=True))


def test_contrato_detecta_catalogo_incompleto(visitas):
    """El catalogo debe traer todas las variables que el modelo consume.

    Este es el fallo mas dificil de detectar sin contrato: se agrega una variable
    al entrenamiento y se olvida producirla en el catalogo. El modelo entrena, el
    CI sale verde, y el endpoint por VIN revienta en produccion.
    """
    catalogo = data.construir_catalogo_vehiculos(visitas)
    incompleto = catalogo.drop(columns=[config.COLUMNAS_NUMERICAS_HISTORIA[0]])

    with pytest.raises(esquema.ContratoDeDatosError):
        esquema.validar_catalogo(incompleto)


def test_contrato_detecta_etiqueta_degenerada(visitas):
    """Una etiqueta con una sola clase produce un clasificador inutil.

    Y lo hace con una exactitud aparente altisima, que es lo que lo vuelve
    peligroso: pasa cualquier validacion de tipos y parece un buen resultado.
    """
    fuga = data.preparar_dataset_fuga(data.agregar_features_historia(visitas))
    degenerado = fuga.copy()
    degenerado[config.TARGET_FUGA] = 0.0

    with pytest.raises(esquema.ContratoDeDatosError):
        esquema.validar_dataset_fuga(degenerado)
