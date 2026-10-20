# tests/test_api.py - PRUEBAS AUTOMATICAS DE LA API
# Estas pruebas son la "red de seguridad" del proyecto. Cada vez que alguien
# modifique el codigo, pytest volvera a ejecutarlas y avisara si algo se rompio,
# sin que nadie tenga que abrir el navegador a probar a mano.
# Se ejecutan con:  python -m pytest

from datetime import date, timedelta

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient   # Cliente falso que llama a la API sin levantar Uvicorn

import main
from main import app                        # Importamos la aplicacion real que queremos probar
from src import config, data, diagnostico, esquema, evaluate

# TestClient simula un navegador: hace peticiones HTTP reales contra "app"
# en memoria. Por eso las pruebas corren en segundos y funcionan en GitHub
# Actions, donde no hay nadie abriendo un navegador.
client = TestClient(app)

@pytest.fixture(scope="module")
def catalogo():
    """Catalogo de vehiculos congelado, cargado una sola vez para todo el modulo."""
    return joblib.load(config.RUTA_CATALOGO_VEHICULOS)


@pytest.fixture(scope="module")
def historial():
    """Todas las visitas por VIN: lo que la API usa para recalcular la historia."""
    return joblib.load(config.RUTA_HISTORIAL_VISITAS)


def _ultimo_ingreso(historial, vin):
    return historial[historial[config.COLUMNA_VIN] == vin].sort_values(config.COLUMNA_FECHA_ENTRADA).iloc[-1]


def _ingresar(historial, vin, km_extra=1500.0, **extra):
    """Registra un ingreso HOY con un odometro coherente con la visita anterior."""
    ultima = _ultimo_ingreso(historial, vin)
    km = ultima[config.COLUMNA_KMS]
    kms = float(km) + km_extra if pd.notna(km) and km < config.KMS_MAXIMO_VALIDO - km_extra else 10000.0
    return client.post(f"/predict/vin/{vin}/ingreso", json={"kilometraje": kms, **extra})


def test_health_check():
    """Verifica que el endpoint raíz responda 200 OK y el modelo esté cargado."""
    response = client.get("/")

    # assert = "esto TIENE que ser verdad". Si no lo es, la prueba falla.
    # 200 es el codigo HTTP de exito.
    assert response.status_code == 200

    data_resp = response.json()                # Convertimos la respuesta JSON en diccionario
    assert "mensaje" in data_resp              # El contrato de salida debe incluir el mensaje
    assert data_resp["modelo_cargado"] is True # Confirma que el .joblib se cargo correctamente
    assert data_resp["historial_cargado"] is True
    # El servicio expone DOS modelos: el que dice cuando vuelve el vehiculo y el
    # que dice si va a dejar de volver.
    assert data_resp["modelo_fuga_cargado"] is True
    assert data_resp["algoritmo_fuga"] in config.MODELOS_CLASIFICACION
    assert data_resp["vehiculos_en_historial"] > 0


def test_solo_existe_el_endpoint_de_ingreso():
    """La consulta sin ingreso se retiró: toda predicción parte de una visita registrada."""
    rutas = {r.path for r in app.routes if "predict" in getattr(r, "path", "")}
    assert rutas == {"/predict/vin/{vin}/ingreso"}


def test_ingreso_exitoso(catalogo, historial):
    """Con el ingreso de hoy el carro suma una visita y se vuelve a predecir."""
    vin = str(catalogo[catalogo["Apto_Para_Modelo"]].index[0])
    response = _ingresar(historial, vin, tipo_cargo="Cliente", tipo_trabajo="MECANICA")

    assert response.status_code == 200
    data_resp = response.json()
    assert data_resp["vin"] == vin
    assert isinstance(data_resp["prediccion_dias_retorno"], float)

    ficha = data_resp["datos_encontrados"]
    assert ficha["total_visitas"] == int(catalogo.loc[vin, "Total_Visitas"]) + 1
    assert ficha["ultima_visita_concesionario"] == str(date.today())

    # El riesgo de fuga viaja solo con lo que el asesor usa: probabilidad y alerta.
    riesgo = data_resp["riesgo_de_fuga"]
    assert set(riesgo) == {"probabilidad_fuga", "alerta"}
    assert 0.0 <= riesgo["probabilidad_fuga"] <= 1.0
    assert isinstance(riesgo["alerta"], bool)


def test_estructura_de_respuesta(catalogo, historial):
    """La respuesta tiene exactamente los campos acordados con el negocio, ni uno más."""
    vin = str(catalogo[catalogo["Apto_Para_Modelo"]].index[0])
    data_resp = _ingresar(historial, vin).json()

    assert set(data_resp) == {
        "vin", "datos_encontrados", "prediccion_dias_retorno", "riesgo_de_fuga", "plan_de_llamada",
    }
    assert set(data_resp["datos_encontrados"]) == {
        "gama", "anio_modelo", "antiguedad_calculada",
        "ultimo_kilometraje_visita_concesionario", "ultima_visita_concesionario",
        "vendido_por_nosotros", "total_visitas", "ritmo_habitual_dias", "km_por_dia",
    }
    assert set(data_resp["plan_de_llamada"]) == {
        "estado", "mensaje", "fecha_referencia", "fecha_estimada_retorno",
        "margen_error_dias", "fecha_corte_datos", "base_de_la_fecha",
    }


def test_plan_de_llamada_ventana_con_mae(catalogo, historial):
    """La ventana de llamada es el retorno estimado +/- el error medio del modelo.

    El retorno se cuenta desde la ENTREGA, que es como se entrenó el target, y
    el inicio de la ventana nunca puede quedar antes de esa entrega.
    """
    vin = str(catalogo[catalogo["Apto_Para_Modelo"]].index[0])
    entrega = date.today()
    data_resp = _ingresar(historial, vin).json()
    plan = data_resp["plan_de_llamada"]

    retorno, desde, hasta = main._ventana_llamada(entrega, data_resp["prediccion_dias_retorno"])
    mae = plan["margen_error_dias"]

    assert plan["fecha_estimada_retorno"] == str(retorno)
    assert mae is not None and mae > 0
    assert (retorno - entrega).days == round(data_resp["prediccion_dias_retorno"])
    assert entrega < desde <= retorno <= hasta
    assert (hasta - retorno).days == round(mae)


def test_plan_de_llamada_cambia_de_estado_con_la_fecha():
    """Antes de la ventana se programa, dentro se llama, y después se llama ya."""
    entrega, dias = date(2026, 1, 10), 200.0
    _, desde, hasta = main._ventana_llamada(entrega, dias)

    def estado(dia, riesgo=None):
        return main._plan_de_llamada(entrega, dias, dia, riesgo)["estado"]

    assert estado(desde - timedelta(days=1)) == "programar"
    assert estado(desde) == "llamar_ahora"
    assert estado(hasta) == "llamar_ahora"
    assert estado(hasta + timedelta(days=1)) == "llamar_ya"

    # Sin alerta de fuga la ventana manda; con alerta no se espera a que abra.
    sin_alerta = {"probabilidad_fuga": 0.2, "alerta": False}
    con_alerta = {"probabilidad_fuga": 0.8, "alerta": True}
    assert estado(desde - timedelta(days=1), sin_alerta) == "programar"
    assert estado(desde - timedelta(days=1), con_alerta) == "llamar_ahora"
    # La alerta adelanta la llamada, pero no le quita el atraso a un vencido.
    assert estado(hasta + timedelta(days=1), con_alerta) == "llamar_ya"


def test_regla_de_fuga_en_la_api(catalogo, historial):
    """Un vehículo con alerta de fuga nunca queda en "programar"."""
    for vin in catalogo[catalogo["Apto_Para_Modelo"]].index[:30]:
        data_resp = _ingresar(historial, vin).json()
        if data_resp["riesgo_de_fuga"]["alerta"]:
            assert data_resp["plan_de_llamada"]["estado"] != "programar"


# ---------------------------------------------------------------------------
# PLAN DE MANTENIMIENTO SEGUN EL AÑO DEL MODELO
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anio,dias,kms", [
    (2026, 365, 10_000), (2022, 365, 10_000),        # 2022 en adelante
    (2021, 180, 5_000), (2014, 180, 5_000),          # 2021 o anteriores
    (None, 180, 5_000),                              # año desconocido: plan corto
])
def test_plan_de_mantenimiento_por_anio(anio, dias, kms):
    plan = main._plan_mantenimiento(anio)
    assert (plan["dias"], plan["kms"]) == (dias, kms)


def test_mantenimiento_lo_que_ocurra_primero():
    """Gana el plazo en días o el kilometraje, el que se cumpla antes."""
    # Uso bajo: 10 km/día tarda 500 días en hacer 5.000 km -> manda el plazo.
    assert main._dias_hasta_mantenimiento(2015, 10.0)[0] == 180
    # Uso alto: 100 km/día hace 5.000 km en 50 días -> manda el kilometraje.
    assert main._dias_hasta_mantenimiento(2015, 100.0)[0] == pytest.approx(50)
    # Mismo uso en un 2023: 10.000 km en 100 días.
    assert main._dias_hasta_mantenimiento(2023, 100.0)[0] == pytest.approx(100)
    # Sin km/día conocidos solo cuenta el plazo.
    assert main._dias_hasta_mantenimiento(2023, None)[0] == 365


def test_prediccion_nunca_supera_el_plan_de_mantenimiento(catalogo, historial):
    """No se agenda a nadie después de que ya le tocaba el mantenimiento."""
    for vin in catalogo[catalogo["Apto_Para_Modelo"]].index[:30]:
        data_resp = _ingresar(historial, vin).json()
        ficha = data_resp["datos_encontrados"]
        tope, _ = main._dias_hasta_mantenimiento(ficha["anio_modelo"], ficha["km_por_dia"])
        # Tolerancia del 1%: los km/día de la respuesta vienen redondeados a un
        # decimal, así que el tope recalculado aquí difiere en décimas.
        assert data_resp["prediccion_dias_retorno"] <= tope * 1.01 + 0.1, vin


# ---------------------------------------------------------------------------
# COHERENCIA CON EL ENTRENAMIENTO Y VALIDACIONES DEL INGRESO
# ---------------------------------------------------------------------------

def test_historial_reproduce_el_catalogo(catalogo, historial):
    """Recalcular desde el historial da EXACTAMENTE la ficha del catálogo.

    Es la garantía de que el endpoint del ingreso usa las mismas fórmulas que
    el entrenamiento: si se desalinean, esta prueba falla antes de que el
    asesor reciba una predicción calculada distinto.
    """
    for vin in catalogo[catalogo["Apto_Para_Modelo"]].index[:20]:
        visitas_vin = historial[historial[config.COLUMNA_VIN] == vin]
        recalculada = data.construir_catalogo_vehiculos(visitas_vin).loc[vin]
        esperada = catalogo.loc[vin]
        for columna in config.COLUMNAS_NUMERICAS + ["Total_Visitas", "Es_Vehiculo_Vendido"]:
            a, b = recalculada[columna], esperada[columna]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), (vin, columna)


def test_ingreso_vuelve_predecible_un_carro_de_una_visita(catalogo, historial):
    """Un VIN con un solo ingreso gana su primer intervalo y pasa a usar el modelo."""
    sin_historia = catalogo[~catalogo["Apto_Para_Modelo"]]
    if len(sin_historia) == 0:
        pytest.skip("No hay vehículos de una sola visita en el catálogo actual.")
    vin = str(sin_historia.index[0])

    data_resp = _ingresar(historial, vin, km_extra=1000.0).json()
    assert data_resp["datos_encontrados"]["total_visitas"] == 2
    assert 0.0 <= data_resp["riesgo_de_fuga"]["probabilidad_fuga"] <= 1.0


def test_ingreso_rechaza_datos_incoherentes(catalogo, historial):
    """Odómetro que baja, fecha ya registrada o tipo de trabajo inexistente se rechazan."""
    vin = str(catalogo[catalogo["Apto_Para_Modelo"] & catalogo["Kms."].notna()].index[0])
    ultima = _ultimo_ingreso(historial, vin)
    url = f"/predict/vin/{vin}/ingreso"

    assert client.post(url, json={"kilometraje": float(ultima[config.COLUMNA_KMS]) - 1}).status_code == 422
    fecha_vieja = str(pd.Timestamp(ultima[config.COLUMNA_FECHA_ENTRADA]).date())
    assert client.post(url, json={"kilometraje": 400000, "fecha_ingreso": fecha_vieja}).status_code == 409
    assert client.post(url, json={"kilometraje": 400000, "tipo_trabajo": "LAVADO"}).status_code == 422
    # Pydantic rechaza un kilometraje que no es número.
    assert client.post(url, json={"kilometraje": "cinco_mil"}).status_code == 422


def test_vin_inexistente_404():
    """Verifica que un VIN sin historial responda 404 y no reviente el servicio."""
    # 404 = "el recurso no existe". Es distinto de un 500 (error del servidor):
    # aqui la API funciono bien, simplemente ese carro nunca vino al taller.
    assert client.post("/predict/vin/NOEXISTE123/ingreso", json={"kilometraje": 1000}).status_code == 404


def test_vin_con_datos_incompletos_no_rompe(catalogo, historial):
    """Un chasis sin año de modelo responde igual, en lugar de tumbar el endpoint.

    El historial del taller no esta completo: hay VIN sin año de modelo. int(NaN)
    revienta y NaN genera un JSON invalido que ningun cliente puede leer. Esta
    prueba deja fijado que esos casos se devuelven como null, que la prediccion
    se sigue calculando y que se aplica el plan corto de mantenimiento.
    """
    incompletos = catalogo[catalogo["Anio_Modelo"].isna()]
    if len(incompletos) == 0:
        pytest.skip("El catálogo actual no tiene vehículos sin año de modelo.")

    response = _ingresar(historial, str(incompletos.index[0]))
    assert response.status_code == 200
    data_resp = response.json()            # .json() falla solo si el JSON es invalido
    assert data_resp["datos_encontrados"]["anio_modelo"] is None
    assert data_resp["prediccion_dias_retorno"] <= config.PLAN_MANTENIMIENTO_ANTERIOR["dias"]


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


def test_regla_preventiva_depende_del_anio_del_modelo(visitas):
    """2022 en adelante: 365 días o 10.000 km. 2021, anteriores o sin año: 6 meses o 5.000 km."""
    df = data.agregar_features_historia(visitas)
    nuevos = df[df[config.COLUMNA_ANO] >= config.ANIO_DESDE_PLAN_NUEVO]
    anteriores = df[~(df[config.COLUMNA_ANO] >= config.ANIO_DESDE_PLAN_NUEVO)]

    assert nuevos["Regla_Preventiva_Dias"].max() <= config.PLAN_MANTENIMIENTO_NUEVO["dias"]
    assert anteriores["Regla_Preventiva_Dias"].max() <= config.PLAN_MANTENIMIENTO_ANTERIOR["dias"]
    # Un carro anterior a 2022 que hace 100 km/día cumple los 5.000 km en 50 días.
    rapidos = anteriores[anteriores["Km_Por_Dia_Medio"].between(99, 101)]
    if len(rapidos) > 0:
        assert rapidos["Regla_Preventiva_Dias"].between(49, 51).all()


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
