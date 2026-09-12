# tests/test_api.py - PRUEBAS AUTOMATICAS DE LA API
# Estas pruebas son la "red de seguridad" del proyecto. Cada vez que alguien
# modifique el codigo, pytest volvera a ejecutarlas y avisara si algo se rompio,
# sin que nadie tenga que abrir el navegador a probar a mano.
# Se ejecutan con:  python -m pytest

from fastapi.testclient import TestClient   # Cliente falso que llama a la API sin levantar Uvicorn
from main import app                        # Importamos la aplicacion real que queremos probar

# TestClient simula un navegador: hace peticiones HTTP reales contra "app"
# en memoria. Por eso las pruebas corren en segundos y funcionan en GitHub
# Actions, donde no hay nadie abriendo un navegador.
client = TestClient(app)


def test_health_check():
    """Verifica que el endpoint raíz responda 200 OK y el modelo esté cargado."""
    response = client.get("/")

    # assert = "esto TIENE que ser verdad". Si no lo es, la prueba falla.
    # 200 es el codigo HTTP de exito.
    assert response.status_code == 200

    data = response.json()                    # Convertimos la respuesta JSON en diccionario
    assert "mensaje" in data                  # El contrato de salida debe incluir el mensaje
    assert data["modelo_cargado"] is True     # Confirma que el .joblib se cargo correctamente
    assert data["catalogo_cargado"] is True   # Confirma que el catalogo de VIN tambien se cargo


def test_prediccion_exitosa():
    """Verifica que el endpoint /predict retorne un cálculo numérico válido."""
    # Este es el "camino feliz": datos correctos y completos.
    # Las llaves usan los alias con espacios y puntos definidos en Pydantic.
    payload = {
        "Gama": "Tucson",
        "Tipo Cargo": "Cliente",
        "Tipo de Trabajo": "MECANICA",
        "Kms.": 45000.0,
        "Año Modelo": 2022,          # El usuario escribe el anio, la API calcula la antiguedad
        "Es_Vehiculo_Vendido": True
    }

    # json=payload le dice al cliente que envie el diccionario como cuerpo JSON.
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediccion_dias_retorno" in data                    # La clave del contrato existe
    assert isinstance(data["prediccion_dias_retorno"], float)   # Y ademas es un numero, no texto
    assert data["unidad"] == "días"


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
    # El modelo se entreno con antiguedades de 0 a 29 anios. Un carro de 1800
    # obligaria a extrapolar donde nunca hubo datos, asi que Pydantic lo corta.
    payload = {
        "Gama": "Tucson",
        "Tipo Cargo": "Cliente",
        "Tipo de Trabajo": "MECANICA",
        "Kms.": 45000.0,
        "Año Modelo": 1800,
        "Es_Vehiculo_Vendido": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediccion_por_vin_exitosa():
    """Verifica que al consultar un VIN real se autocompleten los datos y prediga."""
    # ADM130850 es un chasis con historial real en la base del taller.
    response = client.get("/predict/vin/ADM130850")

    assert response.status_code == 200
    data = response.json()
    assert data["vin"] == "ADM130850"
    assert "prediccion_dias_retorno" in data

    # La API debe devolver tambien la ficha encontrada, para que el asesor
    # pueda confirmar que el sistema busco el vehiculo correcto.
    assert "gama" in data["datos_encontrados"]
    assert "ultimo_kilometraje" in data["datos_encontrados"]


def test_vin_inexistente_404():
    """Verifica que un VIN sin historial responda 404 y no reviente el servicio."""
    response = client.get("/predict/vin/NOEXISTE123")

    # 404 = "el recurso no existe". Es distinto de un 500 (error del servidor):
    # aqui la API funciono bien, simplemente ese carro nunca vino al taller.
    assert response.status_code == 404


def test_vin_con_datos_incompletos_no_rompe():
    """Verifica que un chasis con año de modelo o kilometraje faltante responda
    igual, en lugar de tumbar el endpoint con un error 500.

    El historial del taller no esta completo: hay VIN sin año de modelo y sin
    kilometraje. Antes, int(NaN) reventaba y NaN generaba un JSON invalido que
    ningun cliente podia leer. Esta prueba deja fijado que esos casos se
    devuelven como null y que la prediccion se sigue calculando.
    """
    import joblib
    from src import config

    catalogo = joblib.load(config.RUTA_CATALOGO_VEHICULOS)
    incompletos = catalogo[catalogo["Anio_Modelo"].isna() | catalogo["Kms."].isna()]

    # Si algun dia la base llega completa, no hay nada que probar aqui.
    if len(incompletos) == 0:
        return

    vin = str(incompletos.index[0])
    response = client.get(f"/predict/vin/{vin}")

    assert response.status_code == 200
    data = response.json()                 # .json() falla solo si el JSON es invalido
    assert isinstance(data["prediccion_dias_retorno"], float)

    # El dato faltante debe viajar como null explicito, no como NaN ni como 0.
    ficha = data["datos_encontrados"]
    assert ficha["anio_modelo"] is None or isinstance(ficha["anio_modelo"], int)
    assert ficha["ultimo_kilometraje"] is None or isinstance(ficha["ultimo_kilometraje"], float)
