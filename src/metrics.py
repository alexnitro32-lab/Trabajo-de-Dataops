import math
from threading import Lock

class RegistroMetricas:
    """
    Acumulador en memoria protegido por un cerrojo (Lock) para registrar
    peticiones, errores y latencias de la API de forma thread-safe.
    """

    def __init__(self, limite_historial_latencia: int = 1000):
        self._lock = Lock()
        self.peticiones_totales: int = 0
        self.errores_totales: int = 0
        self.latencias_ms: list[float] = []
        self.limite_historial: int = limite_historial_latencia

    def registrar_peticion(self, latencia_ms: float, exitoso: bool = True) -> None:
        """
        Registra una petición finalizada agregando su latencia y estado.
        """
        with self._lock:
            self.peticiones_totales += 1
            if not exitoso:
                self.errores_totales += 1

            self.latencias_ms.append(latencia_ms)

            # Mantener el historial dentro del límite para evitar consumo excesivo de memoria
            if len(self.latencias_ms) > self.limite_historial:
                self.latencias_ms.pop(0)

    def _calcular_percentil(self, datos_ordenados: list[float], percentil: float) -> float:
        """
        Calcula un percentil sobre una lista de números previamente ordenada.
        """
        if not datos_ordenados:
            return 0.0

        n = len(datos_ordenados)
        k = (n - 1) * percentil
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return datos_ordenados[int(k)]

        d0 = datos_ordenados[int(f)] * (c - k)
        d1 = datos_ordenados[int(c)] * (k - f)
        return d0 + d1

    def obtener_metricas(self) -> dict:
        """
        Calcula y retorna un resumen estructurado de las métricas acumuladas.
        """
        with self._lock:
            peticiones = self.peticiones_totales
            errores = self.errores_totales
            latencias = list(self.latencias_ms)

        tasa_error = (errores / peticiones * 100.0) if peticiones > 0 else 0.0

        if not latencias:
            return {
                "peticiones_totales": peticiones,
                "errores_totales": errores,
                "tasa_error_pct": 0.0,
                "latencia_promedio_ms": 0.0,
                "latencia_p50_ms": 0.0,
                "latencia_p95_ms": 0.0,
            }

        latencias_ordenadas = sorted(latencias)
        promedio = sum(latencias_ordenadas) / len(latencias_ordenadas)
        p50 = self._calcular_percentil(latencias_ordenadas, 0.50)
        p95 = self._calcular_percentil(latencias_ordenadas, 0.95)

        return {
            "peticiones_totales": peticiones,
            "errores_totales": errores,
            "tasa_error_pct": round(tasa_error, 2),
            "latencia_promedio_ms": round(promedio, 2),
            "latencia_p50_ms": round(p50, 2),
            "latencia_p95_ms": round(p95, 2),
        }


# Instancia global (Singleton) para ser consumida por la API
METRICAS_API = RegistroMetricas()