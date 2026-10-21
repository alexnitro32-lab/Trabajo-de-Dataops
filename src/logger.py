from datetime import datetime, timezone
import json
import logging
import sys


class FormateadorJSON(logging.Formatter):
    """
    Formateador personalizado que convierte cada registro de log de Python
    en un objeto JSON de una sola línea, apto para sistemas de monitoreo y nubes (como Render).
    """

    def format(self, record: logging.LogRecord) -> str:
        # Estructura base requerida para observabilidad
        log_data = {
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "level": record.levelname,
            "logger": record.name,
            "mensaje": record.getMessage(),
        }

        # Capturar campos adicionales enviados mediante extra={...}
        campos_reservados = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }

        for clave, valor in record.__dict__.items():
            if clave not in campos_reservados:
                log_data[clave] = valor

        # Si hay un rastreo de excepción (traceback), lo adjuntamos
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def obtener_logger(nombre: str = "automotor_api") -> logging.Logger:
    """
    Inicializa y retorna un logger singleton con el formateador JSON configurado hacia stdout.
    Evita la duplicación de handlers si la función se llama múltiples veces.
    """
    logger = logging.getLogger(nombre)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Handler a salida estándar
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(FormateadorJSON())

        logger.addHandler(handler)
        logger.propagate = False

    return logger
