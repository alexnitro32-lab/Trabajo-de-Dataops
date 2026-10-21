FROM python:3.12-slim

WORKDIR /app

# 1. Copiar e instalar dependencias (capa en caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar solo lo necesario para inferencia y observabilidad
COPY src/ ./src/
COPY models/ ./models/
COPY main.py .

EXPOSE 8000

# 3. Arrancar Uvicorn exponiendo en 0.0.0.0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]