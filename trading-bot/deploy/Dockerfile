# Mesa de trading 24/7 en un contenedor.
#   docker build -f deploy/Dockerfile -t desk .
#   docker run -d --name desk --restart=always --env-file .env -v "$PWD/state:/app/state" desk
#   docker exec -it desk python -m desk send "estado"
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DESK_STATE_DIR=/app/state

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY desk/ desk/

# Usuario sin privilegios; el estado (posiciones paper, diarios, logs) vive en un volumen.
RUN useradd --create-home --uid 10001 desk && mkdir -p /app/state && chown -R desk:desk /app
USER desk
VOLUME ["/app/state"]

# Sana si completa ciclos y lee la cuenta y los fills del exchange (en los últimos 10 minutos; el backoff
# máximo entre ciclos es de 5). Un proceso vivo que no ve el exchange queda "unhealthy".
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
  CMD python -m desk health

# SIGTERM (docker stop) -> la mesa termina el ciclo, guarda el estado y sale; los SL/TP quedan puestos.
STOPSIGNAL SIGTERM
CMD ["python", "-m", "desk", "run", "--no-input"]
