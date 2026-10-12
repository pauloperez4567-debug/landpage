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

# Sano si la mesa completó un ciclo en los últimos 10 minutos (el backoff máximo es de 5).
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import os,sys,time; p='/app/state/heartbeat'; sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 600 else 1)"

# SIGTERM (docker stop) -> la mesa termina el ciclo, guarda el estado y sale; los SL/TP quedan puestos.
STOPSIGNAL SIGTERM
CMD ["python", "-m", "desk", "run", "--no-input"]
