# Core foundation only. GPU/model image is gated on Phase 0 benchmark results.
FROM node:24.14.1-bookworm-slim AS web
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data APP_PORT=8080
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ffmpeg fonts-noto-core fonts-noto-cjk tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml ./
COPY src/ ./src/
ENV PYTHONPATH=/app/src
COPY --from=web /build/dist/ ./frontend/dist/
RUN mkdir -p /data && useradd --uid 10001 --create-home autodub && chown -R autodub:autodub /app /data
USER autodub
EXPOSE 8080
VOLUME ["/data"]
HEALTHCHECK --interval=15s --timeout=3s --start-period=15s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['APP_PORT']+'/healthz',timeout=2)"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "exec uvicorn autodub.main:create_app --factory --host 0.0.0.0 --port ${APP_PORT} --workers 1"]
