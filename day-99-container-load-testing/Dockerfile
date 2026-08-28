# =============================================================
# Multi-Stage Production Dockerfile for MiniViT FastAPI Service
# =============================================================

# Aşama 1: Bağımlılık İnşa Aşaması
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY gereksinimler.txt .
RUN pip install --no-cache-dir --user -r gereksinimler.txt

# Aşama 2: Üretim Çalışma Zamanı (Runtime)
FROM python:3.11-slim as runtime

WORKDIR /app

# Güvenlik: Non-root kullanıcı oluşturma
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/ciktilar && \
    chown -R appuser:appuser /app

# Bağımlılıkları builder'dan kopyalama
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Uygulama kaynak kodlarını kopyalama
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser ana_akis.py /app/

USER appuser

# Kubernetes / Docker Liveness Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Çoklu iş parçacığı (multi-worker) Uvicorn çalıştırma
CMD ["uvicorn", "src.api_uygulamasi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
