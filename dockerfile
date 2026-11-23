FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    MALLOC_ARENA_MAX=2

# Run as a normal user (required by some cloud platforms, e.g. Hugging Face Spaces)
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app

# Install the requirements first (Docker cache: only rebuilt when requirements.txt changes).
# Locust is only used to test the API from outside, so it is not needed in the image.
COPY --chown=appuser:appuser requirements.txt .
RUN grep -v -i locust requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'))"

# The port comes from the PORT variable (cloud platforms set it themselves)
CMD ["sh", "-c", "python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]