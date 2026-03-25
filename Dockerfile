# ── Dragon Oracle ── FastAPI web server
# Requires host display access for mss screen capture.
# See docker-compose.yml for the full run configuration.

FROM python:3.12-slim

# System deps needed by opencv-python (headless build) and mss
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrandr2 \
        libxrender1 \
        x11-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY dragon_oracle/ ./dragon_oracle/

# Non-root user for security
RUN useradd -m -u 1000 oracle && chown -R oracle:oracle /app
USER oracle

EXPOSE 8000

# Persist board state outside the container via a named volume
VOLUME ["/app/data"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "dragon_oracle.web_app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
