# Small official Python image.
# Python 3.12 is required: PyPI only provides xgboost>=3.3 wheels for Python>=3.12.
FROM python:3.12-slim

# Faster, cleaner Python inside containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# XGBoost needs the OpenMP runtime (libgomp) on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Copy only what the API needs: code and the exported production model
COPY src/__init__.py src/config.py src/api.py ./src/
COPY models/ ./models/

# Run as a non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]