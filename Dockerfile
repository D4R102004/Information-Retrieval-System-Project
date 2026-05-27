# Base image: official Python 3.12 slim variant — smaller than the full image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by chromadb and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the manifest first to generate requirements.txt.
# This layer is cached — deps only reinstall when pyproject.toml changes.
COPY pyproject.toml .

# Extract dependencies from pyproject.toml into requirements.txt using
# tomllib (stdlib since Python 3.11) — no extra tools needed.
RUN python -c "\
import tomllib; \
deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
open('requirements.txt', 'w').write('\n'.join(deps))"

# Install dependencies from requirements.txt — fast, cacheable layer
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code after deps are installed — code changes don't bust dep cache
COPY src/ ./src/

# Set PYTHONPATH so all modules under src/ are importable
ENV PYTHONPATH=/app/src

# Default command: run the integration demo
CMD ["python", "src/main.py"]
