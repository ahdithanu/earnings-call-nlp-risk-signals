# Scoring API image. The lexicon .txt files and results/ watchlist live at
# the repo root (referenced relative to the package), so the image copies
# the project rather than installing from a wheel.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching: metadata + package only.
COPY pyproject.toml README.md ./
COPY earnings_signals ./earnings_signals
RUN pip install ".[api]"

# Data the service reads at runtime.
COPY lm_uncertainty_terms.txt lm_negative_terms.txt lm_positive_terms.txt \
     lm_litigious_terms.txt lm_constraining_terms.txt ./
COPY results/latest_uncertainty_signals.csv ./results/latest_uncertainty_signals.csv

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)"

CMD ["uvicorn", "earnings_signals.api:app", "--host", "0.0.0.0", "--port", "8000"]
