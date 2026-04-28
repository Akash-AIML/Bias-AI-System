FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

# Install backend dependencies first for better layer caching.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# Copy backend source only (no frontend build in this image).
COPY backend /app/backend

EXPOSE 7860

# Hugging Face Spaces (Docker) expects the app to listen on port 7860.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
