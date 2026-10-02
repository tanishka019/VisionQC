FROM python:3.11-slim

# OpenCV (pulled in by anomalib) needs these shared libraries
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# Uploads, heatmaps, model and DB live here. Mount a volume at /data to keep
# them (Railway: attach a Volume in the dashboard; volume instructions in this file are rejected)
ENV VISIONQC_DATA_DIR=/data
RUN mkdir -p /data
EXPOSE 8000
# Hosts like Railway/Render inject $PORT; fall back to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
