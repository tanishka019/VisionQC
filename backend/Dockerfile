FROM python:3.11-slim

# OpenCV (pulled in by anomalib) needs these shared libraries
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# Uploads, heatmaps, model and DB live here — mount a volume to keep them
ENV VISIONQC_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
