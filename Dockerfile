FROM python:3.12-slim

# settings (incl. API key) and downloaded Whisper models live in volumes
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SRT_CONFIG_DIR=/config \
    HF_HOME=/models \
    SRT_VIDEO_DIR=/videos

# ffmpeg decodes the audio track of any video container
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY srt_translator.py .

VOLUME ["/config", "/models", "/videos"]
ENTRYPOINT ["python", "/app/srt_translator.py"]
