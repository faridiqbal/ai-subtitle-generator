FROM python:3.12-slim

# settings (incl. API key) and downloaded Whisper models live in volumes
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SRT_CONFIG_DIR=/config \
    HF_HOME=/models \
    HOST_MOUNT=/host

# ffmpeg decodes the audio track of any video container
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY srt_translator.py .

# /host = the folder of your computer shared by run.sh / run.bat
VOLUME ["/config", "/models"]
ENTRYPOINT ["python", "/app/srt_translator.py"]
