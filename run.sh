#!/usr/bin/env bash
# SRT Translator launcher (Linux / macOS).
# Shares your home folder with the container so you can type any path under it.
# Use another folder:   HOST_ROOT=/mnt/media ./run.sh
# Change settings:      ./run.sh --setup
# Rebuild the image:    REBUILD=1 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=srt-translator
HOST_ROOT="${HOST_ROOT:-$HOME}"

if [[ -n "${REBUILD:-}" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building Docker image '$IMAGE' (first time only)..."
  docker build -t "$IMAGE" .
fi

exec docker run -it --rm \
  -v "$HOST_ROOT:/host" -e HOST_ROOT="$HOST_ROOT" \
  -v srt-config:/config \
  -v whisper-models:/models \
  -e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e GEMINI_API_KEY \
  --add-host host.docker.internal:host-gateway \
  "$IMAGE" "$@"
