# SRT Translator

Generate translated subtitles (`.srt`) from video files. It reads the audio, detects the spoken language (Malay, Arabic, Chinese, Japanese, and about 100 others), writes timestamps, translates every line into your target language (for example English), and saves `<video name>.srt` next to the video.

```
my_movie.mp4  ->  my_movie.srt        (translated)
              ->  my_movie.ja.srt     (optional: original language)
```

## How it works

| Step | Library | Notes |
|---|---|---|
| Audio decoding | **FFmpeg** (installed in the Docker image) | mp4, mkv, mov, avi, webm, mp3, wav... |
| Speech to text, language detection, timestamps | **faster-whisper** (OpenAI Whisper on CTranslate2) | Runs locally and free. Silence is skipped with VAD |
| Translation | **anthropic** / **openai** / **google-genai** | Your choice of model and API key |
| Progress % | `tqdm` | Shows separate bars for transcription and translation |

Subtitle lines go to the AI in batches of 40, along with a few previous lines as context. The model must return JSON with one translation per line ID, so the timestamps always stay aligned. Batches that fail are retried. Any line that still fails keeps its original text instead of breaking the file.

### Translation engines (chosen during setup)

| # | Engine | Default model | API key env var |
|---|---|---|---|
| 1 | Anthropic Claude | `claude-opus-5-5`. Use `claude-sonnet-5-5` or `claude-haiku-5-5` for lower cost | `ANTHROPIC_API_KEY` |
| 2 | OpenAI or any OpenAI-compatible API (DeepSeek, OpenRouter, a local Ollama...) | `gpt-4o-mini` | `OPENAI_API_KEY` |
| 3 | Google Gemini | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| 4 | Whisper built-in translation | Free and offline, **English only**, lower quality | none |
| 5 | No translation | Writes subtitles in the original language | none |

You can type any model name your provider offers during setup.

For Claude, the request uses structured JSON output and `effort: low`, because translation is a routine task and this makes it faster and cheaper. It also turns on server-side **refusal fallbacks**: if the model refuses a batch (for example, a false positive on violent movie dialogue), the API retries that batch on a fallback model. If your account or model doesn't support fallbacks, the script turns them off automatically.

### Whisper model sizes

| Model | Speed | Accuracy | Notes |
|---|---|---|---|
| `small` / `medium` | fast | ok | Good for a CPU-only machine |
| `large-v3-turbo` (default) | good | very good | Best balance |
| `large-v3` | slow | best | Use it for difficult audio and mixed languages |

Whisper sometimes confuses Malay with Indonesian. If that happens, set the source language to `ms` in setup instead of `auto`.

---

## Run with Docker (recommended)

1. Put your videos in a `videos/` folder next to `docker-compose.yml`.
2. Build and run:

```bash
docker compose build
```

```bash
docker compose run --rm srt-translator
```

**First run:** a setup wizard asks for the Whisper model, the source and target language, the translation engine, the model name and your API key. The key input is hidden. The settings are saved in the `srt-config` Docker volume, so later runs only ask for the video path. Press Enter to accept `/videos`, or give a single file such as `/videos/drama_ep1.mp4`.

**Change settings later:**

```bash
docker compose run --rm srt-translator --setup
```

**Process one file directly, without prompts:**

```bash
docker compose run --rm srt-translator /videos/drama_ep1.mp4
```

Whisper models download once into the `whisper-models` volume.

If you'd rather not store the API key in the config volume, leave it blank during setup and export it before you run the container:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Without docker compose

```bash
docker build -t srt-translator .
```

```bash
docker run -it --rm -v "$PWD/videos:/videos" -v srt-config:/config -v whisper-models:/models srt-translator
```

### Using a local Ollama model

Choose engine **2** in setup. For the base URL, enter `http://host.docker.internal:11434/v1`. Enter a model name you have pulled, such as `qwen2.5:14b`. The API key can be left blank.

### GPU

The image runs on CPU. A 1-hour video with `large-v3-turbo` takes roughly real-time to 2x real-time on a modern CPU. For an NVIDIA GPU, base the image on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, install Python and the requirements on top of it, then run with `--gpus all` and set the device to `cuda` in setup.

---

## Run without Docker

You need Python 3.10 or newer and `ffmpeg` on your PATH (`apt install ffmpeg`, `brew install ffmpeg`, or `winget install ffmpeg`).

```bash
pip install -r requirements.txt
```

```bash
python srt_translator.py
```

Settings are saved to `~/.srt-translator/config.json`. Use `--setup` to change them.
