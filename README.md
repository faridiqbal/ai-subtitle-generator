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

You only need [Docker](https://docs.docker.com/get-docker/). Start the launcher; it builds the image the first time:

**Linux / macOS**

```bash
./run.sh
```

**Windows** (Command Prompt or PowerShell)

```bat
run.bat
```

What happens:

1. **First run:** a setup wizard asks for the Whisper model, source and target language, translation engine, model name and API key (input hidden). The settings are saved in the `srt-config` Docker volume.
2. The app asks for the **path of a video file or folder on your computer**. Type it, paste it, or drag the file into the terminal:
   ```
   Path to a video file or folder on your computer (inside /home/farid)
   (drag & drop works; Enter = quit): /home/farid/Videos/drama ep1.mp4
   ```
   On Windows, type a path like `C:\Users\Farid\Videos\drama ep1.mp4`. `~/Videos` also works.
3. It transcribes and translates the video with progress bars, then writes `drama ep1.srt` **next to the original file on your computer**.
4. It asks for the next path. The Whisper model stays loaded, so later files start faster. Press Enter on an empty line to quit.

Useful variations:

| Goal | Linux / macOS | Windows |
|---|---|---|
| Change settings | `./run.sh --setup` | `run.bat --setup` |
| Process one path and exit | `./run.sh "/home/me/Videos/ep1.mp4"` | `run.bat "C:\Users\me\Videos\ep1.mp4"` |
| Videos outside your home folder | `HOST_ROOT=/mnt/media ./run.sh` | `set HOST_ROOT=D:\Movies` then `run.bat` |
| Rebuild after updating the code | `REBUILD=1 ./run.sh` | `set REBUILD=1` then `run.bat` |

### How the host path works

A container can only see the folders you share with it. The launcher shares `HOST_ROOT` with the container (read-write, so the `.srt` can be saved), mounted at `/host`. By default that is your home or user folder. The app converts the host path you type, such as `/home/farid/Videos/a.mp4` or `C:\Users\Farid\Videos\a.mp4`, into the matching path inside the container, `/host/Videos/a.mp4`. Paths outside `HOST_ROOT` are rejected with a hint. To limit what the container can access, set `HOST_ROOT` to just your videos folder.

On Linux, new `.srt` files are given the same owner as the video, so they are not owned by root.

### docker compose (alternative)

```bash
docker compose build
```

```bash
docker compose run --rm srt-translator
```

This shares `HOST_ROOT`, or `$HOME` if `HOST_ROOT` is not set. On Windows, set `HOST_ROOT` first, for example `set HOST_ROOT=%USERPROFILE%`.

If you'd rather not store the API key in the config volume, leave it blank during setup and export it before you start the container:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Using a local Ollama model

Choose engine **2** in setup. For the base URL, enter `http://host.docker.internal:11434/v1`. Enter a model name you have pulled, such as `qwen2.5:14b`. The API key can be left blank.

### GPU

The image runs on CPU. A 1-hour video with `large-v3-turbo` takes roughly real-time to 2x real-time on a modern CPU. For an NVIDIA GPU, base the image on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, install Python and the requirements on top of it, then run with `--gpus all` and set the device to `cuda` in setup.

---

## Run without Docker

You need `ffmpeg` on your PATH: `apt install ffmpeg`, `brew install ffmpeg`, or `winget install ffmpeg`.

**Python 3.10 – 3.14**

```bash
pip install -r requirements.txt
```

**Python 3.15 or newer**

onnxruntime does not publish wheels for Python 3.15 yet. faster-whisper requires it, so a plain `pip install` fails with *"Cannot install ... conflicting dependencies"*. Install faster-whisper without its dependencies, then add the rest:

```bash
pip install --no-deps faster-whisper
```

```bash
pip install -r requirements-py315.txt
```

The app detects that onnxruntime is missing and turns off silence filtering (VAD). Subtitles still work, but timestamps around long pauses or music can be a bit looser. For full quality, use Docker (Python 3.12 inside the image) or a Python 3.13 virtual environment, for example with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13 && source .venv/bin/activate && uv pip install -r requirements.txt
```

Then run:

```bash
python srt_translator.py
```

Settings are saved to `~/.srt-translator/config.json`. Use `--setup` to change them.
