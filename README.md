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

### Subtitle timing

Each subtitle line is timed by Whisper's **word timestamps**. A line starts at its first word and ends at its last word, plus 0.3 s so it doesn't vanish abruptly. Subtitles therefore disappear when the speech stops instead of staying on screen through silence or music.

A new line is started when:
- there is a pause of 0.6 s or more,
- a sentence ends, or
- a line would exceed about 2 rows (84 characters) or 7 seconds.

Each line stays visible for at least 0.8 s, and lines never overlap.

### Do I need a Hugging Face token?

**No.** Whisper models are public and are downloaded from Hugging Face **once**, with a progress bar. They are stored in a plain local folder: the `whisper-models` Docker volume, or `~/.srt-translator/models` without Docker. After that the model always loads from disk, the app never contacts Hugging Face again, and transcription works completely offline. The warning *"You are sending unauthenticated requests to the HF Hub"* only meant downloads have lower rate limits without a token.

Setup offers an **optional** token field (press Enter to skip) and can download the model straight away. You can also set `HF_TOKEN` in the environment.

### Large and long videos (1 GB+, 2+ hours)

| Concern | What the app does |
|---|---|
| **Huge files** | ffmpeg decodes **only the first audio track** (`-map 0:a:0 -vn`). The video picture is never decoded, so a 5 GB file costs no more than its soundtrack. A 1.6 GB test file finished audio extraction in under 1 s. |
| **Memory on long films** | Audio is streamed in **20-minute chunks**, and each chunk is cut at the quietest moment so no word is split. RAM stays flat: in tests, a 2.5-hour file peaked at 1.1 GB with the `tiny` model, the same as an 8-minute file. |
| **Speed** | **Batched transcription** processes several speech segments at once. In testing it was 8x faster than sequential mode (12.6 s vs 105.6 s for 8 min of audio, `tiny` model, 4 CPU cores). It uses all CPU cores, and the AI translation runs **4 requests in parallel**. |
| **Crashes / lost connection / Ctrl+C** | Progress is saved after every audio chunk and every translation batch. Run the same file again and it **resumes** where it stopped: no re-transcribing, and no paying twice for lines already translated. The cache is in `<config dir>/cache`. |
| **Half-written files** | Each `.srt` is written to a temporary file first and swapped in only when complete. |
| **No audio track** | The app reports *"this file has no audio track"* and moves on to the next file. |

Tuning (environment variables):
- `SRT_PARALLEL_REQUESTS` (default `4`): lower it if you hit API rate limits, raise it for faster translation.
- `SRT_CHUNK_SECONDS` (default `1200`): lower it to use less RAM.

Rough timing on a modern 8-core CPU with `large-v3-turbo`: a 2-hour film takes about 30–60 minutes to transcribe. Translation of about 2,000 lines takes a few minutes. With an NVIDIA GPU, transcription takes only a few minutes. On Docker Desktop (Windows/macOS), give Docker at least **4 GB of memory**, or **8 GB for `large-v3`** (Settings → Resources).

> **Note on the batched mode:** it requires `onnxruntime`. On Python 3.15, where onnxruntime isn't available yet, the app falls back to slower sequential mode. The Docker image always has the fast mode.

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

**ffmpeg** is needed to read the audio from videos. You don't have to install it yourself: at startup the app checks for it and, if it's missing, offers:

1. **Automatic install with pip.** This installs the `imageio-ffmpeg` package, which bundles a ready-made ffmpeg for Windows, macOS and Linux. No admin rights are needed. This is the default.
2. **System-wide install** with the package manager it finds on your computer: `winget`, Chocolatey or Scoop on Windows, Homebrew on macOS, or apt, dnf, pacman or zypper on Linux.
3. **Manual instructions.**

The Docker image already includes ffmpeg.

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
