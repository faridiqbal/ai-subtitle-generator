#!/usr/bin/env python3
"""
SRT Translator - generate translated subtitles from video files.

Pipeline
  1. Read the audio track of a video (mp4, mkv, mov, ...)
  2. Transcribe it locally with faster-whisper (auto language detection + timestamps)
  3. Translate every subtitle line with the AI provider you choose
     (Anthropic Claude, OpenAI / OpenAI-compatible, Google Gemini) or with
     Whisper's own built-in "translate to English" mode (free, offline)
  4. Write <video name>.srt next to the video

Run:  python srt_translator.py
"""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install -r requirements.txt")


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v",
                    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

# Whisper sizes -> public Hugging Face repos with CTranslate2 weights (approx. download size)
WHISPER_REPOS = {
    "tiny": ("Systran/faster-whisper-tiny", "75 MB"),
    "base": ("Systran/faster-whisper-base", "145 MB"),
    "small": ("Systran/faster-whisper-small", "485 MB"),
    "medium": ("Systran/faster-whisper-medium", "1.5 GB"),
    "large-v3": ("Systran/faster-whisper-large-v3", "3.1 GB"),
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "1.6 GB"),
}
WHISPER_MODELS = list(WHISPER_REPOS)
MODEL_FILES = re.compile(r"^(config\.json|preprocessor_config\.json|model\.bin|"
                         r"tokenizer\.json|vocabulary\.\w+)$")

LANGUAGE_NAMES = {
    "en": "English", "ms": "Malay", "id": "Indonesian", "ar": "Arabic",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ta": "Tamil",
    "hi": "Hindi", "th": "Thai", "vi": "Vietnamese", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
    "tr": "Turkish", "ur": "Urdu", "bn": "Bengali", "tl": "Tagalog",
}

# provider key -> (label, default model, env var holding the API key)
PROVIDERS = {
    "1": ("anthropic", "Anthropic Claude", "claude-opus-5-5", "ANTHROPIC_API_KEY"),
    "2": ("openai", "OpenAI / OpenAI-compatible (DeepSeek, OpenRouter, Ollama...)",
          "gpt-4o-mini", "OPENAI_API_KEY"),
    "3": ("gemini", "Google Gemini", "gemini-2.5-flash", "GEMINI_API_KEY"),
    "4": ("whisper", "Whisper built-in translation (English only, free, offline)",
          None, None),
    "5": ("none", "No translation (original-language subtitles only)", None, None),
}

BATCH_SIZE = 40       # subtitle lines per translation request
CONTEXT_LINES = 5     # previous lines sent as context for better continuity
MAX_RETRIES = 3
PARALLEL_REQUESTS = int(os.environ.get("SRT_PARALLEL_REQUESTS", "4"))  # concurrent AI calls


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    start: float
    end: float
    text: str


def ask(prompt: str, default: str | None = None, choices: list[str] | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip().strip('"').strip("'")
        if not value and default is not None:
            value = default
        if not value:
            print("  Please enter a value.")
            continue
        if choices and value not in choices:
            print(f"  Choose one of: {', '.join(choices)}")
            continue
        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{d}]: ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes")


def lang_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def format_timestamp(seconds: float) -> str:
    ms = int(round(max(seconds, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[Segment], path: Path) -> None:
    """Write atomically: a crash never leaves a half-written .srt behind."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
                    f"{seg.text.strip()}\n\n")
    tmp.replace(path)


def collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*")
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    return []


# --------------------------------------------------------------------------- #
# 1. Transcription (faster-whisper)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ffmpeg: detect, and help install when missing
# --------------------------------------------------------------------------- #
FFMPEG: str | None = None
FFPROBE: str | None = None


def find_ffmpeg() -> bool:
    """Locate ffmpeg on PATH, or the copy bundled by the imageio-ffmpeg pip package."""
    import shutil
    global FFMPEG, FFPROBE
    FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not FFMPEG:
        try:
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
            FFPROBE = None
        except Exception:
            FFMPEG = None
    return FFMPEG is not None


def _system_install_command() -> tuple[str, list[str]] | None:
    """Best ffmpeg install command for this OS's package manager, if one is present."""
    import platform
    import shutil

    system = platform.system()
    if system == "Windows":
        if shutil.which("winget"):
            return "winget", ["winget", "install", "--id", "Gyan.FFmpeg", "-e",
                              "--accept-source-agreements", "--accept-package-agreements"]
        if shutil.which("choco"):
            return "Chocolatey", ["choco", "install", "ffmpeg", "-y"]
        if shutil.which("scoop"):
            return "Scoop", ["scoop", "install", "ffmpeg"]
    elif system == "Darwin":
        if shutil.which("brew"):
            return "Homebrew", ["brew", "install", "ffmpeg"]
    else:
        sudo = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo"]
        for tool, cmd in (("apt-get", ["apt-get", "install", "-y", "ffmpeg"]),
                          ("dnf", ["dnf", "install", "-y", "ffmpeg"]),
                          ("pacman", ["pacman", "-S", "--noconfirm", "ffmpeg"]),
                          ("zypper", ["zypper", "install", "-y", "ffmpeg"])):
            if shutil.which(tool):
                if tool == "apt-get":
                    cmd = ["sh", "-c", " ".join(sudo + ["apt-get", "update"]) + " && "
                           + " ".join(sudo + cmd)]
                    return tool, cmd
                return tool, sudo + cmd
    return None


def _manual_instructions() -> str:
    return ("  Windows : winget install Gyan.FFmpeg   (or download from https://www.gyan.dev/ffmpeg/builds/)\n"
            "  macOS   : brew install ffmpeg          (Homebrew: https://brew.sh)\n"
            "  Ubuntu  : sudo apt install ffmpeg\n"
            "  Fedora  : sudo dnf install ffmpeg\n"
            "  Or run the Docker version, which already includes ffmpeg.")


def ensure_ffmpeg() -> bool:
    """Check for ffmpeg; if missing, offer to install it. Returns True when usable."""
    import subprocess

    if find_ffmpeg():
        return True
    print("\n" + "-" * 64)
    print("  ffmpeg is not installed")
    print("-" * 64)
    print("ffmpeg is needed to read the audio track of your videos.\n")
    system_cmd = _system_install_command()
    options = {"1": "Install automatically with pip (bundled ffmpeg, no admin rights needed)"}
    if system_cmd:
        options["2"] = f"Install system-wide with {system_cmd[0]}:  {' '.join(system_cmd[1])}"
    options[str(len(options) + 1)] = "Show manual install instructions and exit"
    for key, label in options.items():
        print(f"  {key}. {label}")
    choice = ask("Choose", "1", list(options))

    if choice == "1":
        print("\nInstalling imageio-ffmpeg (about 30 MB)...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        if result.returncode == 0:
            import importlib
            importlib.invalidate_caches()
    elif system_cmd and choice == "2":
        print(f"\nRunning: {' '.join(system_cmd[1])}\n(you may be asked for your password)\n")
        try:
            subprocess.run(system_cmd[1])
        except OSError as exc:
            print(f"  ! Could not run the installer: {exc}")
    else:
        print("\nInstall ffmpeg with one of these, then start the app again:\n"
              + _manual_instructions())
        return False

    if find_ffmpeg():
        print(f"ffmpeg is ready: {FFMPEG}")
        return True
    print("\n! ffmpeg still not found. If it was just installed system-wide, open a new\n"
          "  terminal so PATH is refreshed, then start the app again. Otherwise:\n"
          + _manual_instructions())
    return False


def models_dir() -> Path:
    return Path(os.environ.get("SRT_MODELS_DIR", CONFIG_DIR / "models"))


def model_path(size: str) -> Path:
    return models_dir() / size


def model_is_downloaded(size: str) -> bool:
    return (model_path(size) / ".complete").exists()


def download_model(size: str, token: str | None = None) -> Path:
    """Download a Whisper model once into a plain local folder, with a progress bar.

    Hugging Face models used here are public: a token is optional and only
    raises download rate limits.
    """
    import urllib.request

    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()            # hide the "unauthenticated requests" notice
    repo = WHISPER_REPOS.get(size, (size, ""))[0]
    dest = model_path(size)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading Whisper model '{size}' from huggingface.co/{repo} (one time only)")
    info = HfApi(token=token or None).model_info(repo, files_metadata=True)
    files = [(f.rfilename, f.size or 0) for f in info.siblings if MODEL_FILES.match(f.rfilename)]
    with tqdm(total=sum(sz for _, sz in files), unit="B", unit_scale=True, unit_divisor=1024,
              desc="Downloading model") as bar:
        for name, size_bytes in files:
            target = dest / name
            if target.exists() and target.stat().st_size == size_bytes:
                bar.update(size_bytes)               # already there from an earlier attempt
                continue
            req = urllib.request.Request(hf_hub_url(repo, name),
                                         headers={"User-Agent": "srt-translator"})
            if token:                                # not forwarded to the CDN redirect
                req.add_unredirected_header("Authorization", f"Bearer {token}")
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    bar.update(len(chunk))
            tmp.replace(target)
    (dest / ".complete").touch()
    print(f"Model saved to {dest}")
    return dest


class Spinner:
    """Progress line with elapsed time for steps that report no percentage."""

    def __init__(self, desc: str):
        import threading
        self.desc = desc
        self.bar = tqdm(bar_format="{desc}", desc=desc)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = time.time()

    def _show(self, status: str) -> None:
        elapsed = tqdm.format_interval(time.time() - self._t0)
        self.bar.set_description_str(f"{self.desc}: {status} {elapsed}", refresh=True)

    def _run(self):
        frames = "|/-\\"
        i = 0
        while not self._stop.wait(0.2):
            self._show(frames[i % 4])
            i += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._show("done in" if exc[0] is None else "failed after")
        self.bar.close()


def load_whisper(model_size: str, device: str, hf_token: str | None = None):
    from faster_whisper import WhisperModel

    if not model_is_downloaded(model_size):
        download_model(model_size, hf_token)

    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    # Loading from the local folder: no internet connection needed from here on.
    threads = os.cpu_count() or 4                  # faster-whisper defaults to only 4
    with Spinner(f"\nLoading Whisper '{model_size}' into memory ({device}, {compute_type})"):
        model = WhisperModel(str(model_path(model_size)), device=device,
                             compute_type=compute_type, cpu_threads=threads,
                             local_files_only=True)
    model.srt_device = device
    return model


def media_duration(video: Path) -> float:
    import subprocess
    if not FFMPEG and not find_ffmpeg():
        return 0.0
    try:
        if FFPROBE:
            out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nw=1:nk=1", str(video)],
                                 capture_output=True, text=True, timeout=60).stdout
            return float(out.strip())
        # no ffprobe (e.g. pip-bundled ffmpeg): read "Duration: 01:23:45.67" from ffmpeg -i
        err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video)],
                             capture_output=True, text=True, timeout=60).stderr
        h, m, sec = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err).groups()
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except (ValueError, OSError, AttributeError, subprocess.SubprocessError):
        return 0.0


def vad_available() -> bool:
    """Silence filtering (VAD) needs onnxruntime, which has no wheels for every Python."""
    import importlib.util
    return importlib.util.find_spec("onnxruntime") is not None


SAMPLE_RATE = 16000
CHUNK_SECONDS = int(os.environ.get("SRT_CHUNK_SECONDS", 20 * 60))  # audio decoded per step
CUT_WINDOW = 15            # look +-15 s around each chunk boundary for the quietest moment


def _quiet_cut(samples, target: int) -> int:
    """Index near `target` at the quietest 0.1 s frame, so words are not cut in half."""
    import numpy as np

    frame = SAMPLE_RATE // 10
    lo = max(target - CUT_WINDOW * SAMPLE_RATE, 0)
    hi = min(target + CUT_WINDOW * SAMPLE_RATE, len(samples))
    n = (hi - lo) // frame
    if n < 2:
        return target
    win = samples[lo:lo + n * frame].astype(np.float32).reshape(n, frame)
    return lo + int(np.argmin((win ** 2).mean(axis=1))) * frame + frame // 2


def audio_chunks(video: Path, start: float = 0.0):
    """Stream the first audio track through ffmpeg; yield (offset_s, float32 samples).

    Memory stays flat however long the video is: only one chunk (~77 MB for 20 min)
    is held at a time. The video picture is never decoded, so file size barely matters.
    """
    import subprocess

    import numpy as np

    if not FFMPEG and not find_ffmpeg() and not ensure_ffmpeg():
        raise RuntimeError("ffmpeg is required to read audio")
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-threads", "0"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]                # fast seek when resuming
    cmd += ["-i", str(video), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]

    bytes_per_s = SAMPLE_RATE * 2
    target = CHUNK_SECONDS * SAMPLE_RATE
    need = (CHUNK_SECONDS + CUT_WINDOW) * bytes_per_s
    offset = start
    buf = bytearray()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        eof = False
        while not eof:
            data = proc.stdout.read(1 << 20)
            eof = not data
            buf += data
            while len(buf) >= need or (eof and buf):
                samples = np.frombuffer(bytes(buf), np.int16)
                cut = len(samples) if eof and len(buf) < need else _quiet_cut(samples, target)
                chunk = samples[:cut].astype(np.float32) / 32768.0
                del samples
                buf = bytearray(buf[cut * 2:])
                yield offset, chunk
                offset += cut / SAMPLE_RATE
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0 and offset == start:
            if "matches no streams" in err:
                raise RuntimeError("this file has no audio track")
            raise RuntimeError(f"ffmpeg could not read audio: {err[-300:]}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# Subtitle line rules (common broadcast guidelines)
MAX_LINE_CHARS = 84        # about two rows of 42 characters
MAX_LINE_SECONDS = 7.0     # longest time one subtitle stays on screen
SPLIT_PAUSE = 0.6          # a silence this long between words starts a new subtitle
MAX_WORD_SECONDS = 1.5     # Whisper sometimes stretches a word over the following silence
END_PADDING = 0.3          # keep text visible a moment after the last word
MIN_DURATION = 0.8         # shortest time a subtitle is shown (readability)
SENTENCE_END = tuple(".?!。？！…")


def segment_to_lines(seg, offset: float, chunk_len: float) -> list[Segment]:
    """Turn one Whisper segment into subtitle lines timed by its words.

    Each line starts at its first word and ends at its last word, so subtitles
    disappear when the speech stops instead of lingering through silence.
    """
    words = [w for w in (getattr(seg, "words", None) or []) if w.word.strip()]
    if not words:                                   # no word timing: fall back
        text = seg.text.strip()
        return [Segment(offset + seg.start, offset + min(seg.end, chunk_len), text)] if text else []

    lines: list[list] = []
    current: list = []
    for w in words:
        if current:
            text_len = len("".join(x.word for x in current + [w]).strip())
            if (w.start - current[-1].end >= SPLIT_PAUSE
                    or text_len > MAX_LINE_CHARS
                    or w.end - current[0].start > MAX_LINE_SECONDS
                    or current[-1].word.strip().endswith(SENTENCE_END)):
                lines.append(current)
                current = []
        current.append(w)
    if current:
        lines.append(current)

    out = []
    for ws in lines:
        text = "".join(w.word for w in ws).strip()
        last = ws[-1]
        end = min(last.end, last.start + MAX_WORD_SECONDS)
        out.append(Segment(offset + ws[0].start, offset + min(end, chunk_len), text))
    return out


def fix_timing(lines: list[Segment]) -> list[Segment]:
    """Pad ends slightly, enforce a readable minimum duration, never overlap."""
    lines = sorted((s for s in lines if s.text), key=lambda s: s.start)
    for i, s in enumerate(lines):
        next_start = lines[i + 1].start if i + 1 < len(lines) else float("inf")
        end = max(s.end + END_PADDING, s.start + MIN_DURATION)
        s.end = max(min(end, next_start - 0.04), s.start + 0.1)
    return lines


# Phrases Whisper is known to invent over music, noise or silence (from its training
# data: video outros). People also say them for real ("terima kasih kerana menonton" at
# the end of a video), so a phrase is only treated as a hallucination when it keeps
# repeating through the same video.
HALLUCINATIONS = [
    "terima kasih kerana menonton", "terima kasih kerana menonton video ini",
    "terima kasih kerana menonton dan jangan lupa subscribe", "sila langgan", "jangan lupa subscribe",
    "terima kasih telah menonton", "terima kasih sudah menonton",
    "thanks for watching", "thank you for watching", "please subscribe",
    "subtitles by the amaraorg community", "like and subscribe",
    "ご視聴ありがとうございました", "チャンネル登録をお願いします",
    "谢谢观看", "謝謝觀看", "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "字幕由amaraorg社区提供",
    "شكرا للمشاهدة", "اشتركوا في القناة",
]
_HALLU = {re.sub(r"[\W_]+", "", h.lower()) for h in HALLUCINATIONS}


HALLUCINATION_REPEATS = 2


def drop_hallucination_loops(lines: list[Segment]) -> list[Segment]:
    """Remove stock outro phrases that appear HALLUCINATION_REPEATS+ times in one video."""
    from collections import Counter

    key = lambda t: re.sub(r"[\W_]+", "", t.lower())
    counts = Counter(key(s.text) for s in lines if key(s.text) in _HALLU)
    loops = {k for k, c in counts.items() if c >= HALLUCINATION_REPEATS}
    return [s for s in lines if key(s.text) not in loops]


# Speech sensitivity -> Silero VAD threshold (lower = catches quieter speech).
# Measured on a noisy test clip: 0.5 (library default) dropped half of the quiet lines
# before Whisper ever heard them; 0.15 recovered them without adding false lines.
SENSITIVITY = {"normal": 0.5, "high": 0.3, "max": 0.15}
WINDOW_SECONDS = 15        # window size when onnxruntime (VAD) is unavailable


def fixed_windows(audio, size: float = WINDOW_SECONDS) -> list[dict]:
    """Split audio into ~size-second windows cut at the quietest moment (no VAD needed)."""
    import numpy as np

    sr, n = SAMPLE_RATE, len(audio)
    frame, out, start = SAMPLE_RATE // 10, [], 0
    while start < n:
        target = start + int(size * sr)
        if target >= n - sr:
            out.append({"start": start / sr, "end": n / sr})
            break
        lo, hi = max(target - 2 * sr, start + sr), min(target + 2 * sr, n)
        k = (hi - lo) // frame
        energy = (audio[lo:lo + k * frame].reshape(k, frame) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * frame + frame // 2
        out.append({"start": start / sr, "end": cut / sr})
        start = cut
    return out


def transcribe(model, video: Path, language: str | None, task: str,
               resume: dict | None = None, on_chunk=None,
               vad_threshold: float = SENSITIVITY["max"]) -> tuple[list[Segment], str, float]:
    """Transcribe chunk by chunk. `on_chunk(segments, next_offset, lang)` is called after each
    chunk so an interrupted run can `resume` from the last finished chunk."""
    from faster_whisper import BatchedInferencePipeline

    # Always batched: Whisper decodes short pieces several at a time. (Plain sequential
    # decoding of long 30 s windows made large-v3-turbo skip most lines in testing.)
    batch_size = 16 if getattr(model, "srt_device", "cpu") == "cuda" else 8
    pipeline = BatchedInferencePipeline(model)
    common = dict(task=task, beam_size=5, batch_size=batch_size,
                  without_timestamps=False, word_timestamps=True)
    if vad_available():
        from faster_whisper.vad import VadOptions
        vad = VadOptions(threshold=vad_threshold, min_silence_duration_ms=160,
                         max_speech_duration_s=30)

        def run(audio, lang):
            return pipeline.transcribe(audio, language=lang, vad_filter=True,
                                       vad_parameters=vad, **common)
    else:
        print("  (onnxruntime not installed - using fixed 15 s windows instead of "
              "voice detection; works, but slightly less accurate on quiet speech)")

        def run(audio, lang):
            return pipeline.transcribe(audio, language=lang, vad_filter=False,
                                       clip_timestamps=fixed_windows(audio), **common)

    resume = resume or {}
    segments = [Segment(*x) for x in resume.get("segments", [])]
    start = float(resume.get("next_offset", 0.0))
    detected = resume.get("detected") or language
    duration = media_duration(video)

    label = "Transcribing" if task == "transcribe" else "Transcribing+translating"
    fmt = ("{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s audio [{elapsed}<{remaining}]"
           if duration else "{desc}: {n:.0f}s audio [{elapsed}]")
    end = start
    with tqdm(total=round(duration) or None, initial=round(start), unit="s", desc=label,
              bar_format=fmt) as bar:
        for offset, audio in audio_chunks(video, start):
            chunk_len = len(audio) / SAMPLE_RATE
            segs_iter, info = run(audio, detected)
            if detected is None:                       # detect once, keep for every chunk
                detected = info.language
                bar.write(f"Detected language: {lang_name(detected)} ({detected}), "
                          f"confidence {info.language_probability:.0%}")
            for seg in segs_iter:
                segments.extend(segment_to_lines(seg, offset, chunk_len))
                bar.update(max(0, round(offset + min(seg.end, chunk_len)) - bar.n))
            del audio
            end = offset + chunk_len
            bar.update(max(0, round(end) - bar.n))
            if on_chunk:
                on_chunk(segments, end, detected)
        if duration:
            bar.update(max(0, bar.total - bar.n))
    return fix_timing(drop_hallucination_loops(segments)), detected or "unknown", duration or end


# --------------------------------------------------------------------------- #
# 2. Translation providers
# --------------------------------------------------------------------------- #
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a professional subtitle translator. You translate video subtitle lines "
    "from {source} into {target}. Keep the meaning, tone and register of spoken "
    "dialogue; keep each line short and natural for subtitles; keep names as-is. "
    "Translate each line independently so it still matches its own timestamp - never "
    "merge, split, drop or reorder lines. If a line is already in {target}, return it "
    "unchanged. Reply only with JSON of the form "
    '{{"translations": [{{"id": <id>, "text": "<translation>"}}, ...]}} '
    "containing every id you were given."
)


def build_user_prompt(batch: list[tuple[int, str]], context: list[str]) -> str:
    parts = []
    if context:
        parts.append("Previous lines (context only, do not translate):\n"
                     + "\n".join(context))
    parts.append("Lines to translate (JSON):\n"
                 + json.dumps([{"id": i, "text": t} for i, t in batch], ensure_ascii=False))
    return "\n\n".join(parts)


def parse_translations(raw: str) -> dict[int, str]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)       # strip code fences
    match = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(match.group(0) if match else raw)
    items = data.get("translations", data) if isinstance(data, dict) else data
    return {int(item["id"]): str(item["text"]) for item in items}


class Translator:
    def __init__(self, provider: str, api_key: str, model: str, base_url: str | None = None):
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self._use_fallbacks = True
        elif provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)
        elif provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(provider)

    # -- provider calls ------------------------------------------------------ #
    def _call(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(system, user)
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        if self.provider == "gemini":
            from google.genai import types
            resp = self.client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                ),
            )
            return resp.text or ""
        raise ValueError(self.provider)

    def _call_anthropic(self, system: str, user: str) -> str:
        import anthropic

        params = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            # JSON schema guarantees parseable output with every id/text pair
            output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )
        if self.model.startswith(("claude-opus-5", "claude-sonnet-5", "claude-haiku-5",
                                  "claude-fable", "claude-opus-4-8", "claude-opus-4-7")):
            # Translation is a routine task: low effort = faster and cheaper.
            params["output_config"]["effort"] = "low"

        response = None
        if self._use_fallbacks:
            try:
                # If the model declines a line (e.g. a false-positive safety refusal on
                # violent movie dialogue), the API re-runs the request on a fallback model.
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **params,
                )
            except anthropic.BadRequestError:
                self._use_fallbacks = False       # model/account without fallbacks
        if response is None:
            try:
                response = self.client.messages.create(**params)
            except anthropic.BadRequestError:
                # older models without structured outputs / effort: plain prompt
                params.pop("output_config", None)
                response = self.client.messages.create(**params)

        if response.stop_reason == "refusal":
            raise RuntimeError("model declined to translate this batch")
        return next((b.text for b in response.content if b.type == "text"), "")

    # -- public ------------------------------------------------------------ #
    def translate(self, segments: list[Segment], source: str, target: str,
                  done: dict[int, str] | None = None, on_progress=None) -> list[Segment]:
        """Translate in parallel batches.

        done:        translations already finished in an earlier, interrupted run
        on_progress: called with the cumulative {line id: text} after each batch,
                     so progress can be saved and resumed
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        system = SYSTEM_PROMPT.format(source=source, target=target)
        results: dict[int, str] = dict(done or {})
        lock = threading.Lock()
        jobs = []
        for start in range(0, len(segments), BATCH_SIZE):
            batch = [(start + i + 1, s.text)
                     for i, s in enumerate(segments[start:start + BATCH_SIZE])
                     if (start + i + 1) not in results]
            if batch:
                context = [s.text for s in segments[max(0, start - CONTEXT_LINES):start]]
                jobs.append((batch, context))

        with tqdm(total=len(segments), initial=len(results), unit="line",
                  desc=f"Translating -> {target}",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} lines "
                             "[{elapsed}<{remaining}]") as bar, \
                ThreadPoolExecutor(max_workers=max(1, PARALLEL_REQUESTS)) as pool:
            futures = {pool.submit(self._translate_batch, system, b, c): b for b, c in jobs}
            try:
                for fut in as_completed(futures):
                    batch_result = fut.result()
                    with lock:
                        results.update(batch_result)
                        if on_progress:
                            on_progress(results)
                    bar.update(len(futures[fut]))
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

        return [Segment(s.start, s.end, results.get(i, s.text))
                for i, s in enumerate(segments, 1)]

    def _translate_batch(self, system: str, batch: list[tuple[int, str]],
                         context: list[str]) -> dict[int, str]:
        wanted = {i for i, _ in batch}
        result: dict[int, str] = {}
        pending = batch
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self._call(system, build_user_prompt(pending, context))
                result.update({k: v for k, v in parse_translations(raw).items()
                               if k in wanted and v.strip()})
            except Exception as exc:  # network, rate limit, bad JSON...
                code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if code in (401, 403) or "API_KEY_INVALID" in str(exc):
                    raise RuntimeError(f"API key rejected ({exc}). Run with --setup "
                                       "to enter a valid key.") from exc
                tqdm.write(f"  ! translation attempt {attempt} failed: {exc}")
                time.sleep(2 ** attempt)
            pending = [(i, t) for i, t in batch if i not in result]
            if not pending:
                break
        if pending:
            tqdm.write(f"  ! {len(pending)} line(s) kept untranslated after {MAX_RETRIES} attempts")
        return result


# --------------------------------------------------------------------------- #
# Configuration (first-run setup wizard, saved to config.json)
# --------------------------------------------------------------------------- #
CONFIG_DIR = Path(os.environ.get("SRT_CONFIG_DIR", Path.home() / ".srt-translator"))
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = dict(whisper_model="large-v3-turbo", device="auto", source_lang="auto",
                      sensitivity="max",
                      hf_token=None,
                      provider="none", model=None, api_key=None, base_url=None,
                      target="English", keep_original=False, overwrite=False)


def load_config() -> dict | None:
    try:
        return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text(encoding="utf-8"))}
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)   # the file may hold an API key
    except OSError:
        pass
    print(f"\nSettings saved to {CONFIG_FILE}")


def setup_wizard(current: dict | None) -> dict:
    cur = current or DEFAULT_CONFIG
    print("\n" + "-" * 64)
    print("  Setup  (saved for next time; re-run anytime with --setup)")
    print("-" * 64)

    print("\nWhisper speech-recognition model (bigger = more accurate, slower):")
    print("  tiny / base / small / medium / large-v3 / large-v3-turbo")
    print("  Tip: 'large-v3-turbo' is a good balance; 'large-v3' is the most accurate.")
    for name, (_, size) in WHISPER_REPOS.items():
        mark = "  [downloaded]" if model_is_downloaded(name) else ""
        print(f"    {name:<15} ~{size}{mark}")
    whisper_model = ask("Whisper model", cur["whisper_model"], WHISPER_MODELS)
    device = ask("Device (auto / cpu / cuda)", cur["device"], ["auto", "cpu", "cuda"])

    hf_token = cur.get("hf_token")
    if not model_is_downloaded(whisper_model):
        print("\nThe model is downloaded once from Hugging Face, then runs fully offline.")
        print("A Hugging Face token is NOT required (the models are public); it only")
        print("gives higher download rate limits. Get one at huggingface.co/settings/tokens")
        hint = " (Enter = keep saved token)" if hf_token else " (optional, Enter = skip)"
        hf_token = getpass.getpass(f"Hugging Face token, hidden{hint}: ").strip() or hf_token
        if ask_yes_no("Download the model now?", True):
            try:
                download_model(whisper_model, hf_token or os.environ.get("HF_TOKEN"))
            except Exception as exc:
                print(f"  ! Download failed ({exc}); it will be retried on the first run.")
    source_lang = ask("Default source language code (auto, ms, ar, zh, ja, ...)",
                      cur["source_lang"]).lower()
    print("\nSpeech detection sensitivity:")
    print("  max    - catches quiet and short speech (recommended)")
    print("  high   - in between")
    print("  normal - only clear speech (use if you get subtitles over music/noise)")
    sensitivity = ask("Sensitivity", cur.get("sensitivity", "max"), list(SENSITIVITY))

    print("\nTranslation engine:")
    for key, (_, label, model, _) in PROVIDERS.items():
        print(f"  {key}. {label}" + (f"  (default model: {model})" if model else ""))
    current_key = next((k for k, v in PROVIDERS.items() if v[0] == cur["provider"]), "1")
    provider, _, default_model, env_var = PROVIDERS[ask("Choose", current_key, list(PROVIDERS))]

    cfg = dict(DEFAULT_CONFIG, whisper_model=whisper_model, device=device,
               source_lang=source_lang, provider=provider, hf_token=hf_token,
               sensitivity=sensitivity)

    if provider in ("anthropic", "openai", "gemini"):
        same = cur["provider"] == provider
        cfg["target"] = ask("Target language", cur["target"])
        cfg["model"] = ask("Model name", cur["model"] if same and cur["model"] else default_model)
        if provider == "openai":
            cfg["base_url"] = input(
                "Custom base URL (blank = OpenAI; e.g. https://api.deepseek.com, "
                "http://host.docker.internal:11434/v1 for Ollama): ").strip() or None
        if os.environ.get(env_var):
            print(f"  ${env_var} is set in the environment; it will be used at runtime.")
        else:
            hint = " (Enter = keep saved key)" if same and cur["api_key"] else ""
            key = getpass.getpass(f"API key, hidden{hint}: ").strip()
            cfg["api_key"] = key or (cur["api_key"] if same else None)
            if not cfg["api_key"] and not cfg["base_url"]:
                print("  ! No API key given - translation will fail until you add one.")
        cfg["keep_original"] = ask_yes_no(
            "Also save original-language subtitles (<name>.<lang>.srt)?", cur["keep_original"])

    cfg["overwrite"] = ask_yes_no("Overwrite existing .srt files?", cur["overwrite"])
    save_config(cfg)
    return cfg


def describe(cfg: dict) -> str:
    engine = {"anthropic": "Claude", "openai": "OpenAI-compatible", "gemini": "Gemini",
              "whisper": "Whisper built-in (-> English)", "none": "no translation"}
    s = (f"Whisper: {cfg['whisper_model']} on {cfg['device']} | source: {cfg['source_lang']} | "
         f"sensitivity: {cfg.get('sensitivity', 'max')} | "
         f"engine: {engine[cfg['provider']]}")
    if cfg["model"]:
        s += f" ({cfg['model']}) -> {cfg['target']}"
    return s


def resolve_api_key(cfg: dict) -> str | None:
    env_var = next(v[3] for v in PROVIDERS.values() if v[0] == cfg["provider"])
    return os.environ.get(env_var or "") or cfg.get("api_key")


# --------------------------------------------------------------------------- #
# Host <-> container path mapping
# When run in Docker, a folder of the host (HOST_ROOT, e.g. /home/you or
# C:\\Users\\you) is mounted at HOST_MOUNT (/host). The user types host paths;
# we translate them to container paths and back for display.
# --------------------------------------------------------------------------- #
HOST_ROOT = os.environ.get("HOST_ROOT", "").strip()
HOST_MOUNT = os.environ.get("HOST_MOUNT", "/host")


def _norm(p: str) -> str:
    """Normalise a host path for prefix comparison (handles Windows paths)."""
    p = p.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", p):                  # C:/Users -> c:/users
        p = p.lower()
    return p.rstrip("/") or "/"


def clean_input_path(raw: str) -> str:
    raw = raw.strip().strip('"').strip("'")
    if not re.match(r"^[A-Za-z]:[\\/]", raw):          # unix shells escape spaces on drag & drop
        raw = re.sub(r"\\(.)", r"\1", raw)
    return raw


def to_container_path(raw: str) -> Path | None:
    raw = clean_input_path(raw)
    if HOST_ROOT:
        root, given = _norm(HOST_ROOT), _norm(raw)
        if given == root or given.startswith(root.rstrip("/") + "/"):
            rest = raw.replace("\\", "/")[len(root.rstrip("/")):].lstrip("/")
            return Path(HOST_MOUNT, rest)
        if raw.startswith("~"):                     # ~/Videos -> HOST_ROOT/Videos
            return Path(HOST_MOUNT, raw[1:].lstrip("/\\"))
        if not raw.startswith(HOST_MOUNT):
            return None                             # outside the shared folder
    return Path(raw).expanduser()


def to_host_path(path: Path) -> str:
    if HOST_ROOT:
        try:
            rest = path.relative_to(HOST_MOUNT)
        except ValueError:
            return str(path)
        sep = "\\" if re.match(r"^[A-Za-z]:", HOST_ROOT) else "/"
        root = HOST_ROOT.rstrip("/\\")
        return root + (sep + str(rest).replace("/", sep) if str(rest) != "." else "")
    return str(path)


def match_owner(path: Path, like: Path) -> None:
    """In Docker we run as root; give new .srt files the video owner's uid/gid."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        try:
            st = like.stat()
            os.chown(path, st.st_uid, st.st_gid)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
CACHE_VERSION = 3   # bump when subtitle timing logic changes so old caches are not reused


def cache_file(video: Path, cfg: dict, task: str) -> Path:
    """Progress cache keyed by the file (path, size, mtime) and transcription settings."""
    import hashlib
    st = video.stat()
    key = f"v{CACHE_VERSION}|{video.resolve()}|{st.st_size}|{int(st.st_mtime)}|" \
          f"{cfg['whisper_model']}|{cfg['source_lang']}|{task}|{cfg.get('sensitivity')}"
    return CONFIG_DIR / "cache" / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def process_video(video: Path, model, translator: Translator | None, cfg: dict,
                  index: int, total: int) -> None:
    out_path = video.with_suffix(".srt")
    print(f"\n[{index}/{total}] {to_host_path(video)}")
    if out_path.exists() and not cfg["overwrite"]:
        print(f"  Skipped: {out_path.name} already exists (enable overwrite in --setup).")
        return

    task = "translate" if cfg["provider"] == "whisper" else "transcribe"
    source = None if cfg["source_lang"] == "auto" else cfg["source_lang"]
    cache = cache_file(video, cfg, task)
    state: dict = {}
    try:
        state = json.loads(cache.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if "segments" in state:
        segments = [Segment(*s) for s in state["segments"]]
        detected, duration = state["detected"], state["duration"]
        print(f"  Resuming: reusing earlier transcription ({len(segments)} lines, "
              f"{lang_name(detected)}).")
    else:
        partial = state.get("partial")
        if partial:
            print(f"  Resuming transcription from {format_timestamp(partial['next_offset'])[:8]} "
                  "(earlier run was interrupted).")

        def chunk_done(segs: list[Segment], next_offset: float, lang: str | None) -> None:
            state["partial"] = {"segments": [[s.start, s.end, s.text] for s in segs],
                                "next_offset": next_offset, "detected": lang}
            save_json(cache, state)

        segments, detected, duration = transcribe(
            model, video, source, task, resume=partial, on_chunk=chunk_done,
            vad_threshold=SENSITIVITY.get(cfg.get("sensitivity", "max"), 0.15))
        state = {"segments": [[s.start, s.end, s.text] for s in segments],
                 "detected": detected, "duration": duration,
                 "translations": state.get("translations", {})}
        save_json(cache, state)
    if not segments:
        print("  No speech detected - nothing to write.")
        return
    print(f"  {len(segments)} subtitle lines from {duration / 60:.1f} min of audio.")

    if translator:
        if cfg["keep_original"]:
            orig_path = video.with_suffix(f".{detected}.srt")
            write_srt(segments, orig_path)
            match_owner(orig_path, video)
            print(f"  Original subtitles: {to_host_path(orig_path)}")
        target = cfg["target"]
        if target.lower() in (detected.lower(), lang_name(detected).lower()):
            print(f"  Audio is already in {target}; skipping translation.")
        else:
            tkey = f"{cfg['provider']}|{cfg['model']}|{target}"
            done = {int(k): v for k, v in state.get("translations", {}).get(tkey, {}).items()}
            if done:
                print(f"  Resuming translation: {len(done)}/{len(segments)} lines already done.")

            def checkpoint(results: dict[int, str]) -> None:
                state.setdefault("translations", {})[tkey] = results
                save_json(cache, state)

            segments = translator.translate(segments, lang_name(detected), target,
                                            done=done, on_progress=checkpoint)

    write_srt(segments, out_path)
    match_owner(out_path, video)
    print(f"  Done -> {to_host_path(out_path)}")


def prompt_for_videos(first: str | None) -> list[Path] | None:
    """Ask for a host path until it contains videos; None means quit."""
    hint = f" (inside {HOST_ROOT})" if HOST_ROOT else ""
    raw = first
    while True:
        if not raw:
            raw = input(f"\nPath to a video file or folder on your computer{hint}\n"
                        "(drag & drop works; Enter = quit): ").strip()
            if not raw:
                return None
        path = to_container_path(raw)
        if path is None:
            print(f"  That path is outside the shared folder {HOST_ROOT}.\n"
                  "  Restart with HOST_ROOT set to a folder that contains it "
                  "(see README).")
        else:
            videos = collect_videos(path)
            if videos:
                print(f"  Found {len(videos)} file(s).")
                return videos
            print(f"  No video/audio files found at '{raw}'.")
        raw = None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate translated .srt subtitles from videos.")
    parser.add_argument("path", nargs="?", help="video file or folder (asked if omitted)")
    parser.add_argument("--setup", action="store_true", help="run the setup wizard again")
    args = parser.parse_args()

    print("=" * 64)
    print("  SRT Translator - video audio -> translated subtitles")
    print("=" * 64)
    try:
        if not ensure_ffmpeg():
            sys.exit(1)
        cfg = load_config()
        if cfg is None:
            print("\nWelcome! No configuration found - let's set things up first.")
            cfg = setup_wizard(None)
        elif args.setup:
            cfg = setup_wizard(cfg)
        else:
            print(f"\nUsing saved settings: {describe(cfg)}")
            if not args.path and ask_yes_no("Change settings?", False):
                cfg = setup_wizard(cfg)

        translator = None
        if cfg["provider"] in ("anthropic", "openai", "gemini"):
            api_key = resolve_api_key(cfg)
            if not api_key and not cfg["base_url"]:
                sys.exit("No API key configured. Run with --setup or set the "
                         "provider's API key environment variable.")
            translator = Translator(cfg["provider"], api_key or "", cfg["model"],
                                    cfg["base_url"])

        model = None
        first = args.path
        while True:
            videos = prompt_for_videos(first)
            if not videos:
                break
            if model is None:                      # load once, reuse for every file
                model = load_whisper(cfg["whisper_model"], cfg["device"],
                                     cfg.get("hf_token") or os.environ.get("HF_TOKEN"))
            started = time.time()
            for i, video in enumerate(videos, 1):
                try:
                    process_video(video, model, translator, cfg, i, len(videos))
                except Exception as exc:
                    print(f"  ERROR processing {video.name}: {exc}")
            print(f"\nFinished in {(time.time() - started) / 60:.1f} min.")
            if args.path:                          # path given on the command line: one-shot
                break
            first = None
        print("Bye!")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")


if __name__ == "__main__":
    main()
