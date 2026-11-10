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

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

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
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
                    f"{seg.text.strip()}\n\n")


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
def load_whisper(model_size: str, device: str):
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"\nLoading Whisper model '{model_size}' on {device} ({compute_type})... "
          "(first run downloads the model)")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def load_audio(video: Path, sample_rate: int = 16000):
    """Decode the audio track to 16 kHz mono float32 using the ffmpeg CLI."""
    import shutil
    import subprocess

    import numpy as np

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found - install it (https://ffmpeg.org) or use the Docker image.")
    print("  Extracting audio...")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn",
           "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not read audio: {proc.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def transcribe(model, video: Path, language: str | None, task: str) -> tuple[list[Segment], str, float]:
    segments_iter, info = model.transcribe(
        load_audio(video),
        language=language,          # None = auto-detect
        task=task,                  # "transcribe" or "translate" (to English)
        vad_filter=True,            # skip silence / music, improves timestamps
        beam_size=5,
    )
    detected = info.language
    if language is None:
        print(f"Detected language: {lang_name(detected)} ({detected}), "
              f"confidence {info.language_probability:.0%}")

    duration = info.duration or 0.0
    segments: list[Segment] = []
    label = "Transcribing" if task == "transcribe" else "Transcribing+translating"
    with tqdm(total=round(duration, 1), unit="s", desc=label,
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s audio "
                         "[{elapsed}<{remaining}]") as bar:
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append(Segment(seg.start, seg.end, text))
            bar.update(round(min(seg.end, duration) - bar.n, 1))
        bar.update(bar.total - bar.n)
    return segments, detected, duration


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
    def translate(self, segments: list[Segment], source: str, target: str) -> list[Segment]:
        system = SYSTEM_PROMPT.format(source=source, target=target)
        out: list[Segment] = []
        total = len(segments)
        with tqdm(total=total, unit="line", desc=f"Translating -> {target}",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} lines "
                             "[{elapsed}<{remaining}]") as bar:
            for start in range(0, total, BATCH_SIZE):
                chunk = segments[start:start + BATCH_SIZE]
                batch = [(start + i + 1, s.text) for i, s in enumerate(chunk)]
                context = [s.text for s in segments[max(0, start - CONTEXT_LINES):start]]
                result = self._translate_batch(system, batch, context)
                for (idx, original), seg in zip(batch, chunk):
                    out.append(Segment(seg.start, seg.end, result.get(idx, original)))
                bar.update(len(chunk))
        return out

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
DEFAULT_VIDEO_DIR = os.environ.get("SRT_VIDEO_DIR", "")

DEFAULT_CONFIG = dict(whisper_model="large-v3-turbo", device="auto", source_lang="auto",
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
    whisper_model = ask("Whisper model", cur["whisper_model"], WHISPER_MODELS)
    device = ask("Device (auto / cpu / cuda)", cur["device"], ["auto", "cpu", "cuda"])
    source_lang = ask("Default source language code (auto, ms, ar, zh, ja, ...)",
                      cur["source_lang"]).lower()

    print("\nTranslation engine:")
    for key, (_, label, model, _) in PROVIDERS.items():
        print(f"  {key}. {label}" + (f"  (default model: {model})" if model else ""))
    current_key = next((k for k, v in PROVIDERS.items() if v[0] == cur["provider"]), "1")
    provider, _, default_model, env_var = PROVIDERS[ask("Choose", current_key, list(PROVIDERS))]

    cfg = dict(DEFAULT_CONFIG, whisper_model=whisper_model, device=device,
               source_lang=source_lang, provider=provider)

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
         f"engine: {engine[cfg['provider']]}")
    if cfg["model"]:
        s += f" ({cfg['model']}) -> {cfg['target']}"
    return s


def resolve_api_key(cfg: dict) -> str | None:
    env_var = next(v[3] for v in PROVIDERS.values() if v[0] == cfg["provider"])
    return os.environ.get(env_var or "") or cfg.get("api_key")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process_video(video: Path, model, translator: Translator | None, cfg: dict,
                  index: int, total: int) -> None:
    out_path = video.with_suffix(".srt")
    print(f"\n[{index}/{total}] {video}")
    if out_path.exists() and not cfg["overwrite"]:
        print(f"  Skipped: {out_path.name} already exists (enable overwrite in --setup).")
        return

    task = "translate" if cfg["provider"] == "whisper" else "transcribe"
    source = None if cfg["source_lang"] == "auto" else cfg["source_lang"]
    segments, detected, duration = transcribe(model, video, source, task)
    if not segments:
        print("  No speech detected - nothing to write.")
        return
    print(f"  {len(segments)} subtitle lines from {duration / 60:.1f} min of audio.")

    if translator:
        if cfg["keep_original"]:
            orig_path = video.with_suffix(f".{detected}.srt")
            write_srt(segments, orig_path)
            print(f"  Original subtitles: {orig_path}")
        target = cfg["target"]
        if target.lower() in (detected.lower(), lang_name(detected).lower()):
            print(f"  Audio is already in {target}; skipping translation.")
        else:
            segments = translator.translate(segments, lang_name(detected), target)

    write_srt(segments, out_path)
    print(f"  Done -> {out_path}")


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

        path_str = args.path
        while True:
            if not path_str:
                path_str = ask("\nVideo file or folder path", DEFAULT_VIDEO_DIR or None)
            videos = collect_videos(Path(path_str).expanduser())
            if videos:
                break
            print(f"  No video/audio files found at '{path_str}', try again.")
            path_str = None
        print(f"  Found {len(videos)} file(s).")

        translator = None
        if cfg["provider"] in ("anthropic", "openai", "gemini"):
            api_key = resolve_api_key(cfg)
            if not api_key and not cfg["base_url"]:
                sys.exit("No API key configured. Run with --setup or set the "
                         "provider's API key environment variable.")
            translator = Translator(cfg["provider"], api_key or "", cfg["model"],
                                    cfg["base_url"])
        model = load_whisper(cfg["whisper_model"], cfg["device"])

        overall_start = time.time()
        for i, video in enumerate(videos, 1):
            try:
                process_video(video, model, translator, cfg, i, len(videos))
            except Exception as exc:
                print(f"  ERROR processing {video.name}: {exc}")
        print(f"\nAll done in {(time.time() - overall_start) / 60:.1f} min.")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")


if __name__ == "__main__":
    main()
