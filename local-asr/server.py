from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import gc
from pathlib import Path
from typing import Iterator

import httpx
try:
    import mlx_whisper
except ImportError:  # pragma: no cover - optional fallback
    mlx_whisper = None
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get(
        "FLUENTFRAME_QWEN_ASR_MODEL",
        str(PROJECT_ROOT / "models" / "Qwen3-ASR-0.6B"),
    )
)
ALIGNER_PATH = Path(os.environ.get("FLUENTFRAME_ALIGNER_MODEL", str(PROJECT_ROOT / "models" / "Qwen3-ForcedAligner-0.6B")))
WHISPER_MODEL_PATH = Path(
    os.environ.get(
        "FLUENTFRAME_WHISPER_MODEL",
        str(PROJECT_ROOT / "models" / "whisper-large-v3-mlx"),
    )
)
FFMPEG = os.environ.get("FLUENTFRAME_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = os.environ.get("FLUENTFRAME_FFPROBE", shutil.which("ffprobe") or "ffprobe")
CHUNK_SECONDS = float(os.environ.get("FLUENTFRAME_CHUNK_SECONDS", "20"))
QWEN_DEVICE = os.environ.get("FLUENTFRAME_QWEN_DEVICE", "auto")
MAX_UPLOAD_BYTES = int(os.environ.get("FLUENTFRAME_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
LOCAL_TRANSLATOR_PYTHON = os.environ.get(
    "FLUENTFRAME_TRANSLATOR_PYTHON",
    str(PROJECT_ROOT / "local-asr" / ".venv-mlx" / "bin" / "python"),
)

app = FastAPI(title="FluentFrame Local ASR", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_qwen_model = None


def qwen_ready() -> bool:
    return (MODEL_PATH / "config.json").exists() and (MODEL_PATH / "model.safetensors").exists()


def load_qwen_model():
    global _qwen_model
    if _qwen_model is None:
        import torch
        from qwen_asr import Qwen3ASRModel

        device = resolve_qwen_device(torch)

        kwargs = {
            "dtype": torch.float32 if device in {"mps", "cpu"} else torch.bfloat16,
            "device_map": device,
            "max_inference_batch_size": 1,
            "max_new_tokens": 256,
        }
        if ALIGNER_PATH.exists():
            kwargs["forced_aligner"] = str(ALIGNER_PATH)
            kwargs["forced_aligner_kwargs"] = {
                "dtype": torch.float32 if device in {"mps", "cpu"} else torch.bfloat16,
                "device_map": device,
            }
        _qwen_model = Qwen3ASRModel.from_pretrained(str(MODEL_PATH), **kwargs)
    return _qwen_model


def resolve_qwen_device(torch_module) -> str:
    if QWEN_DEVICE != "auto":
        return QWEN_DEVICE
    return "mps" if torch_module.backends.mps.is_available() else "cpu"


@app.get("/health")
def health() -> dict:
    ready = qwen_ready()
    return {
        "ok": ready,
        "engine": "qwen3-asr",
        "model": "Qwen3-ASR-0.6B + ForcedAligner-0.6B",
        "modelPath": str(MODEL_PATH),
        "alignerPath": str(ALIGNER_PATH),
        "streaming": "chunked-qwen",
        "device": QWEN_DEVICE,
    }


@app.post("/transcribe")
async def transcribe(request: Request):
    content_length = int(request.headers.get("content-length") or 0)
    if content_length and content_length > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "文件过大。"}, status_code=413)
    if not qwen_ready() and not ((WHISPER_MODEL_PATH / "config.json").exists() and (WHISPER_MODEL_PATH / "weights.npz").exists()):
        return JSONResponse({"error": f"未找到本地模型：{MODEL_PATH}"}, status_code=503)

    suffix = Path(request.headers.get("x-file-name") or "lesson.mp4").suffix or ".mp4"
    handle = tempfile.NamedTemporaryFile(prefix="fluentframe-", suffix=suffix, delete=False)
    source_path = Path(handle.name)
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > MAX_UPLOAD_BYTES:
                handle.close()
                source_path.unlink(missing_ok=True)
                return JSONResponse({"error": "文件过大。"}, status_code=413)
            handle.write(chunk)
    finally:
        handle.close()

    if received == 0:
        source_path.unlink(missing_ok=True)
        return JSONResponse({"error": "没有收到视频文件。"}, status_code=400)

    return StreamingResponse(
        transcribe_stream(source_path),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def event(kind: str, **payload) -> str:
    return json.dumps({"type": kind, **payload}, ensure_ascii=False) + "\n"


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    candidates = [payload.get("format", {}).get("duration")]
    candidates.extend(stream.get("duration") for stream in payload.get("streams", []))
    for value in candidates:
        if value not in (None, "N/A"):
            return max(float(value), 0.1)
    raise ValueError("无法读取媒体时长")


def extract_chunk(source: Path, output: Path, start: float, duration: float) -> None:
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def transcribe_chunk(wav_path: Path, chunk_start: float) -> list[dict]:
    """Return normalized segments. Qwen timestamps are produced by ForcedAligner."""
    if qwen_ready():
        import soundfile as sf

        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        audio_duration = len(audio) / sample_rate
        result = load_qwen_model().transcribe(audio=(audio, sample_rate), language=None, return_time_stamps=True)[0]
        words = []
        for index, item in enumerate(getattr(result, "time_stamps", None) or []):
            value = str(getattr(item, "text", "") or "").strip()
            if not value:
                continue
            relative_start = min(audio_duration, max(0.0, float(getattr(item, "start_time", 0) or 0)))
            relative_end = min(audio_duration, max(relative_start, float(getattr(item, "end_time", relative_start) or relative_start)))
            if relative_end <= relative_start:
                relative_end = min(audio_duration, relative_start + 0.08)
            start = chunk_start + relative_start
            end = chunk_start + relative_end
            words.append({"text": value, "start": start, "end": end, "highlight": choose_highlight(value, index)})
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            return []
        language = normalize_language(str(getattr(result, "language", "") or ""), text)
        if words:
            return split_aligned_words(words, language)
        return [{"text": text, "start": chunk_start, "end": chunk_start + CHUNK_SECONDS,
                 "words": words, "language": language}]

    if mlx_whisper is None:
        raise RuntimeError("Qwen3-ASR 依赖不可用，且没有 Whisper 降级引擎")
    result = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=str(WHISPER_MODEL_PATH), language=None,
                                    task="transcribe", word_timestamps=True, verbose=None, temperature=0.0)
    output = []
    for segment in result.get("segments") or []:
        text = (segment.get("text") or "").strip()
        if text:
            output.append({"text": text, "start": chunk_start + float(segment.get("start") or 0),
                           "end": chunk_start + float(segment.get("end") or 0), "raw": segment,
                           "language": normalize_language(str(result.get("language") or ""), text)})
    return output


def normalize_language(language: str, text: str) -> str:
    value = language.lower()
    if "chinese" in value or value in {"zh", "yue"}:
        return "zh"
    if "english" in value or value == "en":
        return "en"
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin_chars = sum(char.isascii() and char.isalpha() for char in text)
    return "zh" if chinese_chars >= max(2, latin_chars // 2) else "en"


def split_aligned_words(words: list[dict], language: str) -> list[dict]:
    """Split an aligned chunk on natural pauses, keeping true source timestamps."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        gap = (next_word["start"] - word["end"]) if next_word else 99.0
        span = word["end"] - current[0]["start"]
        enough_content = len(current) >= (5 if language == "zh" else 3)
        if next_word is None or (enough_content and gap >= 0.75) or span >= 10.0:
            groups.append(current)
            current = []
    output = []
    for group in groups:
        if language == "zh":
            group = merge_chinese_words(group)
        joiner = "" if language == "zh" else " "
        text = joiner.join(word["text"] for word in group).strip()
        if text:
            output.append({"text": text, "start": group[0]["start"], "end": group[-1]["end"],
                           "words": group, "language": language})
    return output


def merge_chinese_words(words: list[dict]) -> list[dict]:
    """Turn character-level alignment into readable Chinese word-level highlighting."""
    import jieba

    characters: list[dict] = []
    for word in words:
        text = word["text"]
        duration = max(0.01, word["end"] - word["start"])
        for index, char in enumerate(text):
            characters.append({"text": char, "start": word["start"] + duration * index / len(text),
                               "end": word["start"] + duration * (index + 1) / len(text)})
    plain = "".join(item["text"] for item in characters)
    tokens = [token for token in jieba.cut(plain, cut_all=False) if token]
    output: list[dict] = []
    cursor = 0
    for index, token in enumerate(tokens):
        token_chars = characters[cursor : cursor + len(token)]
        cursor += len(token)
        if not token_chars:
            continue
        output.append({"text": token, "start": round(token_chars[0]["start"], 3),
                       "end": round(token_chars[-1]["end"], 3), "highlight": choose_highlight(token, index)})
    return output or words


def translated_english_words(text: str, start: float, end: float) -> list[dict]:
    parts = text.split()
    unit = (end - start) / max(len(parts), 1)
    return [{"text": word, "start": round(start + unit * index, 3),
             "end": round(start + unit * (index + 1), 3), "highlight": choose_highlight(word, index)}
            for index, word in enumerate(parts)]


def transcribe_stream(source_path: Path) -> Iterator[str]:
    caption_id = 1
    all_captions: list[dict] = []
    use_cloud_translation = cloud_translation_available()
    try:
        duration = probe_duration(source_path)
        yield event(
            "status",
            stage="loading",
            message="正在加载本地 Qwen3-ASR…",
            progress=3,
            model="Qwen3-ASR-0.6B + ForcedAligner-0.6B",
            duration=duration,
        )

        with tempfile.TemporaryDirectory(prefix="fluentframe-chunks-") as chunk_dir:
            start = 0.0
            while start < duration:
                chunk_duration = min(CHUNK_SECONDS, duration - start)
                wav_path = Path(chunk_dir) / f"chunk-{int(start):06d}.wav"
                yield event(
                    "status",
                    stage="transcribing",
                    message=f"正在识别 {format_clock(start)} — {format_clock(start + chunk_duration)}",
                    progress=min(92, 5 + (start / duration) * 87),
                    model="Qwen3-ASR-0.6B + ForcedAligner-0.6B",
                )
                extract_chunk(source_path, wav_path, start, chunk_duration)
                result = transcribe_chunk(wav_path, start)

                chunk_captions: list[dict] = []
                for segment in result:
                    text = (segment.get("text") or "").strip()
                    seg_start = float(segment.get("start") or 0)
                    seg_end = float(segment.get("end") or 0)
                    if not text or seg_end <= seg_start:
                        continue
                    words = normalize_words(segment.get("raw", segment), text, seg_start, seg_end, start, precomputed=segment.get("words"))
                    language = segment.get("language", "en")
                    caption = {
                        "id": caption_id,
                        "start": round(seg_start, 3),
                        "end": round(seg_end, 3),
                        "english": text if language == "en" else "",
                        "chinese": text if language == "zh" else "",
                        "words": words if language == "en" else [],
                        "sourceWords": words,
                        "sourceLanguage": language,
                    }
                    chunk_captions.append(caption)
                    all_captions.append(caption)
                    yield event("segment", caption=caption)
                    caption_id += 1

                if chunk_captions and use_cloud_translation:
                    for source_language in ("en", "zh"):
                        items = [item for item in chunk_captions if item["sourceLanguage"] == source_language]
                        if not items:
                            continue
                        translations = translate_lines(
                            [item["english"] if source_language == "en" else item["chinese"] for item in items],
                            "zh" if source_language == "en" else "en",
                        )
                        for item, translation in zip(items, translations):
                            apply_translation(item, translation)
                            yield translation_event(item)

                start += chunk_duration

        missing_translations = [item for item in all_captions if not item["chinese"] or not item["english"]]
        if missing_translations:
            yield event(
                "status",
                stage="translation",
                message="原文识别已完成，正在使用本地 Qwen 生成中英双向翻译…",
                progress=94,
                model="Qwen3-ASR + ForcedAligner + Qwen2.5-3B",
            )
            release_asr()
            for source_language in ("en", "zh"):
                items = [item for item in missing_translations if item["sourceLanguage"] == source_language]
                if not items:
                    continue
                for offset in range(0, len(items), 6):
                    batch = items[offset : offset + 6]
                    yield event("status", stage="translation",
                                message=f"正在翻译 {offset + 1}–{min(offset + len(batch), len(items))} / {len(items)}…",
                                progress=min(99, 94 + ((offset + len(batch)) / max(len(items), 1)) * 5),
                                model="Qwen3-ASR + ForcedAligner + Qwen2.5-3B")
                    lines = [item["english"] if source_language == "en" else item["chinese"] for item in batch]
                    target_language = "zh" if source_language == "en" else "en"
                    try:
                        translations = translate_local(lines, target_language)
                    except RuntimeError:
                        # A batch can fail because Metal compilation or memory pressure is transient.
                        # Retry each subtitle independently so one bad batch cannot erase the rest.
                        translations = [translate_local([line], target_language)[0] for line in lines]
                    for item, translation in zip(batch, translations):
                        apply_translation(item, translation)
                        yield translation_event(item)

        yield event(
            "done",
            message=f"识别完成，共 {caption_id - 1} 句。",
            progress=100,
            count=caption_id - 1,
            model="Qwen3-ASR-0.6B + ForcedAligner-0.6B",
        )
    except GeneratorExit:
        return
    except Exception as exc:
        yield event("error", message=f"本地识别失败：{exc}")
    finally:
        source_path.unlink(missing_ok=True)


def normalize_words(segment: dict, text: str, seg_start: float, seg_end: float, chunk_start: float, precomputed: list[dict] | None = None) -> list[dict]:
    if precomputed:
        return precomputed
    raw_words = segment.get("words") or []
    if raw_words:
        output = []
        for index, word in enumerate(raw_words):
            value = (word.get("word") or "").strip()
            if not value:
                continue
            output.append(
                {
                    "text": value,
                    "start": round(chunk_start + float(word.get("start") or 0), 3),
                    "end": round(chunk_start + float(word.get("end") or 0), 3),
                    "highlight": choose_highlight(value, index),
                }
            )
        if output:
            return output

    parts = text.split()
    unit = (seg_end - seg_start) / max(len(parts), 1)
    return [
        {
            "text": word,
            "start": round(seg_start + unit * index, 3),
            "end": round(seg_start + unit * (index + 1), 3),
            "highlight": choose_highlight(word, index),
        }
        for index, word in enumerate(parts)
    ]


def choose_highlight(word: str, index: int) -> str:
    clean = "".join(char for char in word if char.isalpha() or char in "'-")
    if len(clean) >= 8:
        return "blue"
    if len(clean) >= 6 and index % 2 == 0:
        return "mint"
    if len(clean) >= 5 and index % 3 == 0:
        return "orange"
    return "none"


def apply_translation(item: dict, translation: str) -> None:
    if item["sourceLanguage"] == "zh":
        item["english"] = translation
        item["words"] = translated_english_words(translation, item["start"], item["end"])
    else:
        item["chinese"] = translation


def translation_event(item: dict) -> str:
    return event("translation", id=item["id"], english=item["english"], chinese=item["chinese"], words=item["words"])


def translate_lines(lines: list[str], target_language: str) -> list[str]:
    settings = load_settings()
    api_key = settings.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ["" for _ in lines]
    model = settings.get("OPENAI_TEXT_MODEL") or os.environ.get("OPENAI_TEXT_MODEL") or "gpt-5-mini"
    schema = {
        "type": "object",
        "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
        "required": ["translations"],
        "additionalProperties": False,
    }
    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "store": False,
                "instructions": (
                    "Translate each Simplified Chinese subtitle into accurate, natural English. Preserve names and specialist terms such as 八字命理. "
                    if target_language == "en" else
                    "Translate each English subtitle into accurate, natural Simplified Chinese. "
                ) + "Preserve order and return only the schema.",
                "input": json.dumps(lines, ensure_ascii=False),
                "text": {"format": {"type": "json_schema", "name": "subtitle_translations", "strict": True, "schema": schema}},
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        output_text = next(
            content["text"]
            for item in payload.get("output", [])
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
        translations = json.loads(output_text).get("translations", [])
        return [str(translations[index]) if index < len(translations) else "" for index in range(len(lines))]
    except Exception:
        return ["" for _ in lines]


def cloud_translation_available() -> bool:
    settings = load_settings()
    return bool(settings.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def release_whisper() -> None:
    try:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        mx.clear_cache()
    except Exception:
        pass


def release_asr() -> None:
    global _qwen_model
    _qwen_model = None
    release_whisper()
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def translate_local(lines: list[str], target_language: str) -> list[str]:
    translator = PROJECT_ROOT / "local-asr" / "translate_mlx.py"
    model = PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct-4bit" / "config.json"
    if not Path(LOCAL_TRANSLATOR_PYTHON).exists() or not model.exists():
        return ["" for _ in lines]
    try:
        result = subprocess.run(
            [LOCAL_TRANSLATOR_PYTHON, str(translator)],
            input=json.dumps({"lines": lines, "target": target_language}, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        translations = json.loads(result.stdout).get("translations", [])
        if len(translations) != len(lines) or any(not str(item).strip() for item in translations):
            raise ValueError(f"翻译返回数量不完整：期望 {len(lines)}，实际 {len(translations)}")
        return [str(item).strip() for item in translations]
    except Exception as exc:
        raise RuntimeError(f"本地 MLX 翻译失败：{exc}") from exc


def load_settings() -> dict[str, str]:
    output: dict[str, str] = {}
    path = PROJECT_ROOT / ".env.local"
    if not path.exists():
        return output
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        output[key.strip()] = value.strip().strip("\"'")
    return output


def format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("FLUENTFRAME_ASR_PORT", "8766")))
