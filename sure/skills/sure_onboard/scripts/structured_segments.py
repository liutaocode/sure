#!/usr/bin/env python3
"""Shared SD/SA-ASR contracts and bounded segment validation."""

from __future__ import annotations

import json
import math
import re
import wave
from pathlib import Path, PureWindowsPath
from typing import Any


STRUCTURED_TASKS = frozenset({"sd", "sa_asr"})
APPROVED_OUTPUT_FIELDS = frozenset({"segments", "num_speakers"})
SD_SEGMENT_FIELDS = frozenset({"speaker", "start", "end", "duration"})
SA_ASR_SEGMENT_FIELDS = frozenset({*SD_SEGMENT_FIELDS, "text"})
PUBLIC_INFERENCE_PARAMETERS = frozenset(
    {
        "batch_size",
        "beam_size",
        "clustering_threshold",
        "language",
        "max_speakers",
        "min_duration_off",
        "min_duration_on",
        "min_speakers",
        "num_speakers",
        "segmentation_threshold",
        "vad_threshold",
    }
)
REFERENCE_FIELD_NAMES = frozenset(
    {
        "answer",
        "expected",
        "ground_truth",
        "reference",
        "reference_annotation",
        "reference_segments",
        "reference_text",
        "rttm",
        "stm",
        "target",
        "target_segments",
        "target_text",
        "uem",
    }
)
STRUCTURED_EVIDENCE_FIELDS = frozenset(
    {
        "audio",
        "audio_is_silence",
        "dataset",
        "duration_sec",
        "id",
        "key",
        "language",
        "output",
        "sample_rate",
    }
)
STRUCTURED_MANIFEST_SAMPLE_FIELDS = frozenset(
    {
        "annotation_fields",
        "audio",
        "audio_is_silence",
        "audio_path",
        "duration_sec",
        "key",
        "sample_rate",
    }
)
URI_SCHEMES = frozenset({"file", "ftp", "git", "gs", "hf", "http", "https", "s3", "ssh"})
URI_PREFIX = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")


def canonical_task(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"speech_enhancement", "acoustic_noise_suppression"}:
        return "se"
    return normalized


def is_structured_task(value: Any) -> bool:
    return canonical_task(value) in STRUCTURED_TASKS


def structured_task_contract(value: Any) -> dict[str, Any]:
    task = canonical_task(value)
    if task not in STRUCTURED_TASKS:
        raise ValueError(f"task {value!r} does not use the structured segments contract")
    sa_asr = task == "sa_asr"
    segment_required = ["speaker", "start", "end"]
    if sa_asr:
        segment_required.append("text")
    segment_schema = {
        "type": "object",
        "required": segment_required,
        "properties": {
            "speaker": {"type": "string", "minLength": 1},
            "start": {"type": "number", "minimum": 0},
            "end": {"type": "number", "exclusiveMinimum": 0},
            "duration": {"type": "number", "exclusiveMinimum": 0},
            **({"text": {"type": "string", "minLength": 1}} if sa_asr else {}),
        },
        "additionalProperties": False,
    }
    segments_type = (
        "array<{speaker:string,start:number,end:number,text:string}>"
        if sa_asr
        else "array<{speaker:string,start:number,end:number}>"
    )
    return {
        "tool_name": "transcribe_with_speakers" if sa_asr else "diarize",
        "predict_method": "transcribe_with_speakers" if sa_asr else "diarize",
        "input_fields": ["audio_path"],
        "public_inference_parameters": sorted(PUBLIC_INFERENCE_PARAMETERS),
        "io_contract": {
            "input_type": "audio_path",
            "output_type": "structured_segments",
            "input": {"audio_path": "string"},
            "output": {
                "segments": segments_type,
                "num_speakers": "optional integer",
            },
            "primary_field": "segments",
            "required_fields": ["segments"],
            "nonempty_fields": ["segments"] if sa_asr else [],
            "allow_empty_primary": not sa_asr,
            "json_serializable": True,
            "allow_empty_segments": False if sa_asr else "silence_only",
            "approved_output_fields": sorted(APPROVED_OUTPUT_FIELDS),
            "segment_schema": segment_schema,
        },
    }


def pcm_wav_info(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise ValueError("audio must be uncompressed PCM WAV")
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            if channels < 1 or sample_width not in {1, 2, 3, 4} or sample_rate < 1 or frame_count < 1:
                raise ValueError("audio must be a non-empty PCM WAV")
            frames = handle.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError(f"audio must be a readable PCM WAV: {path}: {exc}") from exc
    expected_bytes = frame_count * channels * sample_width
    if len(frames) != expected_bytes:
        raise ValueError(
            f"audio PCM data is truncated: expected {expected_bytes} bytes, read {len(frames)}"
        )
    silence_byte = b"\x80" if sample_width == 1 else b"\x00"
    return {
        "duration_sec": frame_count / sample_rate,
        "sample_rate": sample_rate,
        "audio_is_silence": bool(frames) and frames == silence_byte * len(frames),
    }


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _looks_like_absolute_path_or_uri(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    match = URI_PREFIX.match(stripped)
    if match and (match.group(1).lower() in URI_SCHEMES or "://" in stripped):
        return True
    return Path(stripped).is_absolute() or PureWindowsPath(stripped).is_absolute() or stripped.startswith("\\\\")


def _unsafe_string_paths(value: Any, path: str = "output") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and _looks_like_absolute_path_or_uri(value):
        found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_unsafe_string_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_unsafe_string_paths(item, f"{path}[{index}]"))
    return found


def validate_segments(
    segments: Any,
    *,
    task: Any,
    duration_sec: float,
    audio_is_silence: bool,
) -> list[str]:
    normalized_task = canonical_task(task)
    if normalized_task not in STRUCTURED_TASKS:
        return [f"task {task!r} does not use structured segments"]
    if not isinstance(segments, list):
        return ["segments must be an array"]
    if not segments:
        if normalized_task == "sd" and audio_is_silence:
            return []
        if normalized_task == "sd":
            return ["empty SD segments are allowed only for pure-silence audio"]
        return ["SA-ASR segments must be non-empty"]

    violations: list[str] = []
    for index, segment in enumerate(segments, 1):
        prefix = f"segment {index}"
        if not isinstance(segment, dict):
            violations.append(f"{prefix} must be an object")
            continue
        approved_fields = SA_ASR_SEGMENT_FIELDS if normalized_task == "sa_asr" else SD_SEGMENT_FIELDS
        unknown_fields = sorted(str(key) for key in segment if key not in approved_fields)
        if unknown_fields:
            violations.append(f"{prefix} contains unapproved field(s): " + ", ".join(unknown_fields))
        speaker = segment.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            violations.append(f"{prefix} speaker must be a non-empty string")
        start = segment.get("start")
        end = segment.get("end")
        if not _finite_number(start):
            violations.append(f"{prefix} start must be a finite number")
        elif float(start) < 0:
            violations.append(f"{prefix} start must be >= 0")
        if not _finite_number(end):
            violations.append(f"{prefix} end must be a finite number")
        elif _finite_number(start) and float(end) <= float(start):
            violations.append(f"{prefix} end must be greater than start")
        if _finite_number(end) and float(end) > duration_sec + 1e-6:
            violations.append(
                f"{prefix} end {float(end):.6f} exceeds WAV duration {duration_sec:.6f}"
            )
        declared_duration = segment.get("duration")
        if declared_duration is not None:
            if not _finite_number(declared_duration) or float(declared_duration) <= 0:
                violations.append(f"{prefix} duration must be a finite positive number")
            elif _finite_number(start) and _finite_number(end) and not math.isclose(
                float(declared_duration), float(end) - float(start), rel_tol=0, abs_tol=1e-3
            ):
                violations.append(f"{prefix} duration must equal end - start")
        if normalized_task == "sa_asr":
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                violations.append(f"{prefix} text must be a non-empty string for SA-ASR")
    return violations


def _forbidden_output_paths(value: Any, path: str = "output") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child = f"{path}.{key}"
            if (
                normalized in REFERENCE_FIELD_NAMES
                or normalized.startswith("reference_")
                or normalized == "path"
                or normalized.endswith("_path")
            ):
                found.append(child)
            found.extend(_forbidden_output_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_output_paths(item, f"{path}[{index}]"))
    return found


def validate_structured_output(
    output: Any,
    *,
    task: Any,
    duration_sec: float,
    audio_is_silence: bool,
) -> list[str]:
    if not isinstance(output, dict):
        return ["structured prediction must be an object"]
    unknown_fields = sorted(str(key) for key in output if key not in APPROVED_OUTPUT_FIELDS)
    violations = [
        f"model output must not expose reference or path field {path}"
        for path in _forbidden_output_paths(output)
    ]
    if unknown_fields:
        violations.append("structured prediction contains unapproved field(s): " + ", ".join(unknown_fields))
    violations.extend(
        f"structured prediction contains absolute path or URI at {path}"
        for path in _unsafe_string_paths(output)
    )
    violations.extend(
        validate_segments(
            output.get("segments"),
            task=task,
            duration_sec=duration_sec,
            audio_is_silence=audio_is_silence,
        )
    )
    num_speakers = output.get("num_speakers")
    if num_speakers is not None:
        if isinstance(num_speakers, bool) or not isinstance(num_speakers, int) or num_speakers < 0:
            violations.append("num_speakers must be a non-negative integer when provided")
        elif isinstance(output.get("segments"), list):
            observed = {
                segment.get("speaker").strip()
                for segment in output["segments"]
                if isinstance(segment, dict)
                and isinstance(segment.get("speaker"), str)
                and segment.get("speaker").strip()
            }
            if num_speakers != len(observed):
                violations.append("num_speakers must equal the number of distinct output speakers")
    try:
        json.dumps(output, allow_nan=False)
    except (TypeError, ValueError):
        violations.append("structured prediction must contain finite JSON data")
    return violations


def validate_structured_rows(
    rows: Any,
    *,
    task: Any,
    samples: Any,
    fixture_root: Path | None = None,
) -> list[str]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5:
        return ["structured validation requires 1-5 output rows"]
    if not isinstance(samples, list) or len(samples) != len(rows):
        return ["structured output rows must match fixture manifest samples"]

    sample_by_key: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    for index, sample in enumerate(samples, 1):
        if not isinstance(sample, dict):
            violations.append(f"fixture sample {index} must be an object")
            continue
        unexpected = sorted(str(key) for key in sample if key not in STRUCTURED_MANIFEST_SAMPLE_FIELDS)
        if unexpected:
            violations.append(
                f"fixture sample {index} exposes unapproved field(s): " + ", ".join(unexpected)
            )
        if sample.get("annotation_fields") != ["segments"]:
            violations.append(f"fixture sample {index} annotation_fields must equal ['segments']")
        key = sample.get("key")
        if not isinstance(key, str) or not key.strip():
            violations.append(f"fixture sample {index} requires a non-empty key")
        elif key in sample_by_key:
            violations.append(f"fixture sample {index} duplicates key {key!r}")
        else:
            sample_by_key[key] = sample

    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            violations.append(f"structured output row {index} must be an object")
            continue
        unexpected = sorted(str(key) for key in row if key not in STRUCTURED_EVIDENCE_FIELDS)
        if unexpected:
            violations.append(
                f"structured output row {index} exposes non-portable/reference field(s): "
                + ", ".join(unexpected)
            )
        row_id = row.get("id")
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 1:
            violations.append(f"structured output row {index} id must be a positive integer")
        key = row.get("key")
        prefix = f"structured output row {index}"
        if not isinstance(key, str) or not key.strip():
            violations.append(f"{prefix} requires a non-empty key")
            continue
        if _looks_like_absolute_path_or_uri(key):
            violations.append(f"{prefix} key must not contain an absolute path or URI")
        if key in seen:
            violations.append(f"{prefix} duplicates key {key!r}")
            continue
        seen.add(key)
        sample = sample_by_key.get(key)
        if sample is None:
            violations.append(f"{prefix} key {key!r} is absent from the fixture manifest")
            continue
        audio = row.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            violations.append(f"{prefix} {key!r} requires a relative audio evidence path")
        else:
            audio_path = Path(audio)
            if _looks_like_absolute_path_or_uri(audio) or ".." in audio_path.parts:
                violations.append(f"{prefix} {key!r} audio evidence path must be portable")
            elif audio != sample.get("audio"):
                violations.append(f"{prefix} {key!r} audio evidence disagrees with fixture manifest")
        for field in ("dataset", "language"):
            value = row.get(field)
            if value is not None and not isinstance(value, str):
                violations.append(f"{prefix} {key!r} {field} must be a string or null")
            elif isinstance(value, str) and _looks_like_absolute_path_or_uri(value):
                violations.append(f"{prefix} {key!r} {field} must not contain an absolute path or URI")
        sample_audio_path = sample.get("audio_path")
        if not isinstance(sample_audio_path, str) or not sample_audio_path.strip():
            violations.append(f"fixture sample {key!r} lacks resolved audio_path")
            continue
        resolved_sample_audio = Path(sample_audio_path).expanduser().resolve()
        if fixture_root is not None and not resolved_sample_audio.is_relative_to(fixture_root.resolve()):
            violations.append(f"fixture sample {key!r} audio_path escapes staged fixture root")
            continue
        try:
            wav_info = pcm_wav_info(resolved_sample_audio)
        except ValueError as exc:
            violations.append(f"fixture sample {key!r} has invalid WAV evidence: {exc}")
            continue
        duration = wav_info["duration_sec"]
        silence = wav_info["audio_is_silence"]
        for field in ("duration_sec", "sample_rate", "audio_is_silence"):
            if sample.get(field) != wav_info[field]:
                violations.append(f"fixture sample {key!r} {field} disagrees with the WAV")
        if row.get("duration_sec") != duration:
            violations.append(f"{prefix} {key!r} duration evidence disagrees with fixture manifest")
        if row.get("sample_rate") != sample.get("sample_rate"):
            violations.append(f"{prefix} {key!r} sample-rate evidence disagrees with fixture manifest")
        if row.get("audio_is_silence") is not silence:
            violations.append(f"{prefix} {key!r} silence evidence disagrees with fixture manifest")
        violations.extend(
            f"{prefix} {key!r}: {item}"
            for item in validate_structured_output(
                row.get("output"),
                task=task,
                duration_sec=float(duration),
                audio_is_silence=silence,
            )
        )
    expected_keys = [
        sample.get("key")
        for sample in samples
        if isinstance(sample, dict) and isinstance(sample.get("key"), str) and sample.get("key").strip()
    ]
    observed_keys = [
        row.get("key")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("key"), str) and row.get("key").strip()
    ]
    if observed_keys != expected_keys:
        violations.append("structured output rows must preserve fixture key order")
    if seen != set(sample_by_key):
        missing = sorted(set(sample_by_key) - seen)
        if missing:
            violations.append("structured outputs are missing fixture key(s): " + ", ".join(missing))
    return violations
