#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import wave
from pathlib import Path, PureWindowsPath
from typing import Any


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
ANNOTATION_FIELDS = ("ground_truth", "target_text", "text", "segments", "label", "intent")
KWS_OPERATING_THRESHOLD = 0.5
SPEAKER_TASKS = {"sd", "sa_asr"}
STRUCTURED_FIXTURE_TASKS = {"kws", "sa_asr", "sd", "se"}
SPEAKER_OUTPUT_FIELDS = frozenset({"segments", "num_speakers"})
SD_SEGMENT_FIELDS = frozenset({"speaker", "start", "end", "duration"})
SA_ASR_SEGMENT_FIELDS = frozenset({*SD_SEGMENT_FIELDS, "text"})
URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_fixture(resolved: dict, task: str) -> Path:
    explicit = resolved.get("fixture_path")
    if explicit:
        path = Path(str(explicit))
        if task in STRUCTURED_FIXTURE_TASKS and (
            path.is_dir() or (path.is_file() and path.name == "gt.jsonl")
        ):
            return path
        if task not in STRUCTURED_FIXTURE_TASKS and path.is_file():
            return path
        expected = "a directory or gt.jsonl" if task in STRUCTURED_FIXTURE_TASKS else "a file"
        raise ValueError(f"fixture must be {expected}: {path}")
    build_context = Path(str(resolved["build_context"]))
    if task in STRUCTURED_FIXTURE_TASKS:
        structured_candidates = [
            build_context / "examples" / task / "gt.jsonl",
            build_context / "fixture" / task / "gt.jsonl",
            build_context / "fixtures" / task / "gt.jsonl",
        ]
        matches = [candidate for candidate in structured_candidates if candidate.is_file()]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"{task.upper()} fixture could not be selected unambiguously; "
            "pass fixture=/absolute/path/to/gt.jsonl"
        )
    preferred = [
        build_context / "examples" / "smoke.wav",
        build_context / "examples" / "smoke.flac",
        build_context / "smoke.wav",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    examples = build_context / "examples"
    matches = sorted(path for path in examples.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES) if examples.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise ValueError("fixture could not be selected unambiguously; pass fixture=/absolute/audio/path")


def has_annotation_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return value is not None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"fixture row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def kws_expected_detected(row: dict[str, Any], *, key: str) -> bool:
    positive = {"detect", "detected", "positive", "true", "1", "yes"}
    negative = {"reject", "rejected", "negative", "false", "0", "no"}
    values: list[bool] = []
    for field in ("expected", "label", "expected_detected"):
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, bool):
            values.append(value)
            continue
        normalized = str(value).strip().lower()
        if normalized in positive:
            values.append(True)
        elif normalized in negative:
            values.append(False)
        else:
            raise ValueError(f"KWS fixture {key} has unsupported {field}: {value!r}")
    if not values:
        raise ValueError(
            f"KWS fixture {key} must declare expected, label, or expected_detected explicitly"
        )
    if len(set(values)) != 1:
        raise ValueError(f"KWS fixture {key} has conflicting positive/negative annotations")
    return values[0]


def kws_keywords(value: Any, *, key: str) -> list[str]:
    if isinstance(value, str):
        keywords = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        keywords = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if len(keywords) != len(value):
            raise ValueError(f"KWS fixture {key} keywords must contain non-empty strings")
    else:
        keywords = []
    if not keywords:
        raise ValueError(f"KWS fixture {key} requires at least one keyword")
    return keywords


def normalized_keyword(value: str) -> str:
    return "".join(value.upper().split())


def wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            return handle.getnframes() / frame_rate if frame_rate > 0 else None
    except (OSError, EOFError, wave.Error):
        return None


def looks_like_absolute_path_or_uri(value: str) -> bool:
    stripped = value.strip()
    return bool(
        stripped
        and (
            URI_PREFIX.match(stripped)
            or Path(stripped).is_absolute()
            or PureWindowsPath(stripped).is_absolute()
            or stripped.startswith("\\\\")
        )
    )


def pcm_wav_info(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".wav":
        raise ValueError("speaker fixture audio must be a PCM WAV")
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getcomptype() != "NONE"
                or handle.getnchannels() < 1
                or handle.getsampwidth() not in {1, 2, 3, 4}
                or handle.getframerate() < 1
                or handle.getnframes() < 1
            ):
                raise ValueError("speaker fixture audio must be a non-empty PCM WAV")
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            frames = handle.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"speaker fixture audio must be a readable PCM WAV: {error}") from error
    expected_bytes = frame_count * channels * sample_width
    if len(frames) != expected_bytes:
        raise ValueError(
            f"speaker fixture PCM data is truncated: expected {expected_bytes} bytes, read {len(frames)}"
        )
    silence_byte = b"\x80" if sample_width == 1 else b"\x00"
    return {
        "duration_sec": frame_count / sample_rate,
        "sample_rate": sample_rate,
        "audio_is_silence": bool(frames) and frames == silence_byte * len(frames),
    }


def pcm_wav_is_silence(path: Path) -> bool:
    try:
        return bool(pcm_wav_info(path)["audio_is_silence"])
    except ValueError:
        return False


def kws_duration(row: dict[str, Any], audio_path: Path, *, key: str) -> float:
    value = row.get("duration", row.get("duration_sec"))
    if value is None:
        value = wav_duration(audio_path)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(
            f"KWS fixture {key} requires a positive finite duration or a readable WAV header"
        )
    return float(value)


def fixture_tree_identity(staged_dir: Path, relative_files: list[Path]) -> str:
    hashes = {
        relative.as_posix(): sha256(staged_dir / relative)
        for relative in sorted(relative_files, key=lambda item: item.as_posix())
    }
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_speaker_segments(
    value: object,
    *,
    task: str,
    label: str,
    empty_sd_allowed: bool = False,
    duration_sec: float | None = None,
) -> list[dict[str, Any]]:
    if task not in SPEAKER_TASKS:
        raise ValueError(f"unsupported speaker task: {task}")
    if not isinstance(value, list):
        raise ValueError(f"{label} segments must be an array")
    if task == "sa_asr" and not value:
        raise ValueError(f"{label} SA-ASR segments must not be empty")
    if task == "sd" and not value and not empty_sd_allowed:
        raise ValueError(f"{label} empty SD segments are allowed only for pure-silence audio")
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            raise ValueError(f"{label} segments[{index}] must be an object")
        approved_fields = SA_ASR_SEGMENT_FIELDS if task == "sa_asr" else SD_SEGMENT_FIELDS
        unknown = sorted(str(field) for field in segment if field not in approved_fields)
        if unknown:
            raise ValueError(
                f"{label} segments[{index}] contains unapproved field(s): " + ", ".join(unknown)
            )
        speaker = segment.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"{label} segments[{index}].speaker must be a non-empty string")
        if looks_like_absolute_path_or_uri(speaker):
            raise ValueError(f"{label} segments[{index}].speaker must be a safe token")
        start = segment.get("start")
        end = segment.get("end")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(float(start))
            or float(start) < 0
        ):
            raise ValueError(f"{label} segments[{index}].start must be finite and >= 0")
        if (
            isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(end))
            or float(end) <= float(start)
        ):
            raise ValueError(f"{label} segments[{index}].end must be finite and > start")
        if task == "sa_asr":
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{label} segments[{index}].text must be a non-empty string")
        if duration_sec is not None and float(end) > duration_sec + 1e-6:
            raise ValueError(
                f"{label} segments[{index}].end exceeds WAV duration {duration_sec:.6f}"
            )
        duration = segment.get("duration")
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
            or not math.isclose(
                float(duration), float(end) - float(start), rel_tol=0, abs_tol=1e-3
            )
        ):
            raise ValueError(
                f"{label} segments[{index}].duration must be finite, positive, and equal end-start"
            )
        segments.append(dict(segment))
    return segments


def speaker_audio_source(
    source_dir: Path,
    value: object,
    *,
    task: str,
    key: str,
) -> tuple[Path, Path]:
    label = task.upper().replace("_", "-")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} fixture {key} requires a non-empty audio field")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{label} fixture {key} audio path must stay inside the fixture directory")
    current = source_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} fixture {key} audio must not traverse a symlink")
    resolved = (source_dir / relative).resolve()
    if (
        not resolved.is_relative_to(source_dir)
        or not resolved.is_file()
        or resolved.stat().st_size <= 0
        or resolved.suffix.lower() not in AUDIO_SUFFIXES
    ):
        raise ValueError(f"{label} fixture {key} audio is missing or unsafe: {value}")
    return relative, resolved


def prepare_speaker_fixture(
    resolved: dict,
    source: Path,
    run_dir: Path,
    *,
    task: str,
) -> dict[str, Any]:
    if task not in SPEAKER_TASKS:
        raise ValueError(f"unsupported speaker task: {task}")
    label = task.upper().replace("_", "-")
    raw_source_dir = source.parent if source.is_file() else source
    if raw_source_dir.is_symlink():
        raise ValueError(f"{label} fixture root must not be a symlink: {raw_source_dir}")
    if source.is_symlink():
        raise ValueError(f"{label} fixture gt.jsonl must not be a symlink: {source}")
    source_dir = raw_source_dir.resolve()
    source_gt = source if source.is_file() else source_dir / "gt.jsonl"
    if source_gt.name != "gt.jsonl" or source_gt.is_symlink() or not source_gt.is_file():
        raise ValueError(f"{label} fixture must contain gt.jsonl: {source_dir}")
    rows = read_jsonl(source_gt)
    if not 1 <= len(rows) <= 5:
        raise ValueError(f"{label} smoke fixture must contain 1 to 5 bounded samples")

    staged_dir = run_dir / "fixture" / task
    clear_directory(staged_dir, run_dir / "fixture")
    seen_keys: set[str] = set()
    staged_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    relative_files: set[Path] = set()

    for index, row in enumerate(rows, 1):
        key = str(row.get("key") or "").strip()
        if not key:
            raise ValueError(f"{label} fixture row {index} requires a non-empty key")
        if looks_like_absolute_path_or_uri(key):
            raise ValueError(f"{label} fixture key must be a safe token: {key!r}")
        if key in seen_keys:
            raise ValueError(f"{label} fixture contains duplicate key: {key}")
        seen_keys.add(key)
        relative_audio, audio_source = speaker_audio_source(
            source_dir,
            row.get("audio", row.get("wav")),
            task=task,
            key=key,
        )
        wav_info = pcm_wav_info(audio_source)
        segments = validate_speaker_segments(
            row.get("segments"),
            task=task,
            label=f"{label} fixture {key}",
            empty_sd_allowed=task == "sd" and bool(wav_info["audio_is_silence"]),
            duration_sec=float(wav_info["duration_sec"]),
        )
        destination = staged_dir / relative_audio
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ValueError(f"{label} fixture destination must not be a symlink: {destination}")
        shutil.copy2(audio_source, destination)
        relative_files.add(relative_audio)
        staged_row = {
            **row,
            "key": key,
            "task_type": task,
            "audio": relative_audio.as_posix(),
            "segments": segments,
        }
        try:
            json.dumps(staged_row, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} fixture {key} must contain strict JSON: {error}") from error
        staged_rows.append(staged_row)
        samples.append(
            {
                "key": key,
                "audio": relative_audio.as_posix(),
                "audio_path": str(destination),
                "annotation_fields": ["segments"],
                "segment_count": len(segments),
                **wav_info,
                "sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    gt_jsonl = staged_dir / "gt.jsonl"
    with gt_jsonl.open("w", encoding="utf-8") as handle:
        for row in staged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    relative_files.add(Path("gt.jsonl"))
    return {
        "schema": "sure.trans.fixture_manifest.v1",
        "status": "ready",
        "model_id": resolved["model_name"],
        "model_name": resolved["model_name"],
        "model_dir": str(run_dir),
        "task_type": task,
        "source_dir": str(source_dir),
        "staged_dir": str(staged_dir),
        "gt_jsonl": str(gt_jsonl),
        "samples": samples,
        "source_path": str(source_dir),
        "staged_path": str(staged_dir),
        "sha256": fixture_tree_identity(staged_dir, list(relative_files)),
        "gt_sha256": sha256(gt_jsonl),
        "expected_sha256": sha256(gt_jsonl),
        "size_bytes": sum((staged_dir / relative).stat().st_size for relative in relative_files),
        "sample_count": len(samples),
        "link_policy": "copy",
        "annotation_source": {
            "type": "fixture_gt_jsonl",
            "source_path": str(source_gt.resolve()),
            "staged_path": str(gt_jsonl),
            "fallback": False,
        },
    }


def prepare_kws_fixture(resolved: dict, source: Path, run_dir: Path) -> dict[str, Any]:
    source_dir = (source.parent if source.is_file() else source).resolve()
    source_gt = source if source.is_file() else source_dir / "gt.jsonl"
    if source_gt.name != "gt.jsonl" or not source_gt.is_file():
        raise ValueError(f"KWS fixture must contain gt.jsonl: {source_dir}")
    rows = read_jsonl(source_gt)
    if not 2 <= len(rows) <= 5:
        raise ValueError("KWS smoke fixture must contain 2 to 5 bounded samples")

    staged_dir = run_dir / "fixture" / "kws"
    clear_directory(staged_dir, run_dir / "fixture")
    seen_keys: set[str] = set()
    polarities: set[bool] = set()
    staged_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    relative_files: list[Path] = []

    for index, row in enumerate(rows, 1):
        raw_audio = row.get("audio") or row.get("wav")
        if not isinstance(raw_audio, str) or not raw_audio.strip():
            raise ValueError(f"KWS fixture row {index} requires a non-empty audio or wav field")
        relative_audio = Path(raw_audio)
        if relative_audio.is_absolute() or ".." in relative_audio.parts:
            raise ValueError(f"KWS fixture audio path must stay inside the fixture directory: {raw_audio}")
        raw_audio_source = source_dir / relative_audio
        current = source_dir
        has_symlink = False
        for part in relative_audio.parts:
            current = current / part
            if current.is_symlink():
                has_symlink = True
                break
        audio_source = raw_audio_source.resolve()
        if (
            not audio_source.is_relative_to(source_dir)
            or not audio_source.is_file()
            or has_symlink
            or audio_source.suffix.lower() not in AUDIO_SUFFIXES
        ):
            raise ValueError(f"KWS fixture audio is missing or unsafe: {raw_audio}")
        key = str(row.get("key") or "").strip()
        if not key:
            raise ValueError(f"KWS fixture row {index} requires a non-empty key")
        if key in seen_keys:
            raise ValueError(f"KWS fixture contains duplicate key: {key}")
        seen_keys.add(key)
        expected_detected = kws_expected_detected(row, key=key)
        polarities.add(expected_detected)
        keywords = kws_keywords(row.get("keywords"), key=key)
        expected_keyword_value = row.get("expected_keyword")
        if expected_keyword_value is None and expected_detected:
            expected_keyword_value = row.get("text", row.get("txt"))
        if expected_detected:
            if not isinstance(expected_keyword_value, str) or not expected_keyword_value.strip():
                raise ValueError(f"positive KWS fixture {key} requires expected_keyword or text")
            expected_keyword = expected_keyword_value.strip()
            if normalized_keyword(expected_keyword) not in {
                normalized_keyword(keyword) for keyword in keywords
            }:
                raise ValueError(f"positive KWS fixture {key} expected keyword is not in keywords")
        else:
            if expected_keyword_value not in (None, ""):
                raise ValueError(f"negative KWS fixture {key} must not declare expected_keyword")
            expected_keyword = None
        duration = kws_duration(row, audio_source, key=key)
        if "threshold" in row and (
            isinstance(row["threshold"], bool)
            or not isinstance(row["threshold"], (int, float))
            or not math.isfinite(float(row["threshold"]))
            or float(row["threshold"]) != KWS_OPERATING_THRESHOLD
        ):
            raise ValueError(f"KWS fixture {key} threshold must equal {KWS_OPERATING_THRESHOLD}")

        destination = staged_dir / relative_audio
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ValueError(f"KWS fixture destination must not be a symlink: {destination}")
        shutil.copy2(audio_source, destination)
        relative_files.append(relative_audio)
        staged_row = {
            **row,
            "key": key,
            "audio": relative_audio.as_posix(),
            "expected_detected": expected_detected,
            "expected_keyword": expected_keyword,
            "duration": duration,
        }
        staged_rows.append(staged_row)
        annotation_fields = [
            field
            for field in ("expected", "label", "expected_detected", "text", "txt")
            if field in staged_row and has_annotation_value(staged_row[field])
        ]
        samples.append(
            {
                "key": key,
                "audio": relative_audio.as_posix(),
                "audio_path": str(destination),
                "annotation_fields": annotation_fields,
                "expected_detected": expected_detected,
                "expected_keyword": expected_keyword,
                "keywords": row.get("keywords"),
                "duration": duration,
                "sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    if polarities != {False, True}:
        raise ValueError("KWS smoke fixture must contain at least one positive and one negative sample")

    gt_jsonl = staged_dir / "gt.jsonl"
    with gt_jsonl.open("w", encoding="utf-8") as handle:
        for row in staged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    relative_files.append(Path("gt.jsonl"))
    total_bytes = sum((staged_dir / relative).stat().st_size for relative in relative_files)
    return {
        "schema": "sure.trans.fixture_manifest.v1",
        "status": "ready",
        "model_id": resolved["model_name"],
        "model_name": resolved["model_name"],
        "model_dir": str(run_dir),
        "task_type": "kws",
        "source_dir": str(source_dir),
        "staged_dir": str(staged_dir),
        "gt_jsonl": str(gt_jsonl),
        "samples": samples,
        "source_path": str(source_dir),
        "staged_path": str(staged_dir),
        "sha256": fixture_tree_identity(staged_dir, relative_files),
        "gt_sha256": sha256(gt_jsonl),
        "expected_sha256": sha256(gt_jsonl),
        "size_bytes": total_bytes,
        "sample_count": len(samples),
        "link_policy": "copy",
        "annotation_source": {
            "type": "fixture_gt_jsonl",
            "source_path": str(source_gt.resolve()),
            "staged_path": str(gt_jsonl),
            "fallback": False,
        },
    }


def se_audio_source(source_dir: Path, value: object, *, key: str, role: str) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SE fixture {key} requires a non-empty {role}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"SE fixture {key} {role} path must stay inside the fixture directory")
    current = source_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"SE fixture {key} {role} must not traverse a symlink")
    resolved = (source_dir / relative).resolve()
    if (
        not resolved.is_relative_to(source_dir)
        or not resolved.is_file()
        or resolved.suffix.lower() not in AUDIO_SUFFIXES
    ):
        raise ValueError(f"SE fixture {key} {role} is missing or unsafe: {value}")
    return relative, resolved


def prepare_se_fixture(resolved: dict, source: Path, run_dir: Path) -> dict[str, Any]:
    source_dir = (source.parent if source.is_file() else source).resolve()
    source_gt = source if source.is_file() else source_dir / "gt.jsonl"
    if source_gt.name != "gt.jsonl" or not source_gt.is_file():
        raise ValueError(f"SE fixture must contain gt.jsonl: {source_dir}")
    rows = read_jsonl(source_gt)
    if not 1 <= len(rows) <= 5:
        raise ValueError("SE smoke fixture must contain 1 to 5 bounded samples")

    staged_dir = run_dir / "fixture" / "se"
    clear_directory(staged_dir, run_dir / "fixture")
    seen_keys: set[str] = set()
    staged_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    relative_files: set[Path] = set()

    for index, row in enumerate(rows, 1):
        key = str(row.get("sample_id") or row.get("key") or "").strip()
        if not key:
            raise ValueError(f"SE fixture row {index} requires a non-empty sample_id or key")
        if key in seen_keys:
            raise ValueError(f"SE fixture contains duplicate key: {key}")
        seen_keys.add(key)
        noisy_relative, noisy_source = se_audio_source(
            source_dir,
            row.get("noisy_audio", row.get("audio")),
            key=key,
            role="noisy_audio",
        )
        clean_relative, clean_source = se_audio_source(
            source_dir, row.get("reference_audio"), key=key, role="reference_audio"
        )
        if noisy_source.samefile(clean_source):
            raise ValueError(
                f"SE fixture {key} noisy_audio and reference_audio must be independent files"
            )

        for relative, audio_source in (
            (noisy_relative, noisy_source),
            (clean_relative, clean_source),
        ):
            destination = staged_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                raise ValueError(f"SE fixture destination must not be a symlink: {destination}")
            shutil.copy2(audio_source, destination)
            relative_files.add(relative)

        staged_row = {
            **row,
            "key": key,
            "sample_id": key,
            "task_type": "se",
            "audio": noisy_relative.as_posix(),
            "noisy_audio": noisy_relative.as_posix(),
            "reference_audio": clean_relative.as_posix(),
        }
        staged_rows.append(staged_row)
        noisy_destination = staged_dir / noisy_relative
        clean_destination = staged_dir / clean_relative
        samples.append(
            {
                "key": key,
                "audio": noisy_relative.as_posix(),
                "audio_path": str(noisy_destination),
                "noisy_audio": noisy_relative.as_posix(),
                "noisy_audio_path": str(noisy_destination),
                "reference_audio": clean_relative.as_posix(),
                "reference_audio_path": str(clean_destination),
                "annotation_fields": ["reference_audio"],
                "sha256": sha256(noisy_destination),
                "reference_sha256": sha256(clean_destination),
                "size_bytes": noisy_destination.stat().st_size,
                "reference_size_bytes": clean_destination.stat().st_size,
            }
        )

    gt_jsonl = staged_dir / "gt.jsonl"
    with gt_jsonl.open("w", encoding="utf-8") as handle:
        for row in staged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    relative_files.add(Path("gt.jsonl"))
    return {
        "schema": "sure.trans.fixture_manifest.v1",
        "status": "ready",
        "model_id": resolved["model_name"],
        "model_name": resolved["model_name"],
        "model_dir": str(run_dir),
        "task_type": "se",
        "source_dir": str(source_dir),
        "staged_dir": str(staged_dir),
        "gt_jsonl": str(gt_jsonl),
        "samples": samples,
        "source_path": str(source_dir),
        "staged_path": str(staged_dir),
        "sha256": fixture_tree_identity(staged_dir, list(relative_files)),
        "gt_sha256": sha256(gt_jsonl),
        "expected_sha256": sha256(gt_jsonl),
        "size_bytes": sum((staged_dir / relative).stat().st_size for relative in relative_files),
        "sample_count": len(samples),
        "link_policy": "copy",
        "annotation_source": {
            "type": "fixture_gt_jsonl",
            "source_path": str(source_gt.resolve()),
            "staged_path": str(gt_jsonl),
            "fallback": False,
        },
    }


def clear_directory(path: Path, controlled_root: Path) -> None:
    if controlled_root.is_symlink() or path.is_symlink():
        raise ValueError(f"fixture staging directory must not be a symlink: {path}")
    resolved = path.resolve()
    root = controlled_root.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"fixture staging directory must stay below {root}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise ValueError(f"fixture staging contains unsupported entry: {child}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    artifacts = run_dir / "artifacts"
    resolved = read_object(artifacts / "trans_input_resolved.json")
    task = str(resolved["task_type"]).replace("-", "_").lower()
    source = choose_fixture(resolved, task)
    if source.is_symlink():
        raise ValueError(f"fixture path must not be a symlink: {source}")
    source = source.resolve()
    if task == "kws":
        payload = prepare_kws_fixture(resolved, source, run_dir)
        output = artifacts / "fixture_manifest.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output)
        return 0
    if task == "se":
        payload = prepare_se_fixture(resolved, source, run_dir)
        output = artifacts / "fixture_manifest.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output)
        return 0
    if task in SPEAKER_TASKS:
        payload = prepare_speaker_fixture(resolved, source, run_dir, task=task)
        output = artifacts / "fixture_manifest.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output)
        return 0
    staged_dir = run_dir / "fixture" / task
    clear_directory(staged_dir, run_dir / "fixture")
    destination = staged_dir / source.name
    shutil.copy2(source, destination)
    expected_source = source.with_suffix(".expected.json")
    if not expected_source.is_file():
        raise ValueError(
            f"fixture reference annotation is missing: {expected_source}; "
            "provide a same-stem .expected.json instead of deriving ground truth from model output"
        )
    expected = read_object(expected_source)
    annotations = {
        field: expected[field]
        for field in ANNOTATION_FIELDS
        if field in expected and has_annotation_value(expected[field])
    }
    if not annotations:
        raise ValueError(
            f"fixture reference annotation has no non-empty supported field: {expected_source}"
        )
    # prompt_text is a TTS input, not a label: the fixture gate recomputes
    # annotation_fields from ANNOTATION_FIELDS and compares it with what the
    # sample declares, so listing prompt_text there fails every TTS fixture.
    annotation_fields = list(annotations)
    gt_extras: dict[str, object] = {}
    if task == "tts":
        prompt_text = expected.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"TTS fixture annotation requires non-empty prompt_text: {expected_source}")
        gt_extras["prompt_text"] = prompt_text.strip()
    expected_destination = staged_dir / expected_source.name
    shutil.copy2(expected_source, expected_destination)
    gt_jsonl = staged_dir / "gt.jsonl"
    audio_field = "reference_audio" if task in {"tts", "vc"} else "audio"
    gt_row = {audio_field: source.name, "task_type": task, **annotations, **gt_extras}
    gt_jsonl.write_text(json.dumps(gt_row, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {
        "schema": "sure.trans.fixture_manifest.v1",
        "status": "ready",
        "model_id": resolved["model_name"],
        "model_name": resolved["model_name"],
        "model_dir": str(run_dir),
        "task_type": task,
        "source_dir": str(source.parent),
        "staged_dir": str(staged_dir),
        "gt_jsonl": str(gt_jsonl),
        "samples": [
            {
                "key": source.stem,
                "audio": source.name,
                "audio_path": str(destination),
                "annotation_fields": annotation_fields,
            }
        ],
        "source_path": str(source),
        "staged_path": str(destination),
        "sha256": sha256(destination),
        "gt_sha256": sha256(gt_jsonl),
        "expected_sha256": sha256(expected_destination),
        "size_bytes": destination.stat().st_size,
        "sample_count": 1,
        "link_policy": "copy",
        "annotation_source": {
            "type": "fixture_expected_sidecar",
            "source_path": str(expected_source),
            "staged_path": str(expected_destination),
            "fallback": False,
        },
    }
    output = artifacts / "fixture_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
