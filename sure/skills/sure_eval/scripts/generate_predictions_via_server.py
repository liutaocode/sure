#!/usr/bin/env python3
"""
Generate prediction files for one dataset by calling a model-local MCP server.

This script is the execution surface for the `wait_for_predictions` step when
the main flow chooses `direct_server_use`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "runtime" / "harness"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_child_env import model_child_env

from sure_eval.core.config import Config
from sure_eval.core.logging import configure_logging, get_logger
from sure_eval.datasets import DatasetManager

from classification_contract import (
    CLASSIFICATION_TASKS,
    canonical_task as canonical_classification_task,
    normalize_prediction as normalize_classification_prediction,
    prompt_payload as classification_prompt_payload,
)

configure_logging(level="INFO")
logger = get_logger(__name__)

SURE_SUITES_ROOT = Path("data/datasets/sure_benchmark/SURE_Test_Suites")
PREDICTION_SNAPSHOT_INTERVAL = 25
KWS_OPERATING_THRESHOLD = 0.5
TSE_TASK_ALIASES = {
    "TSE",
    "TARGET_SPEAKER_EXTRACTION",
    "TARGET_SPEAKER_EXTRACTOR",
    "TARGET_SPEAKER_EXTRACTION_MODEL",
    "TARGET_SPEAKER",
    "SPEAKER_EXTRACTION",
    "TARGET_VOICE_EXTRACTION",
    "TARGET_VOICE_SEPARATION",
}
TSE_OUTPUT_FIELDS = {"prediction_audio", "sample_id"}
TSE_FORBIDDEN_FIELDS = {
    "answer",
    "expected",
    "ground_truth",
    "input",
    "input_audio",
    "mixture_audio",
    "mixture_audio_path",
    "mixed_audio",
    "reference",
    "reference_audio",
    "reference_audio_path",
    "reference_text",
    "target",
    "target_audio",
    "target_audio_path",
    "target_text",
    "enrollment_audio",
    "enrollment_audio_path",
}
CLASSIFICATION_PROTECTED_ARGUMENTS = {"audio_path", "prompt", "choices"}
CLASSIFICATION_REFERENCE_ARGUMENTS = {
    "answer",
    "expected",
    "ground_truth",
    "input",
    "input_audio",
    "reference",
    "reference_annotation",
    "reference_audio",
    "reference_text",
    "target",
    "target_audio",
    "target_text",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_run_id(run_dir: Path) -> str:
    return os.environ.get("RUN_ID") or run_dir.name


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_server_command(
    model_dir: Path,
    runtime_inventory: dict[str, Any],
) -> list[str]:
    container = runtime_inventory.get("container_runtime")
    model_runtime = runtime_inventory.get("model_runtime")
    policy = runtime_inventory.get("policy")
    if runtime_inventory.get("schema") != "sure.onboard.runtime_inventory.v2":
        raise ValueError(f"approved model has unsupported runtime inventory: {model_dir}")
    if runtime_inventory.get("status") != "ready":
        raise ValueError("approved model runtime inventory is not ready")
    if not isinstance(policy, dict):
        raise ValueError("approved model runtime policy is missing")
    if policy.get("host_python_fallback") is not False or policy.get("image_override_allowed") is not False:
        raise ValueError("approved model runtime policy permits a forbidden fallback or image override")
    if policy.get("eval_runtime") == "python":
        if not isinstance(model_runtime, dict) or model_runtime.get("required") is not True:
            raise ValueError("approved Model Python runtime is missing")
        command = model_runtime.get("server_command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("approved Model Python server_command is invalid")
        actual_python = os.environ.get("MODEL_PYTHON", "")
        runtime_id = os.environ.get("SURE_EVAL_MODEL_RUNTIME_ID", "")
        if not actual_python or runtime_id != model_runtime.get("runtime_id"):
            raise ValueError("active Model Python does not match the approved runtime identity")
        return [actual_python, *command[1:]]
    if policy.get("eval_runtime") != "container_only":
        raise ValueError("approved model has no supported Eval runtime")
    if not isinstance(container, dict):
        raise ValueError("approved model container runtime is missing")
    command = container.get("server_command")
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        raise ValueError("approved model container server_command is invalid")
    expected_image = str(container.get("target_image_ref") or "")
    actual_image = os.environ.get("SURE_EVAL_CONTAINER_IMAGE", "")
    if "@sha256:" not in expected_image or actual_image != expected_image:
        raise ValueError(
            "inference container does not match the approved digest-pinned image: "
            f"expected={expected_image!r} actual={actual_image!r}"
        )
    return command


def _resolve_working_dir(model_dir: Path, runtime_inventory: dict[str, Any]) -> Path:
    policy = runtime_inventory.get("policy") if isinstance(runtime_inventory.get("policy"), dict) else {}
    if policy.get("eval_runtime") == "python":
        raw = os.environ.get("SURE_EVAL_MODEL_WORKING_DIR", "")
        path = Path(raw).expanduser().resolve()
        try:
            path.relative_to(model_dir.resolve())
        except ValueError as exc:
            raise ValueError("approved Model Python working_dir escapes the model bundle") from exc
        if not path.is_dir():
            raise ValueError(f"approved Model Python working_dir does not exist: {path}")
        return path
    container = runtime_inventory.get("container_runtime")
    working_dir = container.get("working_dir") if isinstance(container, dict) else None
    path = Path(str(working_dir or ""))
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"approved container working_dir does not exist: {path}")
    return path


def _resolve_audio_path(repo_root: Path, sample: dict[str, Any]) -> Path:
    sample_value = (
        sample.get("path")
        or sample.get("mixture_audio_path")
        or sample.get("mixture_audio")
        or sample.get("mixed_audio")
        or sample.get("audio")
        or ""
    )
    sample_path = Path(str(sample_value))
    if sample_path.is_absolute():
        return sample_path

    sure_candidate = repo_root / SURE_SUITES_ROOT / sample_path
    if sure_candidate.exists():
        return sure_candidate

    relative_candidate = repo_root / sample_path
    if relative_candidate.exists():
        return relative_candidate

    raise FileNotFoundError(f"Unable to resolve audio path for sample: {sample}")


def _materialize_sample_audio(repo_root: Path, sample: dict[str, Any], scratch_dir: Path) -> Path:
    """Return a normal audio file path for a sample, slicing long audio if needed."""
    # TSE's source is the mixture role.  ``source_audio`` is a legacy alias
    # used by VC datasets and must never become an implicit reference/enrollment
    # fallback for target-speaker extraction.
    if (
        _normalize_task(sample.get("task")) != "TSE"
        and not any(sample.get(field) for field in ("mixture_audio", "mixture_audio_path", "enrollment_audio", "enrollment_audio_path"))
        and sample.get("source_audio")
        and sample.get("begin_time") is not None
        and sample.get("end_time") is not None
    ):
        source = Path(str(sample.get("source_audio") or sample.get("mixture_audio") or sample.get("path") or ""))
        if not source.is_absolute():
            source = repo_root / source
        if not source.exists():
            raise FileNotFoundError(f"Unable to resolve source audio path for sample: {sample}")

        key = (_sample_key(sample) or "sample").replace("/", "_")
        output_path = scratch_dir / f"{key}.wav"
        if not output_path.exists():
            start = float(sample["begin_time"])
            end = float(sample["end_time"])
            duration = max(0.01, end - start)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(source),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(output_path),
                ],
                check=True,
            )
        return output_path

    return _resolve_audio_path(repo_root, sample)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _sample_key(sample: dict[str, Any]) -> str:
    return str(sample.get("key") or sample.get("sample_id") or "")


def _normalize_tts_language(language: str | None) -> str:
    value = str(language or "").strip()
    normalized = value.lower().replace("_", "-")
    mapping = {
        "": "",
        "en": "English",
        "eng": "English",
        "english": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "cmn": "Chinese",
        "yue": "Chinese",
        "chinese": "Chinese",
        "cn": "Chinese",
    }
    return mapping.get(normalized, value)


def _normalize_task(value: Any) -> str:
    normalized = (
        str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    )
    if normalized in {"SPEECH_ACTIVITY_DETECTION", "VOICE_ACTIVITY_DETECTION"}:
        return "VAD"
    if normalized in TSE_TASK_ALIASES:
        return "TSE"
    classification = canonical_classification_task(normalized).upper()
    if classification in CLASSIFICATION_TASKS:
        return classification
    return "SA-ASR" if normalized == "SA_ASR" else normalized


def _split_metrics(value: str | None) -> list[str]:
    out: list[str] = []
    for item in str(value or "").replace(",", " ").split():
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def _kws_metrics_require_scores(metrics: list[str]) -> bool:
    normalized = [str(metric).lower().replace("-", "_") for metric in metrics]
    return any(
        "macro_recall" in metric or "det_curve" in metric or metric == "det"
        for metric in normalized
    )


def _metric_task_hint(metrics: list[str]) -> str:
    hinted: list[str] = []
    for metric in metrics:
        metric_name = str(metric or "").strip().lower()
        if metric_name.startswith("vc_"):
            hinted.append("VC")
        elif metric_name.startswith("tts_"):
            hinted.append("TTS")
        elif metric_name.startswith("tse_"):
            hinted.append("TSE")
    hinted = [task for index, task in enumerate(hinted) if task not in hinted[:index]]
    return hinted[0] if len(hinted) == 1 else ""


def _model_task(model_cfg: dict[str, Any]) -> str:
    model_section = model_cfg.get("model") if isinstance(model_cfg.get("model"), dict) else {}
    return _normalize_task(model_section.get("task") or model_cfg.get("task") or model_cfg.get("task_type"))


def _effective_generation_task(sample_task: str, model_cfg: dict[str, Any], metrics: list[str]) -> str:
    task = _normalize_task(sample_task) or "ASR"
    if task in {"TTS", "VC", "TSE"}:
        metric_task = _metric_task_hint(metrics)
        if metric_task in {"TTS", "VC", "TSE"}:
            return metric_task
        declared_task = _model_task(model_cfg)
        if declared_task in {"TTS", "VC", "TSE"}:
            return declared_task
    return task


def _resolve_audio_field_path(repo_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        repo_root / SURE_SUITES_ROOT / path,
        repo_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return repo_root / path


def _sample_reference_audio_path(repo_root: Path, sample: dict[str, Any], fallback: Path) -> Path:
    value = (
        sample.get("reference_audio")
        or sample.get("reference_audio_path")
        or sample.get("target_audio_path")
        or sample.get("prompt_audio")
        or sample.get("prompt_audio_path")
        or sample.get("prompt_wav_path")
        or sample.get("prompt_wav")
    )
    return _resolve_audio_field_path(repo_root, value) or fallback


def _sample_enrollment_audio_path(repo_root: Path, sample: dict[str, Any]) -> Path | None:
    """Resolve the TSE enrollment role without using scoring references."""

    value = (
        sample.get("enrollment_audio_path")
        or sample.get("enrollment_audio")
        or sample.get("enrollment")
        or sample.get("speaker_audio")
        or sample.get("enroll_audio")
    )
    return _resolve_audio_field_path(repo_root, value)


def _validate_tse_input_path(path: Path, *, role: str) -> Path:
    candidate = path.expanduser()
    if ".." in candidate.parts:
        raise ValueError(f"TSE {role} path must not contain traversal components")
    absolute = candidate.absolute()
    current = Path(absolute.anchor) if absolute.anchor else Path(".")
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"TSE {role} path must not traverse a symlink: {absolute}")
    if not absolute.is_file() or absolute.stat().st_size <= 0:
        raise ValueError(f"TSE {role} path is missing or empty: {absolute}")
    return absolute.resolve()


def _validate_tse_sample_key(value: Any) -> str:
    key = str(value or "").strip()
    if (
        not key
        or "/" in key
        or "\\" in key
        or any(ord(character) < 32 or character.isspace() for character in key)
    ):
        raise ValueError("TSE sample key must be a safe non-empty token")
    return key


def _se_run_output_path(output_audio_dir: Path, key: str) -> Path:
    root = output_audio_dir.expanduser().absolute()
    if root.is_symlink():
        raise ValueError(f"SE output directory must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root:
        raise ValueError(f"SE output directory must not traverse a symlink: {root}")
    output = (root / f"{_safe_filename(key)}.wav").absolute()
    if output.parent != root or output.is_symlink():
        raise ValueError(f"SE output_path must be a direct non-symlink child of {root}")
    return output


def _tse_run_output_path(output_audio_dir: Path, key: str) -> Path:
    root = output_audio_dir.expanduser().absolute()
    if root.is_symlink():
        raise ValueError(f"TSE output directory must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root:
        raise ValueError(f"TSE output directory must not traverse a symlink: {root}")
    output = (root / f"{_safe_filename(key)}.wav").absolute()
    if output.parent != root or output.is_symlink():
        raise ValueError(f"TSE output_path must be a direct non-symlink child of {root}")
    return output


def _validate_pcm_wav(path: Path, *, label: str) -> float:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a real non-empty file: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getcomptype() != "NONE"
                or handle.getnchannels() < 1
                or handle.getsampwidth() not in {1, 2, 3, 4}
                or handle.getframerate() < 1
                or handle.getnframes() < 1
            ):
                raise ValueError(f"{label} must be a readable non-empty PCM WAV: {path}")
            duration = handle.getnframes() / handle.getframerate()
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError(f"{label} must be a readable non-empty PCM WAV: {path}") from exc
    return duration


def _classification_forbidden_argument_fields(
    value: Any,
    path: str = "tool_args",
) -> list[str]:
    """Find reference/path keys before protocol arguments reach the model."""

    found: list[str] = []
    if isinstance(value, dict):
        for field, item in value.items():
            normalized = str(field).strip().lower().replace("-", "_")
            child = f"{path}.{field}"
            if (
                normalized in CLASSIFICATION_REFERENCE_ARGUMENTS
                or normalized == "path"
                or normalized.startswith("reference_")
                or normalized.endswith("_path")
            ):
                found.append(child)
            found.extend(_classification_forbidden_argument_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_classification_forbidden_argument_fields(item, f"{path}[{index}]"))
    return found


def _validate_classification_tool_args(tool_args: dict[str, Any]) -> None:
    protected = sorted(
        str(key)
        for key in tool_args
        if str(key).strip().lower().replace("-", "_") in CLASSIFICATION_PROTECTED_ARGUMENTS
    )
    if protected:
        raise ValueError(
            "classification harness-owned argument(s) cannot be overridden: "
            + ", ".join(protected)
        )
    forbidden = _classification_forbidden_argument_fields(tool_args)
    if forbidden:
        raise ValueError(
            "classification tool arguments must not contain reference/path field(s): "
            + ", ".join(forbidden)
        )
    for key, value in tool_args.items():
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"classification tool argument {key!r} must be a scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"classification tool argument {key!r} must be finite")


def _validate_tse_prediction_audio(
    value: Any,
    *,
    expected_path: str | Path,
    forbidden_inputs: tuple[Path, ...],
) -> Path:
    """Resolve a TSE output and bind it to the harness-assigned file."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("TSE prediction payload must contain prediction_audio")
    expected = Path(expected_path).expanduser().absolute()
    candidate = Path(value.strip()).expanduser()
    if ".." in candidate.parts or "://" in value or (
        len(value) > 1 and value[0].isalpha() and value[1] == ":"
    ):
        raise ValueError("TSE prediction_audio must be a contained basename/path without traversal")
    if not candidate.is_absolute():
        candidate = expected.parent / candidate
    candidate = candidate.absolute()
    if candidate != expected:
        raise ValueError(
            f"TSE prediction_audio must equal the run-local output_path: {candidate}"
        )
    root = expected.parent
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("TSE run-local output directory must not traverse a symlink")
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ValueError("TSE prediction_audio must be a real non-empty file")
    for input_path in forbidden_inputs:
        try:
            if candidate.resolve().samefile(input_path):
                raise ValueError(
                    "TSE prediction_audio must not alias mixture, enrollment, or reference audio"
                )
        except OSError:
            continue
    _validate_pcm_wav(candidate, label="TSE prediction_audio")
    return candidate.resolve()


def _validate_kws_threshold(value: Any, *, source: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"KWS {source} threshold must be a finite number")
    threshold = float(value)
    if threshold != KWS_OPERATING_THRESHOLD:
        raise ValueError(
            f"KWS {source} threshold must equal the formal operating threshold "
            f"{KWS_OPERATING_THRESHOLD}"
        )
    return threshold


def _validate_annotation_prediction_segments(value: Any, *, task: str) -> list[dict[str, Any]]:
    canonical_task = _normalize_task(task)
    if not isinstance(value, list) or (canonical_task == "SA-ASR" and not value):
        raise ValueError(f"{canonical_task} prediction segments must be a list")
    seen: set[tuple[Any, ...]] = set()
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            raise ValueError(f"{canonical_task} prediction segment {index} must be an object")
        allowed_fields = {"speaker", "start", "end", "duration"}
        if canonical_task == "SA-ASR":
            allowed_fields.add("text")
        unknown_fields = sorted(str(field) for field in segment if field not in allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"{canonical_task} prediction segment {index} contains unapproved field(s): "
                + ", ".join(unknown_fields)
            )
        speaker = segment.get("speaker")
        start = segment.get("start")
        end = segment.get("end")
        if (
            not isinstance(speaker, str)
            or not speaker.strip()
            or speaker != speaker.strip()
            or speaker.startswith(";")
            or speaker == "<NA>"
            or any(ch.isspace() or ord(ch) < 32 for ch in speaker)
        ):
            raise ValueError(f"{canonical_task} prediction segment {index} has invalid speaker")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise ValueError(f"{canonical_task} prediction segment {index} requires 0 <= start < end")
        text = segment.get("text")
        if canonical_task == "SA-ASR" and (
            not isinstance(text, str)
            or not text.strip()
            or "\n" in text
            or "\r" in text
        ):
            raise ValueError(f"SA-ASR prediction segment {index} requires non-empty text")
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
                f"{canonical_task} prediction segment {index} duration must equal end - start"
            )
        identity = (
            speaker.strip(),
            float(start),
            float(end),
            str(text or "").strip(),
        )
        if identity in seen:
            raise ValueError(f"{canonical_task} prediction contains a duplicate segment")
        seen.add(identity)
    return value


def _validate_vad_intervals(
    value: Any,
    *,
    role: str,
    duration: float | None = None,
    with_score: bool = False,
) -> list[dict[str, float]]:
    if not isinstance(value, list):
        raise ValueError(f"VAD {role} must be a list")
    if with_score and not value:
        raise ValueError("VAD frame_scores must not be empty when provided")
    allowed_fields = {"start", "end", "score"} if with_score else {"start", "end"}
    normalized: list[dict[str, float]] = []
    previous_end: float | None = None
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"VAD {role}[{index}] must be an object")
        unknown = sorted(str(field) for field in item if field not in allowed_fields)
        if unknown:
            raise ValueError(
                f"VAD {role}[{index}] contains unapproved field(s): "
                + ", ".join(unknown)
            )
        if set(item) != allowed_fields:
            missing = sorted(allowed_fields - set(item))
            raise ValueError(f"VAD {role}[{index}] is missing field(s): {', '.join(missing)}")
        start = item["start"]
        end = item["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
        ):
            raise ValueError(f"VAD {role}[{index}] times must be finite numbers")
        start_value = float(start)
        end_value = float(end)
        if start_value < 0 or end_value <= start_value:
            raise ValueError(f"VAD {role}[{index}] requires 0 <= start < end")
        if duration is not None and end_value > duration + 1e-6:
            raise ValueError(f"VAD {role}[{index}] exceeds audio duration {duration}")
        if previous_end is not None and start_value < previous_end - 1e-12:
            raise ValueError(f"VAD {role} must be ordered and non-overlapping")
        if with_score and (
            (previous_end is None and start_value > 1e-6)
            or (previous_end is not None and abs(start_value - previous_end) > 1e-6)
        ):
            raise ValueError("VAD frame_scores must continuously cover the audio timebase")
        row = {"start": start_value, "end": end_value}
        if with_score:
            score = item["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError(f"VAD frame_scores[{index}].score must be within [0, 1]")
            row["score"] = float(score)
        normalized.append(row)
        previous_end = end_value
    if with_score and duration is not None and abs((previous_end or 0.0) - duration) > 1e-6:
        raise ValueError("VAD frame_scores must continuously cover the audio timebase")
    return normalized


def _build_tool_arguments(
    *,
    repo_root: Path,
    sample: dict[str, Any],
    task: str,
    language: str,
    argument_name: str,
    audio_path: Path,
    output_audio_dir: Path,
    tool_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_name = _normalize_task(task)
    if task_name == "TSE":
        key = _validate_tse_sample_key(sample.get("key") or sample.get("sample_id") or "sample")
        enrollment_candidate = _sample_enrollment_audio_path(repo_root, sample)
        if enrollment_candidate is None:
            raise ValueError(
                f"TSE sample {key} requires an enrollment_audio/enrollment_audio_path role"
            )
        enrollment = _validate_tse_input_path(enrollment_candidate, role="enrollment_audio")
        mixture = _validate_tse_input_path(audio_path, role="mixture_audio")
        _validate_pcm_wav(enrollment, label=f"TSE sample {key} enrollment audio")
        if enrollment == mixture:
            raise ValueError(f"TSE sample {key} mixture and enrollment audio must differ")
        output_audio_path = _tse_run_output_path(output_audio_dir, key)
        arguments: dict[str, Any] = {
            "mixture_audio_path": str(mixture),
            "enrollment_audio_path": str(enrollment),
            "output_path": str(output_audio_path),
        }
        if tool_args:
            forbidden = sorted(
                key
                for key in tool_args
                if str(key).strip().lower().replace("-", "_") in TSE_FORBIDDEN_FIELDS
            )
            if forbidden:
                raise ValueError(
                    "TSE inference arguments must not contain reference/input fields: "
                    + ", ".join(forbidden)
                )
            arguments.update(tool_args)
        # The three contract fields are harness-owned and cannot be overridden.
        for field, expected in (
            ("mixture_audio_path", str(mixture)),
            ("enrollment_audio_path", str(enrollment)),
            ("output_path", str(output_audio_path)),
        ):
            if arguments.get(field) != expected:
                raise ValueError(f"TSE tool argument {field} cannot be overridden")
        return arguments
    if task_name == "SE":
        key = _sample_key(sample) or "sample"
        output_audio_path = str(_se_run_output_path(output_audio_dir, key))
        arguments: dict[str, Any] = {argument_name: str(audio_path)}
        if tool_args:
            arguments.update(tool_args)
        arguments["output_path"] = output_audio_path
        return arguments

    if task_name in {"TTS", "VC"}:
        key = _sample_key(sample) or "sample"
        prompt_audio_path = _sample_reference_audio_path(repo_root, sample, audio_path)

        target_text = (
            sample.get("target")
            or sample.get("reference_text")
            or sample.get("text")
            or sample.get("target_text")
            or ""
        )
        if not target_text:
            raise ValueError(f"TTS/VC sample has no target text: {key}")

        output_audio_dir.mkdir(parents=True, exist_ok=True)
        output_audio_path = str(output_audio_dir / f"{_safe_filename(key)}.wav")
        if task_name == "VC":
            arguments = {
                "source_audio_path": str(audio_path),
                "source": str(audio_path),
                "input_audio_path": str(audio_path),
                "reference_audio_path": str(prompt_audio_path),
                "target_audio_path": str(prompt_audio_path),
                "ref_audio_path": str(prompt_audio_path),
                "prompt_audio_path": str(prompt_audio_path),
                "prompt_wav_path": str(prompt_audio_path),
                "reference_text": str(target_text),
                "target_text": str(target_text),
                "text": str(target_text),
                "language": _normalize_tts_language(language or str(sample.get("language") or "")),
                "output_path": output_audio_path,
                "audio_path": output_audio_path,
                "converted_audio_path": output_audio_path,
            }
        else:
            arguments = {
                "text": str(target_text),
                "prompt_audio_path": str(prompt_audio_path),
                "prompt_wav_path": str(prompt_audio_path),
                "language": _normalize_tts_language(language or str(sample.get("language") or "")),
                "output_path": output_audio_path,
                "audio_path": output_audio_path,
            }
        prompt_text = (
            sample.get("prompt_text")
            or sample.get("ref_text")
            or sample.get("reference_text")
            or sample.get("target")
            or ""
        )
        if prompt_text:
            arguments["prompt_text"] = str(prompt_text)
            arguments["ref_text"] = str(prompt_text)
        if tool_args:
            arguments.update(tool_args)
        return arguments

    if task_name == "KWS":
        arguments = {argument_name: str(audio_path)}
        if "keywords" in sample:
            keywords = sample["keywords"]
            valid_keywords = (
                isinstance(keywords, str) and bool(keywords.strip())
            ) or (
                isinstance(keywords, list)
                and bool(keywords)
                and all(isinstance(keyword, str) and bool(keyword.strip()) for keyword in keywords)
            )
            if not valid_keywords:
                raise ValueError("KWS sample keywords must be a non-empty string or list of strings")
            arguments["keywords"] = keywords
        if "threshold" in sample:
            _validate_kws_threshold(sample["threshold"], source="sample")
            arguments["threshold"] = sample["threshold"]
        if tool_args and "threshold" in tool_args:
            _validate_kws_threshold(tool_args["threshold"], source="tool argument")
        if tool_args:
            arguments.update(tool_args)
        return arguments

    if task_name in CLASSIFICATION_TASKS:
        arguments: dict[str, Any] = {"audio_path": str(audio_path)}
        if language:
            arguments["language"] = language
        if task_name == "SLU":
            arguments.update(classification_prompt_payload(sample))
        if tool_args:
            _validate_classification_tool_args(tool_args)
            arguments.update(tool_args)
        forbidden = {
            "answer",
            "expected",
            "ground_truth",
            "reference",
            "reference_audio",
            "reference_text",
            "target",
            "target_text",
        } & set(arguments)
        if forbidden:
            raise ValueError(
                "classification inference arguments must not contain reference fields: "
                + ", ".join(sorted(forbidden))
            )
        return arguments

    if task_name == "VAD":
        arguments = {argument_name: str(audio_path)}
        if tool_args:
            arguments.update(tool_args)
        return arguments

    arguments: dict[str, Any] = {argument_name: str(audio_path)}
    if language:
        arguments["language"] = language
    if tool_args:
        arguments.update(tool_args)
    return arguments


def _parse_tool_args(values: list[str] | None) -> dict[str, Any]:
    """Parse repeated key=value tool argument overrides."""

    parsed: dict[str, Any] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"--tool-arg must use key=value format, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--tool-arg key must not be empty: {raw}")
        parsed[key] = _parse_tool_arg_value(value)
    return parsed


def _parse_tool_arg_value(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_env_overrides(values: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"--env must use KEY=VALUE format, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env key must not be empty: {raw}")
        parsed[key] = value
    return parsed


SENSITIVE_KEY_PARTS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "ACCESS_KEY", "PRIVATE_KEY", "CREDENTIAL", "COOKIE", "AUTH")
SAFE_ENV_VALUE_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "DEVICE",
    "HF_ENDPOINT",
    "HF_HOME",
    "HARNESS_PYTHON_BIN",
    "MODEL_PATH",
    "MODEL_PYTHON",
    "MODELSCOPE_CACHE",
    "NO_RESUME",
    "PYTHON_BIN",
    "SURE_EVAL_ALLOW_PARTITION_FALLBACK",
    "SURE_EVAL_CONTAINER_IMAGE",
    "SURE_EVAL_CONTAINER_REPO_ROOT",
    "SURE_EVAL_DEVICE_ACTUAL",
    "SURE_EVAL_DEVICE_REQUEST",
    "SURE_EVAL_EXECUTION_GENERATION_METHOD",
    "SURE_EVAL_EXECUTION_JOB_ID",
    "SURE_EVAL_EXECUTION_PATH",
    "SURE_EVAL_EXECUTION_REQUESTED",
    "SURE_EVAL_EXECUTION_SURFACE_TYPE",
    "SURE_EVAL_REQUIRE_VC_SUBMIT",
    "SURE_EVAL_VC_CPU",
    "SURE_EVAL_VC_GPU",
    "SURE_EVAL_VC_MEMORY",
    "SURE_EVAL_VC_NODES",
    "SURE_EVAL_VC_PARTITION",
    "SURE_HARNESS_LOCK_SHA256",
    "SURE_HARNESS_MANIFEST_PATH",
    "SURE_HARNESS_RUNTIME_ID",
    "SURE_HARNESS_RUNTIME_ROOT",
}
PATH_ARGUMENT_HINTS = ("audio", "path", "file", "dir", "jsonl")
TEXT_ARGUMENT_HINTS = ("text", "prompt", "reference", "target", "keyword", "threshold")


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(part in upper for part in SENSITIVE_KEY_PARTS)


def _redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in values.items():
        if _is_sensitive_key(str(key)):
            redacted[str(key)] = "<redacted>"
        else:
            redacted[str(key)] = value
    return redacted


def _safe_env_snapshot(env: dict[str, str], *, extra_keys: set[str] | None = None) -> dict[str, Any]:
    selected_keys = set(SAFE_ENV_VALUE_KEYS)
    if extra_keys:
        selected_keys.update(extra_keys)
    safe_values = {
        key: ("<redacted>" if _is_sensitive_key(key) else env.get(key))
        for key in sorted(selected_keys)
        if key in env
    }
    redacted_keys = sorted(key for key in env if _is_sensitive_key(key))
    return {
        "safe_env_values": safe_values,
        "env_keys": sorted(env.keys()),
        "redacted_env_keys": redacted_keys,
        "policy": "Only allowlisted non-secret values are materialized; all other values are represented by keys.",
    }


def _load_runtime_inventory(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "artifacts" / "runtime_inventory.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _runtime_inventory_summary(model_dir: Path) -> dict[str, Any]:
    inventory = _load_runtime_inventory(model_dir)
    if not inventory:
        return {
            "path": str(model_dir / "artifacts" / "runtime_inventory.json"),
            "status": "missing",
            "runtime": {},
            "evidence": {},
        }
    return {
        "path": str(model_dir / "artifacts" / "runtime_inventory.json"),
        "status": inventory.get("status"),
        "schema": inventory.get("schema"),
        "container_runtime": inventory.get("container_runtime") if isinstance(inventory.get("container_runtime"), dict) else {},
        "model_runtime": inventory.get("model_runtime") if isinstance(inventory.get("model_runtime"), dict) else {},
        "policy": inventory.get("policy") if isinstance(inventory.get("policy"), dict) else {},
        "evidence": inventory.get("evidence") if isinstance(inventory.get("evidence"), dict) else {},
    }


def _harness_runtime_summary(env: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "sure.harness.runtime.binding.v1",
        "runtime_id": env.get("SURE_HARNESS_RUNTIME_ID"),
        "runtime_type": "harness_python",
        "python_executable": env.get("HARNESS_PYTHON_BIN"),
        "process_python_executable": sys.executable,
        "lock_sha256": env.get("SURE_HARNESS_LOCK_SHA256"),
        "manifest_path": env.get("SURE_HARNESS_MANIFEST_PATH"),
        "runtime_root": env.get("SURE_HARNESS_RUNTIME_ROOT"),
    }


def _resolve_protocol_parameters(protocol_id: str, model_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    env["SURE_EVAL_PROTOCOL_ID"] = protocol_id
    from sure_eval.models.registry import ModelRegistry
    from sure_eval.protocols.resolver import ProtocolResolver

    resolver = ProtocolResolver()
    env["SURE_EVAL_PROTOCOL_DEFINITION_PATH"] = str(resolver.protocols_path.resolve())
    registry = ModelRegistry(model_dir.parent)
    model_info = registry.get_model(model_dir.name)
    if model_info is None:
        raise ValueError(f"approved model is not registered from config.yaml: {model_dir}")
    resolved = resolver.resolve(protocol_id, model_info)
    standard_params = dict(resolved.standard_params or {})
    model_params = dict(resolved.model_params or {})
    for key, value in standard_params.items():
        env[f"SURE_EVAL_PROTOCOL_{key.upper()}"] = str(value)
    for key, value in model_params.items():
        env[f"SURE_EVAL_MODEL_{key.upper()}"] = str(value)
    config_path = model_dir / "config.yaml"
    return {
        "enabled": True,
        "status": "resolved",
        "protocol_id": protocol_id,
        "parameter_policy": "upstream_native" if protocol_id == "standard_system" else "strict_mapped",
        "standard_params": standard_params,
        "model_params": model_params,
        "unmapped": dict(resolved.unmapped or {}),
        "parameter_status": dict(resolved.parameter_status or {}),
        "config_sources": [
            {
                "path": str(config_path),
                "sha256": _sha256(config_path),
                "role": "approved_model_runtime_config",
            }
        ],
        "error": None,
    }


def _merge_protocol_tool_args(
    protocol_id: str,
    protocol_resolution: dict[str, Any],
    explicit_tool_args: dict[str, Any],
    allowed_tool_args: set[str] | None = None,
) -> dict[str, Any]:
    protocol_tool_args = dict(protocol_resolution.get("model_params") or {})
    if protocol_id == "standard_system" and explicit_tool_args:
        raise ValueError(
            "standard_system forbids explicit --tool-arg generation overrides; "
            "declare upstream defaults in the approved model package"
        )
    extra_strict_args = sorted(set(explicit_tool_args) - set(protocol_tool_args))
    if protocol_id == "strict_core" and extra_strict_args:
        raise ValueError(
            "strict_core forbids tool arguments outside the resolved protocol mapping: "
            + ", ".join(extra_strict_args)
        )
    undeclared_protocol_args = sorted(set(protocol_tool_args) - (allowed_tool_args or set()))
    if protocol_id == "strict_core" and undeclared_protocol_args:
        raise ValueError(
            "strict_core mappings must name arguments declared by the selected MCP tool input_schema: "
            + ", ".join(undeclared_protocol_args)
        )
    conflicts = {
        key: {"requested": explicit_tool_args[key], "required": value}
        for key, value in protocol_tool_args.items()
        if key in explicit_tool_args and explicit_tool_args[key] != value
    }
    if conflicts:
        raise ValueError(
            "explicit --tool-arg values conflict with strict_core: "
            + json.dumps(conflicts, ensure_ascii=False, sort_keys=True)
        )
    return {**explicit_tool_args, **protocol_tool_args}


def _declared_tool_args(model_cfg: dict[str, Any], tool_name: str) -> set[str]:
    for tool in model_cfg.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("name") != tool_name:
            continue
        schema = tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        return {str(key) for key in properties}
    raise ValueError(f"selected tool {tool_name!r} is not declared in approved config.yaml")


def _is_dynamic_argument_key(key: str) -> bool:
    lower = key.lower()
    return any(hint in lower for hint in PATH_ARGUMENT_HINTS) or any(hint in lower for hint in TEXT_ARGUMENT_HINTS)


def _update_generation_observations(
    status_payload: dict[str, Any],
    *,
    argument_keys_seen: set[str],
    dynamic_argument_fields: set[str],
    raw_response_types: set[str],
    raw_response_keys: set[str],
) -> None:
    generation = status_payload.setdefault("generation", {})
    argument_policy = generation.setdefault("argument_policy", {})
    argument_policy["argument_keys"] = sorted(argument_keys_seen)
    argument_policy["dynamic_argument_fields"] = sorted(dynamic_argument_fields)
    generation["observed_raw_response"] = {
        "source_of_truth": False,
        "payload_types": sorted(raw_response_types),
        "payload_keys": sorted(raw_response_keys),
        "note": "raw_response is model wrapper output and is not used to infer protocol parameters.",
    }


def _remap_legacy_model_env_path(value: str, model_dir: Path) -> str:
    legacy_model_dir = f"/workspace/sure-eval/src/sure_eval/models/{model_dir.name}"
    if value == legacy_model_dir:
        return str(model_dir)
    if value.startswith(legacy_model_dir + "/"):
        return str(model_dir) + value[len(legacy_model_dir):]
    return value


def _send_request(
    process: subprocess.Popen[str],
    request: dict[str, Any],
) -> dict[str, Any]:
    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
    process.stdin.flush()

    while True:
        line = process.stdout.readline()
        if line == "":
            raise RuntimeError("Server exited before returning a response")
        line = line.strip()
        if not line:
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            # Ignore non-JSON stderr-like spillovers accidentally written to stdout.
            continue
        if response.get("id") == request.get("id"):
            return response


def _extract_response_payload(response: dict[str, Any]) -> Any:
    if "error" in response:
        raise RuntimeError(response["error"].get("message", "Unknown server error"))

    result = response.get("result", {})
    if isinstance(result, dict) and result.get("isError"):
        content = result.get("content") or []
        message = ""
        if content and isinstance(content[0], dict):
            message = str(content[0].get("text") or "")
        raise RuntimeError(message or "Tool call returned isError=true")
    content = result.get("content", [])
    if not content:
        return result

    text = content[0].get("text", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_prediction_payload(
    payload: Any,
    *,
    task: str,
    kws_require_score: bool = False,
    expected_audio_output: str | Path | None = None,
    vad_duration: float | None = None,
    forbidden_inputs: tuple[Path, ...] = (),
    sample_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    task_name = _normalize_task(task)
    if isinstance(payload, dict):
        nested_prediction = payload.get("prediction")
        if nested_prediction is not None and not isinstance(nested_prediction, dict):
            raise ValueError(f"{task_name or 'model'} prediction envelope must contain an object")
        prediction = dict(nested_prediction or {})
        if not prediction:
            prediction = dict(payload)
        if task_name == "TSE":
            if nested_prediction is not None:
                envelope_fields = sorted(str(field) for field in payload if field != "prediction")
                if envelope_fields:
                    raise ValueError(
                        "TSE prediction envelope contains unapproved field(s): "
                        + ", ".join(envelope_fields)
                    )
            unknown = sorted(str(field) for field in prediction if field not in TSE_OUTPUT_FIELDS)
            if unknown:
                raise ValueError(
                    "TSE prediction contains unapproved field(s): " + ", ".join(unknown)
                )
            prediction_audio = prediction.get("prediction_audio")
            if prediction_audio is None and "audio_path" in prediction:
                raise ValueError("TSE prediction must use prediction_audio, not audio_path")
            if expected_audio_output is None:
                raise ValueError("TSE prediction requires a harness-assigned output_path")
            resolved = _validate_tse_prediction_audio(
                prediction_audio,
                expected_path=expected_audio_output,
                forbidden_inputs=forbidden_inputs,
            )
            normalized: dict[str, Any] = {"prediction_audio": str(resolved)}
            returned_id = prediction.get("sample_id")
            if returned_id is not None:
                returned_id = _validate_tse_sample_key(returned_id)
                if sample_id is not None and returned_id != sample_id:
                    raise ValueError(
                        f"TSE prediction sample_id {returned_id!r} does not match {sample_id!r}"
                    )
                normalized["sample_id"] = returned_id
            elif sample_id is not None:
                normalized["sample_id"] = _validate_tse_sample_key(sample_id)
            return str(resolved), normalized
        if task_name in {"ASR", "S2TT"}:
            value = prediction.get("text") or prediction.get("transcript") or payload.get("text") or ""
            if isinstance(value, (list, tuple)) and len(value) == 1:
                # A wrapper that hands back {"text": ["…"]} is a normal MCP shape;
                # str() on the list would write the Python literal, brackets and
                # quotes included, straight into the prediction file.
                value = value[0]
            return str(value), {"text": str(value)}
        if task_name in {"TTS", "VC"}:
            value = (
                prediction.get("audio_path")
                or prediction.get("path")
                or prediction.get("generated_audio")
                or prediction.get("converted_audio")
                or payload.get("audio_path")
                or payload.get("path")
                or ""
            )
            normalized = {"audio_path": str(value)}
            if task_name == "VC":
                normalized["converted_audio"] = str(value)
                for key in ("source_audio_path", "reference_audio_path"):
                    if prediction.get(key) is not None:
                        normalized[key] = prediction[key]
            for key in ("sample_rate", "duration_ms"):
                if prediction.get(key) is not None:
                    normalized[key] = prediction[key]
            return str(value), normalized
        if task_name == "SE":
            value = (
                prediction.get("audio_path")
                or prediction.get("enhanced_audio")
                or prediction.get("path")
                or prediction.get("output_path")
            )
            if not value:
                raise ValueError("SE prediction payload must contain audio_path or enhanced_audio")
            audio_path = Path(str(value)).expanduser()
            if not audio_path.is_absolute():
                raise ValueError("SE prediction audio path must be absolute")
            audio_path = audio_path.absolute()
            if expected_audio_output is not None:
                expected_path = Path(expected_audio_output).expanduser().absolute()
                output_root = expected_path.parent
                if output_root.is_symlink() or output_root.resolve() != output_root:
                    raise ValueError("SE run-local output directory must not traverse a symlink")
                if audio_path != expected_path:
                    raise ValueError(
                        f"SE prediction audio path differs from run-local output_path: {audio_path}"
                    )
                if not audio_path.is_relative_to(output_root):
                    raise ValueError("SE prediction audio path escapes the run-local output directory")
            _validate_pcm_wav(audio_path, label="SE prediction audio")
            resolved_audio = str(audio_path)
            return resolved_audio, {
                "audio_path": resolved_audio,
                "enhanced_audio": resolved_audio,
            }
        if task_name in CLASSIFICATION_TASKS:
            # Only the model response is normalized here.  Reference fields
            # from the dataset never participate in this conversion.
            if isinstance(payload.get("prediction"), dict):
                envelope_fields = sorted(str(field) for field in payload if field != "prediction")
                if envelope_fields:
                    raise ValueError(
                        f"{task_name} prediction envelope contains unapproved field(s): "
                        + ", ".join(envelope_fields)
                    )
            raw_prediction: Any = prediction if prediction else payload
            scalar, normalized = normalize_classification_prediction(task_name, raw_prediction)
            return scalar, normalized
        if task_name in {"SD", "SA-ASR"}:
            if isinstance(payload.get("prediction"), dict):
                envelope_fields = sorted(str(field) for field in payload if field != "prediction")
                if envelope_fields:
                    raise ValueError(
                        f"{task_name} prediction envelope contains unapproved field(s): "
                        + ", ".join(envelope_fields)
                    )
            unknown_fields = sorted(
                str(field) for field in prediction if field not in {"segments", "num_speakers"}
            )
            if unknown_fields:
                raise ValueError(
                    f"{task_name} prediction contains unapproved field(s): "
                    + ", ".join(unknown_fields)
                )
            segments = _validate_annotation_prediction_segments(
                prediction.get("segments"), task=task_name
            )
            num_speakers = prediction.get("num_speakers")
            if num_speakers is not None:
                speakers = {str(segment["speaker"]).strip() for segment in segments}
                if (
                    isinstance(num_speakers, bool)
                    or not isinstance(num_speakers, int)
                    or num_speakers < 0
                    or num_speakers != len(speakers)
                ):
                    raise ValueError(
                        f"{task_name} num_speakers must equal distinct segment speakers"
                    )
            normalized = {"segments": segments}
            if num_speakers is not None:
                normalized["num_speakers"] = num_speakers
            return (
                json.dumps(segments, ensure_ascii=False, allow_nan=False),
                normalized,
            )
        if task_name == "VAD":
            if isinstance(payload.get("prediction"), dict):
                envelope_fields = sorted(str(field) for field in payload if field != "prediction")
                if envelope_fields:
                    raise ValueError(
                        "VAD prediction envelope contains unapproved field(s): "
                        + ", ".join(envelope_fields)
                    )
            unknown_fields = sorted(
                str(field)
                for field in prediction
                if field not in {"speech_segments", "frame_scores"}
            )
            if unknown_fields:
                raise ValueError(
                    "VAD prediction contains unapproved field(s): "
                    + ", ".join(unknown_fields)
                )
            if "speech_segments" not in prediction:
                raise ValueError("VAD prediction is missing speech_segments")
            normalized = {
                "speech_segments": _validate_vad_intervals(
                    prediction["speech_segments"],
                    role="speech_segments",
                    duration=vad_duration,
                )
            }
            if "frame_scores" in prediction:
                normalized["frame_scores"] = _validate_vad_intervals(
                    prediction["frame_scores"],
                    role="frame_scores",
                    duration=vad_duration,
                    with_score=True,
                )
            return (
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                normalized,
            )
        if task_name == "KWS":
            required_fields = ("detected", "keyword", "score")
            missing_fields = [field for field in required_fields if field not in prediction]
            if missing_fields:
                raise ValueError(
                    "KWS prediction payload is missing direct field(s): " + ", ".join(missing_fields)
                )
            detected = prediction["detected"]
            keyword = prediction["keyword"]
            score = prediction["score"]
            if not isinstance(detected, bool):
                raise ValueError("KWS prediction detected must be a bool")
            if keyword is not None and not isinstance(keyword, str):
                raise ValueError("KWS prediction keyword must be a string or null")
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError("KWS prediction score must be a finite number or null")
            if score is not None and not 0.0 <= float(score) <= 1.0:
                raise ValueError("KWS prediction score must be within [0, 1]")
            if detected and (not isinstance(keyword, str) or not keyword.strip()):
                raise ValueError("KWS detected prediction keyword must be a non-empty string")
            if detected and score is None:
                raise ValueError("KWS detected prediction score must be a finite number")
            if detected and float(score) < KWS_OPERATING_THRESHOLD:
                raise ValueError(
                    f"KWS detected prediction score must be >= {KWS_OPERATING_THRESHOLD}"
                )
            if not detected and keyword is not None:
                raise ValueError("KWS rejected prediction keyword must be null")
            if not detected and score is not None and float(score) >= KWS_OPERATING_THRESHOLD:
                raise ValueError(
                    f"KWS rejected prediction score must be < {KWS_OPERATING_THRESHOLD}"
                )
            if kws_require_score and score is None:
                raise ValueError("KWS formal score-sweep generation requires a score for every sample")
            normalized = {
                "detected": detected,
                "keyword": keyword,
                "score": score,
            }
            if "events" in prediction:
                events = prediction["events"]
                if not isinstance(events, list):
                    raise ValueError("KWS prediction events must be a list when provided")
                normalized["events"] = events
            return (
                json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                normalized,
            )
        value = payload.get("text", "")
        return str(value), {"text": str(value)}

    value = str(payload)
    if task_name in {"TTS", "VC"}:
        normalized = {"audio_path": value}
        if task_name == "VC":
            normalized["converted_audio"] = value
        return value, normalized
    if task_name == "TSE":
        raise ValueError("TSE prediction payload must be a JSON object with prediction_audio")
    if task_name in CLASSIFICATION_TASKS:
        return normalize_classification_prediction(task_name, value)
    if task_name in {"SD", "SA-ASR"}:
        raise ValueError(f"{task_name} prediction payload must be a JSON object with segments")
    if task_name == "VAD":
        raise ValueError("VAD prediction payload must be a JSON object with speech_segments")
    if task_name == "SE":
        raise ValueError("SE prediction payload must be a JSON object with an audio path")
    if task_name == "KWS":
        raise ValueError("KWS prediction payload must be a JSON object with detected, keyword, and score")
    return value, {"text": value}


def _load_existing_predictions(path: Path, *, exclude_keys: set[str] | None = None) -> dict[str, str]:
    predictions: dict[str, str] = {}
    if not path.exists():
        return predictions
    excluded = exclude_keys or set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                key, value = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                key = parts[0]
                value = parts[1] if len(parts) > 1 else ""
            if key in excluded:
                continue
            if value.strip():
                predictions[key] = value
    return predictions


def _load_existing_structured_predictions(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("key") or row.get("sample_id") or "")
            if key:
                records[key] = row
    return records


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _write_prediction_snapshots(
    *,
    samples: list[dict[str, Any]],
    prediction_path: Path,
    structured_prediction_path: Path,
    prediction_map: dict[str, str],
    structured_map: dict[str, dict[str, Any]],
    canonical_dataset: str,
    sample_task: str,
    sample_language: str,
) -> None:
    prediction_tmp = prediction_path.with_name(f"{prediction_path.name}.tmp")
    structured_tmp = structured_prediction_path.with_name(f"{structured_prediction_path.name}.tmp")

    with open(prediction_tmp, "w", encoding="utf-8") as handle:
        for sample in samples:
            key = _sample_key(sample)
            handle.write(f"{key}\t{prediction_map.get(key, '')}\n")
    prediction_tmp.replace(prediction_path)

    with open(structured_tmp, "w", encoding="utf-8") as handle:
        for sample in samples:
            key = _sample_key(sample)
            row = structured_map.get(
                key,
                {
                    "key": key,
                    **({"sample_id": _validate_tse_sample_key(sample.get("sample_id") or key)} if sample_task == "TSE" else {}),
                    "dataset": canonical_dataset,
                    "task": sample_task,
                    "language": str(sample.get("language") or sample_language),
                    "prediction": {},
                    "normalized_prediction": prediction_map.get(key, ""),
                    "raw_response": None,
                },
            )
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    structured_tmp.replace(structured_prediction_path)


def _write_existing_result_log_entries(
    result_log_handle: Any,
    samples: list[dict[str, Any]],
    predictions: dict[str, str],
) -> None:
    written: set[str] = set()
    for sample in samples:
        key = _sample_key(sample)
        if key in predictions:
            result_log_handle.write(f"{key}\t{predictions[key]}\n")
            written.add(key)
    for key, value in predictions.items():
        if key not in written:
            result_log_handle.write(f"{key}\t{value}\n")


def _upsert_dataset_status(
    status_path: Path,
    default_payload: dict[str, Any],
    dataset_status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = dict(default_payload)
    else:
        payload = dict(default_payload)
    for key, value in default_payload.items():
        if key != "datasets":
            if key == "generated_at" and payload.get("generated_at"):
                continue
            payload[key] = value
    datasets = list(payload.get("datasets") or [])
    dataset_name = dataset_status.get("dataset")
    for index, row in enumerate(datasets):
        if row.get("dataset") == dataset_name:
            merged = dict(row)
            merged.update(dataset_status)
            datasets[index] = merged
            payload["datasets"] = datasets
            return payload, datasets[index]
    datasets.append(dict(dataset_status))
    payload["datasets"] = datasets
    return payload, datasets[-1]


def _write_prediction_manifests(
    *,
    predictions_dir: Path,
    run_dir: Path,
    model_name: str,
    tool_name: str,
    dataset: str,
    task: str,
    language: str,
    prediction_path: Path,
    structured_prediction_path: Path,
    protocol_id: str | None,
    source_samples: int,
    generated_samples: int,
) -> tuple[Path, Path]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = predictions_dir / "manifest.json"
    conversion_path = predictions_dir / "conversion_manifest.json"
    txt_exists = prediction_path.is_file()
    jsonl_exists = structured_prediction_path.is_file()
    row = {
        "dataset": dataset,
        "task": task,
        "language": language,
        "format_used": "jsonl+txt" if jsonl_exists else "txt",
        "txt": str(prediction_path),
        "jsonl": str(structured_prediction_path) if jsonl_exists else None,
        "txt_sha256": _sha256(prediction_path) if txt_exists else None,
        "jsonl_sha256": _sha256(structured_prediction_path) if jsonl_exists else None,
        "num_rows": _count_nonempty_lines(prediction_path),
        "structured_num_rows": _count_nonempty_lines(structured_prediction_path) if jsonl_exists else 0,
        "source_samples": source_samples,
        "generated_samples": generated_samples,
        "protocol_id": protocol_id,
    }
    conversion_row = {
        "dataset": dataset,
        "source_format": "model_mcp_tool_response",
        "format_used": row["format_used"],
        "num_rows": row["num_rows"],
        "source_artifacts": {
            "raw_response_field": "predictions/<dataset>.jsonl:raw_response",
            "structured_jsonl": str(structured_prediction_path) if jsonl_exists else None,
            "compatibility_tsv": str(prediction_path),
        },
        "steps": [
            {
                "name": "raw_response_to_prediction",
                "input": "MCP tools/call JSON-RPC response payload",
                "output": "prediction object and normalized_prediction scalar/path",
                "script": "scripts/generate_predictions_via_server.py:_normalize_prediction_payload",
            },
            {
                "name": "structured_prediction_to_tsv_projection",
                "input": "predictions/<dataset>.jsonl normalized_prediction",
                "output": "predictions/<dataset>.txt key<TAB>normalized_prediction",
                "script": "scripts/generate_predictions_via_server.py:_write_prediction_snapshots",
            },
        ],
        "conversion_trace": None,
    }

    existing_manifest = _load_yaml(manifest_path) if manifest_path.is_file() else {}
    existing_conversion = _load_yaml(conversion_path) if conversion_path.is_file() else {}
    existing_datasets = [
        item
        for item in existing_manifest.get("datasets", [])
        if isinstance(item, dict) and item.get("dataset") != dataset
    ]
    existing_conversion_datasets = [
        item
        for item in existing_conversion.get("datasets", [])
        if isinstance(item, dict) and item.get("dataset") != dataset
    ]
    generated_at = _utc_now()
    prediction_manifest = {
        "schema": "sure.eval.prediction_manifest.v1",
        "generated_at": generated_at,
        "run_id": _artifact_run_id(run_dir),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "tool_name": tool_name,
        "predictions_dir": str(predictions_dir),
        "datasets": existing_datasets + [row],
    }
    conversion_manifest = {
        "schema": "sure.eval.prediction_conversion_manifest.v1",
        "generated_at": generated_at,
        "run_id": _artifact_run_id(run_dir),
        "run_dir": str(run_dir),
        "generated_by": "scripts/generate_predictions_via_server.py",
        "predictions_dir": str(predictions_dir),
        "datasets": existing_conversion_datasets + [conversion_row],
    }
    manifest_path.write_text(json.dumps(prediction_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    conversion_path.write_text(json.dumps(conversion_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path, conversion_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate predictions by calling a model-local MCP server")
    parser.add_argument("--model-dir", required=True, help="Resolved model directory containing config.yaml")
    parser.add_argument("--dataset", required=True, help="Canonical dataset name")
    parser.add_argument("--run-dir", required=True, help="Run directory under eval_runs")
    parser.add_argument("--tool-name", help="Tool name to call; defaults to the first configured tool")
    parser.add_argument("--argument-name", default="audio_path", help="Argument name for the audio path")
    parser.add_argument("--language", help="Optional language argument passed through to the tool")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional limit for quick tests")
    parser.add_argument("--resume", action="store_true", help="Resume and skip keys already present in the prediction file")
    parser.add_argument(
        "--resume-exclude-keys-file",
        help="Optional newline-delimited keys to ignore while loading existing resume predictions.",
    )
    parser.add_argument("--config", help="Optional sure-eval config path")
    parser.add_argument(
        "--protocol",
        choices=("standard_system", "strict_core"),
        default="standard_system",
        help="Inference protocol ID (default: standard_system).",
    )
    parser.add_argument(
        "--device",
        help="Device override for model inference (e.g., cuda:0, cuda:1, cpu). "
             "If set, overrides the DEVICE env var from config.yaml. "
             "When set to cpu, CUDA_VISIBLE_DEVICES is hidden unless already configured.",
    )
    parser.add_argument(
        "--tool-arg",
        action="append",
        default=[],
        help="Extra MCP tool argument in key=value form. Values are parsed as JSON when possible.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra model server environment override in KEY=VALUE form; repeatable.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    model_dir = Path(args.model_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    predictions_dir = run_dir / "predictions"
    logs_dir = predictions_dir / "logs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_yaml(args.config) if args.config else Config.from_env()
    dataset_manager = DatasetManager(cfg)
    expanded = dataset_manager.expand_dataset_names([args.dataset])
    if len(expanded) != 1 or dataset_manager.normalize_dataset_name(args.dataset) != expanded[0]:
        raise ValueError(
            "generate_predictions_via_server.py expects one concrete dataset split; "
            f"{args.dataset!r} expands to {expanded}"
        )
    canonical_dataset = dataset_manager.normalize_dataset_name(args.dataset)
    jsonl_path = dataset_manager.get_jsonl_path(canonical_dataset)
    if not jsonl_path.exists():
        jsonl_path = dataset_manager.download_and_convert(canonical_dataset)

    samples = _load_jsonl(jsonl_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    sample_task = str(samples[0].get("task", "ASR")) if samples else "ASR"
    sample_language = str(samples[0].get("language", "")) if samples else ""

    model_cfg = _load_yaml(model_dir / "config.yaml")
    generation_metrics = _split_metrics(
        os.environ.get("SURE_EVAL_METRICS") or os.environ.get("METRICS")
    )
    sample_task = _effective_generation_task(
        sample_task,
        model_cfg,
        generation_metrics,
    )
    runtime_inventory_document = _load_runtime_inventory(model_dir)
    server_cfg = model_cfg.get("server", {})
    command = _resolve_server_command(model_dir, runtime_inventory_document)
    working_dir = _resolve_working_dir(model_dir, runtime_inventory_document)
    env = model_child_env()
    server_env_config: dict[str, str] = {}
    writable_cache_keys = {
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "MODELSCOPE_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    }
    for key, value in (server_cfg.get("env", {}) or {}).items():
        key = str(key)
        configured = _remap_legacy_model_env_path(str(value), model_dir)
        if key in writable_cache_keys and env.get(key):
            configured = env[key]
        server_env_config[key] = configured
        env[key] = configured

    # Override DEVICE if --device is explicitly provided
    if args.device:
        env["DEVICE"] = str(args.device)
        if str(args.device).lower() == "cpu" and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ""
    env_overrides = _parse_env_overrides(args.env)
    env.update(env_overrides)
    protocol_id = args.protocol
    protocol_resolution = _resolve_protocol_parameters(protocol_id, model_dir, env)

    tools = model_cfg.get("tools", [])
    tool_name = args.tool_name or (tools[0]["name"] if tools else None)
    if not tool_name:
        raise ValueError("No tool name provided and config.yaml has no tools entry")
    runtime_policy = runtime_inventory_document.get("policy") if isinstance(runtime_inventory_document.get("policy"), dict) else {}
    selected_runtime = (
        runtime_inventory_document.get("model_runtime")
        if runtime_policy.get("eval_runtime") == "python"
        else runtime_inventory_document.get("container_runtime")
    )
    approved_tools = selected_runtime.get("tool_names") if isinstance(selected_runtime, dict) else []
    if tool_name not in approved_tools:
        raise ValueError(f"tool {tool_name!r} is not present in the approved runtime inventory: {approved_tools}")
    tool_args = _merge_protocol_tool_args(
        protocol_id,
        protocol_resolution,
        _parse_tool_args(args.tool_arg),
        _declared_tool_args(model_cfg, tool_name),
    )
    runtime_inventory = _runtime_inventory_summary(model_dir)
    safe_env = _safe_env_snapshot(env, extra_keys=set(server_env_config) | set(env_overrides))

    prediction_path = predictions_dir / f"{canonical_dataset}.txt"
    structured_prediction_path = predictions_dir / f"{canonical_dataset}.jsonl"
    output_audio_dir = predictions_dir / "audio" / canonical_dataset
    log_path = logs_dir / f"{canonical_dataset}.log"
    result_log_path = logs_dir / f"{canonical_dataset}_results.log"
    status_path = run_dir / "prediction_generation_status.json"

    resume_exclude_keys: set[str] = set()
    if args.resume and args.resume_exclude_keys_file:
        exclude_path = Path(args.resume_exclude_keys_file)
        resume_exclude_keys = {
            line.strip()
            for line in exclude_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    existing_predictions = _load_existing_predictions(prediction_path, exclude_keys=resume_exclude_keys) if args.resume else {}
    if args.resume:
        existing_predictions.update(_load_existing_predictions(result_log_path, exclude_keys=resume_exclude_keys))
    existing_structured = _load_existing_structured_predictions(structured_prediction_path) if args.resume else {}
    prediction_map = dict(existing_predictions)
    structured_map = dict(existing_structured)

    default_status_payload: dict[str, Any] = {
        "schema": "sure.eval.prediction_generation_status.v2",
        "generated_at": _utc_now(),
        "updated_at": _utc_now(),
        "run_id": _artifact_run_id(run_dir),
        "run_dir": str(run_dir),
        "model_name": model_dir.name,
        "model_dir": str(model_dir),
        "execution_path": env.get("SURE_EVAL_EXECUTION_PATH", "unknown"),
        "execution_requested": env.get("SURE_EVAL_EXECUTION_REQUESTED", ""),
        "execution_job_id": env.get("SURE_EVAL_EXECUTION_JOB_ID", ""),
        "inference_call_mode": "direct_server_use",
        "protocol_id": protocol_id,
        "tool_name": tool_name,
        "host": socket.gethostname(),
        "device_request": env.get("SURE_EVAL_DEVICE_REQUEST", args.device or ""),
        "device_actual": env.get("SURE_EVAL_DEVICE_ACTUAL", args.device or ""),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "runtime": {
            "server_command": command,
            "server_working_dir": str(working_dir),
            "model_python": command[0] if command else None,
            "harness_python": env.get("HARNESS_PYTHON_BIN") or sys.executable,
            "harness_runtime": _harness_runtime_summary(env),
            "server_config": {
                "working_dir": server_cfg.get("working_dir", "."),
                "timeout": server_cfg.get("timeout"),
                "startup_timeout_sec": server_cfg.get("startup_timeout_sec"),
                "env_keys": sorted(server_env_config),
            },
            "runtime_inventory": runtime_inventory,
        },
        "environment": {
            **safe_env,
            "server_env_values": _redact_mapping(server_env_config),
            "cli_env_overrides": _redact_mapping(env_overrides),
            "execution": {
                "path": env.get("SURE_EVAL_EXECUTION_PATH", "unknown"),
                "requested": env.get("SURE_EVAL_EXECUTION_REQUESTED", ""),
                "job_id": env.get("SURE_EVAL_EXECUTION_JOB_ID", ""),
                "surface_type": env.get("SURE_EVAL_EXECUTION_SURFACE_TYPE", ""),
            },
            "device": {
                "request": env.get("SURE_EVAL_DEVICE_REQUEST", args.device or ""),
                "actual": env.get("SURE_EVAL_DEVICE_ACTUAL", args.device or ""),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            },
        },
        "generation": {
            "protocol_id": protocol_id,
            "protocol_resolution": _redact_mapping(protocol_resolution),
            "tool_name": tool_name,
            "tool_args": _redact_mapping(tool_args),
            "argument_policy": {
                "argument_name": args.argument_name,
                "language_argument": args.language,
                "constant_arguments": _redact_mapping(tool_args),
                "dynamic_argument_fields": [],
                "argument_keys": [],
                "per_sample_arguments_materialized": False,
                "note": "Actual MCP tools/call arguments are generated per sample; only key policy and explicit overrides are persisted.",
            },
            "observed_raw_response": {
                "source_of_truth": False,
                "payload_types": [],
                "payload_keys": [],
            },
        },
    }
    dataset_status = {
        "dataset": canonical_dataset,
        "prediction_file": str(prediction_path),
        "structured_prediction_file": str(structured_prediction_path),
        "status": "running",
        "num_expected_samples": len(samples),
        "num_generated_samples": len(prediction_map),
        "log_path": str(log_path),
        "result_log_path": str(result_log_path),
        "error": None,
    }
    status_payload, current_dataset_status = _upsert_dataset_status(status_path, default_status_payload, dataset_status)
    generation_started = monotonic()
    argument_keys_seen: set[str] = set()
    dynamic_argument_fields: set[str] = set()
    raw_response_types: set[str] = set()
    raw_response_keys: set[str] = set()
    _update_generation_observations(
        status_payload,
        argument_keys_seen=argument_keys_seen,
        dynamic_argument_fields=dynamic_argument_fields,
        raw_response_types=raw_response_types,
        raw_response_keys=raw_response_keys,
    )
    status_path.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with open(log_path, "w", encoding="utf-8") as log_handle, open(result_log_path, "w", encoding="utf-8") as result_log_handle:
        if args.resume and existing_predictions:
            _write_existing_result_log_entries(result_log_handle, samples, existing_predictions)
            result_log_handle.flush()

        process = subprocess.Popen(
            command,
            cwd=str(working_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            bufsize=1,
        )

        try:
            initialize = _send_request(
                process,
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
            if "error" in initialize:
                raise RuntimeError(initialize["error"].get("message", "initialize failed"))

            tools_list = _send_request(
                process,
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            if "error" in tools_list:
                raise RuntimeError(tools_list["error"].get("message", "tools/list failed"))

            next_id = 3
            with tempfile.TemporaryDirectory(prefix=f"sure-eval-{canonical_dataset}-audio-") as scratch:
                scratch_dir = Path(scratch)
                for sample in samples:
                    key = _sample_key(sample)
                    sample_identity = (
                        _validate_tse_sample_key(sample.get("sample_id") or key)
                        if sample_task == "TSE"
                        else key
                    )
                    if args.resume and key in prediction_map:
                        continue

                    audio_path = _materialize_sample_audio(repo_root, sample, scratch_dir)
                    arguments = _build_tool_arguments(
                        repo_root=repo_root,
                        sample=sample,
                        task=sample_task,
                        language=args.language or sample_language,
                        argument_name=args.argument_name,
                        audio_path=audio_path,
                        output_audio_dir=output_audio_dir,
                        tool_args=tool_args,
                    )
                    argument_keys_seen.update(str(key) for key in arguments)
                    dynamic_argument_fields.update(
                        str(key)
                        for key in arguments
                        if key not in tool_args and _is_dynamic_argument_key(str(key))
                    )

                    response = _send_request(
                        process,
                        {
                            "jsonrpc": "2.0",
                            "id": next_id,
                            "method": "tools/call",
                            "params": {"name": tool_name, "arguments": arguments},
                        },
                    )
                    next_id += 1
                    raw_payload = _extract_response_payload(response)
                    raw_response_types.add(type(raw_payload).__name__)
                    tse_forbidden_inputs: tuple[Path, ...] = ()
                    if sample_task == "TSE":
                        tse_forbidden = _resolve_audio_field_path(
                            repo_root, sample.get("reference_audio")
                        )
                        tse_forbidden_inputs = tuple(
                            Path(str(arguments[field]))
                            for field in ("mixture_audio_path", "enrollment_audio_path")
                            if arguments.get(field)
                        ) + ((tse_forbidden,) if tse_forbidden is not None else ())
                    prediction, normalized_prediction = _normalize_prediction_payload(
                        raw_payload,
                        task=sample_task,
                        kws_require_score=_kws_metrics_require_scores(generation_metrics),
                        expected_audio_output=(
                            arguments.get("output_path")
                            if sample_task in {"SE", "TSE"}
                            else None
                        ),
                        forbidden_inputs=tse_forbidden_inputs,
                        sample_id=sample_identity if sample_task == "TSE" else None,
                        vad_duration=(
                            _validate_pcm_wav(audio_path, label="VAD input audio")
                            if sample_task == "VAD"
                            else None
                        ),
                    )
                    if isinstance(raw_payload, dict):
                        observed_payload = (
                            normalized_prediction
                            if sample_task in {"SD", "SA-ASR", "VAD", "TSE", *CLASSIFICATION_TASKS}
                            else raw_payload
                        )
                        raw_response_keys.update(str(key) for key in observed_payload)
                    prediction_map[key] = prediction
                    raw_response_evidence = (
                        normalized_prediction
                        if sample_task in {"SD", "SA-ASR", "VAD", "TSE", *CLASSIFICATION_TASKS}
                        else raw_payload
                    )
                    structured_map[key] = {
                        "key": key,
                        **({"sample_id": sample_identity} if sample_task == "TSE" else {}),
                        "dataset": canonical_dataset,
                        "task": sample_task,
                        "language": str(sample.get("language") or sample_language),
                        "prediction": normalized_prediction,
                        "normalized_prediction": prediction,
                        "raw_response": raw_response_evidence,
                    }
                    result_log_handle.write(f"{key}\t{prediction}\n")
                    result_log_handle.flush()

                    current_dataset_status["num_generated_samples"] = len(prediction_map)
                    status_payload["updated_at"] = _utc_now()
                    _update_generation_observations(
                        status_payload,
                        argument_keys_seen=argument_keys_seen,
                        dynamic_argument_fields=dynamic_argument_fields,
                        raw_response_types=raw_response_types,
                        raw_response_keys=raw_response_keys,
                    )
                    status_path.write_text(
                        json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    if len(prediction_map) % PREDICTION_SNAPSHOT_INTERVAL == 0:
                        # The only line this script logged used to be its last one,
                        # so a five-hour generation pass looked identical to a hung
                        # one. Counting lines in the prediction file does not help:
                        # the snapshot materializes every row up front.
                        generated = len(prediction_map)
                        elapsed = max(monotonic() - generation_started, 1e-9)
                        remaining = max(len(samples) - generated, 0)
                        logger.info(
                            "Generating predictions",
                            dataset=canonical_dataset,
                            generated=generated,
                            expected=len(samples),
                            elapsed_seconds=round(elapsed, 1),
                            seconds_per_sample=round(elapsed / generated, 3),
                            eta_seconds=round(remaining * elapsed / generated, 1),
                        )
                        _write_prediction_snapshots(
                            samples=samples,
                            prediction_path=prediction_path,
                            structured_prediction_path=structured_prediction_path,
                            prediction_map=prediction_map,
                            structured_map=structured_map,
                            canonical_dataset=canonical_dataset,
                            sample_task=sample_task,
                            sample_language=sample_language,
                        )

            _write_prediction_snapshots(
                samples=samples,
                prediction_path=prediction_path,
                structured_prediction_path=structured_prediction_path,
                prediction_map=prediction_map,
                structured_map=structured_map,
                canonical_dataset=canonical_dataset,
                sample_task=sample_task,
                sample_language=sample_language,
            )
            manifest_path, conversion_manifest_path = _write_prediction_manifests(
                predictions_dir=predictions_dir,
                run_dir=run_dir,
                model_name=model_dir.name,
                tool_name=tool_name,
                dataset=canonical_dataset,
                task=sample_task,
                language=sample_language,
                prediction_path=prediction_path,
                structured_prediction_path=structured_prediction_path,
                protocol_id=args.protocol if args.protocol.lower() != "none" else None,
                source_samples=len(samples),
                generated_samples=len(prediction_map),
            )

            current_dataset_status["status"] = "completed"
            current_dataset_status["num_generated_samples"] = len(samples)
            current_dataset_status["prediction_manifest"] = str(manifest_path)
            current_dataset_status["conversion_manifest"] = str(conversion_manifest_path)
            status_payload["updated_at"] = _utc_now()
            _update_generation_observations(
                status_payload,
                argument_keys_seen=argument_keys_seen,
                dynamic_argument_fields=dynamic_argument_fields,
                raw_response_types=raw_response_types,
                raw_response_keys=raw_response_keys,
            )
            status_path.write_text(
                json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        except Exception as exc:
            current_dataset_status["status"] = "failed"
            current_dataset_status["error"] = str(exc)
            current_dataset_status["num_generated_samples"] = len(prediction_map)
            status_payload["updated_at"] = _utc_now()
            _update_generation_observations(
                status_payload,
                argument_keys_seen=argument_keys_seen,
                dynamic_argument_fields=dynamic_argument_fields,
                raw_response_types=raw_response_types,
                raw_response_keys=raw_response_keys,
            )
            _write_prediction_snapshots(
                samples=samples,
                prediction_path=prediction_path,
                structured_prediction_path=structured_prediction_path,
                prediction_map=prediction_map,
                structured_map=structured_map,
                canonical_dataset=canonical_dataset,
                sample_task=sample_task,
                sample_language=sample_language,
            )
            status_path.write_text(
                json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise
        finally:
            try:
                _send_request(
                    process,
                    {"jsonrpc": "2.0", "id": 999999, "method": "shutdown", "params": {}},
                )
            except Exception:
                pass
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            process.wait(timeout=30)

    logger.info(
        "Generated predictions via model-local server",
        dataset=canonical_dataset,
        prediction_file=str(prediction_path),
        result_log_file=str(result_log_path),
        status_file=str(status_path),
    )
    print(
        json.dumps(
            {
                "dataset": canonical_dataset,
                "prediction_file": str(prediction_path),
                "structured_prediction_file": str(structured_prediction_path),
                "result_log_file": str(result_log_path),
                "status_file": str(status_path),
                "prediction_manifest": str(predictions_dir / "manifest.json"),
                "conversion_manifest": str(predictions_dir / "conversion_manifest.json"),
                "protocol_id": args.protocol if args.protocol.lower() != "none" else None,
                "num_samples": len(samples),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
