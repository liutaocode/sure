"""Dependency-free TSE fixture and output contract helpers for onboarding."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Any


TSE_TASK = "tse"
TSE_OUTPUT_FIELDS = frozenset({"prediction_audio", "sample_id"})
TSE_REFERENCE_FIELDS = frozenset(
    {
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
)
URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def canonical_task(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return TSE_TASK if normalized in {
        "tse",
        "target_speaker_extraction",
        "target_speaker_extractor",
        "target_speaker_extraction_model",
        "target_speaker",
        "speaker_extraction",
        "target_voice_extraction",
        "target_voice_separation",
    } else normalized


def looks_like_absolute_path_or_uri(value: Any) -> bool:
    if not isinstance(value, str):
        return False
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


def safe_sample_id(value: Any) -> str:
    token = str(value or "").strip()
    if (
        not token
        or "/" in token
        or "\\" in token
        or any(ord(character) < 32 or character.isspace() for character in token)
        or looks_like_absolute_path_or_uri(token)
    ):
        raise ValueError("TSE sample_id must be a safe non-empty token")
    return token


def safe_relative_audio(value: Any, *, role: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TSE fixture {role} is required")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in value
        or looks_like_absolute_path_or_uri(value)
    ):
        raise ValueError(f"TSE fixture {role} path must be relative and contained")
    return relative


def validate_output_object(value: Any, *, sample_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("TSE prediction must be a JSON object")
    unknown = sorted(str(key) for key in value if key not in TSE_OUTPUT_FIELDS)
    if unknown:
        raise ValueError("TSE prediction contains unapproved field(s): " + ", ".join(unknown))
    for key in value:
        normalized = str(key).strip().lower().replace("-", "_")
        if normalized != "prediction_audio" and (
            normalized.endswith("_path") or normalized in TSE_REFERENCE_FIELDS
        ):
            raise ValueError(f"TSE prediction contains forbidden reference/input field: {key}")
    prediction_audio = value.get("prediction_audio")
    if not isinstance(prediction_audio, str) or not prediction_audio.strip():
        raise ValueError("TSE prediction requires a non-empty prediction_audio")
    result: dict[str, Any] = {"prediction_audio": prediction_audio.strip()}
    returned_id = value.get("sample_id")
    if returned_id is not None:
        returned_id = safe_sample_id(returned_id)
        if sample_id is not None and returned_id != sample_id:
            raise ValueError(f"TSE prediction sample_id {returned_id!r} does not match {sample_id!r}")
        result["sample_id"] = returned_id
    elif sample_id is not None:
        result["sample_id"] = safe_sample_id(sample_id)
    return result


def task_contract() -> dict[str, Any]:
    return {
        "tool_name": "extract_target_speaker",
        "predict_method": "predict",
        "input_fields": ["mixture_audio_path", "enrollment_audio_path", "output_path"],
        "public_inference_parameters": [],
        "io_contract": {
            **{
                "input_type": "audio_pair",
                "output_type": "audio",
                "input": {
                    "mixture_audio_path": "string",
                    "enrollment_audio_path": "string",
                    "output_path": "string",
                },
                "output": {"prediction_audio": "string", "sample_id": "optional string"},
                "primary_field": "prediction_audio",
                "required_fields": ["prediction_audio"],
                "nonempty_fields": ["prediction_audio"],
                "approved_output_fields": ["prediction_audio", "sample_id"],
                "json_serializable": True,
            }
        },
    }


__all__ = [
    "TSE_OUTPUT_FIELDS",
    "TSE_REFERENCE_FIELDS",
    "TSE_TASK",
    "canonical_task",
    "safe_relative_audio",
    "safe_sample_id",
    "validate_output_object",
    "task_contract",
]
