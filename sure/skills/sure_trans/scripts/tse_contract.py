"""Strict target-speaker-extraction (TSE) adapter contract.

TSE has two different data surfaces which must not be conflated:

* model inference receives a mixture and an enrollment recording;
* evaluation receives the generated prediction plus clean/reference roles.

This module contains only small, dependency-free helpers so the Trans scripts
and generated adapters can share the same canonical names and path policy.
"""

from __future__ import annotations

import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any


TSE_TASK = "tse"
TSE_TOOL = "extract_target_speaker"
TSE_INPUT_FIELDS = frozenset(
    {"mixture_audio_path", "enrollment_audio_path", "output_path"}
)
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
    """Normalize TSE names while leaving unrelated task names untouched."""

    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "tse": TSE_TASK,
        "target_speaker_extraction": TSE_TASK,
        "target_speaker_extractor": TSE_TASK,
        "target_speaker_extraction_model": TSE_TASK,
        "target_speaker": TSE_TASK,
        "speaker_extraction": TSE_TASK,
        "target_voice_extraction": TSE_TASK,
        "target_voice_separation": TSE_TASK,
    }
    return aliases.get(normalized, normalized)


def tool_name_for(value: Any) -> str:
    if canonical_task(value) != TSE_TASK:
        raise ValueError(f"unsupported TSE task: {value!r}")
    return TSE_TOOL


def input_schema_for(value: Any) -> dict[str, Any]:
    if canonical_task(value) != TSE_TASK:
        raise ValueError(f"unsupported TSE task: {value!r}")
    return {
        "type": "object",
        "properties": {
            "mixture_audio_path": {"type": "string", "minLength": 1},
            "enrollment_audio_path": {"type": "string", "minLength": 1},
            "output_path": {"type": "string", "minLength": 1},
        },
        "required": [
            "mixture_audio_path",
            "enrollment_audio_path",
            "output_path",
        ],
        "additionalProperties": False,
    }


def io_contract_for(value: Any) -> dict[str, Any]:
    if canonical_task(value) != TSE_TASK:
        raise ValueError(f"unsupported TSE task: {value!r}")
    return {
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


def looks_like_absolute_path_or_uri(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return bool(
        URI_PREFIX.match(stripped)
        or Path(stripped).is_absolute()
        or PureWindowsPath(stripped).is_absolute()
        or stripped.startswith("\\\\")
    )


def safe_relative_audio(value: Any, *, role: str) -> Path:
    """Validate an audio role path stored in a fixture JSONL row."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TSE {role} is required")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in value
        or looks_like_absolute_path_or_uri(value)
    ):
        raise ValueError(f"TSE {role} path must be relative and contained")
    return relative


def safe_sample_id(value: Any, *, role: str = "sample_id") -> str:
    token = str(value or "").strip()
    if (
        not token
        or looks_like_absolute_path_or_uri(token)
        or "/" in token
        or "\\" in token
        or any(ord(char) < 32 or char.isspace() for char in token)
    ):
        raise ValueError(f"TSE {role} must be a safe non-empty token")
    return token


def validate_output_object(
    value: Any,
    *,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Validate and return a canonical model output object.

    The output object deliberately has no reference or input fields.  File
    location checks are performed by each execution surface because each has a
    different controlled output root.
    """

    if not isinstance(value, dict):
        raise ValueError("TSE prediction must be a JSON object")
    unknown = sorted(str(key) for key in value if key not in TSE_OUTPUT_FIELDS)
    if unknown:
        raise ValueError("TSE prediction contains unapproved field(s): " + ", ".join(unknown))
    for key in value:
        normalized = str(key).strip().lower().replace("-", "_")
        if normalized in TSE_REFERENCE_FIELDS or normalized.endswith("_path"):
            raise ValueError(f"TSE prediction contains forbidden reference/input field: {key}")
    prediction_audio = value.get("prediction_audio")
    if not isinstance(prediction_audio, str) or not prediction_audio.strip():
        raise ValueError("TSE prediction requires a non-empty prediction_audio")
    output: dict[str, Any] = {"prediction_audio": prediction_audio.strip()}
    returned_id = value.get("sample_id")
    if returned_id is not None:
        returned_id = safe_sample_id(returned_id)
        if sample_id is not None and returned_id != sample_id:
            raise ValueError(
                f"TSE prediction sample_id {returned_id!r} does not match {sample_id!r}"
            )
        output["sample_id"] = returned_id
    elif sample_id is not None:
        output["sample_id"] = safe_sample_id(sample_id)
    return output


def validate_numeric_duration(value: Any, *, role: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"TSE {role} duration must be a finite positive number")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"TSE {role} duration must be a finite positive number")
    return duration


__all__ = [
    "TSE_INPUT_FIELDS",
    "TSE_OUTPUT_FIELDS",
    "TSE_REFERENCE_FIELDS",
    "TSE_TASK",
    "TSE_TOOL",
    "canonical_task",
    "input_schema_for",
    "io_contract_for",
    "looks_like_absolute_path_or_uri",
    "safe_relative_audio",
    "safe_sample_id",
    "tool_name_for",
    "validate_numeric_duration",
    "validate_output_object",
]
