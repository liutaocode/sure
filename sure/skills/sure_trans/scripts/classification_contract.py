"""Shared SER/GR/SLU contracts for the transformation skill.

The standalone evaluator consumes keyed scalar text files, while the model
adapter needs a structured JSON record to preserve task and sample identity.
This module keeps those two representations equivalent and rejects reference
fields before they can reach a model invocation.
"""

from __future__ import annotations

import math
import re
from typing import Any


CLASSIFICATION_TASKS = frozenset({"ser", "gr", "slu"})
CLASSIFICATION_TOOLS = {
    "ser": "emotion_recognize",
    "gr": "gender_recognize",
    "slu": "slu_understand",
}
CLASSIFICATION_PRIMARY_FIELDS = {
    "ser": "label",
    "gr": "label",
    "slu": "answer",
}
CLASSIFICATION_OUTPUT_FIELDS = {
    "ser": frozenset({"label", "score"}),
    "gr": frozenset({"label", "score"}),
    "slu": frozenset({"answer", "label"}),
}
REFERENCE_FIELDS = frozenset(
    {
        "answer",
        "expected",
        "ground_truth",
        "reference",
        "reference_audio",
        "reference_text",
        "target",
        "target_text",
    }
)
PREDICTION_REFERENCE_FIELDS = REFERENCE_FIELDS - {"answer"}
CHOICE_REFERENCE_FIELDS = REFERENCE_FIELDS | {"target_audio", "target_audio_path"}

SER_LABEL_ALIASES = {
    "neu": "neu",
    "neutral": "neu",
    "calm": "neu",
    "hap": "hap",
    "happy": "hap",
    "happiness": "hap",
    "joy": "hap",
    "ang": "ang",
    "angry": "ang",
    "anger": "ang",
    "sad": "sad",
    "sadness": "sad",
}
GR_LABEL_ALIASES = {
    "man": "man",
    "male": "man",
    "m": "man",
    "woman": "woman",
    "female": "woman",
    "f": "woman",
}
SER_NUMERIC_ALIASES = {"0": "neu", "1": "hap", "2": "ang", "3": "sad"}
GR_NUMERIC_ALIASES = {"0": "man", "1": "woman"}


def canonical_task(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "speech_emotion_recognition": "ser",
        "emotion_recognition": "ser",
        "speaker_emotion_recognition": "ser",
        "gender_recognition": "gr",
        "speaker_gender": "gr",
        "spoken_language_understanding": "slu",
    }
    return aliases.get(normalized, normalized)


def require_task(value: Any) -> str:
    task = canonical_task(value)
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"unsupported classification task: {value!r}")
    return task


def tool_name_for(task: Any) -> str:
    return CLASSIFICATION_TOOLS[require_task(task)]


def primary_field_for(task: Any) -> str:
    return CLASSIFICATION_PRIMARY_FIELDS[require_task(task)]


def input_schema_for(task: Any) -> dict[str, Any]:
    normalized = require_task(task)
    properties: dict[str, Any] = {
        "audio_path": {"type": "string", "minLength": 1},
        "language": {"type": "string"},
    }
    required = ["audio_path"]
    if normalized == "slu":
        properties["prompt"] = {"type": "string", "minLength": 1}
        properties["choices"] = {
            "oneOf": [
                {"type": "array", "minItems": 1},
                {"type": "object", "minProperties": 1},
            ]
        }
        required.append("prompt")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def io_contract_for(task: Any) -> dict[str, Any]:
    normalized = require_task(task)
    if normalized in {"ser", "gr"}:
        label_spec = "ser_default" if normalized == "ser" else "gr_default"
        return {
            "input_type": "audio_path",
            "output_type": "classification",
            "input": {"audio_path": "string", "language": "optional string"},
            "output": {"label": "string", "score": "optional number"},
            "primary_field": "label",
            "required_fields": ["label"],
            "nonempty_fields": ["label"],
            "approved_output_fields": ["label", "score"],
            "label_spec": label_spec,
            "json_serializable": True,
        }
    return {
        "input_type": "audio_with_prompt",
        "output_type": "classification_answer",
        "input": {
            "audio_path": "string",
            "prompt": "string",
            "choices": "optional object|array",
        },
        "output": {"answer": "string", "label": "optional string"},
        "primary_field": "answer",
        "required_fields": ["answer"],
        "nonempty_fields": ["answer"],
        "approved_output_fields": ["answer", "label"],
        "json_serializable": True,
    }


def reference_value(row: dict[str, Any], task: Any) -> Any:
    normalized = require_task(task)
    if normalized in {"ser", "gr"}:
        for field in ("ground_truth", "target", "label", "expected"):
            if field in row and row[field] not in (None, ""):
                return row[field]
        raise ValueError(f"{normalized.upper()} fixture row has no reference label")
    for field in ("ground_truth", "target", "answer", "label", "expected"):
        if field in row and row[field] not in (None, ""):
            return row[field]
    raise ValueError("SLU fixture row has no reference answer")


def normalize_label(task: Any, value: Any) -> str:
    normalized = require_task(task)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{normalized.upper()} label {value!r} is unknown")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{normalized.upper()} label {value!r} is unknown")
    text = ("" if value is None else str(value)).strip().lower()
    text = re.sub(r"^[\s\[({<]+|[\s\])}>.,!?;:：，。！？；]+$", "", text)
    aliases = (
        {**SER_LABEL_ALIASES, **SER_NUMERIC_ALIASES}
        if normalized == "ser"
        else {**GR_LABEL_ALIASES, **GR_NUMERIC_ALIASES}
    )
    result = aliases.get(text)
    if result is None:
        raise ValueError(
            f"{normalized.upper()} label {value!r} is unknown; expected one of "
            + ", ".join(sorted(aliases))
        )
    return result


def normalize_answer(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("SLU answer must be a string or finite scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("SLU answer must be a string or finite scalar")
    text = str(value).strip()
    if any(ord(character) < 32 for character in text):
        raise ValueError("SLU answer must not contain control characters")
    text = text.rstrip(".!?。！？")
    if not text:
        raise ValueError("SLU answer must be non-empty")
    # Keep arbitrary choice ids/text intact, but remove common answer wrappers.
    match = re.fullmatch(
        r"(?is)(?:the\s+)?answer\s*(?:is|:|-)?\s*([A-Za-z0-9_+-]+)",
        text,
    )
    if match:
        return match.group(1).strip()
    match = re.fullmatch(r"答案\s*(?:是|为|:|：)?\s*([A-Za-z0-9_+-]+)", text)
    if match:
        return match.group(1).strip()
    return text


def _score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("classification score must be a finite number or null")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("classification score must be within [0, 1]")
    return result


def _reject_reference_fields(value: dict[str, Any]) -> None:
    forbidden = sorted(
        key
        for key in value
        if str(key).strip().lower().replace("-", "_") in PREDICTION_REFERENCE_FIELDS
        or str(key).strip().lower().endswith("_path")
    )
    if forbidden:
        raise ValueError(
            "classification prediction contains reference/path field(s): "
            + ", ".join(forbidden)
        )


def normalize_prediction(task: Any, value: Any) -> tuple[str, dict[str, Any]]:
    """Return the TSV scalar and its closed structured representation."""

    normalized = require_task(task)
    if not isinstance(value, dict) and normalized in {"ser", "gr"} and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, int):
            value = {"label": value}
    elif not isinstance(value, dict) and normalized == "slu" and isinstance(value, (str, int, float)) and not isinstance(value, bool):
        value = {"answer": value}
    if isinstance(value, dict):
        _reject_reference_fields(value)
        unknown = sorted(str(key) for key in value if key not in CLASSIFICATION_OUTPUT_FIELDS[normalized] | {"text"})
        if unknown:
            raise ValueError(
                f"{normalized.upper()} prediction contains unapproved field(s): "
                + ", ".join(unknown)
            )
        raw = (
            value["label"]
            if value.get("label") is not None
            else value["answer"]
            if value.get("answer") is not None
            else value.get("text")
        )
        if normalized in {"ser", "gr"}:
            label = normalize_label(normalized, raw)
            output: dict[str, Any] = {"label": label}
            score = _score(value.get("score"))
            if score is not None:
                output["score"] = score
            return label, output
        answer = normalize_answer(raw)
        output = {"answer": answer}
        if value.get("label") is not None:
            output["label"] = normalize_answer(value["label"])
        return answer, output
    if normalized in {"ser", "gr"}:
        label = normalize_label(normalized, value)
        return label, {"label": label}
    answer = normalize_answer(value)
    return answer, {"answer": answer}


def inference_arguments(task: Any, row: dict[str, Any], audio_path: str) -> dict[str, Any]:
    """Build model-call arguments without copying reference annotations."""

    normalized = require_task(task)
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise ValueError("classification inference requires audio_path")
    arguments: dict[str, Any] = {"audio_path": audio_path}
    language = row.get("language")
    if isinstance(language, str) and language.strip():
        arguments["language"] = language
    if normalized == "slu":
        prompt = row.get("prompt") or row.get("instruction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("SLU fixture row requires a non-empty prompt")
        arguments["prompt"] = prompt
        choices = row.get("choices", row.get("options"))
        if choices is not None:
            if not isinstance(choices, (dict, list)) or not choices:
                raise ValueError("SLU choices must be a non-empty object or array")
            _validate_choice_payload(choices)
            arguments["choices"] = choices
    forbidden = set(arguments) & REFERENCE_FIELDS
    if forbidden:
        raise ValueError("classification inference arguments contain reference fields")
    return arguments


def _validate_choice_payload(value: Any, path: str = "choices") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in CHOICE_REFERENCE_FIELDS or normalized.endswith("_path"):
                raise ValueError(f"SLU choices contain reference/path field at {path}.{key}")
            _validate_choice_payload(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_choice_payload(item, f"{path}[{index}]")


def validate_fixture_row(task: Any, row: dict[str, Any], *, key: str) -> str:
    normalized = require_task(task)
    if not isinstance(row, dict):
        raise ValueError(f"{normalized.upper()} fixture {key} must be an object")
    if not key.strip() or "/" in key or "\\" in key:
        raise ValueError(f"{normalized.upper()} fixture key must be a safe token")
    reference = reference_value(row, normalized)
    if normalized in {"ser", "gr"}:
        return normalize_label(normalized, reference)
    prompt = row.get("prompt") or row.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"SLU fixture {key} requires a non-empty prompt")
    choices = row.get("choices", row.get("options"))
    if choices is not None:
        if not isinstance(choices, (dict, list)) or not choices:
            raise ValueError(f"SLU fixture {key} choices must be a non-empty object or array")
        _validate_choice_payload(choices)
    return normalize_answer(reference)
