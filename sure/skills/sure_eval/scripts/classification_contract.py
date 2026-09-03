"""Canonical SER/GR/SLU prediction contracts for the Eval bridge."""

from __future__ import annotations

import math
import re
from typing import Any


CLASSIFICATION_TASKS = frozenset({"SER", "GR", "SLU"})
SER_LABEL_ALIASES = {
    "neu": "neu", "neutral": "neu", "calm": "neu",
    "hap": "hap", "happy": "hap", "happiness": "hap", "joy": "hap",
    "ang": "ang", "angry": "ang", "anger": "ang",
    "sad": "sad", "sadness": "sad",
}
GR_LABEL_ALIASES = {
    "man": "man", "male": "man", "m": "man",
    "woman": "woman", "female": "woman", "f": "woman",
}
SER_NUMERIC_ALIASES = {"0": "neu", "1": "hap", "2": "ang", "3": "sad"}
GR_NUMERIC_ALIASES = {"0": "man", "1": "woman"}
OUTPUT_FIELDS = {
    "SER": frozenset({"label", "score"}),
    "GR": frozenset({"label", "score"}),
    "SLU": frozenset({"answer", "label"}),
}
CHOICE_REFERENCE_FIELDS = {
    "answer",
    "expected",
    "ground_truth",
    "reference",
    "reference_audio",
    "reference_text",
    "target",
    "target_audio",
    "target_text",
}


def canonical_task(value: Any) -> str:
    normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "SPEECH_EMOTION_RECOGNITION": "SER",
        "EMOTION_RECOGNITION": "SER",
        "SPEAKER_EMOTION_RECOGNITION": "SER",
        "GENDER_RECOGNITION": "GR",
        "SPEAKER_GENDER": "GR",
        "SPOKEN_LANGUAGE_UNDERSTANDING": "SLU",
    }
    return aliases.get(normalized, "SA-ASR" if normalized == "SA_ASR" else normalized)


def require_task(value: Any) -> str:
    task = canonical_task(value)
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"unsupported classification task: {value!r}")
    return task


def normalize_label(task: Any, value: Any) -> str:
    normalized = require_task(task)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{normalized} label is unknown: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{normalized} label is unknown: {value!r}")
    text = ("" if value is None else str(value)).strip().lower()
    text = re.sub(r"^[\s\[({<]+|[\s\])}>.,!?;:：，。！？；]+$", "", text)
    aliases = (
        {**SER_LABEL_ALIASES, **SER_NUMERIC_ALIASES}
        if normalized == "SER"
        else {**GR_LABEL_ALIASES, **GR_NUMERIC_ALIASES}
    )
    if text not in aliases:
        raise ValueError(f"{normalized} label is unknown: {value!r}")
    return aliases[text]


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
    match = re.fullmatch(r"(?is)(?:the\s+)?answer\s*(?:is|:|-)?\s*([A-Za-z0-9_+-]+)", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"答案\s*(?:是|为|:|：)?\s*([A-Za-z0-9_+-]+)", text)
    return match.group(1) if match else text


def normalize_prediction(task: Any, value: Any) -> tuple[str, dict[str, Any]]:
    """Normalize a raw model response into TSV and closed JSON forms."""

    normalized = require_task(task)
    if not isinstance(value, dict):
        if normalized in {"SER", "GR"} and not isinstance(value, bool):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, int):
                value = {"label": value}
        elif normalized == "SLU" and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = {"answer": value}
        else:
            raise ValueError("classification prediction must be an object or a supported scalar")
    if not isinstance(value, dict):
        raise ValueError("classification prediction must be an object or string")
    allowed = OUTPUT_FIELDS[normalized] | {"text"}
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{normalized} prediction contains unapproved field(s): {', '.join(unknown)}")
    forbidden = sorted(
        str(key)
        for key in value
        if str(key).strip().lower() in {
            "expected", "ground_truth", "reference", "reference_audio",
            "reference_text", "target", "target_text",
        }
        or str(key).strip().lower().endswith("_path")
    )
    if forbidden:
        raise ValueError(f"{normalized} prediction contains reference/path field(s): {', '.join(forbidden)}")
    if normalized in {"SER", "GR"}:
        raw_label = value["label"] if value.get("label") is not None else value.get("text")
        label = normalize_label(normalized, raw_label)
        output: dict[str, Any] = {"label": label}
        score = value.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("classification score must be a finite number or null")
            score_value = float(score)
            if not math.isfinite(score_value) or not 0 <= score_value <= 1:
                raise ValueError("classification score must be within [0, 1]")
            output["score"] = score_value
        return label, output
    raw_answer = (
        value["answer"]
        if value.get("answer") is not None
        else value["label"]
        if value.get("label") is not None
        else value.get("text")
    )
    answer = normalize_answer(raw_answer)
    output = {"answer": answer}
    if value.get("label") is not None:
        output["label"] = normalize_answer(value["label"])
    return answer, output


def reference_value(sample: dict[str, Any], task: Any) -> Any:
    normalized = require_task(task)
    fields = ("ground_truth", "target", "label", "expected") if normalized in {"SER", "GR"} else (
        "ground_truth", "target", "answer", "label", "expected"
    )
    for field in fields:
        if field in sample and sample[field] not in (None, ""):
            return sample[field]
    raise ValueError(f"{normalized} dataset sample has no reference value")


def prompt_payload(sample: dict[str, Any]) -> dict[str, Any]:
    prompt = sample.get("prompt") or sample.get("instruction")
    choices = sample.get("choices", sample.get("options"))
    payload: dict[str, Any] = {}
    if isinstance(prompt, str) and prompt.strip():
        payload["prompt"] = prompt
    if choices is not None:
        if not isinstance(choices, (dict, list)) or not choices:
            raise ValueError("SLU choices must be a non-empty object or array")
        _validate_choice_payload(choices)
        payload["choices"] = choices
    if not payload:
        raise ValueError("SLU sample requires prompt or choices")
    return payload


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


def label_spec_payload(task: Any) -> dict[str, Any]:
    normalized = require_task(task)
    if normalized == "SER":
        labels = [
            {"id": "neu", "aliases": ["neutral", "calm"], "numeric_ids": [0]},
            {"id": "hap", "aliases": ["happy", "happiness", "joy"], "numeric_ids": [1]},
            {"id": "ang", "aliases": ["angry", "anger"], "numeric_ids": [2]},
            {"id": "sad", "aliases": ["sadness"], "numeric_ids": [3]},
        ]
        return {"id": "ser_default", "task": "SER", "labels": labels, "unknown_policy": "invalid"}
    if normalized == "GR":
        labels = [
            {"id": "man", "aliases": ["male", "m"], "numeric_ids": [0]},
            {"id": "woman", "aliases": ["female", "f"], "numeric_ids": [1]},
        ]
        return {"id": "gr_default", "task": "GR", "labels": labels, "unknown_policy": "invalid"}
    return {"id": "slu_prompt_choice_id", "task": "SLU", "labels": [], "unknown_policy": "keep"}
