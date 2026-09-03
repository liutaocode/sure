#!/usr/bin/env python3
"""Model-local validation template for SURE /sure_onboard.

The generated model directory should customize the constants below and keep the
CLI contract stable:

    python validate.py --stage import
    python validate.py --stage load
    python validate.py --stage infer
    python validate.py --stage contract
    python validate.py --stage all

Each stage writes artifacts/<stage>_result.json. Inference writes the first
result to artifacts/sample_output.json for contract compatibility and all
fixture results to artifacts/sample_outputs.jsonl.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any


MODEL_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
VALIDATION_LOG = ARTIFACTS_DIR / "validation.log"
SAMPLE_OUTPUT = ARTIFACTS_DIR / "sample_output.json"
SAMPLE_OUTPUTS = ARTIFACTS_DIR / "sample_outputs.jsonl"

# Agent-filled constants.
MODEL_ID = "__MODEL_ID__"
TASK_TYPE = "__TASK_TYPE__"
WRAPPER_MODULE = "model"
WRAPPER_CLASS = "__WRAPPER_CLASS__"
PREDICT_METHOD = "__PREDICT_METHOD__"
_IO_CONTRACT_JSON = r'''__IO_CONTRACT_JSON__'''
IO_CONTRACT: dict[str, Any] = (
    {} if _IO_CONTRACT_JSON.startswith("__") else json.loads(_IO_CONTRACT_JSON)
)
KWS_OPERATING_THRESHOLD = 0.5
STRUCTURED_TASKS = {"vad", "sd", "sa_asr"}
CLASSIFICATION_TASKS = {"ser", "gr", "slu"}
TSE_TASK = "tse"
TSE_OUTPUT_FIELDS = {"prediction_audio", "sample_id"}
TSE_REFERENCE_FIELDS = {
    "answer",
    "expected",
    "ground_truth",
    "input",
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
CLASSIFICATION_CHOICE_REFERENCE_FIELDS = {
    "answer",
    "expected",
    "ground_truth",
    "reference",
    "reference_audio",
    "reference_text",
    "target",
    "target_text",
}
CLASSIFICATION_OUTPUT_ROW_FIELDS = {
    "id",
    "key",
    "task",
    "audio",
    "dataset",
    "ground_truth",
    "prompt",
    "result",
}
CLASSIFICATION_ROW_REFERENCE_FIELDS = {
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
STRUCTURED_SPEAKER_OUTPUT_FIELDS = {"segments", "num_speakers"}
STRUCTURED_VAD_OUTPUT_FIELDS = {"speech_segments", "frame_scores"}
STRUCTURED_SD_SEGMENT_FIELDS = {"speaker", "start", "end", "duration"}
STRUCTURED_SA_ASR_SEGMENT_FIELDS = {*STRUCTURED_SD_SEGMENT_FIELDS, "text"}
STRUCTURED_VAD_SEGMENT_FIELDS = {"start", "end"}
STRUCTURED_VAD_FRAME_SCORE_FIELDS = {"start", "end", "score"}
STRUCTURED_PUBLIC_INFERENCE_PARAMETERS = {
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
STRUCTURED_REFERENCE_FIELDS = {
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
STRUCTURED_EVIDENCE_FIELDS = {
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
STRUCTURED_URI_SCHEMES = {"file", "ftp", "git", "gs", "hf", "http", "https", "s3", "ssh"}
STRUCTURED_URI_PREFIX = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")


def normalized_task_value(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"speech_enhancement", "acoustic_noise_suppression"}:
        return "se"
    if normalized in {"speech_activity_detection", "voice_activity_detection"}:
        return "vad"
    if normalized in {
        "tse",
        "target_speaker_extraction",
        "target_speaker_extractor",
        "target_speaker_extraction_model",
        "target_speaker",
        "speaker_extraction",
        "target_voice_extraction",
        "target_voice_separation",
    }:
        return TSE_TASK
    if normalized in {"speech_emotion_recognition", "speaker_emotion_recognition", "emotion_recognition"}:
        return "ser"
    if normalized in {"gender_recognition", "speaker_gender"}:
        return "gr"
    if normalized == "spoken_language_understanding":
        return "slu"
    return normalized


def structured_reference_segments_field(task: str) -> str:
    return "speech_segments" if task == "vad" else "segments"


def structured_public_inference_parameters(task: str) -> set[str]:
    if task == "vad":
        return {
            "batch_size",
            "min_duration_off",
            "min_duration_on",
            "vad_threshold",
        }
    return STRUCTURED_PUBLIC_INFERENCE_PARAMETERS


def normalized_task_type() -> str:
    return normalized_task_value(TASK_TYPE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(stage: str, status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "stage": stage,
        "status": status,
        "message": message,
    }
    if extra:
        payload.update(extra)
    with VALIDATION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def result_path(stage: str) -> Path:
    return ARTIFACTS_DIR / f"{stage}_result.json"


def write_stage_result(stage: str, passed: bool, started: float, error: str | None = None, **extra: Any) -> None:
    key = f"{stage}_passed" if stage != "contract" else "contract_passed"
    payload: dict[str, Any] = {
        key: passed,
        "duration_ms": round((time.time() - started) * 1000, 3),
        "error": error,
        "model_dir": str(MODEL_DIR),
        "validate_py": "validate.py",
        "validate_args": ["--stage", stage],
        "sample_output_path": "artifacts/sample_output.json",
    }
    payload.update(extra)
    write_json(result_path(stage), payload)


def import_wrapper_class():
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    module = importlib.import_module(WRAPPER_MODULE)
    return getattr(module, WRAPPER_CLASS)


def instantiate_wrapper() -> Any:
    wrapper_cls = import_wrapper_class()
    model_path = os.environ.get("MODEL_PATH", MODEL_ID)
    device = os.environ.get("DEVICE", os.environ.get("SURE_DEVICE", "auto"))
    attempts = [
        lambda: wrapper_cls(model_path=model_path, device=device),
        lambda: wrapper_cls({"model_path": model_path, "device": device}),
        lambda: wrapper_cls(),
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("failed to instantiate wrapper")


def load_wrapper() -> Any:
    wrapper = instantiate_wrapper()
    if hasattr(wrapper, "load"):
        wrapper.load()
    return wrapper


def require_vad_single_link_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"VAD {label} must be a regular file: {path}")
    if path.stat().st_nlink != 1:
        raise ValueError(f"VAD {label} must not be hard-linked: {path}")


def normalize_classification_label(task: str, value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{task.upper()} label is unknown: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{task.upper()} label is unknown: {value!r}")
    text = ("" if value is None else str(value)).strip().lower()
    text = re.sub(r"^[\s\[({<]+|[\s\])}>.,!?;:：，。！？；]+$", "", text)
    aliases = (
        {**SER_LABEL_ALIASES, **SER_NUMERIC_ALIASES}
        if task == "ser"
        else {**GR_LABEL_ALIASES, **GR_NUMERIC_ALIASES}
    )
    if text not in aliases:
        raise ValueError(f"{task.upper()} label is unknown: {value!r}")
    return aliases[text]


def normalize_classification_answer(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("SLU answer must be a string or finite scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("SLU answer must be a string or finite scalar")
    text = str(value).strip()
    text = text.rstrip(".!?。！？")
    if not text or any(ord(character) < 32 for character in text):
        raise ValueError("SLU answer must be non-empty and must not contain control characters")
    match = re.fullmatch(r"(?is)(?:the\s+)?answer\s*(?:is|:|-)?\s*([A-Za-z0-9_+-]+)", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"答案\s*(?:是|为|:|：)?\s*([A-Za-z0-9_+-]+)", text)
    return match.group(1) if match else text


def validate_classification_choices(value: Any, path: str = "choices") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in CLASSIFICATION_CHOICE_REFERENCE_FIELDS or normalized.endswith("_path"):
                raise ValueError(f"SLU choices contain reference/path field at {path}.{key}")
            validate_classification_choices(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_classification_choices(item, f"{path}[{index}]")


def normalize_classification_output(task: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        if task in {"ser", "gr"} and not isinstance(value, bool):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, int):
                value = {"label": value}
        elif task == "slu" and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = {"answer": value}
    if not isinstance(value, dict):
        raise ValueError("classification output must be an object or string")
    allowed = ({"label", "score", "text"} if task in {"ser", "gr"} else {"answer", "label", "text"})
    unknown = sorted(str(field) for field in value if field not in allowed)
    if unknown:
        raise ValueError("classification output contains unapproved field(s): " + ", ".join(unknown))
    forbidden = sorted(
        str(field)
        for field in value
        if str(field).lower() in {"expected", "ground_truth", "reference", "reference_audio", "reference_text", "target", "target_text"}
        or str(field).lower().endswith("_path")
    )
    if forbidden:
        raise ValueError("classification output contains reference/path field(s): " + ", ".join(forbidden))
    if task in {"ser", "gr"}:
        raw_label = value["label"] if value.get("label") is not None else value.get("text")
        label = normalize_classification_label(task, raw_label)
        output: dict[str, Any] = {"label": label}
        score = value.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                raise ValueError("classification score must be finite and within [0, 1]")
            output["score"] = float(score)
        return output
    raw_answer = (
        value["answer"]
        if value.get("answer") is not None
        else value["label"]
        if value.get("label") is not None
        else value.get("text")
    )
    answer = normalize_classification_answer(raw_answer)
    output = {"answer": answer}
    if value.get("label") is not None:
        output["label"] = normalize_classification_answer(value["label"])
    return output


def classification_fixture_payloads() -> list[dict[str, Any]]:
    task = normalized_task_type()
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gt_path in sorted((MODEL_DIR / "fixture" / task).glob("**/gt.jsonl")):
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{task.upper()} fixture line {line_number} must be an object")
            key = str(item.get("key") or item.get("id") or "").strip()
            if (
                not key
                or key in seen
                or "/" in key
                or "\\" in key
                or any(ord(character) < 32 or character.isspace() for character in key)
            ):
                raise ValueError(f"{task.upper()} fixture key is missing or duplicated: {key!r}")
            seen.add(key)
            audio = item.get("audio") or item.get("wav")
            if not isinstance(audio, str) or not audio.strip():
                raise ValueError(f"{task.upper()} fixture {key} requires audio")
            audio_path = (gt_path.parent / audio).resolve()
            if not audio_path.is_file() or not audio_path.is_relative_to(gt_path.parent.resolve()):
                raise ValueError(f"{task.upper()} fixture {key} audio is missing or unsafe")
            input_data: dict[str, Any] = {"audio_path": str(audio_path)}
            if isinstance(item.get("language"), str) and item["language"].strip():
                input_data["language"] = item["language"]
            if task == "slu":
                prompt = item.get("prompt") or item.get("instruction")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"SLU fixture {key} requires a non-empty prompt")
                input_data["prompt"] = prompt
                choices = item.get("choices", item.get("options"))
                if choices is not None:
                    if not isinstance(choices, (dict, list)) or not choices:
                        raise ValueError(f"SLU fixture {key} choices must be non-empty")
                    validate_classification_choices(choices)
                    input_data["choices"] = choices
            if task in {"ser", "gr"}:
                normalize_classification_label(
                    task,
                    item.get("ground_truth", item.get("target", item.get("label"))),
                )
            else:
                normalize_classification_answer(
                    item.get("ground_truth", item.get("target", item.get("answer")))
                )
            payloads.append({
                "input": input_data,
                "fixture": {
                    "key": key,
                    "audio": audio,
                    "dataset": item.get("dataset"),
                    "language": item.get("language"),
                    "ground_truth": item.get("ground_truth", item.get("target", item.get("answer"))),
                    **({"prompt": item.get("prompt") or item.get("instruction")} if task == "slu" else {}),
                },
            })
    if not 1 <= len(payloads) <= 5:
        raise ValueError(f"{task.upper()} validation requires 1 to 5 fixture rows")
    return payloads


def run_classification_fixture(wrapper: Any, task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run every classification fixture row and return a keyed output document."""

    rows: list[dict[str, Any]] = []
    for fixture in classification_fixture_payloads():
        key = str(fixture["fixture"]["key"])
        result = run_predict(wrapper, dict(fixture["input"]), scalar_fallback=False)
        canonical = normalize_classification_output(task, result)
        rows.append(
            {
                "key": key,
                "task": task,
                "audio": fixture["fixture"].get("audio"),
                "dataset": fixture["fixture"].get("dataset"),
                "ground_truth": fixture["fixture"].get("ground_truth"),
                **({"prompt": fixture["fixture"].get("prompt")} if task == "slu" else {}),
                "result": canonical,
            }
        )
    return {"rows": rows}, rows


def classification_forbidden_row_fields(value: Any, path: str = "row") -> list[str]:
    """Find nested reference/path field names in a classification evidence row."""

    found: list[str] = []
    if isinstance(value, dict):
        for field, item in value.items():
            normalized = str(field).strip().lower().replace("-", "_")
            child = f"{path}.{field}"
            root_ground_truth = path == "row" and normalized == "ground_truth"
            if not root_ground_truth and (
                normalized in CLASSIFICATION_ROW_REFERENCE_FIELDS
                or normalized == "path"
                or normalized.startswith("reference_")
                or normalized.endswith("_path")
            ):
                found.append(child)
            found.extend(classification_forbidden_row_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(classification_forbidden_row_fields(item, f"{path}[{index}]"))
    return found


def _validate_classification_output_rows(
    rows: Any,
    task: str,
    *,
    fixtures: list[dict[str, Any]] | None = None,
) -> list[str]:
    if not isinstance(rows, list):
        return [f"{task.upper()} sample_output.json must contain a rows array"]
    fixture_rows = fixtures if fixtures is not None else classification_fixture_payloads()
    expected_keys = [str(item["fixture"]["key"]) for item in fixture_rows]
    allowed_fields = set(CLASSIFICATION_OUTPUT_ROW_FIELDS)
    if task != "slu":
        allowed_fields.discard("prompt")
    violations: list[str] = []
    observed: list[str] = []
    for index, row in enumerate(rows, 1):
        prefix = f"{task.upper()} output row {index}"
        if not isinstance(row, dict):
            violations.append(f"{prefix} must be an object")
            continue
        unknown = sorted(str(field) for field in row if field not in allowed_fields)
        if unknown:
            violations.append(f"{prefix} contains unapproved field(s): " + ", ".join(unknown))
        forbidden = classification_forbidden_row_fields(row)
        if forbidden:
            violations.append(
                f"{prefix} exposes reference/path field(s): " + ", ".join(forbidden)
            )
        missing = sorted(field for field in allowed_fields if field not in row)
        if missing:
            violations.append(f"{prefix} is missing field(s): " + ", ".join(missing))

        raw_key = row.get("key")
        key = raw_key.strip() if isinstance(raw_key, str) else ""
        if (
            not isinstance(raw_key, str)
            or not key
            or raw_key != key
            or "/" in key
            or "\\" in key
            or any(ord(character) < 32 or character.isspace() for character in key)
        ):
            violations.append(f"{prefix} key must be a safe canonical token")
            key = ""
        elif key in observed:
            violations.append(f"{prefix} key is duplicated: {key!r}")
        observed.append(key)

        fixture = fixture_rows[index - 1].get("fixture") if index <= len(fixture_rows) else None
        expected_key = str(fixture.get("key") or "") if isinstance(fixture, dict) else ""
        if key and expected_key and key != expected_key:
            violations.append(
                f"{prefix} key does not preserve fixture order: expected={expected_key!r}, observed={key!r}"
            )
        row_id = row.get("id")
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id != index:
            violations.append(f"{prefix} id must equal its one-based fixture position")
        if row.get("task") != task:
            violations.append(f"{prefix} task must use canonical value {task!r}")

        audio = row.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            violations.append(f"{prefix} audio must be a non-empty relative path")
        else:
            relative_audio = Path(audio)
            if (
                relative_audio.is_absolute()
                or ".." in relative_audio.parts
                or "\\" in audio
                or structured_looks_like_absolute_path_or_uri(audio)
            ):
                violations.append(f"{prefix} audio must be a portable relative path")
            if isinstance(fixture, dict) and audio != fixture.get("audio"):
                violations.append(f"{prefix} audio does not match the fixture")

        if isinstance(fixture, dict) and row.get("dataset") != fixture.get("dataset"):
            violations.append(f"{prefix} dataset metadata changed")
        dataset = row.get("dataset")
        if dataset is not None and (
            not isinstance(dataset, str)
            or structured_looks_like_absolute_path_or_uri(dataset)
        ):
            violations.append(f"{prefix} dataset must be a portable string or null")

        reference = row.get("ground_truth")
        if isinstance(fixture, dict):
            expected_reference = fixture.get("ground_truth")
            try:
                observed_reference = (
                    normalize_classification_label(task, reference)
                    if task in {"ser", "gr"}
                    else normalize_classification_answer(reference)
                )
                canonical_reference = (
                    normalize_classification_label(task, expected_reference)
                    if task in {"ser", "gr"}
                    else normalize_classification_answer(expected_reference)
                )
                if observed_reference != canonical_reference:
                    violations.append(f"{prefix} ground_truth does not match the fixture")
            except (TypeError, ValueError) as error:
                violations.append(f"{prefix} has invalid ground_truth: {error}")

        if task == "slu":
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                violations.append(f"{prefix} prompt must be a non-empty string")
            elif isinstance(fixture, dict) and prompt != fixture.get("prompt"):
                violations.append(f"{prefix} prompt metadata changed")

        result = row.get("result")
        if not isinstance(result, dict):
            violations.append(f"{prefix} result must be an object")
            continue
        try:
            canonical = normalize_classification_output(task, result)
        except (TypeError, ValueError) as error:
            violations.append(f"{prefix} result: {error}")
            continue
        if result != canonical:
            violations.append(f"{prefix} result is not canonical")

    if observed != expected_keys:
        violations.append(
            f"{task.upper()} output keys must preserve fixture order: "
            f"expected={expected_keys}, observed={observed}"
        )
    return violations


def validate_classification_output_document(sample: dict[str, Any], task: str) -> list[str]:
    """Validate every keyed classification result and reject reference leakage."""

    return _validate_classification_output_rows(sample.get("rows"), task)


def fixture_payloads() -> list[dict[str, Any]]:
    task = normalized_task_type()
    raw_payload = os.environ.get("SURE_VALIDATE_INPUT_JSON")
    if raw_payload:
        parsed = json.loads(raw_payload)
        if not isinstance(parsed, dict):
            raise ValueError("SURE_VALIDATE_INPUT_JSON must decode to an object.")
        if task in STRUCTURED_TASKS:
            audio_path = parsed.get("audio_path")
            if not isinstance(audio_path, str) or not audio_path.strip():
                raise ValueError("structured SURE_VALIDATE_INPUT_JSON requires audio_path")
            if task == "vad":
                require_vad_single_link_file(Path(audio_path), "input audio")
            info = structured_wav_info(Path(audio_path))
            reference_field = structured_reference_segments_field(task)
            if reference_field in parsed:
                violations = validate_structured_segments(
                    parsed[reference_field],
                    task=task,
                    duration_sec=float(info["duration_sec"]),
                    audio_is_silence=bool(info["audio_is_silence"]),
                )
                if violations:
                    raise ValueError("invalid explicit reference segments: " + "; ".join(violations))
            if "inference_params" in parsed or any(
                field in parsed for field in STRUCTURED_PUBLIC_INFERENCE_PARAMETERS
            ):
                raise ValueError(
                    "structured inference parameters must be supplied through "
                    "SURE_VALIDATE_PROTOCOL_JSON, not fixture/reference input"
                )
            model_input = {"audio_path": audio_path}
            model_input.update(structured_protocol_arguments())
            return [
                {
                    "input": model_input,
                    "fixture": {
                        "key": str(parsed.get("key") or Path(audio_path).stem),
                        "audio": parsed.get("audio") or Path(audio_path).name,
                        "dataset": parsed.get("dataset"),
                        **info,
                    },
                }
            ]
        if task == TSE_TASK:
            mixture = parsed.get("mixture_audio_path")
            enrollment = parsed.get("enrollment_audio_path")
            if not isinstance(mixture, str) or not mixture.strip():
                raise ValueError("TSE SURE_VALIDATE_INPUT_JSON requires mixture_audio_path")
            if not isinstance(enrollment, str) or not enrollment.strip():
                raise ValueError("TSE SURE_VALIDATE_INPUT_JSON requires enrollment_audio_path")
            mixture_path = Path(mixture).expanduser()
            enrollment_path = Path(enrollment).expanduser()
            if mixture_path.is_symlink() or not mixture_path.is_file():
                raise ValueError("TSE mixture_audio_path must identify a regular file")
            if enrollment_path.is_symlink() or not enrollment_path.is_file():
                raise ValueError("TSE enrollment_audio_path must identify a regular file")
            structured_wav_info(mixture_path)
            structured_wav_info(enrollment_path)
            try:
                if mixture_path.resolve().samefile(enrollment_path.resolve()):
                    raise ValueError("TSE mixture and enrollment audio must be independent files")
            except OSError:
                pass
            key = str(parsed.get("sample_id") or parsed.get("key") or Path(mixture).stem).strip()
            tse_safe_sample_id(key)
            fixture: dict[str, Any] = {
                "key": key,
                "sample_id": key,
                "mixture_audio": parsed.get("mixture_audio") or mixture,
                "enrollment_audio": parsed.get("enrollment_audio") or enrollment,
                "reference_audio": parsed.get("reference_audio"),
                "mixture_audio_path": mixture,
                "enrollment_audio_path": enrollment,
                "reference_audio_path": parsed.get("reference_audio_path") or parsed.get("reference_audio"),
                "audio": parsed.get("audio") or parsed.get("mixture_audio") or mixture,
                "language": parsed.get("language"),
                "dataset": parsed.get("dataset"),
                "reference_text": parsed.get("reference_text"),
            }
            return [
                {
                    "input": {
                        "mixture_audio_path": mixture,
                        "enrollment_audio_path": enrollment,
                    },
                    "fixture": fixture,
                }
            ]
        if task in CLASSIFICATION_TASKS:
            audio_path = parsed.get("audio_path")
            if not isinstance(audio_path, str) or not audio_path.strip():
                raise ValueError("classification SURE_VALIDATE_INPUT_JSON requires audio_path")
            input_data: dict[str, Any] = {"audio_path": audio_path}
            if isinstance(parsed.get("language"), str) and parsed["language"].strip():
                input_data["language"] = parsed["language"]
            fixture: dict[str, Any] = {
                "key": str(parsed.get("key") or Path(audio_path).stem),
                "audio": parsed.get("audio") or Path(audio_path).name,
                "dataset": parsed.get("dataset"),
                "language": parsed.get("language"),
                "ground_truth": parsed.get("ground_truth", parsed.get("target", parsed.get("answer"))),
            }
            if task == "slu":
                prompt = parsed.get("prompt") or parsed.get("instruction")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("SLU SURE_VALIDATE_INPUT_JSON requires a non-empty prompt")
                input_data["prompt"] = prompt
                fixture["prompt"] = prompt
                choices = parsed.get("choices", parsed.get("options"))
                if choices is not None:
                    if not isinstance(choices, (dict, list)) or not choices:
                        raise ValueError("SLU choices must be a non-empty object or array")
                    validate_classification_choices(choices)
                    input_data["choices"] = choices
            return [{"input": input_data, "fixture": fixture}]
        keywords = parsed.get("keywords")
        if (
            task == "kws"
            and keywords is not None
            and not valid_keywords(keywords)
        ):
            raise ValueError("KWS keywords must be non-empty when provided")
        if (
            task == "kws"
            and "threshold" in parsed
            and not valid_kws_threshold(parsed["threshold"])
        ):
            raise ValueError(f"KWS threshold must equal {KWS_OPERATING_THRESHOLD}")
        if task == "se":
            audio_path = parsed.get("audio_path")
            if not isinstance(audio_path, str) or not audio_path.strip():
                raise ValueError("SE SURE_VALIDATE_INPUT_JSON requires audio_path")
            model_input = {"audio_path": audio_path}
            return [
                {
                    "input": model_input,
                    "fixture": {
                        field: parsed[field]
                        for field in ("key", "audio", "reference_audio")
                        if field in parsed
                    },
                }
            ]
        return [
            {
                "input": parsed,
                "fixture": {
                    field: parsed[field]
                    for field in (
                        "key",
                        "audio",
                        "text",
                        "label",
                        "expected",
                        "expected_detected",
                        "expected_keyword",
                    )
                    if field in parsed
                },
            }
        ]

    if task == TSE_TASK:
        return tse_fixture_payloads()

    fixture_root = MODEL_DIR / "fixture"
    for gt_path in sorted(fixture_root.glob("**/gt.jsonl")):
        if task == "vad":
            require_vad_single_link_file(gt_path, "fixture gt.jsonl")
        payloads: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            payload: dict[str, Any] = {}
            audio = (
                item.get("audio") or item.get("wav")
                if task in STRUCTURED_TASKS
                else item.get("audio")
                or item.get("wav")
                or item.get("prompt_audio")
                or item.get("reference_audio")
            )
            if task in STRUCTURED_TASKS:
                if not isinstance(audio, str) or not audio:
                    raise ValueError("structured fixture requires a non-empty audio field")
                if item.get("task") is not None and normalized_task_value(item["task"]) != task:
                    raise ValueError(
                        f"Structured fixture declares task {item['task']!r}, expected {task!r}"
                    )
                if "inference_params" in item:
                    raise ValueError(
                        "structured fixture rows must not declare inference_params; "
                        "use SURE_VALIDATE_PROTOCOL_JSON"
                    )
                audio_path = (gt_path.parent / audio).resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Structured fixture audio does not exist: {audio_path}")
                if task == "vad":
                    relative_audio = Path(audio)
                    if (
                        relative_audio.is_absolute()
                        or ".." in relative_audio.parts
                        or "\\" in audio
                        or not audio_path.is_relative_to(gt_path.parent.resolve())
                    ):
                        raise ValueError("VAD fixture audio must stay inside its fixture directory")
                    require_vad_single_link_file(audio_path, f"fixture audio {audio!r}")
                info = structured_wav_info(audio_path)
                violations = validate_structured_segments(
                    item.get(structured_reference_segments_field(task)),
                    task=task,
                    duration_sec=float(info["duration_sec"]),
                    audio_is_silence=bool(info["audio_is_silence"]),
                )
                if violations:
                    raise ValueError("invalid fixture reference segments: " + "; ".join(violations))
                key = str(item.get("key") or item.get("id") or audio_path.stem).strip()
                if not key:
                    raise ValueError("structured fixture requires a non-empty key")
                if key in seen_keys:
                    raise ValueError(f"structured fixture duplicates key {key!r}")
                seen_keys.add(key)
                payload["audio_path"] = str(audio_path)
                payload.update(structured_protocol_arguments())
            elif task in CLASSIFICATION_TASKS:
                audio = item.get("audio") or item.get("wav")
                if not isinstance(audio, str) or not audio.strip():
                    raise ValueError(f"{task.upper()} fixture requires a non-empty audio field")
                audio_path = (gt_path.parent / audio).resolve()
                if not audio_path.is_file() or not audio_path.is_relative_to(gt_path.parent.resolve()):
                    raise ValueError(f"{task.upper()} fixture audio is missing or unsafe")
                key = str(item.get("key") or item.get("id") or audio_path.stem).strip()
                if (
                    not key
                    or key in seen_keys
                    or "/" in key
                    or "\\" in key
                    or any(ord(character) < 32 or character.isspace() for character in key)
                ):
                    raise ValueError(f"{task.upper()} fixture key is missing or duplicated: {key!r}")
                seen_keys.add(key)
                if item.get("task_type") is not None and normalized_task_value(item["task_type"]) != task:
                    raise ValueError(f"{task.upper()} fixture declares task {item['task_type']!r}, expected {task!r}")
                if task in {"ser", "gr"}:
                    reference_value = item.get("ground_truth", item.get("target", item.get("label")))
                    normalize_classification_label(task, reference_value)
                prompt = item.get("prompt") or item.get("instruction")
                if task == "slu" and (not isinstance(prompt, str) or not prompt.strip()):
                    raise ValueError(f"SLU fixture {key} requires a non-empty prompt")
                payload["audio_path"] = str(audio_path)
                if isinstance(item.get("language"), str) and item["language"].strip():
                    payload["language"] = item["language"]
                if task == "slu":
                    payload["prompt"] = prompt
                    choices = item.get("choices", item.get("options"))
                    if choices is not None:
                        if not isinstance(choices, (dict, list)) or not choices:
                            raise ValueError(f"SLU fixture {key} choices must be non-empty")
                        payload["choices"] = choices
            elif task == "se":
                reference_audio = item.get("reference_audio")
                if not isinstance(audio, str) or not audio:
                    raise ValueError("SE fixture requires a non-empty noisy audio field")
                if not isinstance(reference_audio, str) or not reference_audio:
                    raise ValueError("SE fixture requires a non-empty reference_audio field")
                noisy_path = (gt_path.parent / audio).resolve()
                clean_path = (gt_path.parent / reference_audio).resolve()
                if not noisy_path.is_file() or not clean_path.is_file():
                    raise FileNotFoundError("SE fixture noisy and reference audio files must exist")
                if noisy_path == clean_path:
                    raise ValueError("SE fixture audio and reference_audio must be distinct files")
                payload["audio_path"] = str(noisy_path)
            elif isinstance(audio, str):
                payload["audio_path"] = str((gt_path.parent / audio).resolve())
                payload["prompt_audio_path"] = payload["audio_path"]
                payload["reference_audio_path"] = payload["audio_path"]
                payload["ref_audio"] = payload["audio_path"]
            text = item.get("target_text") or item.get("text") or item.get("prompt_text") or item.get("ground_truth")
            if task not in {"se", *STRUCTURED_TASKS, *CLASSIFICATION_TASKS} and isinstance(text, str):
                payload["text"] = text
                payload["prompt_text"] = item.get("prompt_text", text)
            if task not in {"se", *STRUCTURED_TASKS, *CLASSIFICATION_TASKS} and isinstance(item.get("language"), str):
                payload["language"] = item["language"]
            if task not in {"se", *STRUCTURED_TASKS, *CLASSIFICATION_TASKS} and "keywords" in item:
                payload["keywords"] = item["keywords"]
            if task not in {"se", *STRUCTURED_TASKS, *CLASSIFICATION_TASKS} and "threshold" in item:
                if task == "kws" and not valid_kws_threshold(
                    item["threshold"]
                ):
                    raise ValueError(f"KWS fixture threshold must equal {KWS_OPERATING_THRESHOLD}")
                payload["threshold"] = item["threshold"]
            if payload:
                if task in STRUCTURED_TASKS:
                    fixture_metadata = {
                        "key": key,
                        "audio": audio,
                        "language": item.get("language"),
                        "dataset": item.get("dataset"),
                        **info,
                    }
                elif task in CLASSIFICATION_TASKS:
                    fixture_metadata = {
                        "key": key,
                        "audio": item.get("audio") or item.get("wav"),
                        "language": item.get("language"),
                        "dataset": item.get("dataset"),
                        "ground_truth": item.get("ground_truth", item.get("target", item.get("answer"))),
                    }
                    if task == "slu":
                        fixture_metadata["prompt"] = item.get("prompt") or item.get("instruction")
                        if "choices" in item:
                            fixture_metadata["choices"] = item["choices"]
                        elif "options" in item:
                            fixture_metadata["choices"] = item["options"]
                else:
                    fixture_metadata = {
                        "key": item.get("key"),
                        "audio": item.get("audio"),
                        "reference_audio": item.get("reference_audio"),
                        "language": item.get("language"),
                        "dataset": item.get("dataset"),
                        "ground_truth": item.get("ground_truth"),
                        "text": item.get("text"),
                        "keywords": item.get("keywords"),
                    }
                if task == "se":
                    fixture_metadata["reference_audio_path"] = str(clean_path)
                if task not in { *STRUCTURED_TASKS, *CLASSIFICATION_TASKS }:
                    fixture_metadata.update(
                        {
                            field: item[field]
                            for field in ("label", "expected", "expected_detected", "expected_keyword")
                            if field in item
                        }
                    )
                payloads.append(
                    {
                        "input": payload,
                        "fixture": fixture_metadata,
                    }
                )
        if payloads:
            if len(payloads) > 5:
                raise ValueError(f"Fixture set exceeds the 5-sample validation limit: {gt_path}")
            if task == "kws":
                polarities: list[bool] = []
                for fixture in payloads:
                    metadata = fixture["fixture"]
                    expected = kws_expected_detected(metadata)
                    polarities.append(expected)
                    keywords = fixture["input"].get("keywords")
                    if keywords is not None and not valid_keywords(keywords):
                        raise ValueError(
                            f"KWS fixture {metadata.get('key')!r} has invalid keywords"
                        )
                if True not in polarities or False not in polarities:
                    raise ValueError("KWS fixture set must contain at least one positive and one negative sample")
            return payloads
    raise FileNotFoundError(
        "No validation payload found. Set SURE_VALIDATE_INPUT_JSON or provide fixture/**/gt.jsonl."
    )


def output_summary(outputs: list[dict[str, Any]]) -> str:
    first = outputs[0]
    summarized: dict[str, Any] = {}
    for key, value in first.items():
        if isinstance(value, str):
            summarized[key] = value[:200]
        elif isinstance(value, (int, float, bool)) or value is None:
            summarized[key] = value
        else:
            summarized[key] = {"type": type(value).__name__}
    return json.dumps(
        {"sample_count": len(outputs), "first_output": summarized},
        ensure_ascii=False,
    )


def to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return to_plain(value.to_dict())
    if dataclasses.is_dataclass(value):
        return to_plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def run_predict(
    wrapper: Any,
    payload: dict[str, Any],
    *,
    scalar_fallback: bool = True,
) -> dict[str, Any]:
    task = normalized_task_type()
    if task in STRUCTURED_TASKS:
        method_name = (
            "detect_speech"
            if task == "vad"
            else "transcribe_with_speakers"
            if task == "sa_asr"
            else "diarize"
        )
        predict = getattr(wrapper, method_name, None)
        if predict is None:
            raise AttributeError(f"structured-task wrapper must implement {method_name}().")
        audio_path = payload.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError("structured-task prediction requires audio_path")
        public_args = {key: value for key, value in payload.items() if key != "audio_path"}
        result = predict(audio_path, **public_args)
        plain = to_plain(result)
        return plain if isinstance(plain, dict) else {"result": plain}

    predict = getattr(wrapper, PREDICT_METHOD, None) or getattr(wrapper, "predict", None)
    if predict is None:
        raise AttributeError(f"Wrapper has neither {PREDICT_METHOD!r} nor 'predict'.")
    try:
        result = predict(payload)
    except TypeError:
        if not scalar_fallback or normalized_task_type() in {"kws", TSE_TASK}:
            raise
        if "audio_path" in payload:
            result = predict(payload["audio_path"])
        elif "text" in payload:
            result = predict(payload["text"])
        else:
            raise
    plain = to_plain(result)
    if isinstance(plain, dict):
        return plain
    if isinstance(plain, str):
        return {"text": plain}
    return {"result": plain}


def load_io_contract() -> dict[str, Any]:
    if IO_CONTRACT:
        return IO_CONTRACT
    spec_path = MODEL_DIR / "model.spec.yaml"
    if not spec_path.exists():
        raise FileNotFoundError("model.spec.yaml is required when IO_CONTRACT is not filled.")
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyYAML is required to read model.spec.yaml io_contract.") from exc
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("io_contract"), dict):
        raise ValueError("model.spec.yaml must contain io_contract.")
    return spec["io_contract"]


def string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def structured_protocol_arguments() -> dict[str, Any]:
    raw = os.environ.get("SURE_VALIDATE_PROTOCOL_JSON")
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("SURE_VALIDATE_PROTOCOL_JSON must decode to an object")
    allowed = structured_public_inference_parameters(normalized_task_type())
    unknown = sorted(str(key) for key in parsed if key not in allowed)
    if unknown:
        raise ValueError("unsupported public inference parameter(s): " + ", ".join(unknown))
    arguments: dict[str, Any] = {}
    for key, value in parsed.items():
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"public inference parameter {key!r} must be finite JSON data") from exc
        if key == "language" and (
            not isinstance(value, str)
            or not value.strip()
            or structured_looks_like_absolute_path_or_uri(value)
        ):
            raise ValueError("public inference parameter 'language' must be a non-path string")
        if key in {"batch_size", "beam_size", "min_speakers", "max_speakers", "num_speakers"} and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"public inference parameter {key!r} must be a positive integer")
        if key in {"clustering_threshold", "segmentation_threshold", "vad_threshold"} and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"public inference parameter {key!r} must be within [0, 1]")
        if key in {"min_duration_off", "min_duration_on"} and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"public inference parameter {key!r} must be non-negative")
        arguments[str(key)] = value
    minimum = arguments.get("min_speakers")
    maximum = arguments.get("max_speakers")
    exact = arguments.get("num_speakers")
    if isinstance(minimum, int) and isinstance(maximum, int) and minimum > maximum:
        raise ValueError("min_speakers must be <= max_speakers")
    if isinstance(exact, int) and isinstance(minimum, int) and exact < minimum:
        raise ValueError("num_speakers must be >= min_speakers")
    if isinstance(exact, int) and isinstance(maximum, int) and exact > maximum:
        raise ValueError("num_speakers must be <= max_speakers")
    return arguments


def structured_wav_info(path: Path) -> dict[str, Any]:
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


def structured_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def structured_looks_like_absolute_path_or_uri(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    match = STRUCTURED_URI_PREFIX.match(stripped)
    if match and (match.group(1).lower() in STRUCTURED_URI_SCHEMES or "://" in stripped):
        return True
    return (
        Path(stripped).is_absolute()
        or PureWindowsPath(stripped).is_absolute()
        or stripped.startswith("\\\\")
    )


def structured_unsafe_string_paths(value: Any, path: str = "output") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and structured_looks_like_absolute_path_or_uri(value):
        found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(structured_unsafe_string_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(structured_unsafe_string_paths(item, f"{path}[{index}]"))
    return found


def validate_structured_segments(
    segments: Any,
    *,
    task: str,
    duration_sec: float,
    audio_is_silence: bool,
) -> list[str]:
    if task == "vad":
        return validate_vad_intervals(
            segments,
            field="speech_segments",
            duration_sec=duration_sec,
            audio_is_silence=audio_is_silence,
            require_score=False,
        )
    if not isinstance(segments, list):
        return ["segments must be an array"]
    if not segments:
        if task == "sd" and audio_is_silence:
            return []
        if task == "sd":
            return ["empty SD segments are allowed only for pure-silence audio"]
        return ["SA-ASR segments must be non-empty"]
    violations: list[str] = []
    for index, segment in enumerate(segments, 1):
        prefix = f"segment {index}"
        if not isinstance(segment, dict):
            violations.append(f"{prefix} must be an object")
            continue
        approved_fields = (
            STRUCTURED_SA_ASR_SEGMENT_FIELDS
            if task == "sa_asr"
            else STRUCTURED_SD_SEGMENT_FIELDS
        )
        unknown_fields = sorted(str(key) for key in segment if key not in approved_fields)
        if unknown_fields:
            violations.append(f"{prefix} contains unapproved field(s): " + ", ".join(unknown_fields))
        speaker = segment.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            violations.append(f"{prefix} speaker must be a non-empty string")
        start = segment.get("start")
        end = segment.get("end")
        if not structured_finite_number(start):
            violations.append(f"{prefix} start must be a finite number")
        elif float(start) < 0:
            violations.append(f"{prefix} start must be >= 0")
        if not structured_finite_number(end):
            violations.append(f"{prefix} end must be a finite number")
        elif structured_finite_number(start) and float(end) <= float(start):
            violations.append(f"{prefix} end must be greater than start")
        if structured_finite_number(end) and float(end) > duration_sec + 1e-6:
            violations.append(
                f"{prefix} end {float(end):.6f} exceeds WAV duration {duration_sec:.6f}"
            )
        declared_duration = segment.get("duration")
        if declared_duration is not None:
            if not structured_finite_number(declared_duration) or float(declared_duration) <= 0:
                violations.append(f"{prefix} duration must be a finite positive number")
            elif structured_finite_number(start) and structured_finite_number(end) and not math.isclose(
                float(declared_duration), float(end) - float(start), rel_tol=0, abs_tol=1e-3
            ):
                violations.append(f"{prefix} duration must equal end - start")
        if task == "sa_asr":
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                violations.append(f"{prefix} text must be a non-empty string for SA-ASR")
    return violations


def validate_vad_intervals(
    intervals: Any,
    *,
    field: str,
    duration_sec: float,
    audio_is_silence: bool,
    require_score: bool,
) -> list[str]:
    if not isinstance(intervals, list):
        return [f"{field} must be an array"]
    if not intervals:
        if field == "speech_segments" and audio_is_silence:
            return []
        if field == "speech_segments":
            return ["empty VAD speech_segments are allowed only for pure-silence audio"]
        return ["frame_scores must be non-empty when provided"]

    approved_fields = (
        STRUCTURED_VAD_FRAME_SCORE_FIELDS
        if require_score
        else STRUCTURED_VAD_SEGMENT_FIELDS
    )
    violations: list[str] = []
    previous_start: float | None = None
    previous_end: float | None = None
    for index, interval in enumerate(intervals, 1):
        prefix = f"{field} item {index}"
        if not isinstance(interval, dict):
            violations.append(f"{prefix} must be an object")
            continue
        unknown_fields = sorted(
            str(key) for key in interval if key not in approved_fields
        )
        if unknown_fields:
            violations.append(
                f"{prefix} contains unapproved field(s): "
                + ", ".join(unknown_fields)
            )
        start = interval.get("start")
        end = interval.get("end")
        if not structured_finite_number(start):
            violations.append(f"{prefix} start must be a finite number")
        elif float(start) < 0:
            violations.append(f"{prefix} start must be >= 0")
        if not structured_finite_number(end):
            violations.append(f"{prefix} end must be a finite number")
        elif structured_finite_number(start) and float(end) <= float(start):
            violations.append(f"{prefix} end must be greater than start")
        if structured_finite_number(end) and float(end) > duration_sec + 1e-6:
            violations.append(
                f"{prefix} end {float(end):.6f} exceeds WAV duration {duration_sec:.6f}"
            )
        if structured_finite_number(start):
            current_start = float(start)
            if require_score and index == 1 and not math.isclose(
                current_start, 0.0, rel_tol=0, abs_tol=1e-9
            ):
                violations.append(f"{prefix} must start at 0")
            if previous_start is not None and current_start < previous_start:
                violations.append(f"{prefix} is not ordered by start time")
            if previous_end is not None and current_start < previous_end - 1e-9:
                violations.append(f"{prefix} overlaps the previous interval")
            elif require_score and previous_end is not None and current_start > previous_end + 1e-9:
                violations.append(f"{prefix} leaves a gap after the previous interval")
            previous_start = current_start
        if structured_finite_number(end):
            previous_end = float(end)
        if require_score:
            score = interval.get("score")
            if not structured_finite_number(score):
                violations.append(f"{prefix} score must be a finite number")
            elif not 0 <= float(score) <= 1:
                violations.append(f"{prefix} score must be within [0, 1]")
    if require_score and previous_end is not None and not math.isclose(
        previous_end, duration_sec, rel_tol=0, abs_tol=1e-6
    ):
        violations.append(
            f"frame_scores must end at WAV duration {duration_sec:.6f}"
        )
    return violations


def structured_forbidden_output_paths(value: Any, path: str = "output") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child = f"{path}.{key}"
            if (
                normalized in STRUCTURED_REFERENCE_FIELDS
                or normalized.startswith("reference_")
                or normalized == "path"
                or normalized.endswith("_path")
            ):
                found.append(child)
            found.extend(structured_forbidden_output_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(structured_forbidden_output_paths(item, f"{path}[{index}]"))
    return found


def validate_structured_output(
    output: Any,
    *,
    task: str,
    duration_sec: float,
    audio_is_silence: bool,
) -> list[str]:
    if not isinstance(output, dict):
        return ["structured prediction must be an object"]
    approved_output_fields = (
        STRUCTURED_VAD_OUTPUT_FIELDS
        if task == "vad"
        else STRUCTURED_SPEAKER_OUTPUT_FIELDS
    )
    unknown_fields = sorted(
        str(key) for key in output if key not in approved_output_fields
    )
    violations = [
        f"model output must not expose reference or path field {path}"
        for path in structured_forbidden_output_paths(output)
    ]
    if unknown_fields:
        violations.append("structured prediction contains unapproved field(s): " + ", ".join(unknown_fields))
    violations.extend(
        f"structured prediction contains absolute path or URI at {path}"
        for path in structured_unsafe_string_paths(output)
    )
    violations.extend(
        validate_structured_segments(
            output.get(structured_reference_segments_field(task)),
            task=task,
            duration_sec=duration_sec,
            audio_is_silence=audio_is_silence,
        )
    )
    if task == "vad" and "frame_scores" in output:
        violations.extend(
            validate_vad_intervals(
                output["frame_scores"],
                field="frame_scores",
                duration_sec=duration_sec,
                audio_is_silence=audio_is_silence,
                require_score=True,
            )
        )
    num_speakers = output.get("num_speakers")
    if task != "vad" and num_speakers is not None:
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


def validate_structured_rows(rows: Any, contract: dict[str, Any]) -> list[str]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5:
        return ["structured validation requires 1-5 output rows"]
    task = normalized_task_type()
    violations: list[str] = []
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
        if structured_looks_like_absolute_path_or_uri(key):
            violations.append(f"{prefix} key must not contain an absolute path or URI")
        if key in seen:
            violations.append(f"{prefix} duplicates key {key!r}")
            continue
        seen.add(key)
        audio = row.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            violations.append(f"{prefix} {key!r} requires a relative audio evidence path")
        else:
            audio_path = Path(audio)
            if structured_looks_like_absolute_path_or_uri(audio) or ".." in audio_path.parts:
                violations.append(f"{prefix} {key!r} audio evidence path must be portable")
        for field in ("dataset", "language"):
            value = row.get(field)
            if value is not None and not isinstance(value, str):
                violations.append(f"{prefix} {key!r} {field} must be a string or null")
            elif isinstance(value, str) and structured_looks_like_absolute_path_or_uri(value):
                violations.append(f"{prefix} {key!r} {field} must not contain an absolute path or URI")
        duration = row.get("duration_sec")
        silence = row.get("audio_is_silence")
        if not structured_finite_number(duration) or float(duration) <= 0 or not isinstance(silence, bool):
            violations.append(f"{prefix} {key!r} lacks trusted WAV duration/silence evidence")
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            violations.append(f"{prefix} {key!r} result must be an object")
            continue
        violations.extend(f"{prefix} {key!r}: {item}" for item in validate_contract(output, contract))
        violations.extend(
            f"{prefix} {key!r}: {item}"
            for item in validate_structured_output(
                output,
                task=task,
                duration_sec=float(duration),
                audio_is_silence=silence,
            )
        )
    return violations


def valid_keywords(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(keyword, str) and bool(keyword.strip()) for keyword in value)
    )


def valid_kws_threshold(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == KWS_OPERATING_THRESHOLD
    )


def kws_expected_detected(reference: dict[str, Any]) -> bool:
    positive_values = {"detect", "detected", "positive", "true", "1", "yes"}
    negative_values = {"reject", "rejected", "negative", "false", "0", "no"}
    declared: list[tuple[str, bool]] = []
    for field in ("expected", "label", "expected_detected"):
        if field not in reference:
            continue
        value = reference[field]
        if isinstance(value, bool):
            parsed = value
        else:
            normalized = str(value or "").strip().lower()
            if normalized in positive_values:
                parsed = True
            elif normalized in negative_values:
                parsed = False
            else:
                raise ValueError(f"unsupported {field} value {value!r}")
        declared.append((field, parsed))
    if not declared:
        raise ValueError("expected, label, or expected_detected is required")
    if len({parsed for _field, parsed in declared}) != 1:
        fields = ", ".join(f"{field}={reference[field]!r}" for field, _parsed in declared)
        raise ValueError(f"conflicting KWS polarity fields: {fields}")
    return declared[0][1]


def normalized_keyword(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = "".join(value.upper().split())
    return normalized or None


def validate_kws_output(sample: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    detected = sample.get("detected")
    keyword = sample.get("keyword")
    score = sample.get("score")
    score_is_finite_number = (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
    )
    if not isinstance(detected, bool):
        violations.append("detected must be a boolean")
    if keyword is not None and not isinstance(keyword, str):
        violations.append("keyword must be a string or null")
    if score is not None and not score_is_finite_number:
        violations.append("score must be a finite number or null")
    if score_is_finite_number and not 0 <= float(score) <= 1:
        violations.append("score must be within [0, 1]")
    if detected is True and normalized_keyword(keyword) is None:
        violations.append("detected=true requires a non-empty keyword")
    if detected is True and not score_is_finite_number:
        violations.append("detected=true requires a finite numeric score")
    if detected is True and score_is_finite_number and float(score) < KWS_OPERATING_THRESHOLD:
        violations.append(f"detected=true requires score >= {KWS_OPERATING_THRESHOLD}")
    if detected is False and keyword is not None:
        violations.append("detected=false requires keyword=null")
    if (
        detected is False
        and score_is_finite_number
        and float(score) >= KWS_OPERATING_THRESHOLD
    ):
        violations.append(f"detected=false requires score < {KWS_OPERATING_THRESHOLD}")
    return violations


def validate_kws_rows(rows: list[dict[str, Any]], contract: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    seen_keys: set[str] = set()
    positive_seen = False
    negative_seen = False
    reference_seen = any(
        isinstance(row, dict)
        and any(field in row for field in ("expected", "label", "expected_detected"))
        for row in rows
    )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(f"KWS output row {index + 1} must be an object")
            continue
        key = row.get("key")
        prefix = f"KWS output row {index + 1}"
        if not isinstance(key, str) or not key.strip():
            violations.append(f"{prefix} requires a non-empty key")
        elif key in seen_keys:
            violations.append(f"{prefix} duplicates key {key!r}")
        else:
            seen_keys.add(key)
            prefix = f"KWS output {key!r}"
        output = row.get("output")
        if not isinstance(output, dict):
            violations.append(f"{prefix} result must be an object")
            continue
        violations.extend(f"{prefix}: {item}" for item in validate_contract(output, contract))
        violations.extend(f"{prefix}: {item}" for item in validate_kws_output(output))

        has_reference = any(field in row for field in ("expected", "label", "expected_detected"))
        if not has_reference:
            if reference_seen:
                violations.append(f"{prefix} has no expected KWS polarity")
            continue
        try:
            expected = kws_expected_detected(row)
        except ValueError as error:
            violations.append(f"{prefix} has invalid KWS reference: {error}")
            continue
        if expected:
            positive_seen = True
            if output.get("detected") is not True:
                violations.append(f"{prefix} must detect the positive fixture")
            expected_keyword = row.get("expected_keyword") or row.get("text")
            if expected_keyword is not None and normalized_keyword(output.get("keyword")) != normalized_keyword(
                expected_keyword
            ):
                violations.append(f"{prefix} detected the wrong keyword")
        else:
            negative_seen = True
            if output.get("detected") is not False:
                violations.append(f"{prefix} must reject the negative fixture")
    if reference_seen and (not positive_seen or not negative_seen):
        violations.append("KWS validation requires at least one positive and one negative fixture")
    return violations


def se_outputs_root() -> Path:
    root = ARTIFACTS_DIR / "outputs"
    if root.is_symlink():
        raise ValueError("SE outputs directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise ValueError("SE outputs directory must be writable")
    return root.resolve()


def se_output_path(key: str, index: int) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return se_outputs_root() / f"{index:02d}-{digest}.wav"


def resolve_se_output_path(value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == "artifacts":
        return ARTIFACTS_DIR.joinpath(*raw.parts[1:])
    return se_outputs_root() / raw


def portable_se_output_path(path: Path) -> str:
    relative = path.resolve().relative_to(ARTIFACTS_DIR.resolve())
    return (Path("artifacts") / relative).as_posix()


def validate_se_output(
    sample: dict[str, Any],
    *,
    expected_path: Path | None = None,
    forbidden_inputs: tuple[Path, ...] = (),
) -> list[str]:
    audio_path = sample.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path.strip():
        return ["SE output audio_path must be a non-empty string"]
    output = resolve_se_output_path(audio_path)
    root = se_outputs_root()
    try:
        lexical_relative = output.absolute().relative_to(root)
    except ValueError:
        return [f"SE output audio_path must stay below artifacts/outputs: {audio_path}"]
    current = root
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            return [f"SE output audio_path must not traverse a symlink: {audio_path}"]
    resolved = output.resolve()
    if not resolved.is_relative_to(root):
        return [f"SE output audio_path must stay below artifacts/outputs: {audio_path}"]
    if expected_path is not None and resolved != expected_path.resolve():
        return [f"SE output audio_path differs from the harness-assigned output: {audio_path}"]
    if not resolved.is_file():
        return [f"SE output audio_path does not exist: {audio_path}"]
    if resolved.stat().st_size <= 0:
        return [f"SE output audio_path is empty: {audio_path}"]
    for input_path in forbidden_inputs:
        if input_path.is_file() and os.path.samefile(resolved, input_path):
            return [f"SE output audio_path must not alias an input audio file: {audio_path}"]
    try:
        with wave.open(str(resolved), "rb") as handle:
            if (
                handle.getcomptype() != "NONE"
                or handle.getnchannels() < 1
                or handle.getsampwidth() not in {1, 2, 3, 4}
                or handle.getframerate() < 1
                or handle.getnframes() < 1
            ):
                return [f"SE output audio_path must be a non-empty PCM WAV: {audio_path}"]
    except (EOFError, OSError, wave.Error) as error:
        return [f"SE output audio_path must be a readable PCM WAV: {error}"]
    return []


def validate_se_rows(rows: list[dict[str, Any]], contract: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if not rows:
        return ["SE validation requires at least one output row"]
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            violations.append(f"SE output row {index} must be an object")
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            violations.append(f"SE output row {index} result must be an object")
            continue
        key = str(row.get("key") or index)
        prefix = f"SE output {key!r}"
        if not isinstance(row.get("audio"), str) or not row["audio"].strip():
            violations.append(f"{prefix}: noisy audio role is missing")
        if not isinstance(row.get("reference_audio"), str) or not row["reference_audio"].strip():
            violations.append(f"{prefix}: reference_audio role is missing")
        violations.extend(f"{prefix}: {item}" for item in validate_contract(output, contract))
        violations.extend(
            f"{prefix}: {item}"
            for item in validate_se_output(output, expected_path=se_output_path(key, index))
        )
    return violations


def tse_safe_sample_id(value: Any) -> str:
    token = str(value or "").strip()
    if (
        not token
        or "/" in token
        or "\\" in token
        or any(ord(character) < 32 or character.isspace() for character in token)
        or structured_looks_like_absolute_path_or_uri(token)
    ):
        raise ValueError("TSE sample_id must be a safe non-empty token")
    return token


def validate_tse_output_object(value: Any, sample_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("TSE prediction must be a JSON object")
    unknown = sorted(str(field) for field in value if field not in TSE_OUTPUT_FIELDS)
    if unknown:
        raise ValueError("TSE prediction contains unapproved field(s): " + ", ".join(unknown))
    for field in value:
        normalized = str(field).strip().lower().replace("-", "_")
        if normalized != "prediction_audio" and (
            normalized.endswith("_path") or normalized in TSE_REFERENCE_FIELDS
        ):
            raise ValueError(f"TSE prediction contains forbidden reference/input field: {field}")
    prediction_audio = value.get("prediction_audio")
    if not isinstance(prediction_audio, str) or not prediction_audio.strip():
        raise ValueError("TSE prediction requires a non-empty prediction_audio")
    output: dict[str, Any] = {"prediction_audio": prediction_audio.strip()}
    returned_id = value.get("sample_id")
    if returned_id is not None:
        returned_id = tse_safe_sample_id(returned_id)
        if sample_id is not None and returned_id != sample_id:
            raise ValueError(f"TSE prediction sample_id {returned_id!r} does not match {sample_id!r}")
        output["sample_id"] = returned_id
    elif sample_id is not None:
        output["sample_id"] = tse_safe_sample_id(sample_id)
    try:
        json.dumps(output, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"TSE prediction must contain finite JSON data: {error}") from error
    return output


def tse_fixture_role_path(
    gt_path: Path,
    row: dict[str, Any],
    role: str,
    key: str,
) -> tuple[str, Path]:
    aliases = {
        "mixture_audio": ("mixed_audio", "audio"),
        "enrollment_audio": ("enrollment",),
        "reference_audio": ("target_audio",),
    }
    raw: Any = row.get(role)
    if raw is None:
        for alias in aliases[role]:
            raw = row.get(alias)
            if raw is not None:
                break
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"TSE fixture {key} requires {role}")
    relative = Path(raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in raw
        or structured_looks_like_absolute_path_or_uri(raw)
    ):
        raise ValueError(f"TSE fixture {key} {role} path must be relative and contained")
    current = gt_path.parent
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"TSE fixture {key} {role} traverses a symlink")
    path = (gt_path.parent / relative).resolve()
    if not path.is_file() or path.stat().st_size <= 0 or not path.is_relative_to(gt_path.parent.resolve()):
        raise ValueError(f"TSE fixture {key} {role} is missing or unsafe")
    return raw, path


def tse_fixture_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    fixture_root = MODEL_DIR / "fixture" / TSE_TASK
    if fixture_root.is_symlink():
        raise ValueError("TSE fixture root must not be a symlink")
    if fixture_root.exists() and any(path.is_symlink() for path in fixture_root.rglob("*")):
        raise ValueError("TSE fixture tree must not contain symlinks")
    for gt_path in sorted(fixture_root.glob("**/gt.jsonl")):
        if gt_path.is_symlink():
            raise ValueError("TSE gt.jsonl must not be a symlink")
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"TSE fixture {gt_path}:{line_number} is not valid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"TSE fixture {gt_path}:{line_number} must be an object")
            allowed_fields = {
                "key",
                "sample_id",
                "task_type",
                "audio",
                "mixture_audio",
                "mixed_audio",
                "enrollment_audio",
                "enrollment",
                "reference_audio",
                "target_audio",
                "language",
                "dataset",
                "reference_text",
            }
            unknown_fields = sorted(str(field) for field in row if field not in allowed_fields)
            if unknown_fields:
                raise ValueError(
                    f"TSE fixture {gt_path}:{line_number} contains unapproved field(s): "
                    + ", ".join(unknown_fields)
                )
            key = tse_safe_sample_id(row.get("sample_id") or row.get("key"))
            if key in seen_keys:
                raise ValueError(f"TSE fixture key is duplicated: {key!r}")
            seen_keys.add(key)
            if row.get("task_type") is not None and normalized_task_value(row["task_type"]) != TSE_TASK:
                raise ValueError(f"TSE fixture {key} declares task {row['task_type']!r}")
            role_values: dict[str, str] = {}
            roles: dict[str, Path] = {}
            for role in ("mixture_audio", "enrollment_audio", "reference_audio"):
                raw, path = tse_fixture_role_path(gt_path, row, role, key)
                role_values[role] = raw
                roles[role] = path
            if row.get("audio") is not None and row.get("audio") != role_values["mixture_audio"]:
                raise ValueError(f"TSE fixture {key} audio must equal mixture_audio")
            role_paths = tuple(roles.values())
            if len(set(role_paths)) != 3 or any(
                left.samefile(right)
                for offset, left in enumerate(role_paths)
                for right in role_paths[offset + 1 :]
            ):
                raise ValueError(f"TSE fixture {key} roles must be independent")
            reference_text = row.get("reference_text")
            if reference_text is not None and (
                not isinstance(reference_text, str)
                or any(ord(character) < 32 for character in reference_text)
            ):
                raise ValueError(f"TSE fixture {key} reference_text must be a safe string")
            payloads.append(
                {
                    "input": {
                        "mixture_audio_path": str(roles["mixture_audio"]),
                        "enrollment_audio_path": str(roles["enrollment_audio"]),
                    },
                    "fixture": {
                        "key": key,
                        "sample_id": key,
                        "mixture_audio": role_values["mixture_audio"],
                        "enrollment_audio": role_values["enrollment_audio"],
                        "reference_audio": role_values["reference_audio"],
                        "mixture_audio_path": str(roles["mixture_audio"]),
                        "enrollment_audio_path": str(roles["enrollment_audio"]),
                        "reference_audio_path": str(roles["reference_audio"]),
                        "audio": role_values["mixture_audio"],
                        "language": row.get("language"),
                        "dataset": row.get("dataset"),
                        "reference_text": reference_text,
                    },
                }
            )
    if not 1 <= len(payloads) <= 5:
        raise ValueError("TSE validation requires 1 to 5 fixture rows")
    return payloads


def tse_outputs_root() -> Path:
    root = ARTIFACTS_DIR / "outputs"
    if root.is_symlink():
        raise ValueError("TSE outputs directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise ValueError("TSE outputs directory must be writable")
    return root.resolve()


def tse_output_path(key: str, index: int) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return tse_outputs_root() / f"{index:02d}-{digest}.wav"


def resolve_tse_output_path(value: str) -> Path:
    if structured_looks_like_absolute_path_or_uri(value) and not Path(value).is_absolute():
        raise ValueError(f"TSE prediction_audio must be a local path: {value}")
    raw = Path(value)
    if raw.is_absolute():
        candidate = raw
    elif raw.parts[:1] == ("artifacts",):
        candidate = ARTIFACTS_DIR.joinpath(*raw.parts[1:])
    else:
        candidate = tse_outputs_root() / raw
    root = tse_outputs_root()
    try:
        lexical = candidate.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError("TSE prediction_audio must stay under artifacts/outputs") from error
    current = root
    for part in lexical.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("TSE prediction_audio must not traverse a symlink")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("TSE prediction_audio must stay under artifacts/outputs")
    return resolved


def portable_tse_output_path(path: Path) -> str:
    relative = path.resolve().relative_to(ARTIFACTS_DIR.resolve())
    return (Path("artifacts") / relative).as_posix()


def validate_tse_output(
    sample: dict[str, Any],
    *,
    expected_path: Path,
    forbidden_inputs: tuple[Path, ...] = (),
) -> list[str]:
    try:
        canonical = validate_tse_output_object(sample)
        path = resolve_tse_output_path(canonical["prediction_audio"])
    except (TypeError, ValueError) as error:
        return [str(error)]
    if path != expected_path.resolve() or not path.is_file() or path.stat().st_size <= 0:
        return ["TSE prediction_audio must equal the harness-assigned non-empty output"]
    for input_path in forbidden_inputs:
        try:
            if path.samefile(input_path):
                return ["TSE prediction_audio must not alias an input or reference audio"]
        except (FileNotFoundError, OSError):
            continue
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getcomptype() != "NONE"
                or handle.getnchannels() < 1
                or handle.getsampwidth() not in {1, 2, 3, 4}
                or handle.getframerate() < 1
                or handle.getnframes() < 1
            ):
                return ["TSE prediction_audio must be a non-empty PCM WAV"]
    except (EOFError, OSError, wave.Error) as error:
        return [f"TSE prediction_audio must be a readable PCM WAV: {error}"]
    return []


def run_tse_fixture(wrapper: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index, fixture in enumerate(tse_fixture_payloads(), 1):
        key = str(fixture["fixture"]["key"])
        requested = tse_output_path(key, index)
        if requested.exists() or requested.is_symlink():
            requested.unlink()
        payload = dict(fixture["input"])
        payload["output_path"] = str(requested)
        result = run_predict(wrapper, payload, scalar_fallback=False)
        canonical = validate_tse_output_object(result, sample_id=key)
        input_paths = tuple(
            Path(str(fixture["fixture"][field]))
            for field in ("mixture_audio_path", "enrollment_audio_path", "reference_audio_path")
        )
        violations = validate_tse_output(
            canonical,
            expected_path=requested,
            forbidden_inputs=input_paths,
        )
        if violations:
            raise AssertionError("; ".join(violations))
        canonical["prediction_audio"] = portable_tse_output_path(requested)
        rows.append({"key": key, "sample_id": key, "result": canonical})
    return {"rows": rows}, rows


def validate_tse_output_document(sample: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    rows = sample.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5:
        return ["TSE sample_output.json must contain 1 to 5 rows"]
    try:
        fixtures = tse_fixture_payloads()
    except (OSError, ValueError) as error:
        return [str(error)]
    expected = [str(item["fixture"]["key"]) for item in fixtures]
    observed: list[str] = []
    references = {
        str(item["fixture"]["key"]): item["fixture"] for item in fixtures
    }
    violations: list[str] = []
    referenced_outputs: set[Path] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            violations.append(f"TSE output row {index} must be an object")
            continue
        unknown_row_fields = sorted(str(field) for field in row if field not in {"key", "sample_id", "result"})
        if unknown_row_fields:
            violations.append(
                f"TSE output row {index} contains unapproved field(s): "
                + ", ".join(unknown_row_fields)
            )
        key = str(row.get("key") or row.get("sample_id") or "").strip()
        try:
            tse_safe_sample_id(key)
        except ValueError as error:
            violations.append(f"TSE output row {index}: {error}")
            continue
        if key in observed:
            violations.append(f"TSE output row {index} key is duplicated: {key!r}")
            continue
        observed.append(key)
        row_sample_id = row.get("sample_id")
        if row_sample_id is not None and str(row_sample_id).strip() != key:
            violations.append(f"TSE output {key} sample_id does not match key")
        result = row.get("result")
        try:
            canonical = validate_tse_output_object(result, sample_id=key)
        except (TypeError, ValueError) as error:
            violations.append(f"TSE output {key}: {error}")
            continue
        if result != canonical:
            violations.append(f"TSE output {key} is not canonical")
        if key not in references:
            violations.append(f"unexpected TSE output key: {key}")
            continue
        violations.extend(f"TSE output {key}: {item}" for item in validate_contract(canonical, contract))
        fixture = references[key]
        forbidden = tuple(
            Path(str(fixture[field]))
            for field in ("mixture_audio_path", "enrollment_audio_path", "reference_audio_path")
        )
        fixture_index = expected.index(key) + 1
        expected_output = tse_output_path(key, fixture_index).resolve()
        violations.extend(
            f"TSE output {key}: {item}"
            for item in validate_tse_output(
                canonical,
                expected_path=expected_output,
                forbidden_inputs=forbidden,
            )
        )
        referenced_outputs.add(expected_output)
    if observed != expected:
        violations.append(
            f"TSE output keys must preserve fixture order: expected={expected}, observed={observed}"
        )
    actual_outputs: set[Path] = set()
    for path in tse_outputs_root().rglob("*"):
        if path.is_symlink():
            violations.append(f"TSE outputs must not contain symlinks: {path.name}")
        elif path.is_file():
            actual_outputs.add(path.resolve())
    extra_outputs = sorted(path.name for path in actual_outputs - referenced_outputs)
    missing_outputs = sorted(path.name for path in referenced_outputs - actual_outputs)
    if extra_outputs:
        violations.append("TSE outputs contain unreferenced file(s): " + ", ".join(extra_outputs))
    if missing_outputs:
        violations.append("TSE outputs are missing referenced file(s): " + ", ".join(missing_outputs))
    return violations


def validate_contract(sample: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    required = string_list(contract.get("required_fields"))
    nonempty = string_list(contract.get("nonempty_fields"))
    primary = contract.get("primary_field")
    if isinstance(primary, str) and primary:
        if primary not in required:
            required.append(primary)
        if (
            primary not in nonempty
            and contract.get("allow_empty_primary") is not True
            and contract.get("allow_empty_segments") != "silence_only"
        ):
            nonempty.append(primary)
    for field in required:
        if field not in sample:
            violations.append(f"required field missing: {field}")
    for field in nonempty:
        if field in sample and not is_nonempty(sample[field]):
            violations.append(f"field must be nonempty: {field}")
    audio_evidence_fields = ("audio_path", "prediction_audio", "wavs", "wavs_summary", "sample_rate")
    if contract.get("output_type") == "audio" and not any(
        key in sample for key in audio_evidence_fields
    ):
        violations.append("audio output requires audio_path, wavs, wavs_summary, or sample_rate evidence")
    if contract.get("json_serializable") is True:
        try:
            json.dumps(sample)
        except TypeError as exc:
            violations.append(f"sample output is not JSON serializable: {exc}")
    return violations


def stage_import() -> bool:
    started = time.time()
    try:
        import_wrapper_class()
    except Exception as exc:  # noqa: BLE001
        append_log("VALIDATE_IMPORT", "failed", str(exc))
        write_stage_result("import", False, started, str(exc))
        return False
    append_log("VALIDATE_IMPORT", "passed", "Wrapper import succeeded.")
    write_stage_result("import", True, started)
    return True


def stage_load() -> bool:
    started = time.time()
    try:
        load_wrapper()
    except Exception as exc:  # noqa: BLE001
        append_log("VALIDATE_LOAD", "failed", str(exc))
        write_stage_result("load", False, started, str(exc))
        return False
    append_log("VALIDATE_LOAD", "passed", "Wrapper load succeeded.")
    write_stage_result("load", True, started)
    return True


def stage_infer() -> bool:
    started = time.time()
    try:
        wrapper = load_wrapper()
        task = normalized_task_type()
        payloads = (
            classification_fixture_payloads()
            if task in CLASSIFICATION_TASKS
            else fixture_payloads()
        )
        outputs: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for index, fixture in enumerate(payloads, start=1):
            payload = dict(fixture["input"])
            requested_output: Path | None = None
            if task == "se":
                key = str(fixture["fixture"].get("key") or index)
                requested_output = se_output_path(key, index)
                if requested_output.exists() or requested_output.is_symlink():
                    requested_output.unlink()
                payload["output_path"] = str(requested_output)
            elif task == TSE_TASK:
                key = str(fixture["fixture"].get("key") or index)
                requested_output = tse_output_path(key, index)
                if requested_output.exists() or requested_output.is_symlink():
                    requested_output.unlink()
                payload["output_path"] = str(requested_output)
            sample = (
                run_predict(wrapper, payload, scalar_fallback=False)
                if task == TSE_TASK
                else run_predict(wrapper, payload)
            )
            if not sample:
                raise AssertionError(f"prediction output is empty for fixture {index}")
            if task in STRUCTURED_TASKS:
                violations = validate_structured_output(
                    sample,
                    task=task,
                    duration_sec=float(fixture["fixture"]["duration_sec"]),
                    audio_is_silence=bool(fixture["fixture"]["audio_is_silence"]),
                )
                if violations:
                    raise AssertionError("; ".join(violations))
            elif task == "se":
                forbidden_inputs = [Path(str(payload["audio_path"]))]
                reference_path = fixture["fixture"].get("reference_audio_path")
                if isinstance(reference_path, str) and reference_path:
                    forbidden_inputs.append(Path(reference_path))
                violations = validate_se_output(
                    sample,
                    expected_path=requested_output,
                    forbidden_inputs=tuple(forbidden_inputs),
                )
                if violations:
                    raise AssertionError("; ".join(violations))
                assert requested_output is not None
                sample["audio_path"] = portable_se_output_path(requested_output)
            elif task == TSE_TASK:
                assert requested_output is not None
                canonical = validate_tse_output_object(
                    sample,
                    sample_id=str(fixture["fixture"].get("key") or index),
                )
                forbidden_inputs = tuple(
                    Path(str(fixture["fixture"][field]))
                    for field in (
                        "mixture_audio_path",
                        "enrollment_audio_path",
                        "reference_audio_path",
                    )
                )
                violations = validate_tse_output(
                    canonical,
                    expected_path=requested_output,
                    forbidden_inputs=forbidden_inputs,
                )
                if violations:
                    raise AssertionError("; ".join(violations))
                canonical["prediction_audio"] = portable_tse_output_path(requested_output)
                sample = canonical
            elif task in CLASSIFICATION_TASKS:
                sample = normalize_classification_output(task, sample)
            outputs.append(sample)
            if task in STRUCTURED_TASKS:
                row = {
                    "id": index,
                    "key": fixture["fixture"].get("key") or str(index),
                    "audio": fixture["fixture"].get("audio"),
                    "language": fixture["fixture"].get("language") or payload.get("language"),
                    "dataset": fixture["fixture"].get("dataset"),
                    "duration_sec": fixture["fixture"]["duration_sec"],
                    "sample_rate": fixture["fixture"]["sample_rate"],
                    "audio_is_silence": fixture["fixture"]["audio_is_silence"],
                    "output": sample,
                }
            elif task == TSE_TASK:
                row = {
                    "key": fixture["fixture"].get("key") or str(index),
                    "sample_id": fixture["fixture"].get("sample_id") or str(index),
                    "result": sample,
                }
            elif task in CLASSIFICATION_TASKS:
                row = {
                    "id": index,
                    "key": fixture["fixture"].get("key") or str(index),
                    "task": task,
                    "audio": fixture["fixture"].get("audio"),
                    "dataset": fixture["fixture"].get("dataset"),
                    "ground_truth": fixture["fixture"].get("ground_truth"),
                    **({"prompt": fixture["fixture"].get("prompt")} if task == "slu" else {}),
                    "result": sample,
                }
            else:
                row = {
                    "id": index,
                    "key": fixture["fixture"].get("key")
                    or (str(index) if task in {"kws", "se"} else None),
                    "audio": fixture["fixture"].get("audio"),
                    "reference_audio": fixture["fixture"].get("reference_audio"),
                    "language": fixture["fixture"].get("language") or payload.get("language"),
                    "dataset": fixture["fixture"].get("dataset"),
                    "ground_truth": fixture["fixture"].get("ground_truth"),
                    "text": fixture["fixture"].get("text"),
                    "output": sample,
                }
                row.update(
                    {
                        field: fixture["fixture"][field]
                        for field in ("label", "expected", "expected_detected", "expected_keyword")
                        if field in fixture["fixture"]
                    }
                )
            rows.append(row)
        write_json(
            SAMPLE_OUTPUT,
            {"rows": rows} if task in {TSE_TASK, *CLASSIFICATION_TASKS} else outputs[0],
        )
        write_jsonl(SAMPLE_OUTPUTS, rows)
    except Exception as exc:  # noqa: BLE001
        append_log("VALIDATE_INFER", "failed", str(exc))
        write_stage_result("infer", False, started, str(exc))
        return False
    append_log("VALIDATE_INFER", "passed", f"Inference passed for {len(outputs)} fixture sample(s).")
    inference_evidence: dict[str, Any] = {}
    if task in STRUCTURED_TASKS:
        inference_evidence["protocol_arguments"] = structured_protocol_arguments()
    write_stage_result(
        "infer",
        True,
        started,
        output_summary=output_summary(outputs),
        sample_outputs_path="artifacts/sample_outputs.jsonl",
        validated_sample_count=len(outputs),
        **inference_evidence,
    )
    return True


def stage_contract() -> bool:
    started = time.time()
    task = normalized_task_type()
    protocol_arguments: dict[str, Any] = {}
    try:
        if task in STRUCTURED_TASKS:
            protocol_arguments = structured_protocol_arguments()
        if not SAMPLE_OUTPUT.exists():
            raise FileNotFoundError(f"Missing sample output: {SAMPLE_OUTPUT}")
        sample = json.loads(SAMPLE_OUTPUT.read_text(encoding="utf-8"))
        if not isinstance(sample, dict):
            raise ValueError("sample_output.json must be an object")
        contract = load_io_contract()
        if task == TSE_TASK:
            if not SAMPLE_OUTPUTS.is_file():
                violations = ["TSE contract validation requires sample_outputs.jsonl"]
                rows = []
            else:
                rows = [
                    json.loads(line)
                    for line in SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                violations = validate_tse_output_document(sample, contract)
                if rows != sample.get("rows"):
                    violations.append("TSE sample_outputs.jsonl must exactly mirror sample_output rows")
        elif task in CLASSIFICATION_TASKS:
            violations = validate_classification_output_document(sample, task)
            if SAMPLE_OUTPUTS.is_file():
                rows = [
                    json.loads(line)
                    for line in SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                violations.extend(_validate_classification_output_rows(rows, task))
                if rows != sample.get("rows"):
                    violations.append(
                        "classification sample_outputs.jsonl must exactly mirror sample_output rows"
                    )
            else:
                violations.append("classification contract validation requires sample_outputs.jsonl")
        elif task in STRUCTURED_TASKS:
            if not SAMPLE_OUTPUTS.is_file():
                violations = ["structured contract validation requires sample_outputs.jsonl"]
            else:
                rows = [
                    json.loads(line)
                    for line in SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                violations = validate_structured_rows(rows, contract)
                expected_keys = [
                    str(fixture["fixture"].get("key") or index)
                    for index, fixture in enumerate(fixture_payloads(), 1)
                ]
                observed_keys = [str(row.get("key") or "") for row in rows if isinstance(row, dict)]
                if observed_keys != expected_keys:
                    violations.append(
                        "structured output rows must preserve every fixture key in order: "
                        f"expected={expected_keys}, observed={observed_keys}"
                    )
        elif task == "kws" and SAMPLE_OUTPUTS.is_file():
            rows = [
                json.loads(line)
                for line in SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            violations = validate_kws_rows(rows, contract)
        elif task == "se" and SAMPLE_OUTPUTS.is_file():
            rows = [
                json.loads(line)
                for line in SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            violations = validate_se_rows(rows, contract)
        else:
            violations = validate_contract(sample, contract)
            if task == "kws":
                violations.extend(validate_kws_output(sample))
            elif task == "se":
                violations.extend(validate_se_output(sample))
        if violations:
            raise AssertionError("; ".join(violations))
    except Exception as exc:  # noqa: BLE001
        append_log("VALIDATE_CONTRACT", "failed", str(exc))
        write_stage_result(
            "contract",
            False,
            started,
            str(exc),
            io_contract_satisfied=False,
            violations=[str(exc)],
            io_contract=IO_CONTRACT,
            sample_outputs_path="artifacts/sample_outputs.jsonl",
            **({"protocol_arguments": protocol_arguments} if task in STRUCTURED_TASKS else {}),
        )
        return False
    append_log("VALIDATE_CONTRACT", "passed", "Sample output satisfies io_contract.")
    write_stage_result(
        "contract",
        True,
        started,
        io_contract_satisfied=True,
        violations=[],
        io_contract=contract,
        sample_outputs_path="artifacts/sample_outputs.jsonl",
        validated_sample_count=(
            len(rows)
            if task in {"kws", "se", TSE_TASK, *STRUCTURED_TASKS, *CLASSIFICATION_TASKS} and SAMPLE_OUTPUTS.is_file()
            else 1
        ),
        **({"protocol_arguments": protocol_arguments} if task in STRUCTURED_TASKS else {}),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "import", "load", "infer", "contract"], default="all")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stages = [args.stage] if args.stage != "all" else ["import", "load", "infer", "contract"]
    ok = True
    for stage in stages:
        if stage == "import":
            ok = stage_import() and ok
        elif stage == "load":
            ok = stage_load() and ok
        elif stage == "infer":
            ok = stage_infer() and ok
        elif stage == "contract":
            ok = stage_contract() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
