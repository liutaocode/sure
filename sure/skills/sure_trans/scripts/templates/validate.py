#!/usr/bin/env python3
"""Model-local validation template for SURE /sure_trans.

Keeps the same CLI contract as the /sure_onboard template:

    python validate.py --stage import
    python validate.py --stage load
    python validate.py --stage infer
    python validate.py --stage contract
    python validate.py --stage all

Each stage writes artifacts/<stage>_result.json. Inference writes
artifacts/sample_output.json, and contract validates that sample against the
filled IO_CONTRACT constant. Set SURE_VALIDATE_ARTIFACTS_DIR to redirect the
artifacts directory (in-container runs mount the run artifacts there).
"""

from __future__ import annotations

import argparse
import hashlib
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
ARTIFACTS_DIR = Path(os.environ.get("SURE_VALIDATE_ARTIFACTS_DIR") or (MODEL_DIR / "artifacts"))
VALIDATION_LOG = ARTIFACTS_DIR / "validation.log"
SAMPLE_OUTPUT = ARTIFACTS_DIR / "sample_output.json"

# Agent-filled constants.
MODEL_ID = "__MODEL_NAME__"
TASK_TYPE = "__TASK_TYPE__"
WRAPPER_MODULE = "model"
WRAPPER_CLASS = "ModelWrapper"
_IO_CONTRACT_JSON = r'''__IO_CONTRACT_JSON__'''
IO_CONTRACT: dict[str, Any] = (
    {} if _IO_CONTRACT_JSON.startswith("__") else json.loads(_IO_CONTRACT_JSON)
)
KWS_OPERATING_THRESHOLD = 0.5
REFERENCE_OUTPUT_FIELDS = {
    "answer",
    "expected",
    "ground_truth",
    "reference",
    "reference_segments",
    "reference_text",
    "target",
    "target_segments",
    "target_text",
}
SPEAKER_OUTPUT_FIELDS = {"segments", "num_speakers"}
SD_SEGMENT_FIELDS = {"speaker", "start", "end", "duration"}
SA_ASR_SEGMENT_FIELDS = {*SD_SEGMENT_FIELDS, "text"}
VAD_OUTPUT_FIELDS = {"speech_segments", "frame_scores"}
VAD_SEGMENT_FIELDS = {"start", "end"}
VAD_FRAME_SCORE_FIELDS = {"start", "end", "score"}
URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


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
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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
    module = __import__(WRAPPER_MODULE)
    return getattr(module, WRAPPER_CLASS)


def instantiate_wrapper() -> Any:
    wrapper_cls = import_wrapper_class()
    return wrapper_cls()


def load_wrapper() -> Any:
    wrapper = instantiate_wrapper()
    if hasattr(wrapper, "load"):
        wrapper.load()
    return wrapper


def first_fixture_payload() -> dict[str, Any]:
    raw_payload = os.environ.get("SURE_VALIDATE_INPUT_JSON")
    if raw_payload:
        parsed = json.loads(raw_payload)
        if not isinstance(parsed, dict):
            raise ValueError("SURE_VALIDATE_INPUT_JSON must decode to an object.")
        return parsed

    fixture_root = MODEL_DIR / "fixture"
    for gt_path in sorted(fixture_root.glob("**/gt.jsonl")):
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            payload: dict[str, Any] = {}
            audio = item.get("audio") or item.get("wav") or item.get("prompt_audio") or item.get("reference_audio")
            if isinstance(audio, str):
                payload["audio_path"] = str((gt_path.parent / audio).resolve())
                payload["prompt_audio_path"] = payload["audio_path"]
                payload["reference_audio_path"] = payload["audio_path"]
                payload["ref_audio"] = payload["audio_path"]
            text = item.get("target_text") or item.get("text") or item.get("prompt_text") or item.get("ground_truth")
            if isinstance(text, str):
                payload["text"] = text
                payload["prompt_text"] = item.get("prompt_text", text)
            if isinstance(item.get("language"), str):
                payload["language"] = item["language"]
            if payload:
                return payload
    raise FileNotFoundError(
        "No validation payload found. Set SURE_VALIDATE_INPUT_JSON or provide fixture/**/gt.jsonl."
    )


def kws_fixture_rows() -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for gt_path in sorted((MODEL_DIR / "fixture" / "kws").glob("**/gt.jsonl")):
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{gt_path}:{line_number} must be a JSON object")
            rows.append((gt_path, row))
    if not 2 <= len(rows) <= 5:
        raise ValueError("KWS validation requires 2 to 5 positive/negative fixture rows")
    return rows


def kws_fixture_payload(gt_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    audio = row.get("audio") or row.get("wav")
    if not isinstance(audio, str) or not audio:
        raise ValueError("KWS fixture row requires a non-empty audio or wav field")
    payload: dict[str, Any] = {"audio_path": str((gt_path.parent / audio).resolve())}
    keywords = row.get("keywords")
    if isinstance(keywords, (str, list)):
        payload["keywords"] = keywords
    threshold = row.get("threshold")
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        if not math.isfinite(float(threshold)) or float(threshold) != KWS_OPERATING_THRESHOLD:
            raise ValueError(f"KWS fixture threshold must equal {KWS_OPERATING_THRESHOLD}")
        payload["threshold"] = threshold
    elif threshold is not None:
        raise ValueError(f"KWS fixture threshold must equal {KWS_OPERATING_THRESHOLD}")
    return payload


def normalized_keyword(value: str) -> str:
    return "".join(value.upper().split())


def validate_kws_result(result: Any, reference: dict[str, Any]) -> list[str]:
    if not isinstance(result, dict):
        return ["result must be an object"]
    violations: list[str] = []
    for field in ("detected", "keyword", "score"):
        if field not in result:
            violations.append(f"missing required field: {field}")
    detected = result.get("detected")
    keyword = result.get("keyword")
    score = result.get("score")
    if not isinstance(detected, bool):
        violations.append("detected must be a boolean")
    if keyword is not None and (not isinstance(keyword, str) or not keyword.strip()):
        violations.append("keyword must be a non-empty string or null")
    if score is not None and (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        violations.append("score must be a finite number or null")
    elif isinstance(score, (int, float)) and not isinstance(score, bool) and not 0 <= float(score) <= 1:
        violations.append("score must be within [0, 1]")
    if detected is True:
        if not isinstance(keyword, str) or not keyword.strip():
            violations.append("detected=true requires a keyword")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            violations.append("detected=true requires a finite numeric score")
        elif float(score) < KWS_OPERATING_THRESHOLD:
            violations.append(f"detected=true requires score >= {KWS_OPERATING_THRESHOLD}")
    elif detected is False and keyword is not None:
        violations.append("detected=false requires keyword=null")
    elif (
        detected is False
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        and float(score) >= KWS_OPERATING_THRESHOLD
    ):
        violations.append(f"detected=false requires score < {KWS_OPERATING_THRESHOLD}")

    expected_detected = reference.get("expected_detected")
    if not isinstance(expected_detected, bool):
        violations.append("fixture expected_detected must be a boolean")
    elif isinstance(detected, bool) and detected is not expected_detected:
        violations.append(
            f"detection disagrees with fixture: expected {expected_detected}, got {detected}"
        )
    expected_keyword = reference.get("expected_keyword")
    if expected_detected is True and isinstance(keyword, str):
        if not isinstance(expected_keyword, str) or (
            normalized_keyword(keyword) != normalized_keyword(expected_keyword)
        ):
            violations.append(
                f"keyword disagrees with fixture: expected {expected_keyword!r}, got {keyword!r}"
            )
    return violations


def run_kws_fixture(wrapper: Any) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    polarities: set[bool] = set()
    for gt_path, reference in kws_fixture_rows():
        key = str(reference.get("key") or "").strip()
        if not key or key in seen:
            raise ValueError(f"KWS fixture key is missing or duplicated: {key!r}")
        seen.add(key)
        expected_detected = reference.get("expected_detected")
        if isinstance(expected_detected, bool):
            polarities.add(expected_detected)
        result = run_predict(wrapper, kws_fixture_payload(gt_path, reference), scalar_fallback=False)
        violations = validate_kws_result(result, reference)
        if violations:
            raise AssertionError(f"KWS sample {key}: {'; '.join(violations)}")
        output_rows.append({"key": key, "result": result})
    if polarities != {False, True}:
        raise ValueError("KWS validation requires at least one positive and one negative fixture")
    return {"rows": output_rows}


def se_fixture_rows() -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for gt_path in sorted((MODEL_DIR / "fixture" / "se").glob("**/gt.jsonl")):
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{gt_path}:{line_number} must be a JSON object")
            rows.append((gt_path, row))
    if not 1 <= len(rows) <= 5:
        raise ValueError("SE validation requires 1 to 5 noisy/clean fixture rows")
    return rows


def se_fixture_audio(gt_path: Path, row: dict[str, Any], role: str, key: str) -> Path:
    value = row.get(role, row.get("audio")) if role == "noisy_audio" else row.get(role)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SE fixture {key} requires {role}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"SE fixture {key} {role} path must be relative and contained")
    path = gt_path.parent / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(gt_path.parent.resolve()):
        raise ValueError(f"SE fixture {key} {role} is missing or unsafe")
    return path.resolve()


def validation_outputs_root() -> Path:
    root = ARTIFACTS_DIR / "outputs"
    if root.is_symlink():
        raise ValueError("SE validation outputs directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise ValueError("SE validation outputs directory must be writable")
    return root.resolve()


def se_output_path(key: str, index: int) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return validation_outputs_root() / f"{index:02d}-{digest}.wav"


def resolve_se_generated_audio(
    value: Any,
    *,
    key: str,
    expected_path: Path | None = None,
    forbidden_inputs: tuple[Path, ...] = (),
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SE result {key} requires audio_path")
    raw_path = Path(value)
    path = raw_path if raw_path.is_absolute() else validation_outputs_root() / raw_path
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"SE result {key} audio_path must be a real non-empty file")
    root = validation_outputs_root()
    try:
        lexical_relative = path.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError(f"SE result {key} audio_path must stay below validation outputs") from error
    current = root
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"SE result {key} audio_path must not traverse a symlink")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"SE result {key} audio_path must stay below validation outputs")
    if expected_path is not None and (
        path.absolute() != expected_path.absolute()
        or resolved != expected_path.resolve()
    ):
        raise ValueError(
            f"SE result {key} audio_path must equal the harness-assigned output_path"
        )
    for input_path in forbidden_inputs:
        try:
            aliases_input = resolved.samefile(input_path)
        except OSError:
            aliases_input = False
        if aliases_input:
            raise ValueError(
                f"SE result {key} audio_path must not alias noisy or clean input audio"
            )
    try:
        with wave.open(str(resolved), "rb") as handle:
            if (
                handle.getcomptype() != "NONE"
                or handle.getnchannels() < 1
                or handle.getsampwidth() not in {1, 2, 3, 4}
                or handle.getframerate() < 1
                or handle.getnframes() < 1
            ):
                raise ValueError(f"SE result {key} must be a non-empty PCM WAV")
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError(f"SE result {key} must be a readable PCM WAV: {error}") from error
    return resolved


def run_se_fixture(wrapper: Any) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (gt_path, reference) in enumerate(se_fixture_rows(), 1):
        key = str(reference.get("sample_id") or reference.get("key") or "").strip()
        if not key or key in seen:
            raise ValueError(f"SE fixture key is missing or duplicated: {key!r}")
        seen.add(key)
        noisy_audio = se_fixture_audio(gt_path, reference, "noisy_audio", key)
        clean_audio = se_fixture_audio(gt_path, reference, "reference_audio", key)
        if noisy_audio.samefile(clean_audio):
            raise ValueError(
                f"SE fixture {key} noisy_audio and reference_audio must be independent files"
            )
        requested_output = se_output_path(key, index)
        if requested_output.exists() or requested_output.is_symlink():
            requested_output.unlink()
        result = run_predict(
            wrapper,
            {
                "audio_path": str(noisy_audio),
                "output_path": str(requested_output),
            },
            scalar_fallback=False,
        )
        if not isinstance(result, dict):
            raise ValueError(f"SE sample {key} result must be an object")
        generated = resolve_se_generated_audio(
            result.get("audio_path"),
            key=key,
            expected_path=requested_output,
            forbidden_inputs=(noisy_audio, clean_audio),
        )
        result["audio_path"] = str(generated)
        output_rows.append({"key": key, "sample_id": key, "result": result})
    return {"rows": output_rows}


def vad_fixture_rows() -> list[tuple[Path, dict[str, Any]]]:
    fixture_root = MODEL_DIR / "fixture" / "vad"
    if fixture_root.is_symlink():
        raise ValueError("VAD fixture root must not be a symlink")
    rows: list[tuple[Path, dict[str, Any]]] = []
    for gt_path in sorted(fixture_root.glob("**/gt.jsonl")):
        if gt_path.is_symlink():
            raise ValueError("VAD gt.jsonl must not be a symlink")
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{gt_path}:{line_number} must be a JSON object")
            rows.append((gt_path, row))
    if not 1 <= len(rows) <= 5:
        raise ValueError("VAD validation requires 1 to 5 fixture rows")
    return rows


def vad_fixture_audio(gt_path: Path, row: dict[str, Any], *, key: str) -> Path:
    audio = row.get("audio") or row.get("wav")
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError(f"VAD fixture {key} requires audio or wav")
    relative = Path(audio)
    if relative.is_absolute() or ".." in relative.parts or "\\" in audio:
        raise ValueError(f"VAD fixture {key} audio path must be relative and contained")
    current = gt_path.parent
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"VAD fixture {key} audio must not traverse a symlink")
    audio_path = (gt_path.parent / relative).resolve()
    if (
        not audio_path.is_file()
        or audio_path.stat().st_size <= 0
        or not audio_path.is_relative_to(gt_path.parent.resolve())
    ):
        raise ValueError(f"VAD fixture {key} audio is missing or unsafe")
    return audio_path


def validate_vad_intervals(
    value: Any,
    *,
    label: str,
    frame_scores: bool = False,
    empty_allowed: bool = True,
    duration_sec: float | None = None,
) -> list[str]:
    interval_name = "frame_scores" if frame_scores else "speech_segments"
    if not isinstance(value, list):
        return [f"{label} {interval_name} must be an array"]
    violations: list[str] = []
    if frame_scores and not value:
        violations.append(f"{label} frame_scores must cover the complete audio timebase")
    if not frame_scores and not value and not empty_allowed:
        violations.append(
            f"{label} empty speech_segments are allowed only for pure-silence audio"
        )
    approved_fields = VAD_FRAME_SCORE_FIELDS if frame_scores else VAD_SEGMENT_FIELDS
    previous_end: float | None = None
    for index, interval in enumerate(value):
        if not isinstance(interval, dict):
            violations.append(f"{label} {interval_name}[{index}] must be an object")
            continue
        unknown = sorted(str(field) for field in interval if field not in approved_fields)
        if unknown:
            violations.append(
                f"{label} {interval_name}[{index}] contains unapproved field(s): "
                + ", ".join(unknown)
            )
        missing = sorted(field for field in approved_fields if field not in interval)
        if missing:
            violations.append(
                f"{label} {interval_name}[{index}] is missing field(s): "
                + ", ".join(missing)
            )
        start = interval.get("start")
        end = interval.get("end")
        start_valid = (
            not isinstance(start, bool)
            and isinstance(start, (int, float))
            and math.isfinite(float(start))
            and float(start) >= 0
        )
        if not start_valid:
            violations.append(
                f"{label} {interval_name}[{index}].start must be finite and >= 0"
            )
        end_valid = (
            not isinstance(end, bool)
            and isinstance(end, (int, float))
            and math.isfinite(float(end))
            and start_valid
            and float(end) > float(start)
        )
        if not end_valid:
            violations.append(
                f"{label} {interval_name}[{index}].end must be finite and > start"
            )
        if (
            frame_scores
            and index == 0
            and start_valid
            and not math.isclose(float(start), 0.0, rel_tol=0, abs_tol=1e-6)
        ):
            violations.append(f"{label} frame_scores must start at 0")
        if start_valid and previous_end is not None:
            if frame_scores and not math.isclose(
                float(start), previous_end, rel_tol=0, abs_tol=1e-6
            ):
                violations.append(
                    f"{label} frame_scores must be ordered, contiguous, and non-overlapping"
                )
            if not frame_scores and float(start) < previous_end:
                violations.append(
                    f"{label} {interval_name} must be ordered and non-overlapping"
                )
        if end_valid and duration_sec is not None and float(end) > duration_sec + 1e-6:
            violations.append(
                f"{label} {interval_name}[{index}].end exceeds WAV duration {duration_sec:.6f}"
            )
        if frame_scores:
            score = interval.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0 <= float(score) <= 1
            ):
                violations.append(
                    f"{label} frame_scores[{index}].score must be finite and within [0, 1]"
                )
        if end_valid:
            previous_end = float(end)
    if (
        frame_scores
        and duration_sec is not None
        and previous_end is not None
        and not math.isclose(previous_end, duration_sec, rel_tol=0, abs_tol=1e-6)
    ):
        violations.append(f"{label} frame_scores must end at WAV duration {duration_sec:.6f}")
    return violations


def validate_vad_result(
    result: Any,
    *,
    label: str,
    empty_allowed: bool = False,
    duration_sec: float | None = None,
) -> list[str]:
    if not isinstance(result, dict):
        return [f"{label} result must be an object"]
    if "speech_segments" not in result:
        return [f"{label} result requires speech_segments"]
    violations: list[str] = []
    unknown = sorted(str(field) for field in result if field not in VAD_OUTPUT_FIELDS)
    if unknown:
        violations.append(f"{label} result contains unapproved field(s): " + ", ".join(unknown))
    violations.extend(
        validate_vad_intervals(
            result.get("speech_segments"),
            label=label,
            empty_allowed=empty_allowed,
            duration_sec=duration_sec,
        )
    )
    if "frame_scores" in result:
        violations.extend(
            validate_vad_intervals(
                result.get("frame_scores"),
                label=label,
                frame_scores=True,
                duration_sec=duration_sec,
            )
        )
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        violations.append(f"{label} result must be strict JSON: {error}")
    return violations


def run_vad_fixture(wrapper: Any) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gt_path, reference in vad_fixture_rows():
        key = str(reference.get("key") or "").strip()
        if not key or key in seen:
            raise ValueError(f"VAD fixture key is missing or duplicated: {key!r}")
        if looks_like_absolute_path_or_uri(key):
            raise ValueError(f"VAD fixture key must be a safe identifier: {key!r}")
        seen.add(key)
        audio_path = vad_fixture_audio(gt_path, reference, key=key)
        wav_info = pcm_wav_info(audio_path)
        reference_violations = validate_vad_intervals(
            reference.get("speech_segments"),
            label=f"VAD fixture {key}",
            empty_allowed=bool(wav_info["audio_is_silence"]),
            duration_sec=float(wav_info["duration_sec"]),
        )
        if reference_violations:
            raise ValueError("; ".join(reference_violations))
        result = run_predict(
            wrapper,
            {"audio_path": str(audio_path)},
            scalar_fallback=False,
        )
        violations = validate_vad_result(
            result,
            label=f"VAD sample {key}",
            empty_allowed=bool(wav_info["audio_is_silence"]),
            duration_sec=float(wav_info["duration_sec"]),
        )
        if violations:
            raise AssertionError("; ".join(violations))
        output_rows.append({"key": key, "result": result})
    return {"rows": output_rows}


def speaker_fixture_rows(task: str) -> list[tuple[Path, dict[str, Any]]]:
    if task not in {"sd", "sa_asr"}:
        raise ValueError(f"unsupported speaker task: {task}")
    fixture_root = MODEL_DIR / "fixture" / task
    if fixture_root.is_symlink():
        raise ValueError(f"{task.upper()} fixture root must not be a symlink")
    rows: list[tuple[Path, dict[str, Any]]] = []
    for gt_path in sorted(fixture_root.glob("**/gt.jsonl")):
        if gt_path.is_symlink():
            raise ValueError(f"{task.upper()} gt.jsonl must not be a symlink")
        for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{gt_path}:{line_number} must be a JSON object")
            rows.append((gt_path, row))
    if not 1 <= len(rows) <= 5:
        raise ValueError(f"{task.upper()} validation requires 1 to 5 fixture rows")
    return rows


def speaker_fixture_audio(gt_path: Path, row: dict[str, Any], *, task: str, key: str) -> Path:
    audio = row.get("audio") or row.get("wav")
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError(f"{task.upper()} fixture {key} requires audio or wav")
    relative = Path(audio)
    if relative.is_absolute() or ".." in relative.parts or "\\" in audio:
        raise ValueError(f"{task.upper()} fixture {key} audio path must be relative and contained")
    current = gt_path.parent
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{task.upper()} fixture {key} audio must not traverse a symlink")
    audio_path = (gt_path.parent / relative).resolve()
    if (
        not audio_path.is_file()
        or audio_path.stat().st_size <= 0
        or not audio_path.is_relative_to(gt_path.parent.resolve())
    ):
        raise ValueError(f"{task.upper()} fixture {key} audio is missing or unsafe")
    return audio_path


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
        raise ValueError("structured speaker audio must be a PCM WAV")
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getcomptype() != "NONE"
                or handle.getnchannels() < 1
                or handle.getsampwidth() not in {1, 2, 3, 4}
                or handle.getframerate() < 1
                or handle.getnframes() < 1
            ):
                raise ValueError("structured speaker audio must be a non-empty PCM WAV")
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            frames = handle.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError(f"structured speaker audio must be a readable PCM WAV: {error}") from error
    expected_bytes = frame_count * channels * sample_width
    if len(frames) != expected_bytes:
        raise ValueError(
            f"structured speaker PCM data is truncated: expected {expected_bytes} bytes, read {len(frames)}"
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


def validate_speaker_segments(
    value: Any,
    *,
    task: str,
    label: str,
    empty_sd_allowed: bool = False,
    duration_sec: float | None = None,
) -> list[str]:
    if not isinstance(value, list):
        return [f"{label} segments must be an array"]
    violations: list[str] = []
    if task == "sa_asr" and not value:
        violations.append(f"{label} SA-ASR segments must not be empty")
    if task == "sd" and not value and not empty_sd_allowed:
        violations.append(f"{label} empty SD segments are allowed only for pure-silence audio")
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            violations.append(f"{label} segments[{index}] must be an object")
            continue
        approved_fields = SA_ASR_SEGMENT_FIELDS if task == "sa_asr" else SD_SEGMENT_FIELDS
        unknown = sorted(str(field) for field in segment if field not in approved_fields)
        if unknown:
            violations.append(
                f"{label} segments[{index}] contains unapproved field(s): " + ", ".join(unknown)
            )
        speaker = segment.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            violations.append(f"{label} segments[{index}].speaker must be a non-empty string")
        elif looks_like_absolute_path_or_uri(speaker):
            violations.append(f"{label} segments[{index}].speaker must be a safe token")
        start = segment.get("start")
        end = segment.get("end")
        start_valid = (
            not isinstance(start, bool)
            and isinstance(start, (int, float))
            and math.isfinite(float(start))
            and float(start) >= 0
        )
        if not start_valid:
            violations.append(f"{label} segments[{index}].start must be finite and >= 0")
        end_valid = (
            not isinstance(end, bool)
            and isinstance(end, (int, float))
            and math.isfinite(float(end))
            and start_valid
            and float(end) > float(start)
        )
        if not end_valid:
            violations.append(f"{label} segments[{index}].end must be finite and > start")
        if task == "sa_asr":
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                violations.append(f"{label} segments[{index}].text must be a non-empty string")
        if end_valid and duration_sec is not None and float(end) > duration_sec + 1e-6:
            violations.append(f"{label} segments[{index}].end exceeds WAV duration {duration_sec:.6f}")
        duration = segment.get("duration")
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
            or not start_valid
            or not end_valid
            or not math.isclose(
                float(duration), float(end) - float(start), rel_tol=0, abs_tol=1e-3
            )
        ):
            violations.append(
                f"{label} segments[{index}].duration must be finite, positive, and equal end-start"
            )
    return violations


def forbidden_speaker_output_fields(value: Any, path: str = "result") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child = f"{path}.{key}"
            if (
                normalized in REFERENCE_OUTPUT_FIELDS
                or normalized.startswith("reference_")
                or normalized == "path"
                or normalized.endswith("_path")
            ):
                found.append(child)
            found.extend(forbidden_speaker_output_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_speaker_output_fields(item, f"{path}[{index}]"))
    return found


def validate_speaker_result(
    result: Any,
    *,
    task: str,
    label: str,
    empty_sd_allowed: bool = False,
    duration_sec: float | None = None,
) -> list[str]:
    if not isinstance(result, dict):
        return [f"{label} result must be an object"]
    if "segments" not in result:
        return [f"{label} result requires segments"]
    unknown_fields = sorted(str(field) for field in result if field not in SPEAKER_OUTPUT_FIELDS)
    violations = [
        f"{label} result must not expose reference or path field {path}"
        for path in forbidden_speaker_output_fields(result)
    ]
    if unknown_fields:
        violations.append(f"{label} result contains unapproved field(s): " + ", ".join(unknown_fields))
    for field, item in result.items():
        if field != "segments" and isinstance(item, str) and looks_like_absolute_path_or_uri(item):
            violations.append(f"{label} result contains absolute path or URI at result.{field}")
    violations.extend(validate_speaker_segments(
        result.get("segments"),
        task=task,
        label=label,
        empty_sd_allowed=empty_sd_allowed,
        duration_sec=duration_sec,
    ))
    segments = result.get("segments")
    num_speakers = result.get("num_speakers")
    if num_speakers is not None:
        observed = {
            segment.get("speaker").strip()
            for segment in segments
            if isinstance(segment, dict)
            and isinstance(segment.get("speaker"), str)
            and segment.get("speaker").strip()
        } if isinstance(segments, list) else set()
        if (
            isinstance(num_speakers, bool)
            or not isinstance(num_speakers, int)
            or num_speakers < 0
            or num_speakers != len(observed)
        ):
            violations.append("num_speakers must equal the number of distinct segment speakers")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        violations.append(f"{label} result must be strict JSON: {error}")
    return violations


def run_speaker_fixture(wrapper: Any, *, task: str) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gt_path, reference in speaker_fixture_rows(task):
        key = str(reference.get("key") or "").strip()
        if not key or key in seen:
            raise ValueError(f"{task.upper()} fixture key is missing or duplicated: {key!r}")
        if looks_like_absolute_path_or_uri(key):
            raise ValueError(f"{task.upper()} fixture key must be a safe token: {key!r}")
        seen.add(key)
        audio_path = speaker_fixture_audio(gt_path, reference, task=task, key=key)
        wav_info = pcm_wav_info(audio_path)
        reference_violations = validate_speaker_segments(
            reference.get("segments"),
            task=task,
            label=f"{task.upper()} fixture {key}",
            empty_sd_allowed=task == "sd" and bool(wav_info["audio_is_silence"]),
            duration_sec=float(wav_info["duration_sec"]),
        )
        if reference_violations:
            raise ValueError("; ".join(reference_violations))
        result = run_predict(
            wrapper,
            {"audio_path": str(audio_path)},
            scalar_fallback=False,
        )
        violations = validate_speaker_result(
            result,
            task=task,
            label=f"{task.upper()} sample {key}",
            empty_sd_allowed=task == "sd" and bool(wav_info["audio_is_silence"]),
            duration_sec=float(wav_info["duration_sec"]),
        )
        if violations:
            raise AssertionError("; ".join(violations))
        output_rows.append({"key": key, "result": result})
    return {"rows": output_rows}


def to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return to_plain(value.to_dict())
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
    predict = getattr(wrapper, "predict", None)
    if predict is None:
        raise AttributeError("Wrapper has no 'predict' method.")
    try:
        result = predict(payload)
    except TypeError:
        if not scalar_fallback:
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
    raise FileNotFoundError("IO_CONTRACT was not filled during adapter scaffolding.")


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


def validate_contract(sample: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field in string_list(contract.get("required_fields")):
        if field not in sample:
            violations.append(f"missing required field: {field}")
    for field in string_list(contract.get("nonempty_fields")):
        if field in sample and not is_nonempty(sample.get(field)):
            violations.append(f"field must be non-empty: {field}")
    primary = contract.get("primary_field")
    if (
        isinstance(primary, str)
        and contract.get("allow_empty_primary") is not True
        and not is_nonempty(sample.get(primary))
    ):
        violations.append(f"primary output field must be non-empty: {primary}")
    if contract.get("json_serializable") is True:
        try:
            json.dumps(sample)
        except TypeError as exc:
            violations.append(f"sample output is not JSON serializable: {exc}")
    return violations


def validate_kws_output_document(sample: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    output_rows = sample.get("rows")
    if not isinstance(output_rows, list):
        return ["KWS sample_output.json must be an object with a rows array"]
    references: dict[str, dict[str, Any]] = {}
    for _, reference in kws_fixture_rows():
        key = str(reference.get("key") or "")
        if not key or key in references:
            return [f"KWS fixture key is missing or duplicated: {key!r}"]
        references[key] = reference
    predictions: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    for index, output in enumerate(output_rows):
        if not isinstance(output, dict):
            violations.append(f"rows[{index}] must be an object")
            continue
        key = str(output.get("key") or "")
        if not key:
            violations.append(f"rows[{index}] requires a non-empty key")
            continue
        if key in predictions:
            violations.append(f"duplicate KWS output key: {key}")
            continue
        result = output.get("result")
        if not isinstance(result, dict):
            violations.append(f"KWS output {key} result must be an object")
            continue
        predictions[key] = result
    missing = sorted(set(references) - set(predictions))
    extra = sorted(set(predictions) - set(references))
    if missing:
        violations.append(f"missing KWS output keys: {', '.join(missing)}")
    if extra:
        violations.append(f"unexpected KWS output keys: {', '.join(extra)}")
    for key in sorted(set(references) & set(predictions)):
        result_violations = validate_contract(predictions[key], contract)
        result_violations.extend(validate_kws_result(predictions[key], references[key]))
        violations.extend(f"KWS output {key}: {violation}" for violation in result_violations)
    return violations


def validate_se_output_document(sample: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    rows = sample.get("rows")
    if not isinstance(rows, list):
        return ["SE sample_output.json must be an object with a rows array"]
    references: dict[str, tuple[Path, Path, Path]] = {}
    for fixture_index, (gt_path, reference) in enumerate(se_fixture_rows(), 1):
        key = str(reference.get("sample_id") or reference.get("key") or "")
        if not key or key in references:
            return [f"SE fixture key is missing or duplicated: {key!r}"]
        references[key] = (
            se_fixture_audio(gt_path, reference, "noisy_audio", key),
            se_fixture_audio(gt_path, reference, "reference_audio", key),
            se_output_path(key, fixture_index),
        )
    outputs: set[str] = set()
    violations: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(f"rows[{index}] must be an object")
            continue
        key = str(row.get("sample_id") or row.get("key") or "")
        if not key or key in outputs:
            violations.append(f"SE output key is missing or duplicated: {key!r}")
            continue
        outputs.add(key)
        result = row.get("result")
        if not isinstance(result, dict):
            violations.append(f"SE output {key} result must be an object")
            continue
        violations.extend(f"SE output {key}: {item}" for item in validate_contract(result, contract))
        try:
            if key not in references:
                raise ValueError(f"unexpected SE output key: {key}")
            noisy_audio, clean_audio, expected_path = references[key]
            resolve_se_generated_audio(
                result.get("audio_path"),
                key=key,
                expected_path=expected_path,
                forbidden_inputs=(noisy_audio, clean_audio),
            )
        except ValueError as error:
            violations.append(str(error))
    missing = sorted(set(references) - outputs)
    extra = sorted(outputs - set(references))
    if missing:
        violations.append(f"missing SE output keys: {', '.join(missing)}")
    if extra:
        violations.append(f"unexpected SE output keys: {', '.join(extra)}")
    return violations


def validate_speaker_output_document(
    sample: dict[str, Any],
    contract: dict[str, Any],
    *,
    task: str,
) -> list[str]:
    rows = sample.get("rows")
    if not isinstance(rows, list):
        return [f"{task.upper()} sample_output.json must be an object with a rows array"]
    references: dict[str, tuple[Path, dict[str, Any]]] = {}
    for gt_path, reference in speaker_fixture_rows(task):
        key = str(reference.get("key") or "")
        if not key or key in references:
            return [f"{task.upper()} fixture key is missing or duplicated: {key!r}"]
        try:
            audio_path = speaker_fixture_audio(
                gt_path,
                reference,
                task=task,
                key=key,
            )
            references[key] = (audio_path, pcm_wav_info(audio_path))
        except ValueError as error:
            return [str(error)]
    outputs: set[str] = set()
    violations: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(f"rows[{index}] must be an object")
            continue
        key = str(row.get("key") or "")
        if not key or key in outputs:
            violations.append(f"{task.upper()} output key is missing or duplicated: {key!r}")
            continue
        if looks_like_absolute_path_or_uri(key):
            violations.append(f"{task.upper()} output key must be a safe token: {key!r}")
        outputs.add(key)
        result = row.get("result")
        if not isinstance(result, dict):
            violations.append(f"{task.upper()} output {key} result must be an object")
            continue
        violations.extend(
            f"{task.upper()} output {key}: {item}"
            for item in validate_contract(result, contract)
        )
        wav_info = references.get(key, (Path(), {}))[1]
        violations.extend(
            validate_speaker_result(
                result,
                task=task,
                label=f"{task.upper()} output {key}",
                empty_sd_allowed=(
                    task == "sd"
                    and key in references
                    and bool(wav_info.get("audio_is_silence"))
                ),
                duration_sec=(
                    float(wav_info["duration_sec"])
                    if isinstance(wav_info.get("duration_sec"), (int, float))
                    else None
                ),
            )
        )
    missing = sorted(set(references) - outputs)
    extra = sorted(outputs - set(references))
    if missing:
        violations.append(f"missing {task.upper()} output keys: {', '.join(missing)}")
    if extra:
        violations.append(f"unexpected {task.upper()} output keys: {', '.join(extra)}")
    return violations


def validate_vad_output_document(
    sample: dict[str, Any], contract: dict[str, Any]
) -> list[str]:
    rows = sample.get("rows")
    if not isinstance(rows, list):
        return ["VAD sample_output.json must be an object with a rows array"]
    references: dict[str, dict[str, Any]] = {}
    for gt_path, reference in vad_fixture_rows():
        key = str(reference.get("key") or "")
        if not key or key in references:
            return [f"VAD fixture key is missing or duplicated: {key!r}"]
        try:
            audio_path = vad_fixture_audio(gt_path, reference, key=key)
            references[key] = pcm_wav_info(audio_path)
        except ValueError as error:
            return [str(error)]
    outputs: set[str] = set()
    violations: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(f"rows[{index}] must be an object")
            continue
        key = str(row.get("key") or "")
        if not key or key in outputs:
            violations.append(f"VAD output key is missing or duplicated: {key!r}")
            continue
        if looks_like_absolute_path_or_uri(key):
            violations.append(f"VAD output key must be a safe identifier: {key!r}")
        outputs.add(key)
        result = row.get("result")
        if not isinstance(result, dict):
            violations.append(f"VAD output {key} result must be an object")
            continue
        violations.extend(
            f"VAD output {key}: {item}" for item in validate_contract(result, contract)
        )
        wav_info = references.get(key, {})
        violations.extend(
            validate_vad_result(
                result,
                label=f"VAD output {key}",
                empty_allowed=key in references and bool(wav_info.get("audio_is_silence")),
                duration_sec=(
                    float(wav_info["duration_sec"])
                    if isinstance(wav_info.get("duration_sec"), (int, float))
                    else None
                ),
            )
        )
    missing = sorted(set(references) - outputs)
    extra = sorted(outputs - set(references))
    if missing:
        violations.append(f"missing VAD output keys: {', '.join(missing)}")
    if extra:
        violations.append(f"unexpected VAD output keys: {', '.join(extra)}")
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
        task = TASK_TYPE.lower().replace("-", "_")
        if task == "kws":
            sample = run_kws_fixture(wrapper)
        elif task == "se":
            sample = run_se_fixture(wrapper)
        elif task == "vad":
            sample = run_vad_fixture(wrapper)
        elif task in {"sd", "sa_asr"}:
            sample = run_speaker_fixture(wrapper, task=task)
        else:
            payload = first_fixture_payload()
            sample = run_predict(wrapper, payload)
        if not sample:
            raise AssertionError("prediction output is empty")
        write_json(SAMPLE_OUTPUT, sample)
    except Exception as exc:  # noqa: BLE001
        append_log("VALIDATE_INFER", "failed", str(exc))
        write_stage_result("infer", False, started, str(exc))
        return False
    append_log("VALIDATE_INFER", "passed", "Inference produced sample_output.json.")
    write_stage_result(
        "infer",
        True,
        started,
        output_summary=json.dumps(sample, ensure_ascii=True)[:500],
    )
    return True


def stage_contract() -> bool:
    started = time.time()
    try:
        if not SAMPLE_OUTPUT.exists():
            # infer and contract are coupled through SURE_VALIDATE_ARTIFACTS_DIR:
            # infer writes the sample there and contract reads it back. Pointing
            # the two stages at separate directories fails here every time, so say
            # so rather than reporting a bare missing file.
            seen = sorted(child.name for child in ARTIFACTS_DIR.iterdir()) if ARTIFACTS_DIR.is_dir() else []
            raise FileNotFoundError(
                f"Missing sample output: {SAMPLE_OUTPUT}. The infer stage writes "
                f"sample_output.json into SURE_VALIDATE_ARTIFACTS_DIR, so contract must run "
                f"with the same directory infer used. This one holds: {seen or 'nothing'}"
            )
        sample = json.loads(SAMPLE_OUTPUT.read_text(encoding="utf-8"))
        if not isinstance(sample, dict):
            raise ValueError("sample_output.json must be an object")
        contract = load_io_contract()
        task = TASK_TYPE.lower().replace("-", "_")
        if task == "kws":
            violations = validate_kws_output_document(sample, contract)
        elif task == "se":
            violations = validate_se_output_document(sample, contract)
        elif task == "vad":
            violations = validate_vad_output_document(sample, contract)
        elif task in {"sd", "sa_asr"}:
            violations = validate_speaker_output_document(sample, contract, task=task)
        else:
            violations = validate_contract(sample, contract)
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
