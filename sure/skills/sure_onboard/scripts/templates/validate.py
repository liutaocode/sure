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
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
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


def normalized_task_type() -> str:
    normalized = TASK_TYPE.lower().replace("-", "_")
    if normalized in {"speech_enhancement", "acoustic_noise_suppression"}:
        return "se"
    return normalized


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


def fixture_payloads() -> list[dict[str, Any]]:
    task = normalized_task_type()
    raw_payload = os.environ.get("SURE_VALIDATE_INPUT_JSON")
    if raw_payload:
        parsed = json.loads(raw_payload)
        if not isinstance(parsed, dict):
            raise ValueError("SURE_VALIDATE_INPUT_JSON must decode to an object.")
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

    fixture_root = MODEL_DIR / "fixture"
    for gt_path in sorted(fixture_root.glob("**/gt.jsonl")):
        payloads: list[dict[str, Any]] = []
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            payload: dict[str, Any] = {}
            audio = item.get("audio") or item.get("wav") or item.get("prompt_audio") or item.get("reference_audio")
            if task == "se":
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
            if task != "se" and isinstance(text, str):
                payload["text"] = text
                payload["prompt_text"] = item.get("prompt_text", text)
            if task != "se" and isinstance(item.get("language"), str):
                payload["language"] = item["language"]
            if task != "se" and "keywords" in item:
                payload["keywords"] = item["keywords"]
            if task != "se" and "threshold" in item:
                if task == "kws" and not valid_kws_threshold(
                    item["threshold"]
                ):
                    raise ValueError(f"KWS fixture threshold must equal {KWS_OPERATING_THRESHOLD}")
                payload["threshold"] = item["threshold"]
            if payload:
                fixture_metadata: dict[str, Any] = {
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


def run_predict(wrapper: Any, payload: dict[str, Any]) -> dict[str, Any]:
    predict = getattr(wrapper, PREDICT_METHOD, None) or getattr(wrapper, "predict", None)
    if predict is None:
        raise AttributeError(f"Wrapper has neither {PREDICT_METHOD!r} nor 'predict'.")
    try:
        result = predict(payload)
    except TypeError:
        if TASK_TYPE.lower().replace("-", "_") == "kws":
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


def validate_contract(sample: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    required = string_list(contract.get("required_fields"))
    nonempty = string_list(contract.get("nonempty_fields"))
    primary = contract.get("primary_field")
    if isinstance(primary, str) and primary:
        if primary not in required:
            required.append(primary)
        if primary not in nonempty:
            nonempty.append(primary)
    for field in required:
        if field not in sample:
            violations.append(f"required field missing: {field}")
    for field in nonempty:
        if field in sample and not is_nonempty(sample[field]):
            violations.append(f"field must be nonempty: {field}")
    if contract.get("output_type") == "audio" and not any(
        key in sample for key in ("audio_path", "wavs", "wavs_summary", "sample_rate")
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
        payloads = fixture_payloads()
        outputs: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        task = normalized_task_type()
        for index, fixture in enumerate(payloads, start=1):
            payload = dict(fixture["input"])
            requested_output: Path | None = None
            if task == "se":
                key = str(fixture["fixture"].get("key") or index)
                requested_output = se_output_path(key, index)
                if requested_output.exists() or requested_output.is_symlink():
                    requested_output.unlink()
                payload["output_path"] = str(requested_output)
            sample = run_predict(wrapper, payload)
            if not sample:
                raise AssertionError(f"prediction output is empty for fixture {index}")
            if task == "se":
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
            outputs.append(sample)
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
        write_json(SAMPLE_OUTPUT, outputs[0])
        write_jsonl(SAMPLE_OUTPUTS, rows)
    except Exception as exc:  # noqa: BLE001
        append_log("VALIDATE_INFER", "failed", str(exc))
        write_stage_result("infer", False, started, str(exc))
        return False
    append_log("VALIDATE_INFER", "passed", f"Inference passed for {len(outputs)} fixture sample(s).")
    write_stage_result(
        "infer",
        True,
        started,
        output_summary=output_summary(outputs),
        sample_outputs_path="artifacts/sample_outputs.jsonl",
    )
    return True


def stage_contract() -> bool:
    started = time.time()
    try:
        if not SAMPLE_OUTPUT.exists():
            raise FileNotFoundError(f"Missing sample output: {SAMPLE_OUTPUT}")
        sample = json.loads(SAMPLE_OUTPUT.read_text(encoding="utf-8"))
        if not isinstance(sample, dict):
            raise ValueError("sample_output.json must be an object")
        contract = load_io_contract()
        task = normalized_task_type()
        if task == "kws" and SAMPLE_OUTPUTS.is_file():
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
