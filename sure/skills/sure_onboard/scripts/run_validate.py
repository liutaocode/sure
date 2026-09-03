#!/usr/bin/env python3
"""Gate script for the VALIDATE_{IMPORT,LOAD,INFER,CONTRACT} units.

Routes by --kind to the corresponding minimal-validation test. The gate does
not trust a boolean alone: each validation artifact must provide either:

  - run_command: ["python", "-c", "..."] or a shell string; or
  - validate_py: path/to/validate.py plus optional validate_args.

The command is executed by this gate, stdout/stderr are written to log_path,
and only then is the matching *_passed boolean accepted.

Kinds map to the minimal_validation contract (import/load/infer/contract):
    import    -> the model module imports without error
    load      -> the model object instantiates and loads weights
    infer     -> a minimal inference call produces output
    contract  -> the output satisfies model.spec.yaml io_contract

Called by the Sure hook:
    python3 scripts/run_validate.py --kind <import|load|infer|contract> \
        --run-dir <runDir> --produces <abs>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "runtime" / "harness"))
from model_child_env import model_child_env

from structured_segments import (
    canonical_task,
    is_structured_task,
    structured_task_contract,
    validate_structured_rows,
)
from tse_contract import (
    canonical_task as canonical_tse_task,
    safe_relative_audio,
    safe_sample_id,
    task_contract as tse_task_contract,
    validate_output_object as validate_tse_output_object,
)

KIND_TO_PASS_KEY = {
    "import": "import_passed",
    "load": "load_passed",
    "infer": "infer_passed",
    "contract": "contract_passed",
}

ACTUAL_DEVICES = {"cuda", "cpu", "mps"}
CLASSIFICATION_TASKS = {"ser", "gr", "slu"}
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
CLASSIFICATION_URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def resolve_path(raw: object, base: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return base / path


def command_from_artifact(data: dict, kind: str, run_dir: Path) -> tuple[list[str] | str | None, bool, Path]:
    cwd_raw = data.get("cwd") or data.get("model_dir")
    cwd = resolve_path(cwd_raw, run_dir) or run_dir

    raw_command = data.get("run_command")
    if isinstance(raw_command, list) and all(isinstance(item, str) for item in raw_command):
        return raw_command, False, cwd
    if isinstance(raw_command, str) and raw_command.strip():
        return raw_command, True, cwd

    validate_py = resolve_path(data.get("validate_py"), cwd)
    if validate_py:
        if not validate_py.exists():
            raise FileNotFoundError(f"validate_py does not exist: {validate_py}")
        validate_args = data.get("validate_args")
        extra_args = validate_args if isinstance(validate_args, list) and all(isinstance(item, str) for item in validate_args) else []
        # Generic validate.py files in the reference repo commonly run the full
        # import/load/infer/contract chain without a --stage flag. If a model
        # supports stage-specific args, the artifact can provide validate_args.
        return [sys.executable, str(validate_py), *extra_args], False, validate_py.parent

    return None, False, cwd


def log_path_for(data: dict, kind: str, run_dir: Path, cwd: Path) -> Path:
    raw = data.get("log_path")
    if raw:
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            return path
        if str(raw).startswith("artifacts/"):
            return cwd / path
        return run_dir / "artifacts" / path
    return run_dir / "artifacts" / f"{kind}_execution.log"


def env_for(data: dict, validation_device: str | None = None) -> dict[str, str]:
    env = model_child_env()
    if validation_device in ACTUAL_DEVICES:
        env.setdefault("SURE_DEVICE", validation_device)
        env.setdefault("DEVICE", validation_device)
    raw_env = data.get("env")
    if isinstance(raw_env, dict):
        for key, value in raw_env.items():
            if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                env[key] = str(value)
    return env


def maybe_use_model_local_python(command: list[str] | str, *, shell: bool, cwd: Path) -> list[str] | str:
    if shell or not isinstance(command, list) or not command:
        return command
    executable = Path(command[0]).name
    python_names = {"python", "python3", "python3.10", "python3.11", "python3.12"}
    if executable not in python_names and command[0] != sys.executable:
        return command
    local_python = cwd / ".venv" / "bin" / "python"
    if local_python.exists():
        return [str(local_python), *command[1:]]
    return command


def repo_root_for(run_dir: Path) -> Path:
    resolved = run_dir.expanduser().resolve()
    if resolved.parent.name == "runs" and resolved.parent.parent.name == ".sure":
        return resolved.parent.parent.parent
    return Path.cwd().resolve()


def normalize_repo_relative_text(value: str, repo_root: Path) -> str:
    replacement = str(repo_root / "sure" / "models") + "/"
    # Pass a function, not the string, as the replacement. re.sub treats
    # backslashes in a *string* replacement as escapes (\s, \1, ...), which
    # raises on Windows where replacement contains path backslashes. A
    # function's return value is inserted literally, with no escape parsing.
    return re.sub(r"(?<![A-Za-z0-9_./-])sure/models/", lambda _match: replacement, value)


def normalize_repo_relative_command(command: list[str] | str, repo_root: Path) -> list[str] | str:
    if isinstance(command, str):
        return normalize_repo_relative_text(command, repo_root)
    return [normalize_repo_relative_text(part, repo_root) for part in command]


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object.")
    return data


def validation_device_for(run_dir: Path) -> str | None:
    env_compat_path = run_dir / "artifacts" / "env_compat_result.json"
    if env_compat_path.exists():
        try:
            compat = read_json(env_compat_path)
        except ValueError:
            compat = {}
        device = str(compat.get("device") or "").lower()
        if device in ACTUAL_DEVICES:
            return device

    resolved_path = run_dir / "artifacts" / "model_input_resolved.json"
    if resolved_path.exists():
        try:
            resolved = read_json(resolved_path)
        except ValueError:
            resolved = {}
        requested = str(resolved.get("device") or "").lower()
        if requested in ACTUAL_DEVICES:
            return requested
    return None


def declared_device_in_artifact(data: dict) -> str | None:
    device = data.get("device")
    if isinstance(device, str) and device.strip():
        return device.strip().lower()
    raw_env = data.get("env")
    if isinstance(raw_env, dict):
        for key in ("DEVICE", "SURE_DEVICE"):
            value = raw_env.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return None


def validate_artifact_device(data: dict, target: str | None) -> str | None:
    if target not in ACTUAL_DEVICES:
        return None
    declared = declared_device_in_artifact(data)
    if declared in ACTUAL_DEVICES and declared != target:
        return (
            f"validation artifact requests device={declared!r}, but "
            f"env_compat_result selected device={target!r}. Keep validation "
            "on the same device, or rerun VALIDATE_ENV_COMPAT with documented "
            "CPU fallback evidence first."
        )
    return None


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def sample_output_path_for(data: dict, run_dir: Path) -> Path | None:
    model_dir = resolve_path(data.get("model_dir"), run_dir)
    raw = data.get("sample_output_path")
    candidates: list[Path] = []
    if raw:
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(run_dir / "artifacts" / path)
            candidates.append(run_dir / path)
            if model_dir:
                candidates.append(model_dir / path)
                if str(raw).startswith("artifacts/"):
                    candidates.append(model_dir / path)
    candidates.append(run_dir / "artifacts" / "sample_output.json")
    if model_dir:
        candidates.append(model_dir / "artifacts" / "sample_output.json")
    return first_existing(candidates)


def load_sample_output(data: dict, run_dir: Path) -> tuple[Path, dict]:
    path = sample_output_path_for(data, run_dir)
    if not path:
        raise FileNotFoundError(
            "sample_output.json is required for VALIDATE_INFER/VALIDATE_CONTRACT. "
            "Write it under <run_dir>/artifacts/sample_output.json or declare sample_output_path."
        )
    return path, read_json(path)


def sample_outputs_path_for(data: dict, run_dir: Path) -> Path | None:
    model_dir = resolve_path(data.get("model_dir"), run_dir)
    raw = data.get("sample_outputs_path")
    candidates: list[Path] = []
    if raw:
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend((run_dir / "artifacts" / path, run_dir / path))
            if model_dir:
                candidates.extend((model_dir / path, model_dir / "artifacts" / path.name))
    candidates.append(run_dir / "artifacts" / "sample_outputs.jsonl")
    if model_dir:
        candidates.append(model_dir / "artifacts" / "sample_outputs.jsonl")
    return first_existing(candidates)


def read_jsonl_objects(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object")
        rows.append(row)
    return rows


def structured_task_for(data: dict, run_dir: Path) -> str:
    resolved_path = run_dir / "artifacts" / "model_input_resolved.json"
    if resolved_path.is_file():
        task = canonical_task(read_json(resolved_path).get("task_type"))
        if is_structured_task(task) or canonical_tse_task(task) == "tse":
            return task
    task = canonical_task(data.get("task_type"))
    return "tse" if canonical_tse_task(task) == "tse" else task


def classification_task_for(data: dict, run_dir: Path) -> str | None:
    """Resolve a classification task from the approved run input or artifact."""

    candidates: list[object] = []
    resolved_path = run_dir / "artifacts" / "model_input_resolved.json"
    if resolved_path.is_file():
        candidates.append(read_json(resolved_path).get("task_type"))
    candidates.extend((data.get("task_type"), data.get("task")))
    for candidate in candidates:
        task = canonical_task(candidate)
        if task in CLASSIFICATION_TASKS:
            return task
    return None


def _classification_normalize_label(task: str, value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{task.upper()} label is unknown: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{task.upper()} label is unknown: {value!r}")
    text = str(value).strip().lower()
    text = re.sub(r"^[\s\[({<]+|[\s\])}>.,!?;:：，。！？；]+$", "", text)
    aliases = (
        {**SER_LABEL_ALIASES, **SER_NUMERIC_ALIASES}
        if task == "ser"
        else {**GR_LABEL_ALIASES, **GR_NUMERIC_ALIASES}
    )
    if text not in aliases:
        raise ValueError(f"{task.upper()} label is unknown: {value!r}")
    return aliases[text]


def _classification_normalize_answer(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("SLU answer must be a string or finite scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("SLU answer must be a string or finite scalar")
    text = str(value).strip().rstrip(".!?。！？")
    if not text or any(ord(character) < 32 for character in text):
        raise ValueError("SLU answer must be non-empty and must not contain control characters")
    match = re.fullmatch(r"(?is)(?:the\s+)?answer\s*(?:is|:|-)?\s*([A-Za-z0-9_+-]+)", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"答案\s*(?:是|为|:|：)?\s*([A-Za-z0-9_+-]+)", text)
    return match.group(1) if match else text


def _classification_normalize_result(task: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        if task in {"ser", "gr"} and not isinstance(value, bool):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, int):
                value = {"label": value}
        elif task == "slu" and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = {"answer": value}
    if not isinstance(value, dict):
        raise ValueError("classification output must be an object or supported scalar")
    allowed = {"label", "score", "text"} if task in {"ser", "gr"} else {"answer", "label", "text"}
    unknown = sorted(str(field) for field in value if field not in allowed)
    if unknown:
        raise ValueError("classification output contains unapproved field(s): " + ", ".join(unknown))
    forbidden = sorted(
        str(field)
        for field in value
        if str(field).strip().lower().replace("-", "_") in CLASSIFICATION_ROW_REFERENCE_FIELDS
        or str(field).strip().lower().endswith("_path")
    )
    if forbidden:
        raise ValueError("classification output contains reference/path field(s): " + ", ".join(forbidden))
    if task in {"ser", "gr"}:
        raw = value["label"] if value.get("label") is not None else value.get("text")
        output: dict[str, Any] = {"label": _classification_normalize_label(task, raw)}
        score = value.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("classification score must be finite and within [0, 1]")
            score_value = float(score)
            if not math.isfinite(score_value) or not 0 <= score_value <= 1:
                raise ValueError("classification score must be finite and within [0, 1]")
            output["score"] = score_value
        return output
    raw_answer = (
        value["answer"]
        if value.get("answer") is not None
        else value["label"]
        if value.get("label") is not None
        else value.get("text")
    )
    output = {"answer": _classification_normalize_answer(raw_answer)}
    if value.get("label") is not None:
        output["label"] = _classification_normalize_answer(value["label"])
    return output


def _classification_forbidden_row_fields(value: object, path: str = "row") -> list[str]:
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
            found.extend(_classification_forbidden_row_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_classification_forbidden_row_fields(item, f"{path}[{index}]"))
    return found


def _classification_fixture_expectations(
    manifest: dict, task: str
) -> list[dict[str, object]]:
    """Read canonical classification references without exposing them to inference."""

    if canonical_task(manifest.get("task_type")) != task:
        raise ValueError("classification fixture manifest task_type disagrees with validation task")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 5:
        raise ValueError("classification fixture manifest must contain 1 to 5 samples")
    staged_raw = Path(str(manifest.get("staged_dir") or "")).expanduser()
    gt_raw = Path(str(manifest.get("gt_jsonl") or "")).expanduser()
    if not staged_raw.is_absolute() or staged_raw.is_symlink() or not staged_raw.is_dir():
        raise ValueError("classification fixture staged_dir must be a real absolute directory")
    staged_dir = staged_raw.resolve()
    raw_model_dir = Path(str(manifest.get("model_dir") or "")).expanduser()
    if not raw_model_dir.is_absolute() or raw_model_dir.is_symlink():
        raise ValueError("classification fixture manifest model_dir must be absolute")
    expected_root = raw_model_dir.resolve() / "fixture" / task
    if not staged_dir.is_relative_to(expected_root):
        raise ValueError("classification fixture staged_dir must stay under model_dir/fixture/task")
    if gt_raw.is_symlink() or not gt_raw.is_file() or gt_raw.resolve().parent != staged_dir:
        raise ValueError("classification fixture gt_jsonl must be a real file directly in staged_dir")
    for entry in staged_dir.rglob("*"):
        if entry.is_symlink():
            raise ValueError("classification fixture tree must not contain symlinks")
    references = read_jsonl_objects(gt_raw)
    if len(references) != len(samples) or manifest.get("sample_count") != len(samples):
        raise ValueError("classification fixture samples must mirror gt_jsonl")

    expectations: list[dict[str, object]] = []
    for index, (reference, sample) in enumerate(zip(references, samples, strict=True), 1):
        if not isinstance(sample, dict):
            raise ValueError(f"classification fixture sample {index} must be an object")
        key = reference.get("key")
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError(f"classification fixture row {index} key is unsafe")
        if "/" in key or "\\" in key or any(ord(character) < 32 or character.isspace() for character in key):
            raise ValueError(f"classification fixture row {index} key is unsafe")
        declared_task = reference.get("task_type", reference.get("task"))
        if canonical_task(declared_task) != task:
            raise ValueError(f"classification fixture row {key} task does not match {task}")
        audio = reference.get("audio") or reference.get("wav")
        if not isinstance(audio, str) or not audio.strip():
            raise ValueError(f"classification fixture row {key} requires audio")
        relative_audio = Path(audio)
        if (
            relative_audio.is_absolute()
            or ".." in relative_audio.parts
            or "\\" in audio
            or CLASSIFICATION_URI_PREFIX.match(audio)
            or not (staged_dir / relative_audio).is_file()
            or not (staged_dir / relative_audio).resolve().is_relative_to(staged_dir)
        ):
            raise ValueError(f"classification fixture row {key} audio path is unsafe")
        if sample.get("key") != key or sample.get("audio") != relative_audio.as_posix():
            raise ValueError(f"classification fixture sample {key} identity changed")
        sample_audio_path = Path(str(sample.get("audio_path") or "")).expanduser()
        if (
            not sample_audio_path.is_absolute()
            or sample_audio_path.resolve() != (staged_dir / relative_audio).resolve()
        ):
            raise ValueError(f"classification fixture sample {key} audio_path changed")
        forbidden = _classification_forbidden_row_fields(reference)
        if forbidden:
            raise ValueError(
                f"classification fixture row {key} exposes reference/path field(s): "
                + ", ".join(forbidden)
            )
        reference_value = reference.get("ground_truth")
        if task in {"ser", "gr"}:
            normalized_reference = _classification_normalize_label(task, reference_value)
        else:
            normalized_reference = _classification_normalize_answer(reference_value)
            prompt = reference.get("prompt") or reference.get("instruction")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"classification fixture row {key} requires a non-empty prompt")
        expectations.append(
            {
                "key": key,
                "audio": relative_audio.as_posix(),
                "dataset": reference.get("dataset"),
                "ground_truth": normalized_reference,
                **({"prompt": prompt} if task == "slu" else {}),
            }
        )
    return expectations


def _validate_classification_rows(
    rows: object,
    expectations: list[dict[str, object]],
    task: str,
    *,
    label: str,
) -> list[str]:
    if not isinstance(rows, list):
        return [f"{label} must be an array of classification rows"]
    allowed_fields = set(CLASSIFICATION_OUTPUT_ROW_FIELDS)
    if task != "slu":
        allowed_fields.discard("prompt")
    expected_keys = [str(item["key"]) for item in expectations]
    observed_keys: list[str] = []
    violations: list[str] = []
    for index, row in enumerate(rows, 1):
        prefix = f"{label} row {index}"
        if not isinstance(row, dict):
            violations.append(f"{prefix} must be an object")
            continue
        unknown = sorted(str(field) for field in row if field not in allowed_fields)
        if unknown:
            violations.append(f"{prefix} contains unapproved field(s): " + ", ".join(unknown))
        forbidden = _classification_forbidden_row_fields(row)
        if forbidden:
            violations.append(f"{prefix} exposes reference/path field(s): " + ", ".join(forbidden))
        missing = sorted(field for field in allowed_fields if field not in row)
        if missing:
            violations.append(f"{prefix} is missing field(s): " + ", ".join(missing))

        key_value = row.get("key")
        key = key_value.strip() if isinstance(key_value, str) else ""
        if (
            not isinstance(key_value, str)
            or key_value != key
            or not key
            or "/" in key
            or "\\" in key
            or any(ord(character) < 32 or character.isspace() for character in key)
        ):
            violations.append(f"{prefix} key must be a safe canonical token")
            key = ""
        if key in observed_keys:
            violations.append(f"{prefix} key is duplicated: {key!r}")
        observed_keys.append(key)

        expected = expectations[index - 1] if index <= len(expectations) else {}
        expected_key = str(expected.get("key") or "")
        if key and key != expected_key:
            violations.append(f"{prefix} key does not preserve fixture order")
        if row.get("id") != index or isinstance(row.get("id"), bool):
            violations.append(f"{prefix} id must equal its one-based fixture position")
        if row.get("task") != task:
            violations.append(f"{prefix} task must use canonical value {task!r}")
        if row.get("audio") != expected.get("audio"):
            violations.append(f"{prefix} audio does not match the fixture")
        audio = row.get("audio")
        if not isinstance(audio, str) or (
            Path(audio).is_absolute()
            or ".." in Path(audio).parts
            or "\\" in audio
            or CLASSIFICATION_URI_PREFIX.match(audio)
        ):
            violations.append(f"{prefix} audio must be a portable relative path")
        if row.get("dataset") != expected.get("dataset"):
            violations.append(f"{prefix} dataset metadata changed")
        dataset = row.get("dataset")
        if dataset is not None and (
            not isinstance(dataset, str)
            or CLASSIFICATION_URI_PREFIX.match(dataset)
            or Path(dataset).is_absolute()
        ):
            violations.append(f"{prefix} dataset must be a portable string or null")
        if row.get("ground_truth") != expected.get("ground_truth"):
            try:
                observed_reference = (
                    _classification_normalize_label(task, row.get("ground_truth"))
                    if task in {"ser", "gr"}
                    else _classification_normalize_answer(row.get("ground_truth"))
                )
            except (TypeError, ValueError) as error:
                violations.append(f"{prefix} has invalid ground_truth: {error}")
            else:
                if observed_reference != expected.get("ground_truth"):
                    violations.append(f"{prefix} ground_truth does not match the fixture")
        if task == "slu" and row.get("prompt") != expected.get("prompt"):
            violations.append(f"{prefix} prompt metadata changed")

        result = row.get("result")
        if not isinstance(result, dict):
            violations.append(f"{prefix} result must be an object")
            continue
        try:
            canonical = _classification_normalize_result(task, result)
        except (TypeError, ValueError) as error:
            violations.append(f"{prefix} result: {error}")
        else:
            if result != canonical:
                violations.append(f"{prefix} result is not canonical")
    if observed_keys != expected_keys:
        violations.append(
            f"{label} keys must preserve fixture order: expected={expected_keys}, observed={observed_keys}"
        )
    return violations


def validate_classification_evidence(data: dict, run_dir: Path) -> list[str]:
    """Validate classification sample outputs at the outer gate boundary."""

    task = classification_task_for(data, run_dir)
    if task is None:
        return []
    violations: list[str] = []
    try:
        contract = io_contract_from(data, run_dir)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    primary = "label" if task in {"ser", "gr"} else "answer"
    if contract.get("input_type") != "audio_path":
        violations.append("classification io_contract input_type must be audio_path")
    if contract.get("output_type") not in {"json", "classification", "classification_answer"}:
        violations.append("classification io_contract output_type is unsupported")
    if contract.get("primary_field") != primary:
        violations.append(f"classification io_contract primary_field must be {primary!r}")
    required = contract.get("required_fields")
    nonempty = contract.get("nonempty_fields")
    if not isinstance(required, list) or primary not in required:
        violations.append(f"classification io_contract required_fields must contain {primary!r}")
    if not isinstance(nonempty, list) or primary not in nonempty:
        violations.append(f"classification io_contract nonempty_fields must contain {primary!r}")
    if contract.get("json_serializable") is not True:
        violations.append("classification io_contract must require JSON-serializable output")
    try:
        manifest = fixture_manifest_for(data, run_dir)
        expectations = _classification_fixture_expectations(manifest, task)
        sample_path, sample = load_sample_output(data, run_dir)
        outputs_path = sample_outputs_path_for(data, run_dir)
        if outputs_path is None:
            raise FileNotFoundError("classification validation requires sample_outputs.jsonl")
        rows = read_jsonl_objects(outputs_path)
    except (OSError, ValueError) as exc:
        return [*violations, str(exc)]
    sample_rows = sample.get("rows") if isinstance(sample, dict) else None
    violations.extend(
        _validate_classification_rows(
            sample_rows,
            expectations,
            task,
            label=f"{sample_path} classification output",
        )
    )
    violations.extend(
        _validate_classification_rows(
            rows,
            expectations,
            task,
            label=f"{outputs_path} classification output",
        )
    )
    if rows != sample_rows:
        violations.append("classification sample_outputs.jsonl must exactly mirror sample_output rows")
    return violations


def validate_tse_evidence(data: dict, run_dir: Path) -> list[str]:
    """Validate keyed TSE outputs and role isolation for Onboard gates."""

    task = structured_task_for(data, run_dir)
    if task != "tse":
        return []
    violations: list[str] = []
    try:
        contract = io_contract_from(data, run_dir)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    expected_contract = tse_task_contract()["io_contract"]
    if contract != expected_contract:
        violations.append("io_contract must equal the canonical TSE contract")
    try:
        manifest = fixture_manifest_for(data, run_dir)
        outputs_path = sample_outputs_path_for(data, run_dir)
        if outputs_path is None:
            raise FileNotFoundError("TSE validation requires sample_outputs.jsonl")
        rows = read_jsonl_objects(outputs_path)
    except (OSError, ValueError) as exc:
        return [*violations, str(exc)]
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        return [*violations, "TSE fixture manifest must contain samples"]
    staged_dir_raw = manifest.get("staged_dir")
    staged_dir = Path(str(staged_dir_raw)).expanduser() if staged_dir_raw else None
    if staged_dir is None or not staged_dir.is_absolute() or not staged_dir.is_dir():
        violations.append("TSE fixture manifest staged_dir must be an existing absolute directory")
        staged_dir = None
    expected_keys: list[str] = []
    for sample_index, item in enumerate(samples, 1):
        if not isinstance(item, dict):
            violations.append(f"TSE fixture manifest sample {sample_index} must be an object")
            continue
        try:
            key = safe_sample_id(item.get("sample_id") or item.get("key"))
        except ValueError as exc:
            violations.append(f"TSE fixture manifest sample {sample_index}: {exc}")
            continue
        expected_keys.append(key)
        role_paths: list[Path] = []
        for role in ("mixture_audio", "enrollment_audio", "reference_audio"):
            try:
                relative = safe_relative_audio(item.get(role), role=role)
            except ValueError as exc:
                violations.append(f"TSE fixture {key}: {exc}")
                continue
            if staged_dir is None:
                continue
            current = staged_dir
            for part in relative.parts:
                current /= part
                if current.is_symlink():
                    violations.append(f"TSE fixture {key} {role} traverses a symlink")
                    break
            resolved_role = (staged_dir / relative).resolve()
            if (
                not resolved_role.is_file()
                or not resolved_role.is_relative_to(staged_dir.resolve())
            ):
                violations.append(f"TSE fixture {key} {role} is missing or unsafe")
            role_paths.append(resolved_role)
        if len(role_paths) == 3 and (
            len(set(role_paths)) != 3
            or any(
                left.samefile(right)
                for offset, left in enumerate(role_paths)
                for right in role_paths[offset + 1 :]
                if left.is_file() and right.is_file()
            )
        ):
            violations.append(f"TSE fixture {key} roles must be independent files")
    observed_keys: list[str] = []
    output_root_path = outputs_path.parent / "outputs"
    if output_root_path.is_symlink() or not output_root_path.is_dir():
        return [*violations, "TSE output directory must be a real directory"]
    output_root = output_root_path.resolve()
    referenced_outputs: set[Path] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            violations.append(f"TSE output row {index} must be an object")
            continue
        key = str(row.get("key") or row.get("sample_id") or "").strip()
        if not key or key in observed_keys or "/" in key or "\\" in key or any(character.isspace() for character in key):
            violations.append(f"TSE output row {index} has an unsafe or duplicate key")
            continue
        if row.get("sample_id") is not None and str(row.get("sample_id")).strip() != key:
            violations.append(f"TSE output {key} sample_id does not match key")
        observed_keys.append(key)
        result = row.get("result")
        try:
            canonical = validate_tse_output_object(result, sample_id=key)
        except (TypeError, ValueError) as exc:
            violations.append(f"TSE output {key}: {exc}")
            continue
        if result != canonical:
            violations.append(f"TSE output {key} is not canonical")
            continue
        raw_path = Path(str(canonical["prediction_audio"]))
        if raw_path.is_absolute():
            path = raw_path
        elif raw_path.parts[:2] == ("artifacts", "outputs"):
            path = outputs_path.parent.parent / raw_path
        elif raw_path.parts[:1] == ("outputs",):
            path = output_root / Path(*raw_path.parts[1:])
        else:
            path = output_root / raw_path
        resolved_path = path.resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or not resolved_path.is_relative_to(output_root)
        ):
            violations.append(f"TSE output {key} prediction_audio must stay under outputs")
        else:
            referenced_outputs.add(resolved_path)
    if observed_keys != expected_keys:
        violations.append(f"TSE output keys must preserve fixture order: expected={expected_keys}, observed={observed_keys}")
    output_entries = list(output_root_path.rglob("*"))
    output_symlinks = [entry for entry in output_entries if entry.is_symlink()]
    if output_symlinks:
        violations.append(f"TSE output tree must not contain symlinks: {output_symlinks[0]}")
    actual_outputs = {entry.resolve() for entry in output_entries if entry.is_file()}
    extras = sorted(entry.name for entry in actual_outputs - referenced_outputs)
    missing = sorted(entry.name for entry in referenced_outputs - actual_outputs)
    if extras:
        violations.append("TSE output tree contains unreferenced file(s): " + ", ".join(extras))
    if missing:
        violations.append("TSE output tree is missing referenced file(s): " + ", ".join(missing))
    return violations


def fixture_manifest_for(data: dict, run_dir: Path) -> dict:
    model_dir = resolve_path(data.get("model_dir"), run_dir)
    candidates = [run_dir / "artifacts" / "fixture_manifest.json"]
    if model_dir:
        candidates.append(model_dir / "artifacts" / "fixture_manifest.json")
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError("structured-task validation requires fixture_manifest.json")
    return read_json(path)


def validate_structured_evidence(data: dict, run_dir: Path) -> list[str]:
    classification_task = classification_task_for(data, run_dir)
    if classification_task is not None:
        return validate_classification_evidence(data, run_dir)
    task = structured_task_for(data, run_dir)
    if task == "tse":
        return validate_tse_evidence(data, run_dir)
    if not is_structured_task(task):
        return []
    expected_contract = structured_task_contract(task)["io_contract"]
    actual_contract = io_contract_from(data, run_dir)
    violations: list[str] = []
    if actual_contract != expected_contract:
        violations.append("io_contract must equal the canonical structured task contract")
    outputs_path = sample_outputs_path_for(data, run_dir)
    if outputs_path is None:
        return [*violations, "structured-task validation requires sample_outputs.jsonl"]
    try:
        rows = read_jsonl_objects(outputs_path)
        manifest = fixture_manifest_for(data, run_dir)
    except (OSError, ValueError) as exc:
        return [*violations, str(exc)]
    if canonical_task(manifest.get("task_type")) != task:
        violations.append("fixture manifest task_type disagrees with structured validation task")
    fixture_root = Path(str(manifest.get("staged_dir") or "")).expanduser()
    if not fixture_root.is_absolute() or not fixture_root.is_dir():
        violations.append("fixture manifest staged_dir must be an existing absolute directory")
        fixture_root = None
    violations.extend(
        validate_structured_rows(
            rows,
            task=task,
            samples=manifest.get("samples"),
            fixture_root=fixture_root,
        )
    )
    try:
        _, first_output = load_sample_output(data, run_dir)
    except (OSError, ValueError) as exc:
        violations.append(str(exc))
    else:
        if rows and rows[0].get("output") != first_output:
            violations.append("sample_output.json must equal the first structured output row")
    return violations


def io_contract_from(data: dict, run_dir: Path) -> dict:
    contract = data.get("io_contract")
    if isinstance(contract, dict):
        return contract
    resolved_path = run_dir / "artifacts" / "model_input_resolved.json"
    if resolved_path.exists():
        resolved = read_json(resolved_path)
        normalized = resolved.get("normalized_model_input")
        if isinstance(normalized, dict) and isinstance(normalized.get("io_contract"), dict):
            return normalized["io_contract"]
    raise ValueError(
        "io_contract is required for VALIDATE_CONTRACT. "
        "Provide it in contract_result.json or in model_input_resolved.normalized_model_input.io_contract."
    )


def is_nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def string_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def validate_output_contract(sample_output: dict, contract: dict) -> list[str]:
    violations: list[str] = []
    required_fields = string_list(contract.get("required_fields"))
    nonempty_fields = string_list(contract.get("nonempty_fields"))
    primary_field = contract.get("primary_field")
    if isinstance(primary_field, str) and primary_field and primary_field not in required_fields:
        required_fields.append(primary_field)
    if (
        isinstance(primary_field, str)
        and primary_field
        and primary_field not in nonempty_fields
        and contract.get("allow_empty_primary") is not True
        and contract.get("allow_empty_segments") != "silence_only"
    ):
        nonempty_fields.append(primary_field)

    for field in required_fields:
        if field not in sample_output:
            violations.append(f"required field missing: {field}")
    for field in nonempty_fields:
        if field in sample_output and not is_nonempty(sample_output.get(field)):
            violations.append(f"field must be nonempty: {field}")

    output_type = contract.get("output_type")
    if output_type == "audio":
        if not any(field in sample_output for field in ("audio_path", "wavs", "sample_rate", "wavs_summary")):
            violations.append("audio output requires audio_path, wavs, wavs_summary, or sample_rate evidence")
    if output_type in {"json", "text"} and not required_fields:
        violations.append("json/text output contract must declare required_fields or primary_field")

    json_serializable = contract.get("json_serializable")
    if json_serializable is True:
        try:
            json.dumps(sample_output)
        except TypeError as exc:
            violations.append(f"sample_output is not JSON serializable: {exc}")
    return violations


def run_validation_command(
    data: dict,
    kind: str,
    run_dir: Path,
    validation_device: str | None,
) -> tuple[int, float, Path, str, Path]:
    command, shell, cwd = command_from_artifact(data, kind, run_dir)
    if command is None:
        raise ValueError(
            "validation artifacts must include run_command or validate_py so "
            "the gate can execute the import/load/infer/contract check for real."
        )
    if not cwd.exists():
        raise FileNotFoundError(f"validation cwd does not exist: {cwd}")
    repo_root = repo_root_for(run_dir)
    command = maybe_use_model_local_python(command, shell=shell, cwd=cwd)
    command = normalize_repo_relative_command(command, repo_root)

    log_path = log_path_for(data, kind, run_dir, cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timeout_raw = data.get("timeout_seconds")
    timeout = float(timeout_raw) if isinstance(timeout_raw, (int, float)) and timeout_raw > 0 else 1800.0
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env_for(data, validation_device),
            capture_output=True,
            text=True,
            shell=shell,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        rendered = command if isinstance(command, str) else " ".join(shlex.quote(part) for part in command)
        log_path.write_text(
            f"$ {rendered}\n\nTIMEOUT after {elapsed:.3f}s\n\nSTDOUT:\n{exc.stdout or ''}\n\nSTDERR:\n{exc.stderr or ''}\n",
            encoding="utf-8",
        )
        return 124, elapsed, log_path, f"validation command timed out after {elapsed:.3f}s", cwd

    elapsed = time.time() - started
    rendered = command if isinstance(command, str) else " ".join(shlex.quote(part) for part in command)
    log_path.write_text(
        f"$ {rendered}\n"
        f"cwd={cwd}\n"
        f"exit_code={proc.returncode}\n"
        f"duration_seconds={elapsed:.3f}\n\n"
        f"STDOUT:\n{proc.stdout or ''}\n\nSTDERR:\n{proc.stderr or ''}\n",
        encoding="utf-8",
    )
    detail = (proc.stderr or proc.stdout or "").strip()
    return proc.returncode, elapsed, log_path, detail, cwd


def generated_result_candidates(data: dict, kind: str, run_dir: Path, cwd: Path, produces: Path) -> list[Path]:
    filename = f"{kind}_result.json"
    candidates: list[Path] = []
    model_dir = resolve_path(data.get("model_dir") or data.get("wrapper_path"), run_dir)
    if model_dir and model_dir.is_file():
        model_dir = model_dir.parent
    if model_dir:
        candidates.append(model_dir / "artifacts" / filename)
    candidates.append(cwd / "artifacts" / filename)
    validate_py = resolve_path(data.get("validate_py"), cwd)
    if validate_py:
        candidates.append(validate_py.parent / "artifacts" / filename)
    candidates.append(run_dir / "artifacts" / filename)
    candidates.append(produces)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        try:
            key = candidate.expanduser().resolve()
        except FileNotFoundError:
            key = candidate.expanduser().absolute()
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def snapshot_key(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def snapshot_generated_results(
    data: dict,
    kind: str,
    run_dir: Path,
    cwd: Path,
    produces: Path,
) -> dict[str, tuple[int, bytes]]:
    snapshots: dict[str, tuple[int, bytes]] = {}
    for candidate in generated_result_candidates(data, kind, run_dir, cwd, produces):
        if not candidate.exists():
            continue
        try:
            stat = candidate.stat()
            snapshots[snapshot_key(candidate)] = (stat.st_mtime_ns, candidate.read_bytes())
        except OSError:
            continue
    return snapshots


def first_generated_result(
    data: dict,
    kind: str,
    run_dir: Path,
    cwd: Path,
    produces: Path,
    *,
    before: dict[str, tuple[int, bytes]],
) -> tuple[Path, dict] | None:
    for candidate in generated_result_candidates(data, kind, run_dir, cwd, produces):
        if not candidate.exists():
            continue
        # Model-local artifact directories may contain older adopted validation
        # results from a reference workspace. They are useful only when the
        # command executed by this gate refreshed them; otherwise they can
        # incorrectly override the current run artifact.
        try:
            stat = candidate.stat()
            current = candidate.read_bytes()
            previous = before.get(snapshot_key(candidate))
            if previous is not None and previous == (stat.st_mtime_ns, current):
                continue
        except OSError:
            continue
        try:
            result = json.loads(current.decode("utf-8"))
        except ValueError:
            continue
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            continue
        return candidate, result
    return None


def normalize_validation_result(
    original: dict,
    generated: dict,
    kind: str,
    pass_key: str,
    elapsed: float,
    log_path: Path,
) -> dict:
    merged = dict(original)
    merged.update(generated)
    merged[pass_key] = bool(generated.get(pass_key))
    merged.setdefault("duration_ms", round(elapsed * 1000, 3))
    merged["error"] = generated.get("error")
    merged["log_path"] = str(log_path)
    merged["executed"] = True

    # The shared validate.py template writes sample_output_path for every stage,
    # but import/load schemas intentionally only describe those stages. Keep the
    # durable run artifacts schema-clean while preserving sample paths for
    # infer/contract, where package_gate and contract checks need them.
    if kind in {"import", "load"}:
        for key in ("sample_output_path", "output_summary", "io_contract", "io_contract_satisfied", "violations"):
            merged.pop(key, None)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=list(KIND_TO_PASS_KEY))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--produces", required=True)
    args = parser.parse_args()

    pass_key = KIND_TO_PASS_KEY[args.kind]
    run_dir = Path(args.run_dir)
    path = Path(args.produces)
    if not path.exists():
        print(f"{args.kind}_result.json not found at {path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{args.kind}_result.json is not valid JSON: {exc}", file=sys.stderr)
        return 1

    try:
        validation_device = validation_device_for(run_dir)
        device_violation = validate_artifact_device(data, validation_device)
        if device_violation:
            print(f"VALIDATE_{args.kind.upper()} gate failed: {device_violation}", file=sys.stderr)
            return 1
        _, _, expected_cwd = command_from_artifact(data, args.kind, run_dir)
        before_results = snapshot_generated_results(data, args.kind, run_dir, expected_cwd, path)
        exit_code, elapsed, log_path, detail, validation_cwd = run_validation_command(
            data,
            args.kind,
            run_dir,
            validation_device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"VALIDATE_{args.kind.upper()} gate failed before execution: {exc}", file=sys.stderr)
        return 1
    if exit_code != 0:
        print(
            f"VALIDATE_{args.kind.upper()} gate execution failed with exit_code={exit_code}. "
            f"log_path={log_path}\n{detail}",
            file=sys.stderr,
        )
        return 1

    generated = first_generated_result(data, args.kind, run_dir, validation_cwd, path, before=before_results)
    if generated:
        _, generated_data = generated
        data = normalize_validation_result(data, generated_data, args.kind, pass_key, elapsed, log_path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not data.get(pass_key):
        error = data.get("error")
        detail = f"\n  error: {error}" if error else ""
        print(
            f"VALIDATE_{args.kind.upper()} gate failed: {pass_key} is false.{detail}",
            file=sys.stderr,
        )
        return 1

    # Cross-check: when the kind is load/infer/contract and a wrapper path /
    # model dir is declared, ensure the wrapper file exists.
    if args.kind in ("load", "infer", "contract"):
        model_dir = data.get("model_dir") or data.get("wrapper_path")
        if model_dir:
            candidate = Path(model_dir)
            # model_dir may point to the wrapper dir or a file; check parent.
            targets = [candidate, candidate.parent] if candidate.is_file() else [candidate]
            if not any(p.is_dir() and (p / "model.py").exists() for p in targets):
                print(
                    f"VALIDATE_{args.kind.upper()} gate: declared model dir/wrapper "
                    f"missing model.py: {model_dir}",
                    file=sys.stderr,
                )
                return 1

    if args.kind == "infer":
        try:
            sample_path, sample_output = load_sample_output(data, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"VALIDATE_INFER gate failed: {exc}", file=sys.stderr)
            return 1
        if not sample_output:
            print(f"VALIDATE_INFER gate failed: sample_output is empty: {sample_path}", file=sys.stderr)
            return 1
        structured_violations = validate_structured_evidence(data, run_dir)
        if structured_violations:
            print(
                "VALIDATE_INFER gate failed: " + "; ".join(structured_violations),
                file=sys.stderr,
            )
            return 1
        classification_task = classification_task_for(data, run_dir)
        if (
            classification_task is not None
            or is_structured_task(structured_task_for(data, run_dir))
            or structured_task_for(data, run_dir) == "tse"
        ):
            outputs_path = sample_outputs_path_for(data, run_dir)
            assert outputs_path is not None
            rows = read_jsonl_objects(outputs_path)
            data["sample_outputs_path"] = str(outputs_path)
            data["validated_sample_count"] = len(rows)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.kind == "contract":
        try:
            sample_path, sample_output = load_sample_output(data, run_dir)
            contract = io_contract_from(data, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"VALIDATE_CONTRACT gate failed: {exc}", file=sys.stderr)
            return 1
        classification_task = classification_task_for(data, run_dir)
        task_for_contract = classification_task or structured_task_for(data, run_dir)
        if classification_task is not None:
            violations = validate_classification_evidence(data, run_dir)
        elif task_for_contract == "tse":
            violations = validate_tse_evidence(data, run_dir)
        else:
            violations = validate_output_contract(sample_output, contract)
        if violations:
            print(
                "VALIDATE_CONTRACT gate failed: sample_output does not satisfy io_contract "
                f"({sample_path}): " + "; ".join(violations),
                file=sys.stderr,
            )
            return 1
        structured_violations = (
            []
            if classification_task is not None
            else validate_structured_evidence(data, run_dir)
        )
        if structured_violations:
            print(
                "VALIDATE_CONTRACT gate failed: " + "; ".join(structured_violations),
                file=sys.stderr,
            )
            return 1
        if (
            classification_task is not None
            or is_structured_task(structured_task_for(data, run_dir))
            or structured_task_for(data, run_dir) == "tse"
        ):
            outputs_path = sample_outputs_path_for(data, run_dir)
            assert outputs_path is not None
            rows = read_jsonl_objects(outputs_path)
            data["sample_outputs_path"] = str(outputs_path)
            data["validated_sample_count"] = len(rows)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if data.get("io_contract_satisfied") is False:
            print(
                "VALIDATE_CONTRACT gate failed: io_contract_satisfied is explicitly false.",
                file=sys.stderr,
            )
            return 1

    print(
        f"run_validate OK: kind={args.kind}, {pass_key}=true, "
        f"executed=true, duration_seconds={elapsed:.3f}, log_path={log_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
