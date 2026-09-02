#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from vc_exec import (
    DEFAULT_CPUS,
    DEFAULT_GPUS,
    DEFAULT_MEMORY_GB,
    default_partition,
    diagnose_oom,
    docker_run_to_vc,
    ensure_registry_image,
    recorded_push_digest,
    registry_image,
    run_vc_job,
)

GIB = 1024 ** 3
RAM_SAFETY_FACTOR = 2
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
SPEAKER_OUTPUT_FIELDS = frozenset({"segments", "num_speakers"})
SD_SEGMENT_FIELDS = frozenset({"speaker", "start", "end", "duration"})
SA_ASR_SEGMENT_FIELDS = frozenset({*SD_SEGMENT_FIELDS, "text"})
URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


PASS_KEYS = {
    "original_inference": "inference_passed",
    "import": "import_passed",
    "load": "load_passed",
    "infer": "infer_passed",
    "contract": "contract_passed",
    "mcp": "mcp_passed",
    "equivalence": "equivalent",
}


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def command_for(value: object) -> tuple[list[str] | str, bool]:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value, False
    if isinstance(value, str) and value.strip():
        return value, True
    raise ValueError("validation artifact must contain a non-empty run_command")


def vc_resources(resolved: dict) -> tuple[str, int, int, int]:
    partition = str(resolved.get("vc_partition") or default_partition())
    gpus = int(resolved.get("vc_gpus") or DEFAULT_GPUS)
    memory_gb = int(resolved.get("vc_memory_gb") or DEFAULT_MEMORY_GB)
    return partition, gpus, memory_gb, DEFAULT_CPUS


def model_payload_bytes(run_dir: Path, resolved: dict, kind: str) -> int:
    """Return the model payload size the job will load into RAM.

    ``import`` only imports the wrapper module and never loads weights.
    Prefer the staged manifest total; fall back to a metadata-only walk of
    the supplied model path.
    """
    if kind == "import":
        return 0
    manifest = run_dir / "artifacts" / "model_payload_manifest.json"
    if manifest.is_file():
        try:
            total = read_object(manifest).get("total_bytes")
            if isinstance(total, int):
                return total
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    model_path = str(resolved.get("model_path") or "")
    path = Path(model_path).expanduser()
    if not model_path or not path.is_dir():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def prepare_container_outputs(spec: object, run_dir: Path) -> None:
    """Clear the directory the container writes its stage output into.

    The target is the mount SURE_VALIDATE_ARTIFACTS_DIR names, which is what
    SKILL.md prescribes and what container_stage_error already reads; guessing
    at /output and /artifacts instead cleared nothing on the documented path and
    would have emptied the run's own artifacts directory on an undocumented one.

    The mounts come from the run_command the agent under test wrote into its
    own <stage>_result.json, so a host path here is untrusted input. Refuse
    anything that resolves outside the run directory, and refuse the run's own
    artifacts directory, which holds gate products rather than container output.
    """
    root = run_dir.resolve()
    target = str((getattr(spec, "env", None) or {}).get("SURE_VALIDATE_ARTIFACTS_DIR", "") or "")
    if not target:
        return
    mounts = getattr(spec, "mounts", ())
    for mount in mounts:
        parts = str(mount).split(":")
        if len(parts) < 2 or parts[1] != target:
            continue
        output_dir = Path(parts[0]).expanduser()
        if not output_dir.is_absolute():
            raise ValueError(f"validation output mount host path must be absolute: {mount!r}")
        resolved = output_dir.resolve()
        if root not in resolved.parents:
            raise ValueError(
                f"validation output mount must stay inside the run directory: {mount!r}"
            )
        if resolved == root / "artifacts":
            raise ValueError(
                f"validation output mount must not target the run artifacts directory: {mount!r}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in output_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


MCP_STEPS = ("initialize", "tools_list", "tools_call")
GPU_OOM_MAX_ATTEMPTS = 8
# Leave the gate room to collect diagnostics and write the stage result after
# the last attempt it is allowed to start.
GATE_BUDGET_RESERVE_SECONDS = 120.0


def gate_budget_seconds() -> float:
    """Wall clock the hook allows this script before it kills it.

    checkpoints.ts spawns the gate with a fixed timeout and exports it here, so
    the two stay in step. Eight OOM retries at the default per-stage timeout run
    four times past that budget; a retry started with no room to finish is
    killed mid-flight, which loses the stage result, orphans the vc job and
    spends a state-machine retry on work that was still running.
    """
    try:
        return max(0.0, float(os.environ.get("SURE_TRANS_GATE_BUDGET_SECONDS", "") or 0.0))
    except ValueError:
        return 0.0


def is_gpu_oom(result: object, stage_error: str = "") -> bool:
    """Decide whether a failed job died of a GPU out-of-memory condition.

    The exit code has to agree. A recovered OutOfMemoryError leaves its
    traceback in the log of a job that goes on to exit 0, and resubmitting that
    job wastes a GPU allocation and then judges the stage on a later attempt.
    stage_error carries what the container wrote to <stage>_result.json, which
    is the only place templates/validate.py records a caught exception, so a
    real OOM that never reached the job log still counts.
    """
    if getattr(result, "exit_code", None) in (0, None):
        return False
    evidence = "\n".join(
        [stage_error] + [
            str(getattr(result, field, "") or "")
            for field in ("stdout", "stderr", "vc_diagnostics")
        ]
    )
    return "cuda out of memory" in evidence.lower()


def validate_mcp_evidence(evidence_path: Path, tool_name: str) -> str | None:
    """Require deterministic mcp_smoke.py protocol evidence for the mcp gate.

    Returns None when the evidence proves initialize/tools/list/tools/call;
    otherwise a repair message. Placeholder run_commands never produce the
    evidence file, so they are rejected here.
    """
    if not evidence_path.is_file():
        return (
            "MCP smoke must be driven by scripts/mcp_smoke.py: run_command must execute "
            "`python /opt/sure_trans/mcp_smoke.py --audio <fixture> --tool <tool_name> "
            f"--produces {evidence_path}` so the gate can verify protocol evidence; "
            "placeholder commands are rejected."
        )
    protocol = read_object(evidence_path)
    if protocol.get("status") != "passed":
        return f"mcp_smoke evidence did not pass: {protocol.get('error') or protocol.get('status')}"
    for step in MCP_STEPS:
        entry = protocol.get(step)
        if not isinstance(entry, dict) or entry.get("ok") is not True:
            return f"mcp_smoke evidence must prove {step} passed"
    call = protocol.get("tools_call") or {}
    if call.get("output_nonempty") is not True and call.get("text_nonempty") is not True:
        return "mcp_smoke evidence must return a non-empty primary output from tools/call"
    smoke_tool = str(protocol.get("tool") or "")
    if tool_name and smoke_tool and smoke_tool != tool_name:
        return f"mcp_smoke tool {smoke_tool!r} does not match declared tool {tool_name!r}"
    if tool_name == "kws_predict":
        samples = call.get("samples")
        if not isinstance(samples, list) or not 2 <= len(samples) <= 5:
            return "KWS MCP smoke must record 2 to 5 keyed positive/negative samples"
        if call.get("expected_samples") != len(samples) or call.get("num_samples") != len(samples):
            return "KWS MCP smoke sample counts do not match"
        polarities: set[bool] = set()
        keys: set[str] = set()
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("ok") is not True:
                return "every KWS MCP smoke sample must pass"
            key = str(sample.get("key") or "")
            result = sample.get("result")
            if not key or key in keys or not isinstance(result, dict):
                return "KWS MCP smoke samples require unique keys and structured results"
            keys.add(key)
            detected = result.get("detected")
            if not isinstance(detected, bool):
                return "KWS MCP smoke detected values must be boolean"
            polarities.add(detected)
        if polarities != {False, True}:
            return "KWS MCP smoke must prove one positive detection and one negative rejection"
    if tool_name == "enhance_speech":
        samples = call.get("samples")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 5:
            return "SE MCP smoke must record 1 to 5 keyed noisy/clean samples"
        if call.get("expected_samples") != len(samples) or call.get("num_samples") != len(samples):
            return "SE MCP smoke sample counts do not match"
        keys: set[str] = set()
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("ok") is not True:
                return "every SE MCP smoke sample must pass"
            key = str(sample.get("key") or "")
            if not key or key in keys:
                return "SE MCP smoke samples require unique keys"
            keys.add(key)
            result = sample.get("result")
            if not isinstance(result, dict):
                return f"SE MCP smoke sample {key} must record generated audio evidence"
            audio_path = result.get("audio_path")
            audio_sha256 = result.get("audio_sha256")
            noisy_sha256 = sample.get("audio_sha256")
            if (
                not isinstance(audio_path, str)
                or not audio_path.startswith("outputs/")
                or not isinstance(audio_sha256, str)
                or len(audio_sha256) != 64
                or not isinstance(noisy_sha256, str)
                or len(noisy_sha256) != 64
            ):
                return f"SE MCP smoke sample {key} must preserve portable output identity"
            reference_sha256 = sample.get("reference_audio_sha256")
            if not isinstance(reference_sha256, str) or len(reference_sha256) != 64:
                return f"SE MCP smoke sample {key} clean reference hash is invalid"
    if tool_name in {"diarize", "transcribe_with_speakers"}:
        task = "sd" if tool_name == "diarize" else "sa_asr"
        samples = call.get("samples")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 5:
            return f"{task.upper()} MCP smoke must record 1 to 5 keyed samples"
        if call.get("expected_samples") != len(samples) or call.get("num_samples") != len(samples):
            return f"{task.upper()} MCP smoke sample counts do not match"
        keys: set[str] = set()
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("ok") is not True:
                return f"every {task.upper()} MCP smoke sample must pass"
            key = str(sample.get("key") or "")
            if not key or key in keys:
                return f"{task.upper()} MCP smoke samples require unique keys"
            if looks_like_absolute_path_or_uri(key):
                return f"{task.upper()} MCP smoke sample keys must be safe tokens"
            keys.add(key)
            result = sample.get("result")
            if not isinstance(result, dict) or "segments" not in result:
                return f"{task.upper()} MCP smoke sample {key} must record structured segments"
            try:
                forbidden = forbidden_speaker_output_fields(result)
                if forbidden:
                    return (
                        f"{task.upper()} MCP smoke sample {key} exposes reference or path fields: "
                        + ", ".join(forbidden)
                    )
                unknown_fields = sorted(
                    str(field) for field in result if field not in SPEAKER_OUTPUT_FIELDS
                )
                if unknown_fields:
                    return (
                        f"{task.upper()} MCP smoke sample {key} contains unapproved field(s): "
                        + ", ".join(unknown_fields)
                    )
                duration_sec = sample.get("audio_duration_sec")
                if (
                    isinstance(duration_sec, bool)
                    or not isinstance(duration_sec, (int, float))
                    or not math.isfinite(float(duration_sec))
                    or float(duration_sec) <= 0
                ):
                    return f"{task.upper()} MCP smoke sample {key} lacks WAV duration evidence"
                validate_speaker_segments(
                    result.get("segments"),
                    task=task,
                    label=f"{task.upper()} sample {key}",
                    empty_sd_allowed=(
                        task == "sd" and sample.get("audio_is_silence") is True
                    ),
                    duration_sec=float(duration_sec),
                )
                num_speakers = result.get("num_speakers")
                if num_speakers is not None:
                    observed = {
                        segment["speaker"].strip()
                        for segment in result["segments"]
                        if isinstance(segment, dict)
                        and isinstance(segment.get("speaker"), str)
                        and segment["speaker"].strip()
                    }
                    if (
                        isinstance(num_speakers, bool)
                        or not isinstance(num_speakers, int)
                        or num_speakers < 0
                        or num_speakers != len(observed)
                    ):
                        return (
                            f"{task.upper()} MCP smoke sample {key} num_speakers must match "
                            "distinct speakers"
                        )
                json.dumps(result, allow_nan=False)
            except (TypeError, ValueError) as error:
                return str(error)
    return None


EQUIVALENCE_POLICIES = ("exact", "normalized_whitespace")


def output_text(path: Path, primary_field: str) -> str:
    """Pull the comparable text out of a recorded inference output."""
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        field = value.get(primary_field)
        if isinstance(field, str):
            return field
        raise ValueError(
            f"{path} carries no string {primary_field!r} field to compare; write the adapter "
            f"io_contract primary field into both recorded outputs"
        )
    raise ValueError(f"{path} is neither a string nor an object holding {primary_field!r}")


def kws_equivalence_rows(path: Path) -> dict[str, dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a KWS rows array")
    normalized: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} rows[{index}] must be an object")
        key = str(row.get("key") or "")
        if not key or key in normalized:
            raise ValueError(f"{path} has a missing or duplicate KWS key: {key!r}")
        result = row.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"{path} KWS result for {key} must be an object")
        for field in ("detected", "keyword", "score"):
            if field not in result:
                raise ValueError(f"{path} KWS result for {key} is missing {field}")
        detected = result["detected"]
        keyword = result["keyword"]
        score = result["score"]
        if not isinstance(detected, bool):
            raise ValueError(f"{path} KWS detected for {key} must be boolean")
        if keyword is not None and (not isinstance(keyword, str) or not keyword.strip()):
            raise ValueError(f"{path} KWS keyword for {key} must be string or null")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"{path} KWS score for {key} must be finite or null")
        if score is not None and not 0 <= float(score) <= 1:
            raise ValueError(f"{path} KWS score for {key} must be within [0, 1]")
        if detected and (keyword is None or score is None or float(score) < KWS_OPERATING_THRESHOLD):
            raise ValueError(
                f"{path} KWS detection for {key} requires keyword and score >= {KWS_OPERATING_THRESHOLD}"
            )
        if not detected and (
            keyword is not None
            or (score is not None and float(score) >= KWS_OPERATING_THRESHOLD)
        ):
            raise ValueError(
                f"{path} KWS rejection for {key} requires null keyword and score below {KWS_OPERATING_THRESHOLD}"
            )
        normalized[key] = {
            "detected": detected,
            "keyword": "".join(keyword.upper().split()) if isinstance(keyword, str) else None,
            "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        }
    return normalized


def compare_kws_equivalence(
    baseline_path: Path,
    adapter_path: Path,
    policy: str,
) -> tuple[dict, str | None]:
    baseline = kws_equivalence_rows(baseline_path)
    adapter = kws_equivalence_rows(adapter_path)
    missing = sorted(set(baseline) - set(adapter))
    extra = sorted(set(adapter) - set(baseline))
    mismatches: dict[str, dict[str, object]] = {}
    for key in sorted(set(baseline) & set(adapter)):
        expected = baseline[key]
        actual = adapter[key]
        fields: list[str] = []
        if expected["detected"] is not actual["detected"]:
            fields.append("detected")
        if expected["keyword"] != actual["keyword"]:
            fields.append("keyword")
        expected_score = expected["score"]
        actual_score = actual["score"]
        scores_match = expected_score is None and actual_score is None
        if isinstance(expected_score, float) and isinstance(actual_score, float):
            scores_match = math.isclose(expected_score, actual_score, rel_tol=1e-6, abs_tol=1e-8)
        if not scores_match:
            fields.append("score")
        if fields:
            mismatches[key] = {
                "fields": fields,
                "baseline": expected,
                "adapter": actual,
            }
    match = not missing and not extra and not mismatches
    evidence = {
        "policy": policy,
        "comparison": "keyed_kws_detected_keyword_score",
        "primary_field": "detected",
        "baseline_rows": baseline,
        "adapter_rows": adapter,
        "missing_keys": missing,
        "extra_keys": extra,
        "mismatches": mismatches,
        "match": match,
    }
    if match:
        return evidence, None
    return evidence, (
        "baseline and adapter KWS outputs differ: "
        f"missing={missing}, extra={extra}, mismatched={sorted(mismatches)}"
    )


def validate_speaker_segments(
    value: object,
    *,
    task: str,
    label: str,
    empty_sd_allowed: bool = False,
    duration_sec: float | None = None,
) -> list[dict]:
    if task not in {"sd", "sa_asr"}:
        raise ValueError(f"unsupported speaker task: {task}")
    if not isinstance(value, list):
        raise ValueError(f"{label} segments must be an array")
    if task == "sa_asr" and not value:
        raise ValueError(f"{label} SA-ASR segments must not be empty")
    if task == "sd" and not value and not empty_sd_allowed:
        raise ValueError(f"{label} empty SD segments are allowed only for pure-silence audio")
    segments: list[dict] = []
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
            raise ValueError(f"{label} segments[{index}].end exceeds WAV duration {duration_sec:.6f}")
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
        segments.append(segment)
    return segments


def forbidden_speaker_output_fields(value: object, path: str = "result") -> list[str]:
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


def speaker_equivalence_rows(
    path: Path,
    *,
    task: str,
    audio_info_by_key: dict[str, dict[str, object]],
) -> dict[str, dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a {task.upper()} rows array")
    if not 1 <= len(rows) <= 5:
        raise ValueError(f"{path} must contain 1 to 5 {task.upper()} rows")
    normalized: dict[str, dict] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} rows[{index}] must be an object")
        key = str(row.get("key") or "")
        if not key or key in normalized:
            raise ValueError(f"{path} has a missing or duplicate {task.upper()} key: {key!r}")
        result = row.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"{path} {task.upper()} result for {key} must be an object")
        if "segments" not in result:
            raise ValueError(f"{path} {task.upper()} result for {key} is missing segments")
        forbidden = forbidden_speaker_output_fields(result)
        if forbidden:
            raise ValueError(
                f"{path} {task.upper()} result for {key} exposes reference or path fields: "
                + ", ".join(forbidden)
            )
        unknown_fields = sorted(str(field) for field in result if field not in SPEAKER_OUTPUT_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{path} {task.upper()} result for {key} contains unapproved field(s): "
                + ", ".join(unknown_fields)
            )
        info = audio_info_by_key.get(key)
        if not isinstance(info, dict):
            raise ValueError(f"{path} {task.upper()} result for {key} has no fixture WAV evidence")
        validate_speaker_segments(
            result.get("segments"),
            task=task,
            label=f"{path} {task.upper()} result for {key}",
            empty_sd_allowed=task == "sd" and bool(info.get("audio_is_silence")),
            duration_sec=float(info["duration_sec"]),
        )
        num_speakers = result.get("num_speakers")
        if num_speakers is not None:
            observed = {
                segment["speaker"].strip()
                for segment in result["segments"]
                if isinstance(segment, dict)
                and isinstance(segment.get("speaker"), str)
                and segment["speaker"].strip()
            }
            if (
                isinstance(num_speakers, bool)
                or not isinstance(num_speakers, int)
                or num_speakers < 0
                or num_speakers != len(observed)
            ):
                raise ValueError(
                    f"{path} {task.upper()} num_speakers for {key} must match distinct speakers"
                )
        try:
            json.dumps(result, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} {task.upper()} result for {key} is not strict JSON: {error}") from error
        normalized[key] = result
    return normalized


def compare_speaker_equivalence(
    baseline_path: Path,
    adapter_path: Path,
    policy: str,
    *,
    task: str,
    audio_info_by_key: dict[str, dict[str, object]],
) -> tuple[dict, str | None]:
    baseline = speaker_equivalence_rows(
        baseline_path,
        task=task,
        audio_info_by_key=audio_info_by_key,
    )
    adapter = speaker_equivalence_rows(
        adapter_path,
        task=task,
        audio_info_by_key=audio_info_by_key,
    )
    missing = sorted(set(baseline) - set(adapter))
    extra = sorted(set(adapter) - set(baseline))
    mismatches = {
        key: {"baseline": baseline[key], "adapter": adapter[key]}
        for key in sorted(set(baseline) & set(adapter))
        if json.dumps(baseline[key], allow_nan=False, sort_keys=True, separators=(",", ":"))
        != json.dumps(adapter[key], allow_nan=False, sort_keys=True, separators=(",", ":"))
    }
    match = not missing and not extra and not mismatches
    evidence = {
        "policy": policy,
        "comparison": f"keyed_{task}_full_structured_json",
        "primary_field": "segments",
        "baseline_rows": baseline,
        "adapter_rows": adapter,
        "missing_keys": missing,
        "extra_keys": extra,
        "mismatches": mismatches,
        "match": match,
    }
    if match:
        return evidence, None
    return evidence, (
        f"baseline and adapter {task.upper()} outputs differ: "
        f"missing={missing}, extra={extra}, mismatched={sorted(mismatches)}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_values(data: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [value - 128 for value in data]
    return [
        int.from_bytes(data[offset : offset + sample_width], "little", signed=True)
        for offset in range(0, len(data), sample_width)
    ]


def compare_audio_content(baseline_path: Path, adapter_path: Path) -> tuple[dict[str, object], bool]:
    hashes = {
        "baseline_sha256": file_sha256(baseline_path),
        "adapter_sha256": file_sha256(adapter_path),
    }
    try:
        with wave.open(str(baseline_path), "rb") as baseline, wave.open(
            str(adapter_path), "rb"
        ) as adapter:
            baseline_params = {
                "channels": baseline.getnchannels(),
                "sample_width_bytes": baseline.getsampwidth(),
                "sample_rate": baseline.getframerate(),
                "frames": baseline.getnframes(),
                "compression": baseline.getcomptype(),
            }
            adapter_params = {
                "channels": adapter.getnchannels(),
                "sample_width_bytes": adapter.getsampwidth(),
                "sample_rate": adapter.getframerate(),
                "frames": adapter.getnframes(),
                "compression": adapter.getcomptype(),
            }
            if (
                baseline_params["compression"] != "NONE"
                or adapter_params["compression"] != "NONE"
                or baseline_params["sample_width_bytes"] not in {1, 2, 3, 4}
                or adapter_params["sample_width_bytes"] not in {1, 2, 3, 4}
            ):
                raise wave.Error("not comparable integer PCM")
            parameters_match = baseline_params == adapter_params
            max_abs_error_lsb = 0
            if parameters_match:
                sample_width = int(baseline_params["sample_width_bytes"])
                while True:
                    baseline_chunk = baseline.readframes(4096)
                    adapter_chunk = adapter.readframes(4096)
                    if not baseline_chunk and not adapter_chunk:
                        break
                    baseline_values = pcm_values(baseline_chunk, sample_width)
                    adapter_values = pcm_values(adapter_chunk, sample_width)
                    if len(baseline_values) != len(adapter_values):
                        parameters_match = False
                        break
                    if baseline_values:
                        max_abs_error_lsb = max(
                            max_abs_error_lsb,
                            max(
                                abs(left - right)
                                for left, right in zip(baseline_values, adapter_values)
                            ),
                        )
            tolerance_lsb = 1
            match = parameters_match and max_abs_error_lsb <= tolerance_lsb
            return {
                "method": "integer_pcm_samples",
                "pcm_integer_lsb_tolerance": tolerance_lsb,
                "max_abs_error_lsb": max_abs_error_lsb,
                "baseline_params": baseline_params,
                "adapter_params": adapter_params,
                **hashes,
            }, match
    except (EOFError, OSError, wave.Error):
        match = hashes["baseline_sha256"] == hashes["adapter_sha256"]
        return {
            "method": "exact_content_sha256_fallback",
            "reason": "audio was not comparable integer PCM WAV",
            **hashes,
        }, match


def se_audio_roots(run_dir: Path, role: str) -> tuple[Path, ...]:
    if role == "baseline":
        return (
            run_dir / "original_output",
            run_dir / "artifacts" / "original_output",
        )
    if role == "adapter":
        return (
            run_dir / "artifacts" / "adapter_validation" / "outputs",
            run_dir / "artifacts" / "outputs",
        )
    raise ValueError(f"unknown SE equivalence role: {role}")


def resolve_recorded_audio(
    raw_path: object,
    document: Path,
    run_dir: Path,
    role: str,
) -> Path:
    raw = str(raw_path or "")
    if not raw:
        raise ValueError(f"{document} SE result is missing audio_path")
    declared = Path(raw)
    candidates: list[Path] = []
    if declared.is_absolute():
        if declared.parts[:2] == ("/", "validation"):
            candidates.append(
                run_dir / "artifacts" / "adapter_validation" / Path(*declared.parts[2:])
            )
        candidates.append(declared)
    else:
        candidates.extend(
            [
                document.parent / declared,
                run_dir / declared,
                *(root / declared for root in se_audio_roots(run_dir, role)),
            ]
        )
    run_root = run_dir.resolve()
    allowed_roots = se_audio_roots(run_root, role)
    for candidate in candidates:
        absolute = candidate.absolute()
        matching_root = next(
            (
                root
                for root in allowed_roots
                if absolute == root or absolute.is_relative_to(root)
            ),
            None,
        )
        if matching_root is None:
            continue
        relative_to_run = absolute.relative_to(run_root)
        current = run_root
        for part in relative_to_run.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"{document} SE audio_path must not traverse a parent symlink")
        if not absolute.is_file() or absolute.is_symlink() or absolute.stat().st_size <= 0:
            continue
        resolved = absolute.resolve()
        if not resolved.is_relative_to(matching_root.resolve()):
            raise ValueError(f"{document} SE audio_path escapes its controlled root")
        return resolved
    roots = ", ".join(root.relative_to(run_root).as_posix() for root in allowed_roots)
    raise ValueError(
        f"{document} SE {role} audio_path must be a real non-empty file under: {roots}"
    )


def se_equivalence_rows(path: Path, run_dir: Path, role: str) -> dict[str, Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain an SE rows array")
    normalized: dict[str, Path] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} rows[{index}] must be an object")
        key = str(row.get("sample_id") or row.get("key") or "")
        if not key or key in normalized:
            raise ValueError(f"{path} has a missing or duplicate SE key: {key!r}")
        result = row.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"{path} SE result for {key} must be an object")
        normalized[key] = resolve_recorded_audio(
            result.get("audio_path"),
            path,
            run_dir,
            role,
        )
    return normalized


def compare_se_equivalence(
    run_dir: Path,
    baseline_path: Path,
    adapter_path: Path,
    policy: str,
) -> tuple[dict, str | None]:
    baseline = se_equivalence_rows(baseline_path, run_dir, "baseline")
    adapter = se_equivalence_rows(adapter_path, run_dir, "adapter")
    missing = sorted(set(baseline) - set(adapter))
    extra = sorted(set(adapter) - set(baseline))
    comparisons: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []
    for key in sorted(set(baseline) & set(adapter)):
        if baseline[key].samefile(adapter[key]):
            comparison = {
                "method": "rejected_same_recorded_file",
                "reason": "baseline and adapter must be independent run-owned files",
            }
            row_matches = False
        else:
            comparison, row_matches = compare_audio_content(baseline[key], adapter[key])
        comparison["match"] = row_matches
        comparisons[key] = comparison
        if not row_matches:
            mismatches.append(key)
    match = not missing and not extra and not mismatches
    evidence = {
        "policy": policy,
        "comparison": "keyed_se_pcm_or_exact_content",
        "primary_field": "audio_path",
        "path_strings_compared": False,
        "missing_keys": missing,
        "extra_keys": extra,
        "mismatched_keys": mismatches,
        "rows": comparisons,
        "match": match,
    }
    if match:
        return evidence, None
    return evidence, (
        "baseline and adapter SE audio differ: "
        f"missing={missing}, extra={extra}, mismatched={mismatches}"
    )


def pcm_wav_info(path: Path) -> dict[str, object]:
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


def speaker_fixture_wav_info(run_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = run_dir / "artifacts" / "fixture_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = read_object(manifest_path)
    info_by_key: dict[str, dict[str, object]] = {}
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        return info_by_key
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        key = str(sample.get("key") or "")
        audio_path = Path(str(sample.get("audio_path") or ""))
        if key and audio_path.is_file():
            info_by_key[key] = pcm_wav_info(audio_path)
    return info_by_key


def controlled_speaker_output_document(path: Path, run_dir: Path, role: str) -> Path:
    run_root = run_dir.resolve()
    if role == "baseline":
        exact_files = {run_root / "artifacts" / "original_output.json"}
        roots = (
            run_root / "original_output",
            run_root / "artifacts" / "original_output",
        )
    elif role == "adapter":
        exact_files = {run_root / "artifacts" / "adapter_validation" / "sample_output.json"}
        roots = (run_root / "artifacts" / "adapter_validation",)
    else:
        raise ValueError(f"unknown structured equivalence role: {role}")
    absolute = path.expanduser().absolute()
    allowed = absolute in exact_files or any(absolute.is_relative_to(root) for root in roots)
    if not allowed:
        raise ValueError(f"{role}_output must stay under its run-owned structured output root")
    relative = absolute.relative_to(run_root)
    current = run_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{role}_output must not traverse a symlink")
    if not absolute.is_file() or absolute.stat().st_size <= 0:
        raise ValueError(f"{role}_output must be a real non-empty file")
    resolved = absolute.resolve()
    resolved_allowed = resolved in exact_files or any(
        resolved.is_relative_to(root.resolve()) for root in roots
    )
    if not resolved_allowed:
        raise ValueError(f"{role}_output escapes its run-owned structured output root")
    return resolved


def adapter_primary_field(run_dir: Path) -> str:
    manifest = run_dir / "artifacts" / "adapter_manifest.json"
    if not manifest.is_file():
        return "text"
    contract = read_object(manifest).get("io_contract")
    if isinstance(contract, dict) and isinstance(contract.get("primary_field"), str):
        return contract["primary_field"]
    return "text"


def compare_equivalence_outputs(run_dir: Path, data: dict) -> tuple[dict | None, str | None]:
    """Decide equivalence by reading both recorded outputs.

    The run_command only proves that something ran: a `/bin/true` command once
    carried this gate to passed while neither output file was ever opened. The
    comparison happens here so the verdict rests on the recorded evidence
    rather than on an exit code the agent chooses.
    """
    policy = str(data.get("comparison_policy") or "normalized_whitespace")
    if policy not in EQUIVALENCE_POLICIES:
        return None, (
            f"comparison_policy must be one of {list(EQUIVALENCE_POLICIES)}; got {policy!r}"
        )
    paths: dict[str, Path] = {}
    for key in ("baseline_output", "adapter_output"):
        raw = str(data.get(key) or "")
        path = Path(raw)
        if not raw or not path.is_file():
            return None, (
                f"{key} must be the path of the recorded output file, not the transcript itself; "
                f"got {raw!r}. Point it at the JSON or text file the run wrote."
            )
        paths[key] = path
    resolved_path = run_dir / "artifacts" / "trans_input_resolved.json"
    resolved = read_object(resolved_path) if resolved_path.is_file() else {}
    task_type = str(resolved.get("task_type") or "").lower().replace("-", "_")
    if not task_type:
        manifest_path = run_dir / "artifacts" / "adapter_manifest.json"
        manifest = read_object(manifest_path) if manifest_path.is_file() else {}
        contract = manifest.get("io_contract") if isinstance(manifest.get("io_contract"), dict) else {}
        if contract.get("output_type") == "keyword_detection":
            task_type = "kws"
    if task_type == "kws":
        try:
            return compare_kws_equivalence(
                paths["baseline_output"],
                paths["adapter_output"],
                policy,
            )
        except (json.JSONDecodeError, ValueError) as error:
            return None, str(error)
    if task_type == "se":
        try:
            return compare_se_equivalence(
                run_dir,
                paths["baseline_output"],
                paths["adapter_output"],
                policy,
            )
        except (json.JSONDecodeError, ValueError) as error:
            return None, str(error)
    if task_type in {"sd", "sa_asr"}:
        try:
            baseline_path = controlled_speaker_output_document(
                paths["baseline_output"], run_dir, "baseline"
            )
            adapter_path = controlled_speaker_output_document(
                paths["adapter_output"], run_dir, "adapter"
            )
            if baseline_path.samefile(adapter_path):
                raise ValueError(
                    "structured baseline_output and adapter_output must be independent files"
                )
            return compare_speaker_equivalence(
                baseline_path,
                adapter_path,
                policy,
                task=task_type,
                audio_info_by_key=speaker_fixture_wav_info(run_dir),
            )
        except (json.JSONDecodeError, ValueError) as error:
            return None, str(error)

    primary_field = adapter_primary_field(run_dir)
    texts: dict[str, str] = {}
    for key, path in paths.items():
        try:
            texts[key] = output_text(path, primary_field)
        except ValueError as error:
            return None, str(error)

    def normalized(text: str) -> str:
        return text if policy == "exact" else " ".join(text.split())

    baseline, adapter = texts["baseline_output"], texts["adapter_output"]
    match = normalized(baseline) == normalized(adapter)
    evidence = {
        "policy": policy,
        "primary_field": primary_field,
        "baseline_text": baseline,
        "adapter_text": adapter,
        "match": match,
    }
    if not match:
        return evidence, (
            f"baseline and adapter outputs differ under {policy}: {baseline!r} vs {adapter!r}"
        )
    return evidence, None


def ensure_validation_image(
    run_dir: Path, resolved: dict, kind: str, artifacts: Path
) -> tuple[str, str, str | None]:
    version = str(resolved.get("image_version") or "0.1.0")
    model_name = str(resolved.get("model_name") or "")
    task_type = str(resolved.get("task_type") or "asr")
    delivery = resolved.get("container_delivery")
    if kind == "original_inference":
        registry_ref = (
            str(delivery.get("source_image"))
            if isinstance(delivery, dict) and delivery.get("source_image")
            else registry_image(model_name, version, "source", task_type=task_type)
        )
        return registry_ref, "source_push.log", None
    registry_ref = (
        str(delivery.get("target_image"))
        if isinstance(delivery, dict) and delivery.get("target_image")
        else registry_image(model_name, version, task_type=task_type)
    )
    push_digest: str | None = None
    if kind == "import":
        adapter = read_object(artifacts / "adapter_image_result.json")
        local_image = str(adapter.get("image_id") or adapter.get("target_image") or "")
        if not local_image:
            raise ValueError("adapter image identity is missing")
        push_log = run_dir / "artifacts" / "vc_logs" / "adapter_push.log"
        push_digest = ensure_registry_image(
            local_image,
            registry_ref,
            push_log,
            known_digest=recorded_push_digest(adapter, registry_ref),
        ) or None
        adapter["registry_ref"] = registry_ref
        adapter["registry_push"] = {
            "log_path": str(push_log),
            "digest": push_digest,
            "pushed_at": datetime.now(timezone.utc).isoformat(),
        }
        (artifacts / "adapter_image_result.json").write_text(
            json.dumps(adapter, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return registry_ref, "adapter_push.log", push_digest


def container_stage_error(run_command: object, kind: str) -> str:
    """Reason the container recorded for a failed validation stage.

    templates/validate.py catches every stage exception, writes it to
    <artifacts>/<stage>_result.json in its mounted output directory, and prints
    nothing. The job log therefore carries no error at all, so diagnose_oom
    never saw the CUDA OOM it exists to catch and the gate sent the agent to a
    log with nothing wrong in it. Read the file the container actually wrote.
    """
    try:
        # Only the mounts and env matter here, and resolving the image
        # entrypoint would shell out to docker for a command nobody runs.
        spec = docker_run_to_vc(run_command, resolve_entrypoint=lambda _image: ((), ()))
        target = spec.env.get("SURE_VALIDATE_ARTIFACTS_DIR", "")
        for mount in spec.mounts:
            parts = mount.split(":")
            if len(parts) < 2 or parts[1] != target:
                continue
            result = Path(parts[0]) / f"{kind}_result.json"
            if result.is_file():
                return str(read_object(result).get("error") or "")
    except (OSError, TypeError, ValueError):
        return ""
    return ""


def run_vc_validation(
    run_dir: Path,
    resolved: dict,
    data: dict,
    kind: str,
    artifacts: Path,
    timeout: float,
) -> tuple[int, dict, str]:
    command, _ = command_for(data.get("run_command"))
    spec = docker_run_to_vc(command)
    prepare_container_outputs(spec, run_dir)
    registry_ref, _, _ = ensure_validation_image(run_dir, resolved, kind, artifacts)
    env: dict[str, str] = dict(spec.env)
    for key, value in (data.get("env") or {}).items():
        if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
            env[key] = str(value)
    env["SURE_DEVICE"] = "cuda"
    env["DEVICE"] = "cuda"
    partition, gpus, memory_gb, cpus = vc_resources(resolved)
    payload_bytes = model_payload_bytes(run_dir, resolved, kind)
    required_gb = math.ceil(payload_bytes * RAM_SAFETY_FACTOR / GIB)
    if required_gb > memory_gb:
        raise ValueError(
            f"model payload is {payload_bytes / GIB:.1f} GiB; with {RAM_SAFETY_FACTOR}x loading "
            f"headroom the job needs about {required_gb} GiB RAM but vc_memory_gb={memory_gb} "
            f"(the partition caps 32 GiB per GPU). Set vc_gpus=2 vc_memory_gb=64 in the slash "
            f"command or trans_input_resolved.json, then rerun the gate."
        )
    log_dir = run_dir / "artifacts" / "vc_logs" / kind
    attempts: list[dict[str, object]] = []
    result = None
    budget = gate_budget_seconds()
    budget_started = time.monotonic()
    budget_exhausted = False
    for attempt in range(1, GPU_OOM_MAX_ATTEMPTS + 1):
        if attempt > 1 and budget:
            spent = time.monotonic() - budget_started
            if budget - GATE_BUDGET_RESERVE_SECONDS - spent < timeout:
                budget_exhausted = True
                break
        prepare_container_outputs(spec, run_dir)
        attempt_log_dir = log_dir if attempt == 1 else log_dir / f"oom-attempt-{attempt}"
        result = run_vc_job(
            image=registry_ref,
            command=shlex.join(spec.command),
            log_dir=attempt_log_dir,
            mounts=spec.mounts,
            workdir=spec.workdir,
            env=env,
            partition=partition,
            gpus=gpus,
            memory_gb=memory_gb,
            cpus=cpus,
            job_name=f"sure-trans-{resolved.get('model_name')}-{kind}-attempt-{attempt}",
            timeout_seconds=timeout,
        )
        gpu_oom = is_gpu_oom(result, container_stage_error(data.get("run_command"), kind))
        attempts.append({
            "attempt": attempt,
            "job_id": result.job_id,
            "log_path": str(result.log_dir),
            "exit_code": result.exit_code,
            "gpu_oom": gpu_oom,
        })
        if not gpu_oom:
            break
    assert result is not None
    exit_code = -1 if result.exit_code is None else result.exit_code
    extra = {
        "execution_surface": "vc",
        "vc_job_id": result.job_id,
        "vc_partition": partition,
        "vc_memory_gb": memory_gb,
        "vc_gpus": gpus,
        "vc_submit_command": result.submit_command,
        "vc_log_path": str(result.log_dir),
        "vc_timed_out": result.timed_out,
        "registry_ref": registry_ref,
        "vc_diagnostics": result.vc_diagnostics[:4000],
        "vc_attempts": attempts,
        "gpu_oom_attempts": len(attempts) if attempts[-1]["gpu_oom"] else len(attempts) - 1,
        "gpu_oom_retry_exhausted": bool(attempts[-1]["gpu_oom"]),
        "gpu_oom_retry_budget_exhausted": budget_exhausted,
    }
    rendered = (
        f"$ {' '.join(result.submit_command)}\n$ {shlex.join(spec.command)}\n"
        f"{result.stdout}\n{result.stderr}\n"
    )
    return exit_code, extra, rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--produces", required=True)
    parser.add_argument("--kind", choices=tuple(PASS_KEYS), required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.produces)
    data = read_object(output)
    command, shell = command_for(data.get("run_command"))
    cwd = Path(str(data.get("cwd") or run_dir)).resolve()
    if not cwd.is_dir():
        raise ValueError(f"validation cwd does not exist: {cwd}")
    compat = read_object(run_dir / "artifacts" / "execution_compat.json")
    env = os.environ.copy()
    selected_device = str(compat.get("selected_device") or "")
    if selected_device:
        env["SURE_DEVICE"] = selected_device
        env["DEVICE"] = selected_device
    for key, value in (data.get("env") or {}).items():
        if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
            env[key] = str(value)
    timeout = float(data.get("timeout_seconds") or 1800)
    log_path = Path(str(data.get("log_path") or run_dir / "artifacts" / f"{args.kind}_execution.log"))
    if not log_path.is_absolute():
        log_path = run_dir / "artifacts" / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    extra: dict = {}
    if selected_device == "cuda":
        resolved = read_object(run_dir / "artifacts" / "trans_input_resolved.json")
        exit_code, extra, rendered = run_vc_validation(
            run_dir, resolved, data, args.kind, run_dir / "artifacts", timeout
        )
        log_path.write_text(rendered, encoding="utf-8")
        duration_ms = round((time.monotonic() - started) * 1000, 3)
    else:
        process = subprocess.run(command, shell=shell, cwd=cwd, env=env, check=False, capture_output=True, text=True, timeout=timeout)
        duration_ms = round((time.monotonic() - started) * 1000, 3)
        exit_code = process.returncode
        rendered = command if isinstance(command, str) else " ".join(command)
        log_path.write_text(f"$ {rendered}\n{process.stdout}\n{process.stderr}", encoding="utf-8")
    passed = exit_code == 0
    stage_error = "" if passed else container_stage_error(data.get("run_command"), args.kind)
    evidence = f"{rendered}\n{extra.get('vc_diagnostics', '')}\n{stage_error}"
    hint = "" if passed else (diagnose_oom(exit_code, evidence) or "")
    if not passed and not hint and "permission denied" in evidence.lower():
        hint = (
            "container hit Permission denied writing to a mounted path; the host mount "
            "source is likely owned by another uid. Recreate the empty output dir as your "
            "user (rm the dir and let the gate create it) or point the mount at a "
            "user-owned directory, then rerun the gate."
        )
    if not passed and exit_code == 124 and not hint:
        hint = (
            "container command hit its hard timeout (exit 124); reduce the workload or "
            "raise the command timeout, then rerun the gate."
        )
    if args.kind == "mcp" and selected_device == "cuda":
        evidence_path = run_dir / "artifacts" / "vc_logs" / "mcp" / "mcp_smoke.json"
        mcp_error = validate_mcp_evidence(evidence_path, str(data.get("tool_name") or ""))
        if mcp_error:
            passed = False
            hint = f"{hint} {mcp_error}".strip() if hint else mcp_error
        elif evidence_path.is_file():
            data["protocol"] = read_object(evidence_path)
    if args.kind == "equivalence":
        comparison, equivalence_error = compare_equivalence_outputs(run_dir, data)
        if comparison is not None:
            data["comparison_evidence"] = comparison
        if equivalence_error:
            passed = False
            hint = f"{hint} {equivalence_error}".strip() if hint else equivalence_error
    # A gate that rejects the evidence must not report it as a job failure: the
    # command exited 0 and pointing the agent at the job log sends it to the
    # wrong place.
    failure_text = (
        f"vc job failed or timed out; inspect {log_path}"
        if exit_code != 0
        else f"the command succeeded but the gate rejected its evidence; inspect {log_path}"
    )
    error_text = " ".join(part for part in (failure_text, stage_error, hint) if part)
    data.update({
        "status": "passed" if passed else "failed",
        PASS_KEYS[args.kind]: passed,
        "executed": True,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "log_path": str(log_path),
        "selected_device": selected_device or ("cuda" if "vc_job_id" in extra else "cpu"),
        "error": None if passed else error_text,
        **extra,
    })
    if args.kind == "original_inference":
        data["model_loaded"] = passed
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError(f"{args.kind} validation failed: {error_text}")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
