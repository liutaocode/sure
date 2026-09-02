#!/usr/bin/env python3
"""
Validate deterministic prediction files before formal evaluation.

Checks:
- dataset resolves to a canonical JSONL
- prediction file exists
- all expected keys are present
- optionally require non-empty predictions
- report missing / extra / duplicate keys in a machine-readable way
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sure_eval.core.config import Config
from sure_eval.core.logging import configure_logging, get_logger
from sure_eval.datasets import DatasetManager

configure_logging(level="INFO")
logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
KWS_POSITIVE_LABELS = {"detect", "detected", "positive", "true", "1", "yes"}
KWS_NEGATIVE_LABELS = {"reject", "rejected", "negative", "false", "0", "no"}
KWS_OPERATING_THRESHOLD = 0.5
ANNOTATION_TASKS = {"SD", "SA-ASR"}
STRUCTURED_REQUIRED_TASKS = {"KWS", "SE", "VAD", *ANNOTATION_TASKS}


def _normalize_task(value: Any) -> str:
    normalized = (
        str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    )
    if normalized in {"SPEECH_ACTIVITY_DETECTION", "VOICE_ACTIVITY_DETECTION"}:
        return "VAD"
    return "SA-ASR" if normalized == "SA_ASR" else normalized


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _load_predictions(path: Path) -> tuple[dict[str, str], list[str]]:
    predictions: dict[str, str] = {}
    duplicate_keys: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                key, prediction = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                key = parts[0]
                prediction = parts[1] if len(parts) > 1 else ""

            if key in predictions:
                duplicate_keys.append(key)
            predictions[key] = prediction

    return predictions, duplicate_keys


def _load_structured_predictions(path: Path) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    predictions: dict[str, dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    invalid_rows: list[str] = []
    if not path.exists():
        return predictions, duplicate_keys, invalid_rows
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                invalid_rows.append(f"line:{index}")
                continue
            if not isinstance(row, dict):
                invalid_rows.append(f"line:{index}:not_object")
                continue
            key = str(row.get("key", ""))
            if not key:
                invalid_rows.append(f"line:{index}:missing_key")
                continue
            if key in predictions:
                duplicate_keys.append(key)
            predictions[key] = row
    return predictions, duplicate_keys, invalid_rows


def _resolve_audio_path(value: Any, base_dir: Path | None) -> Path:
    path = Path(str(value))
    if path.is_absolute() and not path.exists():
        workspace_root = Path("/workspace/sure-eval")
        try:
            relative_to_workspace = path.relative_to(workspace_root)
        except ValueError:
            pass
        else:
            remapped = REPO_ROOT / relative_to_workspace
            if remapped.exists():
                path = remapped
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _valid_se_pcm_output(value: Any, base_dir: Path | None) -> bool:
    if base_dir is None or value in (None, ""):
        return False
    root = (base_dir / "audio").expanduser().absolute()
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).absolute()
    else:
        path = path.absolute()
    if root.is_symlink() or root.resolve() != root:
        return False
    if path.resolve() != path:
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getcomptype() == "NONE"
                and handle.getnchannels() >= 1
                and handle.getsampwidth() in {1, 2, 3, 4}
                and handle.getframerate() >= 1
                and handle.getnframes() >= 1
            )
    except (EOFError, OSError, wave.Error):
        return False


def _kws_reference_contract_violation(sample: dict[str, Any]) -> bool:
    labels: list[bool] = []
    for field in ("expected", "label", "expected_detected"):
        if field not in sample:
            continue
        value = sample[field]
        if isinstance(value, bool):
            labels.append(value)
        elif isinstance(value, int) and value in {0, 1}:
            labels.append(bool(value))
        elif isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in KWS_POSITIVE_LABELS:
                labels.append(True)
            elif normalized in KWS_NEGATIVE_LABELS:
                labels.append(False)
            else:
                return True
        else:
            return True
    if labels:
        return any(label != labels[0] for label in labels[1:])
    return not any(str(sample.get(field) or "").strip() for field in ("text", "txt"))


def _valid_annotation_segments(
    value: Any,
    *,
    task: str,
    duration: float | None = None,
) -> bool:
    canonical_task = _normalize_task(task)
    if not isinstance(value, list) or (canonical_task == "SA-ASR" and not value):
        return False
    seen: set[tuple[Any, ...]] = set()
    for segment in value:
        if not isinstance(segment, dict):
            return False
        allowed_fields = {"speaker", "start", "end", "duration"}
        if canonical_task == "SA-ASR":
            allowed_fields.add("text")
        if any(field not in allowed_fields for field in segment):
            return False
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
            return False
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
            return False
        if duration is not None and float(end) > duration + 0.001:
            return False
        text = segment.get("text")
        if canonical_task == "SA-ASR" and (
            not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text
        ):
            return False
        declared_duration = segment.get("duration")
        if declared_duration is not None and (
            isinstance(declared_duration, bool)
            or not isinstance(declared_duration, (int, float))
            or not math.isfinite(float(declared_duration))
            or float(declared_duration) <= 0
            or not math.isclose(
                float(declared_duration), float(end) - float(start), rel_tol=0, abs_tol=1e-3
            )
        ):
            return False
        identity = (
            speaker.strip(),
            float(start),
            float(end),
            str(text or "").strip(),
        )
        if identity in seen:
            return False
        seen.add(identity)
    return True


def _valid_vad_intervals(
    value: Any,
    *,
    duration: float,
    with_score: bool = False,
) -> bool:
    if not isinstance(value, list) or (with_score and not value):
        return False
    allowed = {"start", "end", "score"} if with_score else {"start", "end"}
    previous_end: float | None = None
    for item in value:
        if not isinstance(item, dict) or set(item) != allowed:
            return False
        start = item.get("start")
        end = item.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
        ):
            return False
        start_value = float(start)
        end_value = float(end)
        if (
            start_value < 0
            or end_value <= start_value
            or end_value > duration + 1e-6
            or (previous_end is not None and start_value < previous_end - 1e-12)
        ):
            return False
        if with_score and (
            (previous_end is None and start_value > 1e-6)
            or (previous_end is not None and abs(start_value - previous_end) > 1e-6)
        ):
            return False
        if with_score:
            score = item.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                return False
        previous_end = end_value
    return not with_score or abs((previous_end or 0.0) - duration) <= 1e-6


def _vad_sample_duration(sample: dict[str, Any]) -> float | None:
    values: list[float] = []
    for field in ("duration_sec", "duration_seconds", "duration"):
        if field not in sample:
            continue
        value = sample[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            return None
        values.append(float(value))
    if not values or any(
        not math.isclose(value, values[0], rel_tol=0, abs_tol=1e-6)
        for value in values[1:]
    ):
        return None
    return values[0]


def _task_contract_violations(
    samples: list[dict[str, Any]],
    structured: dict[str, dict[str, Any]],
    *,
    base_dir: Path | None = None,
    kws_require_score: bool = False,
) -> list[str]:
    violations: list[str] = []
    sample_by_key = {str(sample.get("key", "")): sample for sample in samples}
    for key, row in structured.items():
        sample = sample_by_key.get(key)
        if not isinstance(sample, dict):
            violations.append(key)
            continue
        sample_task = _normalize_task(sample.get("task"))
        row_task = _normalize_task(row.get("task"))
        if not sample_task or row_task != sample_task:
            violations.append(key)
            continue
        task = sample_task
        prediction = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        normalized = str(row.get("normalized_prediction") or "")
        if task in {"ASR", "S2TT"} and not (prediction.get("text") or normalized):
            violations.append(key)
        elif task in {"TTS", "VC"}:
            audio_path = prediction.get("audio_path") or normalized
            if not audio_path or not _resolve_audio_path(audio_path, base_dir).exists():
                violations.append(key)
        elif task == "SE":
            audio_path = prediction.get("audio_path")
            enhanced_audio = prediction.get("enhanced_audio")
            if not audio_path or not enhanced_audio or str(audio_path) != str(enhanced_audio):
                violations.append(key)
                continue
            if not _valid_se_pcm_output(audio_path, base_dir):
                violations.append(key)
        elif task in {"SER", "GR"} and not (prediction.get("label") or normalized):
            violations.append(key)
        elif task == "SLU" and not (prediction.get("text") or prediction.get("label") or normalized):
            violations.append(key)
        elif task in ANNOTATION_TASKS:
            if any(field not in {"segments", "num_speakers"} for field in prediction):
                violations.append(key)
                continue
            duration_field = next(
                (field for field in ("duration_sec", "duration_seconds", "duration") if field in sample),
                None,
            )
            raw_duration = sample.get(duration_field) if duration_field else None
            duration_valid = duration_field is None or (
                isinstance(raw_duration, (int, float))
                and not isinstance(raw_duration, bool)
                and math.isfinite(float(raw_duration))
                and float(raw_duration) > 0
            )
            if not duration_valid:
                violations.append(key)
                continue
            duration = float(raw_duration) if duration_field else None
            reference_segments = sample.get("segments")
            prediction_segments = prediction.get("segments")
            if not _valid_annotation_segments(
                reference_segments, task=task, duration=duration
            ) or not _valid_annotation_segments(
                prediction_segments, task=task, duration=duration
            ):
                violations.append(key)
                continue
            num_speakers = prediction.get("num_speakers")
            if num_speakers is not None:
                speakers = {
                    segment["speaker"].strip()
                    for segment in prediction_segments
                    if isinstance(segment, dict) and isinstance(segment.get("speaker"), str)
                }
                if (
                    isinstance(num_speakers, bool)
                    or not isinstance(num_speakers, int)
                    or num_speakers < 0
                    or num_speakers != len(speakers)
                ):
                    violations.append(key)
        elif task == "VAD":
            if any(field not in {"speech_segments", "frame_scores"} for field in prediction):
                violations.append(key)
                continue
            duration = _vad_sample_duration(sample)
            if duration is None:
                violations.append(key)
                continue
            if not _valid_vad_intervals(
                sample.get("speech_segments"), duration=duration
            ) or not _valid_vad_intervals(
                prediction.get("speech_segments"), duration=duration
            ):
                violations.append(key)
                continue
            if "frame_scores" in prediction and not _valid_vad_intervals(
                prediction["frame_scores"],
                duration=duration,
                with_score=True,
            ):
                violations.append(key)
        elif task == "KWS":
            if _kws_reference_contract_violation(sample) or not {
                "detected",
                "keyword",
                "score",
            }.issubset(prediction):
                violations.append(key)
                continue
            detected = prediction["detected"]
            keyword = prediction["keyword"]
            score = prediction["score"]
            events = prediction.get("events")
            if (
                not isinstance(detected, bool)
                or (keyword is not None and not isinstance(keyword, str))
                or (
                    score is not None
                    and (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                    )
                )
                or ("events" in prediction and not isinstance(events, list))
                or (detected and (not isinstance(keyword, str) or not keyword.strip()))
                or (detected and score is None)
                or (not detected and keyword is not None)
                or (score is not None and not 0.0 <= float(score) <= 1.0)
                or (
                    detected
                    and score is not None
                    and float(score) < KWS_OPERATING_THRESHOLD
                )
                or (
                    not detected
                    and score is not None
                    and float(score) >= KWS_OPERATING_THRESHOLD
                )
                or (kws_require_score and score is None)
            ):
                violations.append(key)
    return sorted(set(violations))


def _structured_projection_mismatches(
    predictions: dict[str, str],
    structured: dict[str, dict[str, Any]],
    expected_keys: set[str],
) -> list[str]:
    mismatches: list[str] = []
    for key in sorted(expected_keys & set(predictions) & set(structured)):
        normalized = str(structured[key].get("normalized_prediction") or "")
        if predictions[key] != normalized:
            mismatches.append(key)
    return mismatches


def validate_prediction_file(
    dataset_manager: DatasetManager,
    dataset_name: str,
    prediction_path: Path,
    require_nonempty: bool,
    max_samples: int = 0,
    kws_require_score: bool = False,
) -> dict[str, Any]:
    canonical_name = dataset_manager.normalize_dataset_name(dataset_name)
    jsonl_path = dataset_manager.get_jsonl_path(canonical_name)
    if not jsonl_path.exists():
        jsonl_path = dataset_manager.download_and_convert(canonical_name)

    all_samples = _load_jsonl(jsonl_path)
    samples = all_samples[:max_samples] if max_samples > 0 else all_samples
    predictions, duplicate_keys = _load_predictions(prediction_path)
    structured_path = prediction_path.with_suffix(".jsonl")
    structured_predictions, structured_duplicate_keys, invalid_structured_rows = _load_structured_predictions(structured_path)
    structured_required = any(
        _normalize_task(sample.get("task")) in STRUCTURED_REQUIRED_TASKS
        for sample in samples
    )

    expected_keys = [str(sample.get("key", "")) for sample in samples]
    expected_key_set = set(expected_keys)
    prediction_key_set = set(predictions.keys())

    missing_keys = sorted(expected_key_set - prediction_key_set)
    extra_keys = sorted(prediction_key_set - expected_key_set)
    empty_prediction_keys = sorted(
        key for key, prediction in predictions.items() if key in expected_key_set and prediction.strip() == ""
    )
    structured_key_set = set(structured_predictions.keys())
    structured_missing_keys = (
        sorted(expected_key_set - structured_key_set)
        if structured_path.exists() or structured_required
        else []
    )
    structured_extra_keys = sorted(structured_key_set - expected_key_set) if structured_path.exists() else []
    contract_violation_keys = (
        _task_contract_violations(
            samples,
            structured_predictions,
            base_dir=structured_path.parent,
            kws_require_score=kws_require_score,
        )
        if structured_path.exists()
        else []
    )
    structured_projection_mismatch_keys = (
        _structured_projection_mismatches(predictions, structured_predictions, expected_key_set)
        if structured_path.exists()
        else []
    )

    is_valid = not missing_keys and not extra_keys and not duplicate_keys
    if require_nonempty and empty_prediction_keys:
        is_valid = False
    if (structured_path.exists() or structured_required) and (
        structured_missing_keys
        or structured_extra_keys
        or structured_duplicate_keys
        or invalid_structured_rows
        or contract_violation_keys
        or structured_projection_mismatch_keys
    ):
        is_valid = False

    return {
        "dataset": canonical_name,
        "jsonl_path": str(jsonl_path),
        "prediction_path": str(prediction_path),
        "prediction_jsonl_path": str(structured_path) if structured_path.exists() else None,
        "format_used": "jsonl+txt" if structured_path.exists() else "txt",
        "structured_required": structured_required,
        "total_dataset_samples": len(all_samples),
        "max_samples": max_samples if max_samples > 0 else None,
        "expected_samples": len(expected_keys),
        "provided_predictions": len(predictions),
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "duplicate_keys": sorted(set(duplicate_keys)),
        "empty_prediction_keys": empty_prediction_keys,
        "structured_missing_keys": structured_missing_keys,
        "structured_extra_keys": structured_extra_keys,
        "structured_duplicate_keys": sorted(set(structured_duplicate_keys)),
        "invalid_structured_rows": invalid_structured_rows,
        "structured_projection_mismatch_keys": structured_projection_mismatch_keys,
        "contract_violation_keys": contract_violation_keys,
        "require_nonempty": require_nonempty,
        "is_valid": is_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate deterministic prediction files")
    parser.add_argument("--dataset", nargs="+", required=True, help="Dataset names to validate")
    parser.add_argument("--pred-dir", type=str, help="Directory containing <dataset>.txt prediction files")
    parser.add_argument("--pred", action="append", nargs=2, metavar=("DATASET", "FILE"), help="Explicit dataset-to-prediction mapping")
    parser.add_argument("--config", type=str, help="Config path")
    parser.add_argument("--require-nonempty", action="store_true", help="Fail when any expected prediction is empty")
    parser.add_argument("--max-samples", type=int, default=0, help="Validate only the first N dataset samples when running a bounded evaluation")
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config.from_env()
    dataset_manager = DatasetManager(cfg)

    explicit_preds = {dataset_manager.normalize_dataset_name(name): Path(path) for name, path in (args.pred or [])}
    pred_dir = Path(args.pred_dir) if args.pred_dir else None

    results: list[dict[str, Any]] = []
    overall_valid = True

    for requested_dataset in dataset_manager.expand_dataset_names(args.dataset):
        canonical_name = dataset_manager.normalize_dataset_name(requested_dataset)
        prediction_path = explicit_preds.get(canonical_name)
        if prediction_path is None:
            if pred_dir is None:
                raise ValueError(f"No prediction file provided for dataset: {canonical_name}")
            prediction_path = pred_dir / f"{canonical_name}.txt"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

        result = validate_prediction_file(
            dataset_manager=dataset_manager,
            dataset_name=canonical_name,
            prediction_path=prediction_path,
            require_nonempty=args.require_nonempty,
            max_samples=max(0, args.max_samples),
        )
        overall_valid = overall_valid and result["is_valid"]
        results.append(result)

    payload = {"is_valid": overall_valid, "results": results}
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    print(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        logger.info("Wrote validation payload", path=str(output_path))

    return 0 if overall_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
