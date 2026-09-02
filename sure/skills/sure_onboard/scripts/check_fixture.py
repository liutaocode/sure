#!/usr/bin/env python3
"""Gate script for PREPARE_FIXTURE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from structured_segments import (
    STRUCTURED_MANIFEST_SAMPLE_FIELDS,
    canonical_task,
    is_structured_task,
    pcm_wav_info,
    reference_segments_field,
    validate_segments,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_relative_child(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def fail(message: str) -> int:
    print(f"CHECK_FIXTURE failed: {message}", file=sys.stderr)
    return 1


def annotation_is_nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def vad_hard_link_error(path: Path, label: str) -> str | None:
    if path.is_symlink() or not path.is_file():
        return f"VAD {label} must be a regular file: {path}"
    if path.stat().st_nlink != 1:
        return f"VAD {label} must not be hard-linked: {path}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--produces", required=True)
    args = parser.parse_args()

    path = Path(args.produces)
    if not path.exists():
        return fail(f"fixture_manifest.json not found at {path}")
    try:
        data = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return fail(f"fixture_manifest.json is not valid JSON: {exc}")

    if not isinstance(data, dict):
        return fail("fixture_manifest.json must be a JSON object")
    required = ["model_dir", "task_type", "staged_dir", "gt_jsonl", "samples", "sample_count"]
    missing = [field for field in required if field not in data]
    if missing:
        return fail("missing required fields: " + ", ".join(missing))

    model_dir = Path(str(data["model_dir"])).resolve()
    task = canonical_task(str(data["task_type"]))
    raw_staged_dir = Path(str(data["staged_dir"])).expanduser()
    raw_gt_jsonl = Path(str(data["gt_jsonl"])).expanduser()
    if is_structured_task(task) and raw_staged_dir.is_symlink():
        return fail("structured staged fixture directory must not be a symlink")
    if is_structured_task(task) and raw_gt_jsonl.is_symlink():
        return fail("structured staged gt.jsonl must not be a symlink")
    staged_dir = raw_staged_dir.resolve()
    gt_jsonl = raw_gt_jsonl.resolve()

    if not model_dir.exists():
        return fail(f"model_dir does not exist: {model_dir}")
    expected_root = model_dir / "fixture"
    if not is_relative_child(staged_dir, expected_root):
        return fail(f"staged_dir must be under model_dir/fixture: {staged_dir}")
    expected_task_root = expected_root / task
    if not is_relative_child(staged_dir, expected_task_root):
        return fail(f"staged_dir must be under model_dir/fixture/{task}: {staged_dir}")
    if not staged_dir.exists() or not staged_dir.is_dir():
        return fail(f"staged_dir does not exist or is not a directory: {staged_dir}")
    if is_structured_task(task):
        symlinks = [entry for entry in staged_dir.rglob("*") if entry.is_symlink()]
        if symlinks:
            return fail(f"structured staged fixture tree must not contain symlinks: {symlinks[0]}")
    if not gt_jsonl.exists():
        return fail(f"gt_jsonl does not exist: {gt_jsonl}")
    if gt_jsonl.parent.resolve() != staged_dir.resolve():
        return fail("gt_jsonl must be located directly inside staged_dir")
    if task == "vad" and (error := vad_hard_link_error(gt_jsonl, "gt.jsonl")):
        return fail(error)

    samples = data.get("samples")
    if not isinstance(samples, list):
        return fail("samples must be an array")
    if data.get("sample_count") != len(samples):
        return fail("sample_count must match len(samples)")
    if not (1 <= len(samples) <= 5):
        return fail("sample_count must be between 1 and 5")
    parsed_rows = []
    allowed_structured_files = {gt_jsonl} if is_structured_task(task) else set()
    seen_keys: set[str] = set()
    for line_no, line in enumerate(gt_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            return fail(f"{gt_jsonl}:{line_no} is not valid JSON: {exc}")
        if not isinstance(row, dict):
            return fail(f"{gt_jsonl}:{line_no} must be a JSON object")
        audio = (
            row.get("audio") or row.get("wav")
            if is_structured_task(task)
            else row.get("audio")
            or row.get("wav")
            or row.get("prompt_audio")
            or row.get("reference_audio")
        )
        if not isinstance(audio, str) or not audio:
            return fail(f"{gt_jsonl}:{line_no} must contain a non-empty audio/wav field")
        audio_path = Path(audio)
        if audio_path.is_absolute() or ".." in audio_path.parts:
            return fail(f"{gt_jsonl}:{line_no} audio path must be relative and stay inside staged_dir")
        resolved_audio = (gt_jsonl.parent / audio_path).resolve()
        if not is_relative_child(resolved_audio, staged_dir):
            return fail(f"{gt_jsonl}:{line_no} audio path escapes staged_dir: {audio}")
        if not resolved_audio.is_file():
            return fail(f"{gt_jsonl}:{line_no} referenced audio does not exist: {audio}")
        if task == "vad" and (error := vad_hard_link_error(resolved_audio, f"audio {audio!r}")):
            return fail(error)
        if is_structured_task(task):
            allowed_structured_files.add(resolved_audio)
        key = str(row.get("key") or row.get("id") or audio_path.stem).strip()
        if not key:
            return fail(f"{gt_jsonl}:{line_no} requires a non-empty key")
        if key in seen_keys:
            return fail(f"{gt_jsonl}:{line_no} duplicates fixture key {key!r}")
        seen_keys.add(key)
        reference_audio = row.get("reference_audio")
        if task == "se":
            if not isinstance(reference_audio, str) or not reference_audio:
                return fail(f"{gt_jsonl}:{line_no} task se requires reference_audio")
            reference_path = Path(reference_audio)
            if reference_path.is_absolute() or ".." in reference_path.parts:
                return fail(
                    f"{gt_jsonl}:{line_no} reference_audio path must be relative and stay inside staged_dir"
                )
            resolved_reference = (gt_jsonl.parent / reference_path).resolve()
            if not is_relative_child(resolved_reference, staged_dir):
                return fail(
                    f"{gt_jsonl}:{line_no} reference_audio path escapes staged_dir: {reference_audio}"
                )
            if not resolved_reference.is_file():
                return fail(
                    f"{gt_jsonl}:{line_no} referenced reference_audio does not exist: {reference_audio}"
                )
            if resolved_reference.samefile(resolved_audio):
                return fail(
                    f"{gt_jsonl}:{line_no} task se audio and reference_audio must be independent files"
                )
            if len(samples) <= len(parsed_rows):
                return fail("samples array has fewer entries than gt.jsonl")
            manifest_sample = samples[len(parsed_rows)]
            if not isinstance(manifest_sample, dict):
                return fail(f"samples[{len(parsed_rows)}] must be an object")
            if manifest_sample.get("reference_audio") != reference_audio:
                return fail("SE manifest samples must preserve reference_audio")
            if Path(str(manifest_sample.get("reference_audio_path") or "")).resolve() != resolved_reference:
                return fail("SE manifest samples must preserve resolved reference_audio_path")
        if task == "sa_asr" and "segments" not in row:
            return fail(f"{gt_jsonl}:{line_no} task sa_asr requires speaker-attributed segments")
        if task == "sd" and "segments" not in row:
            return fail(f"{gt_jsonl}:{line_no} task sd requires speaker segments")
        if task == "vad" and "speech_segments" not in row:
            return fail(f"{gt_jsonl}:{line_no} task vad requires speech_segments")
        if is_structured_task(task) and row.get("task") is not None and canonical_task(row["task"]) != task:
            return fail(f"{gt_jsonl}:{line_no} declares task {row['task']!r}, expected {task!r}")

        wav_info: dict[str, Any] | None = None
        if is_structured_task(task):
            try:
                wav_info = pcm_wav_info(resolved_audio)
            except ValueError as exc:
                return fail(f"{gt_jsonl}:{line_no} {exc}")
            segment_violations = validate_segments(
                row.get(reference_segments_field(task)),
                task=task,
                duration_sec=float(wav_info["duration_sec"]),
                audio_is_silence=bool(wav_info["audio_is_silence"]),
            )
            if segment_violations:
                return fail(
                    f"{gt_jsonl}:{line_no} invalid {task} reference segments: "
                    + "; ".join(segment_violations)
                )
            declared_duration = row.get("duration_sec")
            if declared_duration is not None and (
                isinstance(declared_duration, bool)
                or not isinstance(declared_duration, (int, float))
                or not math.isfinite(float(declared_duration))
                or not math.isclose(
                    float(declared_duration),
                    float(wav_info["duration_sec"]),
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                return fail(
                    f"{gt_jsonl}:{line_no} duration_sec disagrees with the PCM WAV"
                )
            if row.get("sample_rate") is not None and row["sample_rate"] != wav_info["sample_rate"]:
                return fail(
                    f"{gt_jsonl}:{line_no} sample_rate disagrees with the PCM WAV"
                )

        annotation_fields = [
            field
            for field in (
                "ground_truth",
                "target_text",
                "text",
                "segments",
                "speech_segments",
                "label",
                "intent",
            )
            if field in row
            and (
                is_structured_task(task) and field == reference_segments_field(task)
                or annotation_is_nonempty(row[field])
            )
        ]
        if task == "se" and annotation_is_nonempty(reference_audio):
            annotation_fields.append("reference_audio")
        if not annotation_fields:
            return fail(
                f"{gt_jsonl}:{line_no} must contain at least one annotation field "
                "(ground_truth, target_text, text, segments, speech_segments, label, intent, or reference_audio)"
            )
        if is_structured_task(task):
            if len(samples) <= len(parsed_rows):
                return fail("samples array has fewer entries than gt.jsonl")
            manifest_sample = samples[len(parsed_rows)]
            if not isinstance(manifest_sample, dict):
                return fail(f"samples[{len(parsed_rows)}] must be an object")
            unexpected = sorted(
                str(field)
                for field in manifest_sample
                if field not in STRUCTURED_MANIFEST_SAMPLE_FIELDS
            )
            if unexpected:
                return fail(
                    "structured manifest sample exposes unapproved field(s): "
                    + ", ".join(unexpected)
                )
            if manifest_sample.get("key") != key:
                return fail("structured manifest samples must preserve fixture key order")
            if task == "vad":
                if manifest_sample.get("audio") != audio:
                    return fail("VAD manifest sample audio must equal its gt.jsonl row audio")
                manifest_audio_path = manifest_sample.get("audio_path")
                if (
                    not isinstance(manifest_audio_path, str)
                    or not Path(manifest_audio_path).is_absolute()
                    or Path(manifest_audio_path).resolve() != resolved_audio
                ):
                    return fail(
                        "VAD manifest sample audio_path must resolve to staged_dir/row.audio"
                    )
            expected_annotation_fields = [reference_segments_field(task)]
            if manifest_sample.get("annotation_fields") != expected_annotation_fields:
                return fail(
                    "structured manifest annotation_fields must equal "
                    f"{expected_annotation_fields!r}"
                )
            assert wav_info is not None
            for field in ("duration_sec", "sample_rate", "audio_is_silence"):
                if manifest_sample.get(field) != wav_info[field]:
                    return fail(f"structured manifest samples must preserve actual WAV {field}")
        parsed_rows.append(row)

    if len(parsed_rows) != len(samples):
        return fail("samples array must mirror non-empty gt.jsonl rows")
    if is_structured_task(task):
        staged_files = {entry.resolve() for entry in staged_dir.rglob("*") if entry.is_file()}
        extras = sorted(staged_files - allowed_structured_files)
        if extras:
            return fail(f"structured staged fixture contains an unreferenced sidecar: {extras[0]}")
        if data.get("validation_protocol_env") != "SURE_VALIDATE_PROTOCOL_JSON":
            return fail("structured fixture manifest must declare SURE_VALIDATE_PROTOCOL_JSON")

    discoverable = list((model_dir / "fixture").glob("**/gt.jsonl"))
    if not discoverable:
        return fail("validate.py cannot discover any model_dir/fixture/**/gt.jsonl")

    print(f"check_fixture OK: {len(samples)} sample(s), staged_dir={staged_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
