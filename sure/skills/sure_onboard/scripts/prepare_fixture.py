#!/usr/bin/env python3
"""Stage task fixtures into the model-local fixture directory.

This helper is intentionally narrow: it chooses a fixture source from
spec_validation/model_input evidence, copies it under
sure/models/<model>/fixture/<task>/<fixture_name>/, and writes
fixture_manifest.json for the PREPARE_FIXTURE gate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from structured_segments import (
    canonical_task,
    is_structured_task,
    pcm_wav_info,
    reference_segments_field,
    validate_segments,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def infer_repo_root(model_dir: Path) -> Path:
    parts = model_dir.resolve().parts
    for idx in range(len(parts) - 2):
        if parts[idx] == "sure" and parts[idx + 1] == "models":
            return Path(*parts[:idx])
    return Path(__file__).resolve().parents[4]


def candidate_from_spec(run_dir: Path, repo_root: Path, task: str) -> Path | None:
    spec_path = run_dir / "artifacts" / "spec_validation.json"
    if not spec_path.exists():
        return None
    try:
        data = load_json(spec_path)
    except Exception:
        return None
    fixture = (((data.get("checks") or {}).get("fixture_availability") or {}).get("fixture_path"))
    if not isinstance(fixture, str) or not fixture.strip():
        return None
    raw = Path(fixture)
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, repo_root / "fixtures" / "tasks" / canonical_task(task) / raw]
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "gt.jsonl":
            return candidate.parent
        if candidate.is_dir():
            return candidate
    return None


def default_fixture_dir(repo_root: Path, task: str) -> Path | None:
    task_dir = repo_root / "fixtures" / "tasks" / canonical_task(task)
    if not task_dir.exists():
        return None
    if (task_dir / "gt.jsonl").exists():
        return task_dir
    options = sorted(path.parent for path in task_dir.glob("*/gt.jsonl"))
    return options[0] if options else None


def first_symlink_component(path: Path) -> Path | None:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
    return None


def validate_structured_source_tree(source_dir: Path) -> None:
    symlink_component = first_symlink_component(source_dir)
    if symlink_component is not None:
        raise ValueError(
            f"structured fixture source path must not traverse a symlink: {symlink_component}"
        )
    if not source_dir.is_dir():
        raise ValueError(f"structured fixture source must be a directory: {source_dir}")
    gt = source_dir / "gt.jsonl"
    if gt.is_symlink() or not gt.is_file():
        raise ValueError(f"structured fixture must contain a regular gt.jsonl: {source_dir}")
    symlinks = [path for path in source_dir.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"structured fixture source tree must not contain symlinks: {symlinks[0]}")


def require_vad_single_link_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"VAD {label} must be a regular file: {path}")
    if path.stat().st_nlink != 1:
        raise ValueError(f"VAD {label} must not be hard-linked: {path}")


def load_samples(source_dir: Path, task: str) -> list[dict[str, Any]]:
    task = canonical_task(task)
    if is_structured_task(task):
        validate_structured_source_tree(source_dir)
    gt = source_dir / "gt.jsonl"
    if task == "vad":
        require_vad_single_link_file(gt, "gt.jsonl")
    samples: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for line_no, line in enumerate(gt.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{gt}:{line_no} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{gt}:{line_no} must be a JSON object")
        audio = (
            row.get("audio") or row.get("wav")
            if is_structured_task(task)
            else row.get("audio")
            or row.get("wav")
            or row.get("prompt_audio")
            or row.get("reference_audio")
        )
        if not isinstance(audio, str) or not audio:
            raise ValueError(f"{gt}:{line_no} must contain a non-empty relative audio/wav field")
        audio_path = Path(audio)
        if audio_path.is_absolute() or ".." in audio_path.parts:
            raise ValueError(f"{gt}:{line_no} audio path must be relative and stay inside the fixture directory")
        resolved_audio = source_dir / audio_path
        if not resolved_audio.is_file():
            raise FileNotFoundError(f"Fixture audio referenced by {gt}:{line_no} does not exist: {audio}")
        if task == "vad":
            require_vad_single_link_file(resolved_audio, f"audio {audio!r}")
        reference_audio = row.get("reference_audio")
        reference_path: Path | None = None
        if task == "se":
            if not isinstance(reference_audio, str) or not reference_audio:
                raise ValueError(f"{gt}:{line_no} task se requires a non-empty reference_audio")
            reference_path = Path(reference_audio)
            if reference_path.is_absolute() or ".." in reference_path.parts:
                raise ValueError(
                    f"{gt}:{line_no} reference_audio must be relative and stay inside the fixture directory"
                )
            if not (source_dir / reference_path).is_file():
                raise FileNotFoundError(
                    f"Fixture reference_audio referenced by {gt}:{line_no} does not exist: {reference_audio}"
                )
            if (source_dir / reference_path).samefile(source_dir / audio_path):
                raise ValueError(
                    f"{gt}:{line_no} task se audio and reference_audio must be independent files"
                )
        key = str(row.get("key") or row.get("id") or audio_path.stem).strip()
        if not key:
            raise ValueError(f"{gt}:{line_no} requires a non-empty key")
        if key in seen_keys:
            raise ValueError(f"{gt}:{line_no} duplicates fixture key {key!r}")
        seen_keys.add(key)

        if task == "sa_asr" and "segments" not in row:
            raise ValueError(f"{gt}:{line_no} task sa_asr requires speaker-attributed segments")
        if task == "sd" and "segments" not in row:
            raise ValueError(f"{gt}:{line_no} task sd requires speaker segments")
        if task == "vad" and "speech_segments" not in row:
            raise ValueError(f"{gt}:{line_no} task vad requires speech_segments")
        if is_structured_task(task) and row.get("task") is not None and canonical_task(row["task"]) != task:
            raise ValueError(f"{gt}:{line_no} declares task {row['task']!r}, expected {task!r}")

        wav_info: dict[str, Any] | None = None
        if is_structured_task(task):
            wav_info = pcm_wav_info(resolved_audio)
            segment_violations = validate_segments(
                row.get(reference_segments_field(task)),
                task=task,
                duration_sec=float(wav_info["duration_sec"]),
                audio_is_silence=bool(wav_info["audio_is_silence"]),
            )
            if segment_violations:
                raise ValueError(
                    f"{gt}:{line_no} invalid {task} reference segments: "
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
                raise ValueError(
                    f"{gt}:{line_no} duration_sec disagrees with the PCM WAV"
                )
            declared_sample_rate = row.get("sample_rate")
            if declared_sample_rate is not None and declared_sample_rate != wav_info["sample_rate"]:
                raise ValueError(
                    f"{gt}:{line_no} sample_rate disagrees with the PCM WAV"
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
        ]
        if task == "se" and isinstance(reference_audio, str) and reference_audio:
            annotation_fields.append("reference_audio")
        if not annotation_fields:
            raise ValueError(
                f"{gt}:{line_no} must contain at least one annotation field "
                "(ground_truth, target_text, text, segments, speech_segments, label, intent, or reference_audio)"
            )
        sample = {
            "key": key,
            "audio": audio,
            "audio_path": str(resolved_audio.resolve()),
            "annotation_fields": annotation_fields,
        }
        if task == "se" and reference_path is not None:
            sample["reference_audio"] = str(reference_audio)
            sample["reference_audio_path"] = str((source_dir / reference_path).resolve())
        if wav_info is not None:
            sample.update(wav_info)
        elif isinstance(row.get("duration_sec"), (int, float)):
            sample["duration_sec"] = row["duration_sec"]
        if wav_info is None and isinstance(row.get("sample_rate"), (int, float)):
            sample["sample_rate"] = row["sample_rate"]
        samples.append(sample)
    if not samples:
        raise ValueError(f"No samples found in {gt}")
    if len(samples) > 5:
        raise ValueError(f"{gt} has {len(samples)} samples; local validation allows at most 5")
    return samples


def replace_tree(source_dir: Path, staged_dir: Path) -> None:
    if staged_dir.exists() or staged_dir.is_symlink():
        if staged_dir.is_symlink() or staged_dir.is_file():
            staged_dir.unlink()
        else:
            shutil.rmtree(staged_dir)
    staged_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, staged_dir)


def replace_structured_tree(
    source_dir: Path,
    staged_dir: Path,
    samples: list[dict[str, Any]],
    *,
    task: str | None = None,
) -> None:
    validate_structured_source_tree(source_dir)
    vad = canonical_task(task) == "vad"
    source_gt = source_dir / "gt.jsonl"
    if vad:
        require_vad_single_link_file(source_gt, "source gt.jsonl")
    if staged_dir.exists() or staged_dir.is_symlink():
        if staged_dir.is_symlink() or staged_dir.is_file():
            staged_dir.unlink()
        else:
            shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True)
    staged_gt = staged_dir / "gt.jsonl"
    shutil.copy2(source_gt, staged_gt)
    if vad:
        require_vad_single_link_file(staged_gt, "staged gt.jsonl")
    copied: set[Path] = set()
    for sample in samples:
        relative = Path(str(sample["audio"]))
        if relative in copied:
            continue
        copied.add(relative)
        source = source_dir / relative
        if vad:
            require_vad_single_link_file(source, f"source audio {relative.as_posix()!r}")
        destination = staged_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if vad:
            require_vad_single_link_file(destination, f"staged audio {relative.as_posix()!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--produces", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--link-policy", choices=["copy"], default="copy")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    resolved_path = run_dir / "artifacts" / "model_input_resolved.json"
    if not resolved_path.exists():
        print(f"model_input_resolved.json not found: {resolved_path}", file=sys.stderr)
        return 1
    resolved = load_json(resolved_path)
    model_dir_raw = resolved.get("model_dir")
    task_raw = resolved.get("task_type")
    if not isinstance(model_dir_raw, str) or not isinstance(task_raw, str):
        print("model_input_resolved.json must contain model_dir and task_type", file=sys.stderr)
        return 1
    task = canonical_task(task_raw)
    declared_model_dir = Path(model_dir_raw).expanduser()
    if is_structured_task(task) and declared_model_dir.is_symlink():
        print("structured-task model_dir must not be a symlink", file=sys.stderr)
        return 1
    model_dir = declared_model_dir.resolve()
    repo_root = infer_repo_root(model_dir)

    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.is_absolute():
            source_dir = repo_root / source_dir
        symlink_component = first_symlink_component(source_dir) if is_structured_task(task) else None
        if symlink_component is not None:
            print(
                f"structured fixture source path must not traverse a symlink: {symlink_component}",
                file=sys.stderr,
            )
            return 1
        source_dir = source_dir.resolve()
    else:
        source_dir = candidate_from_spec(run_dir, repo_root, task_raw) or default_fixture_dir(repo_root, task_raw)
    if source_dir is None or not source_dir.exists():
        print(f"No fixture source found for task {task}. Expected fixtures/tasks/{task}/<fixture>/gt.jsonl", file=sys.stderr)
        return 1
    if source_dir.is_file() and source_dir.name == "gt.jsonl":
        source_dir = source_dir.parent
    if not (source_dir / "gt.jsonl").exists():
        print(f"Fixture source must contain gt.jsonl: {source_dir}", file=sys.stderr)
        return 1

    samples = load_samples(source_dir, task)
    staged_dir = model_dir / "fixture" / task / source_dir.name
    if is_structured_task(task):
        for parent in (model_dir / "fixture", model_dir / "fixture" / task):
            if parent.is_symlink():
                print(f"structured staged fixture parent must not be a symlink: {parent}", file=sys.stderr)
                return 1
        replace_structured_tree(source_dir, staged_dir, samples, task=task)
    else:
        replace_tree(source_dir, staged_dir)

    staged_samples = []
    for sample in load_samples(staged_dir, task):
        staged_samples.append(sample)

    manifest = {
        "model_id": resolved.get("model_id", ""),
        "model_name": resolved.get("model_name", ""),
        "model_dir": str(model_dir),
        "task_type": task,
        "source_dir": str(source_dir),
        "staged_dir": str(staged_dir),
        "gt_jsonl": str(staged_dir / "gt.jsonl"),
        "sample_count": len(staged_samples),
        "link_policy": args.link_policy,
        "samples": staged_samples,
        "validation_payload_env": "SURE_VALIDATE_INPUT_JSON",
        "validation_protocol_env": (
            "SURE_VALIDATE_PROTOCOL_JSON" if is_structured_task(task) else None
        ),
        "notes": "Fixture staged into model-local fixture directory for validate.py discovery.",
    }
    write_json(Path(args.produces), manifest)
    print(f"Prepared {len(samples)} fixture sample(s): {staged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
