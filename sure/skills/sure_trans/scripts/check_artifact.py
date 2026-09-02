#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path, PurePosixPath

import yaml

from mcp_smoke import validate_speaker_output
from prepare_fixture import (
    SPEAKER_TASKS,
    looks_like_absolute_path_or_uri,
    pcm_wav_info,
    validate_speaker_segments,
)
from scaffold_adapter import io_contract_for
from vc_exec import default_partition


LEGACY_PATH = re.compile(r"/(?:mnt/cloudstorfs|hpc_stor\d+|hpc_\d+)/")
ANNOTATION_FIELDS = ("ground_truth", "target_text", "text", "segments", "label", "intent")
KWS_ANNOTATION_FIELDS = ("expected", "label", "expected_detected", "text", "txt")
KWS_OPERATING_THRESHOLD = 0.5
TRANS_RESERVED_ROOTS = {
    "model.py",
    "server.py",
    "__init__.py",
    "validate.py",
    "config.yaml",
    "model.spec.yaml",
    "Dockerfile.sure",
    "Dockerfile",
    "artifacts",
    "fixture",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"artifact must be a JSON object: {path}")
    return value


def artifact_time(value: dict, label: str) -> datetime:
    raw = value.get("generated_at") or value.get("timestamp")
    require(isinstance(raw, str) and bool(raw.strip()), f"{label} must record generated_at or timestamp")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} timestamp is invalid: {raw}") from error


def has_annotation_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return value is not None


def normalized_keyword(value: str) -> str:
    return "".join(value.upper().split())


def kws_keywords(value: object, key: str) -> list[str]:
    if isinstance(value, str):
        keywords = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        keywords = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        require(len(keywords) == len(value), f"KWS fixture {key} keywords must be non-empty strings")
    else:
        keywords = []
    require(bool(keywords), f"KWS fixture {key} requires at least one keyword")
    return keywords


def fixture_tree_identity(staged_dir: Path, relative_files: set[Path]) -> str:
    hashes = {
        relative.as_posix(): sha256_file(staged_dir / relative)
        for relative in sorted(relative_files, key=lambda item: item.as_posix())
    }
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_kws_fixture_manifest(
    value: dict,
    *,
    model_dir: Path,
    staged_dir: Path,
    gt_jsonl: Path,
) -> None:
    require(staged_dir.is_relative_to(model_dir / "fixture"), "fixture staged_dir must stay under model_dir/fixture")
    require(Path(str(value.get("staged_path") or "")).resolve() == staged_dir, "KWS staged_path must equal staged_dir")
    require(gt_jsonl.is_file() and gt_jsonl.parent == staged_dir, "KWS gt_jsonl must exist directly inside staged_dir")
    require(value.get("gt_sha256") == sha256_file(gt_jsonl), "KWS fixture ground-truth checksum changed")
    require(value.get("expected_sha256") == sha256_file(gt_jsonl), "KWS reference checksum changed")

    rows: list[dict] = []
    for line_number, line in enumerate(gt_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"KWS fixture gt_jsonl line {line_number} is invalid JSON: {error}") from error
        require(isinstance(row, dict), f"KWS fixture gt_jsonl line {line_number} must be an object")
        rows.append(row)
    require(2 <= len(rows) <= 5, "KWS smoke fixture must contain 2 to 5 samples")
    samples = value.get("samples")
    require(isinstance(samples, list) and len(samples) == len(rows), "KWS samples must mirror gt_jsonl rows")
    require(value.get("sample_count") == len(rows), "KWS sample_count must match gt_jsonl")

    seen_keys: set[str] = set()
    polarities: set[bool] = set()
    relative_files = {Path("gt.jsonl")}
    for index, (row, sample) in enumerate(zip(rows, samples)):
        require(isinstance(sample, dict), f"KWS fixture sample {index} must be an object")
        key = row.get("key")
        require(isinstance(key, str) and bool(key.strip()), f"KWS fixture row {index} requires a non-empty key")
        require(key not in seen_keys, f"KWS fixture contains duplicate key: {key}")
        seen_keys.add(key)
        require(sample.get("key") == key, f"KWS fixture sample {key} does not mirror gt_jsonl key")
        audio = row.get("audio") or row.get("wav")
        require(isinstance(audio, str) and bool(audio.strip()), f"KWS fixture {key} requires audio or wav")
        relative_audio = Path(audio)
        require(
            not relative_audio.is_absolute() and ".." not in relative_audio.parts,
            f"KWS fixture {key} audio path must be relative and contained",
        )
        audio_path = staged_dir / relative_audio
        require(
            audio_path.is_file() and not audio_path.is_symlink() and audio_path.resolve().is_relative_to(staged_dir),
            f"KWS fixture {key} audio is missing or unsafe",
        )
        require(sample.get("audio") == relative_audio.as_posix(), f"KWS fixture sample {key} audio path changed")
        require(
            Path(str(sample.get("audio_path") or "")).resolve() == audio_path.resolve(),
            f"KWS fixture sample {key} audio_path changed",
        )
        require(sample.get("sha256") == sha256_file(audio_path), f"KWS fixture sample {key} checksum changed")
        require(sample.get("size_bytes") == audio_path.stat().st_size, f"KWS fixture sample {key} size changed")
        relative_files.add(relative_audio)

        expected_detected = row.get("expected_detected")
        require(isinstance(expected_detected, bool), f"KWS fixture {key} requires boolean expected_detected")
        require(sample.get("expected_detected") is expected_detected, f"KWS sample {key} polarity changed")
        polarities.add(expected_detected)
        keywords = kws_keywords(row.get("keywords"), key)
        require(sample.get("keywords") == row.get("keywords"), f"KWS sample {key} keywords changed")
        expected_keyword = row.get("expected_keyword")
        if expected_detected:
            require(
                isinstance(expected_keyword, str) and bool(expected_keyword.strip()),
                f"positive KWS fixture {key} requires expected_keyword",
            )
            require(
                normalized_keyword(expected_keyword) in {normalized_keyword(keyword) for keyword in keywords},
                f"positive KWS fixture {key} expected_keyword is not in keywords",
            )
            require(sample.get("expected_keyword") == expected_keyword, f"KWS sample {key} expected_keyword changed")
        else:
            require(expected_keyword is None, f"negative KWS fixture {key} must have null expected_keyword")
            require(sample.get("expected_keyword") is None, f"negative KWS sample {key} expected_keyword changed")
        duration = row.get("duration")
        require(
            not isinstance(duration, bool)
            and isinstance(duration, (int, float))
            and math.isfinite(float(duration))
            and float(duration) > 0,
            f"KWS fixture {key} duration must be positive and finite",
        )
        require(sample.get("duration") == duration, f"KWS sample {key} duration changed")
        if "threshold" in row:
            threshold = row["threshold"]
            require(
                not isinstance(threshold, bool)
                and isinstance(threshold, (int, float))
                and math.isfinite(float(threshold))
                and float(threshold) == KWS_OPERATING_THRESHOLD,
                f"KWS fixture {key} threshold must equal {KWS_OPERATING_THRESHOLD}",
            )
        declared_annotations = sample.get("annotation_fields")
        actual_annotations = [
            field for field in KWS_ANNOTATION_FIELDS if field in row and has_annotation_value(row[field])
        ]
        require(actual_annotations, f"KWS fixture {key} has no explicit reference annotation")
        require(declared_annotations == actual_annotations, f"KWS sample {key} annotation_fields changed")

    require(polarities == {False, True}, "KWS smoke fixture must contain positive and negative samples")
    actual_files: set[Path] = set()
    for path in staged_dir.rglob("*"):
        require(not path.is_symlink(), f"KWS fixture tree must not contain symlinks: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(staged_dir))
    require(actual_files == relative_files, "KWS fixture tree must contain only gt.jsonl and referenced audio")
    require(value.get("sha256") == fixture_tree_identity(staged_dir, relative_files), "KWS fixture tree checksum changed")
    require(
        value.get("size_bytes") == sum((staged_dir / relative).stat().st_size for relative in relative_files),
        "KWS fixture tree size changed",
    )
    annotation_source = value.get("annotation_source")
    require(isinstance(annotation_source, dict), "KWS fixture annotation_source must be an object")
    require(
        annotation_source.get("type") == "fixture_gt_jsonl"
        and annotation_source.get("fallback") is False,
        "KWS ground truth must come from fixture gt.jsonl",
    )
    require(
        Path(str(annotation_source.get("staged_path") or "")).resolve() == gt_jsonl,
        "KWS annotation_source staged_path must match gt_jsonl",
    )


def validate_se_fixture_manifest(
    value: dict,
    *,
    model_dir: Path,
    staged_dir: Path,
    gt_jsonl: Path,
) -> None:
    require(staged_dir.is_relative_to(model_dir / "fixture"), "fixture staged_dir must stay under model_dir/fixture")
    require(Path(str(value.get("staged_path") or "")).resolve() == staged_dir, "SE staged_path must equal staged_dir")
    require(gt_jsonl.is_file() and gt_jsonl.parent == staged_dir, "SE gt_jsonl must exist directly inside staged_dir")
    require(value.get("gt_sha256") == sha256_file(gt_jsonl), "SE fixture ground-truth checksum changed")
    require(value.get("expected_sha256") == sha256_file(gt_jsonl), "SE reference checksum changed")

    rows: list[dict] = []
    for line_number, line in enumerate(gt_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"SE fixture gt_jsonl line {line_number} is invalid JSON: {error}") from error
        require(isinstance(row, dict), f"SE fixture gt_jsonl line {line_number} must be an object")
        rows.append(row)
    require(1 <= len(rows) <= 5, "SE smoke fixture must contain 1 to 5 samples")
    samples = value.get("samples")
    require(isinstance(samples, list) and len(samples) == len(rows), "SE samples must mirror gt_jsonl rows")
    require(value.get("sample_count") == len(rows), "SE sample_count must match gt_jsonl")

    seen_keys: set[str] = set()
    relative_files = {Path("gt.jsonl")}
    for index, (row, sample) in enumerate(zip(rows, samples)):
        require(isinstance(sample, dict), f"SE fixture sample {index} must be an object")
        key = row.get("sample_id") or row.get("key")
        require(isinstance(key, str) and bool(key.strip()), f"SE fixture row {index} requires a non-empty key")
        require(row.get("key") == key and row.get("sample_id") == key, f"SE fixture row {key} key aliases changed")
        require(key not in seen_keys, f"SE fixture contains duplicate key: {key}")
        seen_keys.add(key)
        require(sample.get("key") == key, f"SE fixture sample {key} does not mirror gt_jsonl key")

        paths: dict[str, tuple[Path, Path]] = {}
        for role in ("noisy_audio", "reference_audio"):
            raw_audio = row.get(role, row.get("audio")) if role == "noisy_audio" else row.get(role)
            require(isinstance(raw_audio, str) and bool(raw_audio.strip()), f"SE fixture {key} requires {role}")
            relative_audio = Path(raw_audio)
            require(
                not relative_audio.is_absolute() and ".." not in relative_audio.parts,
                f"SE fixture {key} {role} path must be relative and contained",
            )
            audio_path = staged_dir / relative_audio
            require(
                audio_path.is_file()
                and not audio_path.is_symlink()
                and audio_path.resolve().is_relative_to(staged_dir),
                f"SE fixture {key} {role} is missing or unsafe",
            )
            paths[role] = (relative_audio, audio_path)
            relative_files.add(relative_audio)
        require(
            not paths["noisy_audio"][1].samefile(paths["reference_audio"][1]),
            f"SE fixture {key} noisy and clean audio must be independent files",
        )

        noisy_relative, noisy_path = paths["noisy_audio"]
        clean_relative, clean_path = paths["reference_audio"]
        require(row.get("audio") == noisy_relative.as_posix(), f"SE fixture row {key} audio alias changed")
        require(row.get("noisy_audio") == noisy_relative.as_posix(), f"SE fixture row {key} noisy_audio changed")
        require(sample.get("audio") == noisy_relative.as_posix(), f"SE fixture sample {key} audio path changed")
        require(sample.get("noisy_audio") == noisy_relative.as_posix(), f"SE fixture sample {key} noisy_audio changed")
        require(sample.get("reference_audio") == clean_relative.as_posix(), f"SE fixture sample {key} reference_audio changed")
        require(Path(str(sample.get("audio_path") or "")).resolve() == noisy_path.resolve(), f"SE fixture sample {key} audio_path changed")
        require(Path(str(sample.get("noisy_audio_path") or "")).resolve() == noisy_path.resolve(), f"SE fixture sample {key} noisy_audio_path changed")
        require(Path(str(sample.get("reference_audio_path") or "")).resolve() == clean_path.resolve(), f"SE fixture sample {key} reference_audio_path changed")
        require(sample.get("sha256") == sha256_file(noisy_path), f"SE fixture sample {key} noisy checksum changed")
        require(sample.get("reference_sha256") == sha256_file(clean_path), f"SE fixture sample {key} clean checksum changed")
        require(sample.get("size_bytes") == noisy_path.stat().st_size, f"SE fixture sample {key} noisy size changed")
        require(sample.get("reference_size_bytes") == clean_path.stat().st_size, f"SE fixture sample {key} clean size changed")
        require(sample.get("annotation_fields") == ["reference_audio"], f"SE sample {key} annotation_fields changed")

    actual_files: set[Path] = set()
    for path in staged_dir.rglob("*"):
        require(not path.is_symlink(), f"SE fixture tree must not contain symlinks: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(staged_dir))
    require(actual_files == relative_files, "SE fixture tree must contain only gt.jsonl and referenced audio")
    require(value.get("sha256") == fixture_tree_identity(staged_dir, relative_files), "SE fixture tree checksum changed")
    require(
        value.get("size_bytes") == sum((staged_dir / relative).stat().st_size for relative in relative_files),
        "SE fixture tree size changed",
    )
    annotation_source = value.get("annotation_source")
    require(isinstance(annotation_source, dict), "SE fixture annotation_source must be an object")
    require(
        annotation_source.get("type") == "fixture_gt_jsonl"
        and annotation_source.get("fallback") is False,
        "SE ground truth must come from fixture gt.jsonl",
    )
    require(
        Path(str(annotation_source.get("staged_path") or "")).resolve() == gt_jsonl,
        "SE annotation_source staged_path must match gt_jsonl",
    )


def validate_speaker_fixture_manifest(
    value: dict,
    *,
    model_dir: Path,
    staged_dir: Path,
    gt_jsonl: Path,
    task: str,
) -> None:
    require(task in SPEAKER_TASKS, f"unsupported speaker fixture task: {task}")
    label = task.upper().replace("_", "-")
    require(
        staged_dir.is_relative_to(model_dir / "fixture"),
        "fixture staged_dir must stay under model_dir/fixture",
    )
    require(
        Path(str(value.get("staged_path") or "")).resolve() == staged_dir,
        f"{label} staged_path must equal staged_dir",
    )
    require(
        gt_jsonl.is_file() and gt_jsonl.parent == staged_dir,
        f"{label} gt_jsonl must exist directly inside staged_dir",
    )
    require(value.get("gt_sha256") == sha256_file(gt_jsonl), f"{label} ground-truth checksum changed")
    require(value.get("expected_sha256") == sha256_file(gt_jsonl), f"{label} reference checksum changed")

    rows: list[dict] = []
    for line_number, line in enumerate(gt_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} fixture gt_jsonl line {line_number} is invalid JSON: {error}") from error
        require(isinstance(row, dict), f"{label} fixture gt_jsonl line {line_number} must be an object")
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} fixture gt_jsonl line {line_number} must contain strict JSON: {error}"
            ) from error
        rows.append(row)
    require(1 <= len(rows) <= 5, f"{label} smoke fixture must contain 1 to 5 samples")
    samples = value.get("samples")
    require(
        isinstance(samples, list) and len(samples) == len(rows),
        f"{label} samples must mirror gt_jsonl rows",
    )
    require(value.get("sample_count") == len(rows), f"{label} sample_count must match gt_jsonl")

    seen_keys: set[str] = set()
    relative_files = {Path("gt.jsonl")}
    for index, (row, sample) in enumerate(zip(rows, samples)):
        require(isinstance(sample, dict), f"{label} fixture sample {index} must be an object")
        key = row.get("key")
        require(
            isinstance(key, str) and bool(key.strip()),
            f"{label} fixture row {index} requires a non-empty key",
        )
        require(not looks_like_absolute_path_or_uri(key), f"{label} fixture key must be a safe token")
        require(key not in seen_keys, f"{label} fixture contains duplicate key: {key}")
        seen_keys.add(key)
        require(row.get("task_type") == task, f"{label} fixture row {key} must use canonical task_type={task}")
        require(sample.get("key") == key, f"{label} fixture sample {key} does not mirror gt_jsonl key")
        require("segments" not in sample, f"{label} manifest samples must not duplicate reference segments")
        audio = row.get("audio")
        require(isinstance(audio, str) and bool(audio.strip()), f"{label} fixture {key} requires audio")
        relative_audio = Path(audio)
        require(
            not relative_audio.is_absolute()
            and ".." not in relative_audio.parts
            and "\\" not in audio,
            f"{label} fixture {key} audio path must be relative and contained",
        )
        audio_path = staged_dir / relative_audio
        require(
            audio_path.is_file()
            and not audio_path.is_symlink()
            and audio_path.stat().st_size > 0
            and audio_path.resolve().is_relative_to(staged_dir),
            f"{label} fixture {key} audio is missing or unsafe",
        )
        relative_files.add(relative_audio)
        require(sample.get("audio") == relative_audio.as_posix(), f"{label} sample {key} audio path changed")
        require(
            Path(str(sample.get("audio_path") or "")).resolve() == audio_path.resolve(),
            f"{label} sample {key} audio_path changed",
        )
        require(sample.get("sha256") == sha256_file(audio_path), f"{label} sample {key} checksum changed")
        require(sample.get("size_bytes") == audio_path.stat().st_size, f"{label} sample {key} size changed")
        wav_info = pcm_wav_info(audio_path)
        for field in ("duration_sec", "sample_rate", "audio_is_silence"):
            require(sample.get(field) == wav_info[field], f"{label} sample {key} {field} changed")
        segments = validate_speaker_segments(
            row.get("segments"),
            task=task,
            label=f"{label} fixture {key}",
            empty_sd_allowed=task == "sd" and bool(wav_info["audio_is_silence"]),
            duration_sec=float(wav_info["duration_sec"]),
        )
        require(sample.get("segment_count") == len(segments), f"{label} sample {key} segment_count changed")
        require(sample.get("annotation_fields") == ["segments"], f"{label} sample {key} annotation_fields changed")

    actual_files: set[Path] = set()
    for path in staged_dir.rglob("*"):
        require(not path.is_symlink(), f"{label} fixture tree must not contain symlinks: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(staged_dir))
    require(
        actual_files == relative_files,
        f"{label} fixture tree must contain only gt.jsonl and referenced audio",
    )
    require(
        value.get("sha256") == fixture_tree_identity(staged_dir, relative_files),
        f"{label} fixture tree checksum changed",
    )
    require(
        value.get("size_bytes")
        == sum((staged_dir / relative).stat().st_size for relative in relative_files),
        f"{label} fixture tree size changed",
    )
    annotation_source = value.get("annotation_source")
    require(isinstance(annotation_source, dict), f"{label} annotation_source must be an object")
    require(
        annotation_source.get("type") == "fixture_gt_jsonl"
        and annotation_source.get("fallback") is False,
        f"{label} ground truth must come from fixture gt.jsonl",
    )
    require(
        Path(str(annotation_source.get("staged_path") or "")).resolve() == gt_jsonl,
        f"{label} annotation_source staged_path must match gt_jsonl",
    )


def validate_fixture_manifest(value: dict) -> None:
    require(value.get("status") == "ready", "fixture manifest is not ready")
    for key in ("model_dir", "staged_dir", "gt_jsonl", "samples", "annotation_source"):
        require(key in value, f"fixture manifest is missing {key}")
    model_dir = Path(str(value["model_dir"])).resolve()
    staged_dir = Path(str(value["staged_dir"])).resolve()
    staged = Path(str(value.get("staged_path", ""))).resolve()
    gt_jsonl = Path(str(value["gt_jsonl"])).resolve()
    task = str(value.get("task_type") or "").replace("-", "_").lower()
    require(model_dir.is_dir(), "fixture model_dir is missing")
    require(staged_dir.is_dir(), "fixture staged_dir is missing")
    if task == "kws":
        validate_kws_fixture_manifest(
            value,
            model_dir=model_dir,
            staged_dir=staged_dir,
            gt_jsonl=gt_jsonl,
        )
        return
    if task == "se":
        validate_se_fixture_manifest(
            value,
            model_dir=model_dir,
            staged_dir=staged_dir,
            gt_jsonl=gt_jsonl,
        )
        return
    if task in SPEAKER_TASKS:
        validate_speaker_fixture_manifest(
            value,
            model_dir=model_dir,
            staged_dir=staged_dir,
            gt_jsonl=gt_jsonl,
            task=task,
        )
        return
    require(staged_dir.is_relative_to(model_dir / "fixture"), "fixture staged_dir must stay under model_dir/fixture")
    require(staged.is_file(), "staged fixture is missing")
    require(staged.parent == staged_dir, "staged fixture must be directly inside staged_dir")
    require(value.get("sha256") == sha256_file(staged), "staged fixture checksum changed")
    require(gt_jsonl.is_file() and gt_jsonl.parent == staged_dir, "gt_jsonl must exist directly inside staged_dir")
    require(value.get("gt_sha256") == sha256_file(gt_jsonl), "fixture ground-truth checksum changed")

    samples = value.get("samples")
    require(isinstance(samples, list) and len(samples) == 1, "trans smoke fixture must declare one sample")
    sample = samples[0]
    require(isinstance(sample, dict), "fixture sample must be an object")
    require(sample.get("audio") == staged.name, "fixture sample must mirror staged_path")
    require(Path(str(sample.get("audio_path") or "")).resolve() == staged, "fixture sample audio_path must match staged_path")
    require(int(value.get("sample_count", 0)) == 1, "trans smoke fixture must contain exactly one bounded sample")

    rows = [line for line in gt_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 1, "trans smoke fixture gt_jsonl must contain exactly one non-empty row")
    try:
        row = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"fixture gt_jsonl is invalid JSON: {error}") from error
    require(isinstance(row, dict), "fixture gt_jsonl row must be an object")
    audio_field = "reference_audio" if task in {"tts", "vc"} else "audio"
    require(row.get(audio_field) == staged.name, f"fixture gt_jsonl {audio_field} must mirror staged_path")
    declared_annotations = sample.get("annotation_fields")
    actual_annotations = [
        field for field in ANNOTATION_FIELDS if field in row and has_annotation_value(row[field])
    ]
    require(actual_annotations, "fixture gt_jsonl must contain a non-empty reference annotation")
    require(declared_annotations == actual_annotations, "fixture sample annotation_fields must mirror gt_jsonl")
    if task == "tts":
        require(
            isinstance(row.get("prompt_text"), str) and bool(row["prompt_text"].strip()),
            "TTS fixture gt_jsonl requires non-empty prompt_text",
        )

    annotation_source = value.get("annotation_source")
    require(isinstance(annotation_source, dict), "fixture annotation_source must be an object")
    require(
        annotation_source.get("type") == "fixture_expected_sidecar"
        and annotation_source.get("fallback") is False,
        "fixture ground truth must come from a reference .expected.json sidecar",
    )
    expected_path = Path(str(annotation_source.get("staged_path") or "")).resolve()
    require(expected_path.is_file() and expected_path.parent == staged_dir, "staged fixture annotation sidecar is missing")
    require(value.get("expected_sha256") == sha256_file(expected_path), "fixture annotation sidecar checksum changed")
    expected = read_object(expected_path)
    for field in actual_annotations:
        require(row.get(field) == expected.get(field), f"fixture gt_jsonl {field} disagrees with reference sidecar")
    if task == "tts":
        require(row.get("prompt_text") == expected.get("prompt_text"), "fixture prompt_text disagrees with reference sidecar")


def infer_repo_root(run_dir: Path) -> Path:
    resolved = run_dir.expanduser().resolve()
    if resolved.parent.name == "runs" and resolved.parent.parent.name == ".sure":
        return resolved.parent.parent.parent
    return Path.cwd().resolve()


def harness_model_dir(run_dir: Path) -> Path:
    resolved = read_object(Path(run_dir) / "artifacts" / "trans_input_resolved.json")
    model_name = str(resolved.get("model_name") or "")
    if not model_name or "/" in model_name or "\\" in model_name:
        raise ValueError("model_name must be a single directory segment")
    path_policy = resolved.get("path_policy") if isinstance(resolved.get("path_policy"), dict) else {}
    raw_root = path_policy.get("allowed_model_root")
    if raw_root:
        allowed_root = Path(str(raw_root)).expanduser().resolve()
    else:
        allowed_root = (infer_repo_root(Path(run_dir)) / "sure" / "models").resolve()
    return (allowed_root / model_name).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--produces", required=True)
    parser.add_argument("--kind", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    path = Path(args.produces)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "artifact must be a JSON object")
    kind = args.kind
    if kind == "input":
        for key in ("dockerfile", "build_context", "model_path", "inference_entrypoint"):
            candidate = Path(str(value.get(key, "")))
            require(candidate.is_absolute() and candidate.exists(), f"{key} must exist and be absolute")
        require(value.get("framework") == "pytorch", "framework must normalize to pytorch")
        require(
            isinstance(value.get("model_framework"), str) and bool(value["model_framework"].strip()),
            "model_framework is required",
        )
        expected_model_dir = harness_model_dir(run_dir)
        declared_model_dir = Path(str(value.get("model_dir") or "")).expanduser()
        try:
            declared_model_dir = declared_model_dir.resolve()
        except OSError:
            declared_model_dir = declared_model_dir.absolute()
        if declared_model_dir.exists() and declared_model_dir.is_symlink():
            raise ValueError("model_dir must be a real harness-owned directory, not a whole-directory symlink")
        require(
            declared_model_dir == expected_model_dir,
            f"model_dir must be the harness-owned bundle {expected_model_dir}; got {declared_model_dir}",
        )
    elif kind == "dependencies":
        require(value.get("status") == "ready", "dependency inspection is blocked")
        require(value.get("unresolved") == [], "dependency report contains unresolved paths")
        require(value.get("external_paths") == [], "dependency report contains undeclared external paths")
    elif kind == "framework":
        resolved = read_object(run_dir / "artifacts" / "trans_input_resolved.json")
        require(value.get("status") == "ready", "primary computation framework must be PyTorch")
        require(
            value.get("declared_framework") == resolved.get("framework") == "pytorch",
            "declared computation framework must match the resolved PyTorch input",
        )
        require(value.get("detected_framework") == "pytorch", "static inspection must detect PyTorch")
        require(value.get("framework_requirement_met") is True, "PyTorch framework requirement was not met")
        declared_model_framework = value.get("declared_model_framework")
        detected_model_framework = value.get("detected_model_framework")
        require(
            isinstance(declared_model_framework, str) and bool(declared_model_framework.strip()),
            "declared_model_framework is required",
        )
        require(
            declared_model_framework == resolved.get("model_framework"),
            "declared_model_framework must match the resolved input",
        )
        require(
            detected_model_framework in {"transformers", "custom"},
            "ready framework detection must identify the model framework category",
        )
        needs_clarification = (
            declared_model_framework != "transformers"
            or detected_model_framework != "transformers"
            or value.get("model_framework_matches") is not True
        )
        require(
            value.get("clarification_required") is needs_clarification,
            "clarification_required does not match the framework evidence",
        )
        clarification = value.get("architecture_clarification")
        if needs_clarification:
            require(
                isinstance(clarification, str) and bool(clarification.strip()),
                "non-Transformers or mismatched model frameworks require architecture clarification",
            )
        else:
            require(clarification is None, "matching Transformers models must not carry a stale clarification")
    elif kind == "fixture":
        validate_fixture_manifest(value)
    elif kind == "source_image":
        require(value.get("status") == "passed", "source image materialization did not pass")
        require(value.get("source_image_policy") in {"load", "build"}, "source image policy must be load or build")
        require(value.get("source_image_policy") == value.get("requested_source_image_policy", value.get("source_image_policy")) or value.get("requested_source_image_policy") == "auto", "source image policy violates requested policy")
        require(Path(str(value.get("source_image_log_path", ""))).is_file(), "source image log is missing")
        require(value.get("image_id", "").startswith("sha256:"), "source image image_id must be a live sha256 ID")
        if value.get("source_image_policy") == "build":
            require(value.get("build_executed") is True, "source image build was not executed")
            require(value.get("build_exit_code") == 0, "docker build did not exit successfully")
            require(isinstance(value.get("build_command"), list) and value["build_command"][0:2] == ["docker", "build"], "source image must record docker build command")
            require(Path(str(value.get("build_log_path", ""))).is_file(), "source image build log is missing")
        else:
            require(value.get("load_executed") is True, "source image load was not executed")
            require(value.get("load_exit_code") == 0, "docker load did not exit successfully")
            require(isinstance(value.get("load_command"), list) and value["load_command"][0:2] == ["docker", "load"], "source image must record docker load command")
            image_tar = Path(str(value.get("image_tar", ""))).resolve()
            build_context = Path(str(value.get("build_context", ""))).resolve()
            require(image_tar.is_file() and image_tar.is_relative_to(build_context), "loaded image tar must be inside build context")
            require(value.get("tar_sha256") == sha256_file(image_tar), "loaded image tar checksum changed")
            require(value.get("load_verified") is True, "loaded image was not verified")
    elif kind == "adapter_image":
        require(value.get("status") == "passed", "adapter image build must pass")
    elif kind == "registry":
        require(value.get("status") == "passed", "registry package must pass")
        require(value.get("pull_verified") is True, "registry package must prove exact digest pull verification")
        require("@sha256:" in str(value.get("target_image_ref", "")), "registry target_image_ref must be digest-pinned")
        require(str(value.get("target_image_digest", "")).startswith("sha256:"), "registry target_image_digest must be a sha256 digest")
        compat_path = Path(run_dir) / "artifacts" / "execution_compat.json"
        selected_device = "cpu"
        if compat_path.is_file():
            selected_device = str(read_object(compat_path).get("selected_device") or "cpu")
        if selected_device == "cuda":
            smoke = value.get("post_pull_smoke")
            require(
                isinstance(smoke, dict),
                "GPU-validated models must repeat the MCP smoke test on VC after the exact digest pull and record post_pull_smoke evidence",
            )
            require(smoke.get("vc_job_id"), "post_pull_smoke must record the vc job id")
            expected_partition = default_partition()
            require(
                smoke.get("vc_partition") == expected_partition,
                f"post-pull MCP smoke must run on the site's dedicated partition {expected_partition}",
            )
            # vc submit takes repo:tag only and answers 镜像不存在 to any
            # repo@sha256:... reference, so the job cannot carry the pin in the
            # reference it runs. Requiring that made this unit unsatisfiable on
            # GPU. The submission proves the pin instead: vc_exec.py resolves
            # what the tag serves and refuses to submit on a mismatch.
            require(
                str(smoke.get("resolved_digest", "")) == str(value.get("target_image_digest", "")),
                "post_pull_smoke.resolved_digest must be the digest the submitted tag resolved to "
                "and must equal target_image_digest; submit through vc_exec.py --expect-digest so "
                "the pin is proven rather than asserted",
            )
            require(smoke.get("exit_code") == 0, "post-pull MCP smoke must exit 0")
            smoke_log = Path(str(smoke.get("log_path") or "")).expanduser()
            require(
                smoke_log.exists(),
                f"post_pull_smoke log path is missing: {smoke_log}",
            )
            evidence = smoke_log / "mcp_smoke.json" if smoke_log.is_dir() else smoke_log.parent / "mcp_smoke.json"
            require(
                evidence.is_file(),
                "post_pull_smoke must record mcp_smoke.json protocol evidence (initialize/tools/list/tools/call)",
            )
            protocol = read_object(evidence)
            require(protocol.get("status") == "passed", "post-pull MCP smoke evidence must pass")
            for step in ("initialize", "tools_list", "tools_call"):
                entry = protocol.get(step)
                require(
                    isinstance(entry, dict) and entry.get("ok") is True,
                    f"post-pull MCP smoke must prove {step} passed",
                )
            require(
                bool((protocol.get("tools_call") or {}).get("output_nonempty"))
                or bool((protocol.get("tools_call") or {}).get("text_nonempty")),
                "post-pull MCP smoke must return a non-empty primary output from tools/call",
            )
            if protocol.get("tool") == "kws_predict":
                call = protocol.get("tools_call") if isinstance(protocol.get("tools_call"), dict) else {}
                samples = call.get("samples")
                require(
                    isinstance(samples, list) and 2 <= len(samples) <= 5,
                    "post-pull KWS MCP smoke must cover 2 to 5 samples",
                )
                keys: set[str] = set()
                polarities: set[bool] = set()
                for sample in samples:
                    require(
                        isinstance(sample, dict) and sample.get("ok") is True,
                        "every post-pull KWS MCP smoke sample must pass",
                    )
                    key = str(sample.get("key") or "")
                    result = sample.get("result")
                    require(
                        bool(key) and key not in keys and isinstance(result, dict),
                        "post-pull KWS MCP smoke samples require unique keys and results",
                    )
                    keys.add(key)
                    detected = result.get("detected")
                    require(isinstance(detected, bool), "post-pull KWS detected must be boolean")
                    polarities.add(detected)
                require(
                    polarities == {False, True},
                    "post-pull KWS MCP smoke must prove positive and negative behavior",
                )
            if protocol.get("tool") in {"diarize", "transcribe_with_speakers"}:
                tool = str(protocol["tool"])
                task = "sd" if tool == "diarize" else "sa_asr"
                call = protocol.get("tools_call") if isinstance(protocol.get("tools_call"), dict) else {}
                samples = call.get("samples")
                require(
                    isinstance(samples, list) and 1 <= len(samples) <= 5,
                    f"post-pull {task.upper()} MCP smoke must cover 1 to 5 samples",
                )
                require(
                    call.get("expected_samples") == len(samples)
                    and call.get("num_samples") == len(samples),
                    f"post-pull {task.upper()} MCP smoke sample counts must match",
                )
                keys: set[str] = set()
                for sample in samples:
                    require(
                        isinstance(sample, dict) and sample.get("ok") is True,
                        f"every post-pull {task.upper()} MCP smoke sample must pass",
                    )
                    key = str(sample.get("key") or "")
                    require(
                        bool(key) and key not in keys,
                        f"post-pull {task.upper()} MCP smoke samples require unique keys",
                    )
                    keys.add(key)
                    violations = validate_speaker_output(
                        sample.get("result"),
                        tool=tool,
                        empty_sd_allowed=(
                            task == "sd" and sample.get("audio_is_silence") is True
                        ),
                    )
                    require(
                        not violations,
                        f"post-pull {task.upper()} MCP smoke sample {key} is malformed: "
                        + "; ".join(violations),
                    )
    elif kind == "model_payload":
        require(value.get("status") == "ready", "model payload was not staged")
        require(Path(str(value.get("destination", ""))).is_dir(), "staged model directory is missing")
        require(int(value.get("file_count", 0)) > 0, "staged model payload is empty")
        expected_model_dir = harness_model_dir(run_dir)
        declared_destination = Path(str(value.get("destination", ""))).expanduser().resolve()
        require(
            declared_destination == expected_model_dir,
            f"model payload must land in the harness-owned bundle {expected_model_dir}; got {declared_destination}",
        )
        files = value.get("files")
        require(isinstance(files, dict) and files, "model payload manifest must list every staged file")
        require(len(files) == int(value.get("file_count", 0)), "model payload file_count must match files")
        verified_hashes: dict[str, str] = {}
        total_bytes = 0
        for raw_path, entry in files.items():
            relative = Path(str(raw_path))
            require(
                str(raw_path) and not relative.is_absolute() and ".." not in relative.parts,
                f"model payload path must be portable: {raw_path}",
            )
            require(isinstance(entry, dict), f"model payload entry must be an object: {raw_path}")
            target = (declared_destination / relative).resolve()
            require(
                target.is_relative_to(declared_destination) and target.is_file() and not target.is_symlink(),
                f"staged model payload file is missing or unsafe: {raw_path}",
            )
            size = target.stat().st_size
            digest = sha256_file(target)
            require(entry.get("size_bytes") == size, f"model payload size changed: {raw_path}")
            require(entry.get("sha256") == digest, f"model payload checksum changed: {raw_path}")
            verified_hashes[relative.as_posix()] = digest
            total_bytes += size
        require(value.get("total_bytes") == total_bytes, "model payload total_bytes must match files")
        identity = hashlib.sha256(
            json.dumps(verified_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        require(value.get("payload_identity_sha256") == identity, "model payload identity does not match files")
        actual_payload: set[str] = set()
        for target in declared_destination.rglob("*"):
            relative = target.relative_to(declared_destination)
            if relative.parts[0] in TRANS_RESERVED_ROOTS:
                continue
            require(not target.is_symlink(), f"model payload must not contain symlinks: {relative}")
            if target.is_file():
                actual_payload.add(relative.as_posix())
        require(actual_payload == set(verified_hashes), "model payload manifest must exactly cover staged payload files")
    elif kind == "adapter":
        require(value.get("status") == "ready", "adapter manifest must be ready")
        require(value.get("harness_runtime_embedded") is True, "adapter image must embed the common Harness Runtime")
        harness = value.get("harness_runtime") if isinstance(value.get("harness_runtime"), dict) else {}
        require(
            all(harness.get(key) for key in ("runtime_id", "lock_sha256", "python_executable", "manifest_path", "runtime_root")),
            "adapter manifest must declare the embedded Harness Runtime binding",
        )
        python_executable = str(value.get("container_python_executable") or "")
        require(
            PurePosixPath(python_executable).is_absolute(),
            "adapter manifest container_python_executable must be absolute",
        )
        server_command = value.get("server_command")
        require(
            isinstance(server_command, list)
            and len(server_command) >= 2
            and server_command[0] == python_executable
            and all(isinstance(item, str) and item for item in server_command),
            "adapter manifest server_command must start with container_python_executable",
        )
        require(
            PurePosixPath(server_command[1]).is_absolute(),
            "adapter manifest server path must be absolute",
        )
        require(
            PurePosixPath(str(value.get("working_dir") or "")).is_absolute(),
            "adapter manifest working_dir must be absolute",
        )
        source_reference = str(value.get("source_image_reference") or "")
        require(source_reference, "adapter manifest source_image_reference is required")
        source_image = read_object(run_dir / "artifacts" / "source_image_result.json")
        require(
            value.get("source_image_id") == source_image.get("image_id"),
            "adapter manifest source_image_id must match source image evidence",
        )
        source_local = str(source_image.get("image") or "")
        source_push = source_image.get("registry_push") if isinstance(source_image.get("registry_push"), dict) else {}
        source_registry = str(source_image.get("registry_ref") or "")
        source_digest = str(source_push.get("digest") or "")
        if source_registry and source_digest:
            repository = source_registry.rsplit(":", 1)[0]
            require(
                source_reference == f"{repository}@{source_digest}",
                "adapter source_image_reference must pin the source registry digest",
            )
        else:
            require(source_reference == source_local, "adapter source_image_reference must match the verified local source image")
        for key in ("model_py", "init_py", "validate_py", "server_py", "config_yaml", "model_spec", "dockerfile", "mcp_smoke_py"):
            candidate = Path(str(value.get(key, "")))
            require(candidate.is_file(), f"adapter file missing: {key}")
        dockerfile = Path(str(value.get("dockerfile", "")))
        require(dockerfile.is_file(), "adapter Dockerfile is missing")
        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        require(
            dockerfile_text.splitlines()[0] == f"FROM {source_reference}",
            "adapter Dockerfile base image must match source_image_reference",
        )
        require(
            f"ENTRYPOINT {json.dumps(server_command)}" in dockerfile_text,
            "adapter Dockerfile ENTRYPOINT must match server_command",
        )
        config = yaml.safe_load(Path(str(value["config_yaml"])).read_text(encoding="utf-8"))
        require(
            isinstance(config, dict)
            and isinstance(config.get("server"), dict)
            and config["server"].get("command") == server_command,
            "adapter config server.command must match adapter manifest",
        )
        resolved_input_path = run_dir / "artifacts" / "trans_input_resolved.json"
        resolved_input = read_object(resolved_input_path) if resolved_input_path.is_file() else {}
        resolved_task = str(resolved_input.get("task_type") or "").lower().replace("-", "_")
        if resolved_task == "kws":
            contract = value.get("io_contract") if isinstance(value.get("io_contract"), dict) else {}
            require(
                contract.get("output_type") == "keyword_detection"
                and contract.get("primary_field") == "detected"
                and contract.get("required_fields") == ["detected", "keyword", "score"]
                and contract.get("nonempty_fields") == ["detected"],
                "KWS adapter io_contract must require detected/keyword/score with detected as primary",
            )
            tools = config.get("tools") if isinstance(config.get("tools"), list) else []
            require(
                any(isinstance(tool, dict) and tool.get("name") == "kws_predict" for tool in tools),
                "KWS adapter config must expose kws_predict",
            )
        if resolved_task == "se":
            contract = value.get("io_contract") if isinstance(value.get("io_contract"), dict) else {}
            require(
                contract.get("input_type") == "audio_path"
                and contract.get("output_type") == "audio"
                and contract.get("primary_field") == "audio_path"
                and contract.get("required_fields") == ["audio_path"]
                and contract.get("nonempty_fields") == ["audio_path"],
                "SE adapter io_contract must require generated audio_path",
            )
            tools = config.get("tools") if isinstance(config.get("tools"), list) else []
            require(
                any(isinstance(tool, dict) and tool.get("name") == "enhance_speech" for tool in tools),
                "SE adapter config must expose enhance_speech",
            )
        if resolved_task in SPEAKER_TASKS:
            contract = value.get("io_contract") if isinstance(value.get("io_contract"), dict) else {}
            require(
                contract == io_contract_for(resolved_task),
                f"{resolved_task.upper()} adapter io_contract must equal the canonical contract",
            )
            expected_nonempty = [] if resolved_task == "sd" else ["segments"]
            require(
                contract.get("input_type") == "audio_path"
                and contract.get("output_type") == "structured_segments"
                and contract.get("primary_field") == "segments"
                and contract.get("required_fields") == ["segments"]
                and contract.get("nonempty_fields") == expected_nonempty
                and contract.get("allow_empty_primary") is (resolved_task == "sd")
                and contract.get("allow_empty_segments")
                == ("silence_only" if resolved_task == "sd" else False),
                f"{resolved_task.upper()} adapter io_contract must require structured speaker segments",
            )
            require(
                contract.get("approved_output_fields") == ["num_speakers", "segments"]
                and isinstance(contract.get("segment_schema"), dict)
                and contract["segment_schema"].get("additionalProperties") is False,
                f"{resolved_task.upper()} adapter io_contract must close structured output fields",
            )
            expected_tool = "diarize" if resolved_task == "sd" else "transcribe_with_speakers"
            tools = config.get("tools") if isinstance(config.get("tools"), list) else []
            require(
                any(isinstance(tool, dict) and tool.get("name") == expected_tool for tool in tools),
                f"{resolved_task.upper()} adapter config must expose {expected_tool}",
            )
        require(
            "COPY --from=sure_harness_runtime" in dockerfile_text,
            "adapter Dockerfile must copy the locked Harness Runtime with the sure_harness_runtime build context",
        )
        build_context = str(value.get("harness_runtime_build_context") or "directory")
        if build_context.startswith("docker-image://"):
            require(
                re.fullmatch(r"docker-image://.+@sha256:[0-9a-f]{64}", build_context) is not None,
                "image-backed Harness Runtime build context must be digest-pinned",
            )
        for key in ("model_py", "init_py", "server_py", "config_yaml", "model_spec", "validate_py", "mcp_smoke_py"):
            declared = Path(str(value.get(key, "")))
            require(
                declared.name in dockerfile_text,
                f"adapter Dockerfile must COPY {declared.name} into the image; the manifest declares {key} but "
                "the Dockerfile does not reference it. Fix the COPY line (templates/Dockerfile.sure), rebuild the "
                "adapter image, and re-run the import gate",
            )
        model_source = Path(str(value["model_py"])).read_text(encoding="utf-8")
        require("NotImplementedError" not in model_source and "TODO" not in model_source, "model.py is still a scaffold")
    elif kind == "runtime_inventory":
        require(value.get("schema") == "sure.onboard.runtime_inventory.v2", "runtime inventory schema is incompatible with sure_eval")
        require(value.get("status") == "ready", "runtime inventory is not ready")
        container = value.get("container_runtime") or {}
        model_runtime = value.get("model_runtime") or {}
        policy = value.get("policy") or {}
        require("@sha256:" in str(container.get("target_image_ref", "")), "runtime image must be digest-pinned")
        require(policy.get("eval_runtime") == "container_only", "Eval runtime must be container_only")
        require(policy.get("host_python_fallback") is False, "host Python fallback must be disabled")
        require(policy.get("nfs_models_mutable_by_eval") is False, "Eval must not mutate the approved model bundle")
        model_python = str(model_runtime.get("python_executable") or "")
        container_python = str(container.get("python_executable") or "")
        require(
            PurePosixPath(model_python).is_absolute(),
            "model runtime Python executable must be absolute",
        )
        require(
            PurePosixPath(container_python).is_absolute(),
            "container runtime Python executable must be absolute",
        )
        require(
            model_python == container_python,
            "model and container runtime Python executables must match",
        )
        require(
            PurePosixPath(str(container.get("working_dir") or "")).is_absolute(),
            "container working directory must be absolute",
        )
        server_command = container.get("server_command")
        require(
            isinstance(server_command, list)
            and len(server_command) >= 2
            and server_command[0] == container_python
            and all(isinstance(item, str) and item for item in server_command),
            "container server_command must start with its Python executable",
        )
        require(
            PurePosixPath(server_command[1]).is_absolute(),
            "container server path must be absolute",
        )
        harness = value.get("harness_runtime") if isinstance(value.get("harness_runtime"), dict) else {}
        require(harness.get("required") is True, "trans adapter image must embed the Harness Runtime")
        require(harness.get("schema") == "sure.harness.runtime.binding.v1", "required Harness Runtime binding must use the common schema")
        require(
            all(harness.get(key) for key in ("runtime_id", "lock_sha256", "python_executable", "manifest_path", "runtime_root")),
            "required Harness Runtime binding is missing identity or path fields",
        )
        require(
            not LEGACY_PATH.search(json.dumps(harness, ensure_ascii=False)),
            "host Harness Runtime paths cannot be declared as the container runtime",
        )
        mount_policy = container.get("mount_policy") or {}
        require((mount_policy.get("model_bundle") or {}).get("read_only") is True, "model bundle mount must be read-only")
        require((mount_policy.get("result_workspace") or {}).get("read_only") is False, "result workspace mount must be writable")
    elif kind == "verdict":
        require(value.get("status") == "success", "verdict is not terminal-success")
        readiness = value.get("readiness")
        require(
            isinstance(readiness, dict)
            and readiness.get("bundle_ready") is True
            and readiness.get("registry_ready") is True,
            "verdict readiness must prove bundle and registry readiness",
        )
    elif kind == "deployment_ready":
        require(value.get("schema") == "sure.onboard.deployment_ready.v1", "deployment schema is incompatible with sure_eval")
        if value.get("status") == "blocked":
            require(
                str(value.get("blocked_reason") or "").strip() != "",
                "blocked deployment marker must record why the run stopped",
            )
            blocked_policy = value.get("execution_policy") if isinstance(value.get("execution_policy"), dict) else {}
            require(
                blocked_policy.get("container_only") is False,
                "blocked deployment marker must not claim container-only Eval readiness",
            )
            print(f"{kind} OK: {path}")
            return 0
        require(value.get("status") == "ready", "deployment is not ready")
        require(
            value.get("integrity_profile") == "manifest-complete-v1",
            "ready deployment must use the manifest-complete-v1 integrity profile",
        )
        require("@sha256:" in str(value.get("target_image_ref", "")), "deployment image must be digest-pinned")
        model_dir = harness_model_dir(run_dir)
        model_copy = model_dir / "artifacts" / "deployment_ready.json"
        require(
            model_copy.is_file() and model_copy.read_bytes() == path.read_bytes(),
            "deployment_ready.json must be written identically to the run and model bundle",
        )
        policy = value.get("execution_policy") if isinstance(value.get("execution_policy"), dict) else {}
        require(
            policy.get("container_only") is True
            and policy.get("nfs_models_read_only") is True
            and policy.get("host_python_fallback") is False
            and policy.get("approved_image_override") is False,
            "final execution policy must be container-only with NFS read-only and no host fallback",
        )
        hashes = value.get("required_artifact_sha256")
        require(isinstance(hashes, dict) and hashes, "required_artifact_sha256 must list finalized artifacts")
        for raw, expected in hashes.items():
            relative = Path(str(raw))
            require(not relative.is_absolute() and ".." not in relative.parts, f"invalid finalized artifact path: {raw}")
            artifact = (model_dir / relative).resolve()
            require(
                artifact.is_relative_to(model_dir) and artifact.is_file() and sha256_file(artifact) == expected,
                f"finalized artifact hash mismatch: {raw}",
            )
        bundle_hash = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        require(value.get("bundle_identity_sha256") == bundle_hash, "bundle_identity_sha256 does not match finalized artifact hashes")
        manifest = read_object(model_dir / "artifacts" / "artifact_manifest.json")
        require(
            manifest.get("status") == "finalized" and manifest.get("model_dir") == ".",
            "artifact_manifest.json must be refreshed into portable finalized form",
        )
        manifest_artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
        required_entries = manifest_artifacts.get("required") if isinstance(manifest_artifacts.get("required"), dict) else {}
        declared_paths = {
            str(entry.get("path"))
            for entry in required_entries.values()
            if isinstance(entry, dict) and entry.get("path") != "artifacts/deployment_ready.json"
        }
        require(
            declared_paths == set(hashes),
            "deployment_ready hashes must cover exactly every artifact_manifest required file except deployment_ready.json",
        )
        validate_fixture_manifest(read_object(model_dir / "artifacts" / "fixture_manifest.json"))
        package = read_object(model_dir / "artifacts" / "package_gate.json")
        require(package.get("status") == "passed", "package_gate must be passed")
        gate_readiness = package.get("readiness") if isinstance(package.get("readiness"), dict) else {}
        require(
            gate_readiness.get("bundle_ready") is True and gate_readiness.get("registry_ready") is True,
            "package_gate readiness must prove bundle and registry readiness",
        )
        docker = package.get("docker") if isinstance(package.get("docker"), dict) else {}
        dockerfile = model_dir / str(docker.get("dockerfile_path") or "Dockerfile.sure")
        require(
            dockerfile.is_file() and docker.get("dockerfile_sha256") == sha256_file(dockerfile),
            "package gate Dockerfile hash does not match the model bundle",
        )
        inventory = read_object(model_dir / "artifacts" / "runtime_inventory.json")
        verdict = read_object(model_dir / "artifacts" / "verdict.json")
        timeline = [
            ("artifact_manifest", artifact_time(manifest, "artifact_manifest")),
            ("package_gate", artifact_time(package, "package_gate")),
            ("runtime_inventory", artifact_time(inventory, "runtime_inventory")),
            ("verdict", artifact_time(verdict, "verdict")),
            ("deployment_ready", artifact_time(value, "deployment_ready")),
        ]
        for (earlier_name, earlier), (later_name, later) in zip(timeline, timeline[1:]):
            require(earlier < later, f"terminal timeline is inverted: {earlier_name} must precede {later_name}")
        declared_binding = value.get("harness_runtime") if isinstance(value.get("harness_runtime"), dict) else {}
        if declared_binding:
            source_binding = inventory.get("harness_runtime") if isinstance(inventory.get("harness_runtime"), dict) else {}
            projected = {key: source_binding.get(key) for key in declared_binding}
            require(declared_binding == projected, "deployment Harness Runtime binding disagrees with runtime inventory")
            require(
                declared_binding.get("schema") == "sure.harness.runtime.binding.v1" and declared_binding.get("runtime_id"),
                "ready deployment must expose the common Harness Runtime binding",
            )
            require(
                not LEGACY_PATH.search(json.dumps(declared_binding, ensure_ascii=False)),
                "deployment Harness Runtime binding must reference an in-image runtime, not host paths",
            )
        portable = [
            read_object(model_dir / "artifacts" / name)
            for name in (
                "runtime_inventory.json",
                "package_gate.json",
                "artifact_manifest.json",
                "deployment_ready.json",
                "mcp_result.json",
            )
        ]
        require(
            not LEGACY_PATH.search(json.dumps(portable, ensure_ascii=False)),
            "finalized deployment sidecars contain legacy host absolute paths",
        )
    print(f"{kind} OK: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error))
        raise SystemExit(1)
