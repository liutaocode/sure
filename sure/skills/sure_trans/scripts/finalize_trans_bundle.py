#!/usr/bin/env python3
"""Finalize a portable SURE trans model bundle.

Seals the harness-owned `sure/models/<model_name>/` bundle with the same
product layout and terminal sidecars as /sure_onboard:
  model.spec.yaml, model.py, server.py, __init__.py, validate.py, config.yaml
  fixture/<task>/ (fixture + gt.jsonl)
  artifacts/{package_gate, runtime_inventory, verdict, docker_registry_result,
             artifact_manifest, deployment_ready}.json
The last written file is deployment_ready.json, written identically to the run
directory and the model bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from check_artifact import validate_fixture_manifest


REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_ROOT = REPO_ROOT / "sure" / "models"

REQUIRED_ARTIFACTS = [
    "trans_input_resolved.json",
    "inference_dependency_report.json",
    "framework_detection.json",
    "fixture_manifest.json",
    "execution_compat.json",
    "original_inference_result.json",
    "source_image_result.json",
    "model_payload_manifest.json",
    "adapter_manifest.json",
    "adapter_image_result.json",
    "import_result.json",
    "load_result.json",
    "infer_result.json",
    "contract_result.json",
    "mcp_result.json",
    "equivalence_result.json",
    "docker_registry_result.json",
    "runtime_inventory.json",
    "verdict.json",
    "sample_output.json",
]

WRAPPER_FILES = ("model.py", "server.py", "__init__.py", "validate.py", "config.yaml", "model.spec.yaml", "Dockerfile.sure")
CORE_MANIFEST_FILES = WRAPPER_FILES

TERMINAL_FILES = (
    "package_gate.json",
    "runtime_inventory.json",
    "verdict.json",
    "docker_registry_result.json",
    "artifact_manifest.json",
    "deployment_ready.json",
)
HOST_SHARED_PATH = re.compile(r"/(?:mnt/cloudstorfs|hpc_stor\d+|hpc_\d+)/")
KWS_OPERATING_THRESHOLD = 0.5


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def write_identical(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def resolve_model_dir(resolved: dict) -> Path:
    path_policy = resolved.get("path_policy") if isinstance(resolved.get("path_policy"), dict) else {}
    allowed_root = Path(str(path_policy.get("allowed_model_root") or MODELS_ROOT)).expanduser().resolve()
    declared_model_dir = Path(str(resolved.get("model_dir") or "")).expanduser()
    if declared_model_dir.is_symlink():
        raise ValueError("model_dir must be a real harness-owned directory, not a symlink")
    model_dir = declared_model_dir
    try:
        model_dir = model_dir.resolve()
    except OSError:
        model_dir = model_dir.absolute()
    if model_dir.parent != allowed_root or model_dir.name != str(resolved["model_name"]):
        raise ValueError(
            f"model_dir must be the harness-owned bundle {allowed_root / str(resolved['model_name'])}; "
            f"refusing external destination {model_dir}"
        )
    return model_dir


def ensure_safe_bundle_parent(model_dir: Path, destination: Path) -> None:
    model_dir = model_dir.resolve()
    try:
        relative_parent = destination.parent.relative_to(model_dir)
    except ValueError as error:
        raise ValueError(f"bundle destination escapes model_dir: {destination}") from error
    current = model_dir
    for part in relative_parent.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"bundle destination parent must not be a symlink: {current}")
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise ValueError(f"bundle destination parent is not a directory: {current}")
    if destination.is_symlink():
        raise ValueError(f"bundle destination must not be a symlink: {destination}")


def stage_wrapper(adapter_dir: Path, model_dir: Path) -> None:
    for name in WRAPPER_FILES:
        source = adapter_dir / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"adapter file missing: {source}")
        destination = model_dir / name
        ensure_safe_bundle_parent(model_dir, destination)
        shutil.copy2(source, destination)


def clear_directory(path: Path, controlled_root: Path) -> None:
    if controlled_root.is_symlink() or path.is_symlink():
        raise ValueError(f"bundle fixture directory must not be a symlink: {path}")
    resolved = path.resolve()
    root = controlled_root.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"bundle fixture directory must stay below {root}: {path}")
    if not path.exists():
        path.mkdir(parents=True)
        return
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise ValueError(f"bundle fixture contains unsupported entry: {child}")


def fixture_tree_identity(staged_dir: Path, relative_files: set[Path]) -> str:
    hashes = {
        relative.as_posix(): sha256(staged_dir / relative)
        for relative in sorted(relative_files, key=lambda item: item.as_posix())
    }
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stage_kws_fixture(
    run_dir: Path,
    model_dir: Path,
    resolved: dict,
    fixture_manifest: dict,
) -> None:
    samples = fixture_manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("KWS fixture manifest must declare samples")
    declared_staged_dir = Path(str(fixture_manifest["staged_dir"])).resolve()
    source_candidates = (run_dir / "fixture" / "kws", declared_staged_dir)
    staged_dir = next(
        (candidate.resolve() for candidate in source_candidates if (candidate / "gt.jsonl").is_file()),
        None,
    )
    if staged_dir is None:
        raise ValueError("prepared KWS fixture source is missing")
    source_samples: list[dict] = []
    relative_files = {Path("gt.jsonl")}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("KWS fixture samples must be objects")
        relative_audio = Path(str(sample.get("audio") or ""))
        if not str(relative_audio) or relative_audio.is_absolute() or ".." in relative_audio.parts:
            raise ValueError(f"KWS fixture audio path must be portable: {relative_audio}")
        source_samples.append(
            {
                **sample,
                "audio": relative_audio.as_posix(),
                "audio_path": str(staged_dir / relative_audio),
            }
        )
        relative_files.add(relative_audio)
    annotation_source = fixture_manifest.get("annotation_source")
    if not isinstance(annotation_source, dict):
        raise ValueError("KWS fixture manifest is missing annotation_source")
    source_manifest = {
        **fixture_manifest,
        "model_dir": str(run_dir if staged_dir.is_relative_to(run_dir) else model_dir),
        "staged_dir": str(staged_dir),
        "staged_path": str(staged_dir),
        "gt_jsonl": str(staged_dir / "gt.jsonl"),
        "samples": source_samples,
        "sha256": fixture_tree_identity(staged_dir, relative_files),
        "gt_sha256": sha256(staged_dir / "gt.jsonl"),
        "expected_sha256": sha256(staged_dir / "gt.jsonl"),
        "size_bytes": sum((staged_dir / relative).stat().st_size for relative in relative_files),
        "sample_count": len(source_samples),
        "annotation_source": {
            **annotation_source,
            "staged_path": str(staged_dir / "gt.jsonl"),
        },
    }
    validate_fixture_manifest(source_manifest)

    fixture_dir = model_dir / "fixture" / "kws"
    clear_directory(fixture_dir, model_dir / "fixture")
    for source in sorted(staged_dir.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"KWS fixture tree must not contain symlinks: {source}")
        relative = source.relative_to(staged_dir)
        destination = fixture_dir / relative
        if source.is_dir():
            ensure_safe_bundle_parent(model_dir, destination / ".fixture-placeholder")
            continue
        if not source.is_file():
            raise ValueError(f"KWS fixture tree contains an unsupported entry: {source}")
        ensure_safe_bundle_parent(model_dir, destination)
        shutil.copy2(source, destination)

    finalized_samples = [
        {
            **sample,
            "audio_path": str(fixture_dir / str(sample["audio"])),
            "sha256": sha256(fixture_dir / str(sample["audio"])),
            "size_bytes": (fixture_dir / str(sample["audio"])).stat().st_size,
        }
        for sample in source_samples
    ]
    finalized_annotation_source = {
        **annotation_source,
        "staged_path": str(fixture_dir / "gt.jsonl"),
        "bundled_path": str(fixture_dir / "gt.jsonl"),
    }
    finalized_manifest = {
        **fixture_manifest,
        "model_id": resolved["model_name"],
        "model_name": resolved["model_name"],
        "model_dir": str(model_dir),
        "task_type": "kws",
        "staged_dir": str(fixture_dir),
        "staged_path": str(fixture_dir),
        "gt_jsonl": str(fixture_dir / "gt.jsonl"),
        "samples": finalized_samples,
        "sha256": fixture_tree_identity(fixture_dir, relative_files),
        "gt_sha256": sha256(fixture_dir / "gt.jsonl"),
        "expected_sha256": sha256(fixture_dir / "gt.jsonl"),
        "size_bytes": sum((fixture_dir / relative).stat().st_size for relative in relative_files),
        "sample_count": len(finalized_samples),
        "annotation_source": finalized_annotation_source,
    }
    write_identical(
        run_dir / "artifacts" / "fixture_manifest.json",
        json_bytes(finalized_manifest),
    )
    validate_fixture_manifest(finalized_manifest)


def stage_fixture(run_dir: Path, model_dir: Path, resolved: dict) -> None:
    fixture_manifest = read_object(run_dir / "artifacts" / "fixture_manifest.json")
    task = str(resolved.get("task_type") or "asr").lower()
    if task == "kws":
        stage_kws_fixture(run_dir, model_dir, resolved, fixture_manifest)
        return
    annotation_source_value = fixture_manifest.get("annotation_source")
    if not isinstance(annotation_source_value, dict):
        raise ValueError("fixture manifest is missing annotation_source")
    samples_value = fixture_manifest.get("samples")
    if not isinstance(samples_value, list) or not samples_value or not isinstance(samples_value[0], dict):
        raise ValueError("fixture manifest must declare at least one sample")
    declared_staged_dir = Path(str(fixture_manifest["staged_dir"])).resolve()
    staged_name = Path(str(fixture_manifest["staged_path"])).name
    expected_name = Path(str(annotation_source_value.get("staged_path") or "")).name
    if not expected_name:
        raise ValueError("fixture annotation_source is missing staged_path")
    source_candidates = (run_dir / "fixture" / task, declared_staged_dir)
    staged_dir = next(
        (
            candidate.resolve()
            for candidate in source_candidates
            if (candidate / staged_name).is_file()
            and (candidate / expected_name).is_file()
            and (candidate / "gt.jsonl").is_file()
        ),
        None,
    )
    if staged_dir is None:
        raise ValueError("prepared fixture source is missing from the run and declared staging directory")
    staged = staged_dir / staged_name
    source_manifest = {
        **fixture_manifest,
        "model_dir": str(run_dir if staged_dir.is_relative_to(run_dir) else model_dir),
        "staged_dir": str(staged_dir),
        "staged_path": str(staged),
        "gt_jsonl": str(staged_dir / "gt.jsonl"),
        "samples": [
            {
                **samples_value[0],
                "audio": staged.name,
                "audio_path": str(staged),
            }
        ],
        "annotation_source": {
            **annotation_source_value,
            "staged_path": str(staged_dir / expected_name),
        },
    }
    # The prepare-fixture gate is the only place allowed to establish ground
    # truth. Revalidate its complete manifest before copying anything into the
    # sealed bundle; finalization must never infer labels from model output.
    validate_fixture_manifest(source_manifest)
    fixture_dir = model_dir / "fixture" / task
    clear_directory(fixture_dir, model_dir / "fixture")
    destination = fixture_dir / staged.name
    for source in sorted(staged_dir.iterdir()):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"fixture staged directory may contain regular files only: {source}")
        fixture_destination = fixture_dir / source.name
        ensure_safe_bundle_parent(model_dir, fixture_destination)
        shutil.copy2(source, fixture_destination)
    expected_destination = fixture_dir / expected_name
    if not expected_destination.is_file():
        raise ValueError(f"fixture annotation sidecar was not staged: {expected_destination}")
    gt_jsonl = fixture_dir / "gt.jsonl"
    if not gt_jsonl.is_file():
        raise ValueError(f"prepared fixture ground truth is missing: {gt_jsonl}")
    annotation_source = dict(fixture_manifest["annotation_source"])
    annotation_source["staged_path"] = str(expected_destination)
    annotation_source["bundled_path"] = str(expected_destination)
    finalized_manifest = {
        **fixture_manifest,
        "model_id": resolved["model_name"],
        "model_name": resolved["model_name"],
        "model_dir": str(model_dir),
        "task_type": task,
        "source_dir": str(fixture_manifest.get("source_dir") or ""),
        "staged_dir": str(fixture_dir),
        "gt_jsonl": str(gt_jsonl),
        "samples": [
            {
                "key": staged.stem,
                "audio": staged.name,
                "audio_path": str(destination),
                "annotation_fields": list(fixture_manifest["samples"][0].get("annotation_fields") or []),
            }
        ],
        "staged_path": str(destination),
        "sample_count": 1,
        "annotation_source": annotation_source,
        "sha256": sha256(destination),
        "gt_sha256": sha256(gt_jsonl),
        "expected_sha256": sha256(expected_destination),
    }
    write_identical(
        run_dir / "artifacts" / "fixture_manifest.json",
        json_bytes(finalized_manifest),
    )
    validate_fixture_manifest(finalized_manifest)


def sample_value_is_nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def normalized_keyword(value: str) -> str:
    return "".join(value.upper().split())


def validate_kws_sample_output(sample: dict, contract: dict, fixture_manifest: dict) -> None:
    rows = sample.get("rows")
    if not isinstance(rows, list):
        raise ValueError("KWS sample_output.json must be an object with a rows array")
    fixture_samples = fixture_manifest.get("samples")
    if not isinstance(fixture_samples, list):
        raise ValueError("KWS fixture manifest has no samples")
    references = {
        str(item.get("key") or ""): item
        for item in fixture_samples
        if isinstance(item, dict) and item.get("key")
    }
    if len(references) != len(fixture_samples):
        raise ValueError("KWS fixture manifest keys must be non-empty and unique")
    outputs: dict[str, dict] = {}
    required = contract.get("required_fields") if isinstance(contract.get("required_fields"), list) else []
    nonempty = contract.get("nonempty_fields") if isinstance(contract.get("nonempty_fields"), list) else []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"KWS sample_output rows[{index}] must be an object")
        key = str(row.get("key") or "")
        if not key or key in outputs:
            raise ValueError(f"KWS sample_output key is missing or duplicated: {key!r}")
        result = row.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"KWS sample_output result for {key} must be an object")
        for field in required:
            if isinstance(field, str) and field not in result:
                raise ValueError(f"KWS sample_output result for {key} is missing {field}")
        for field in nonempty:
            if isinstance(field, str) and not sample_value_is_nonempty(result.get(field)):
                raise ValueError(f"KWS sample_output result for {key} has empty {field}")
        detected = result.get("detected")
        keyword = result.get("keyword")
        score = result.get("score")
        if not isinstance(detected, bool):
            raise ValueError(f"KWS sample_output detected for {key} must be boolean")
        if keyword is not None and (not isinstance(keyword, str) or not keyword.strip()):
            raise ValueError(f"KWS sample_output keyword for {key} must be string or null")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"KWS sample_output score for {key} must be finite or null")
        if score is not None and not 0 <= float(score) <= 1:
            raise ValueError(f"KWS sample_output score for {key} must be within [0, 1]")
        if detected:
            if not isinstance(keyword, str) or not keyword.strip():
                raise ValueError(f"KWS sample_output detection for {key} requires a keyword")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ValueError(f"KWS sample_output detection for {key} requires a finite score")
            if float(score) < KWS_OPERATING_THRESHOLD:
                raise ValueError(
                    f"KWS sample_output detection for {key} requires score >= {KWS_OPERATING_THRESHOLD}"
                )
        elif keyword is not None:
            raise ValueError(f"KWS sample_output rejection for {key} requires keyword=null")
        elif score is not None and float(score) >= KWS_OPERATING_THRESHOLD:
            raise ValueError(
                f"KWS sample_output rejection for {key} requires score < {KWS_OPERATING_THRESHOLD}"
            )
        outputs[key] = result
    missing = sorted(set(references) - set(outputs))
    extra = sorted(set(outputs) - set(references))
    if missing or extra:
        raise ValueError(f"KWS sample_output key mismatch: missing={missing}, extra={extra}")
    for key, reference in references.items():
        result = outputs[key]
        expected_detected = reference.get("expected_detected")
        if not isinstance(expected_detected, bool):
            raise ValueError(f"KWS fixture expected_detected for {key} must be boolean")
        if result["detected"] is not expected_detected:
            raise ValueError(f"KWS sample_output polarity disagrees with fixture for {key}")
        expected_keyword = reference.get("expected_keyword")
        if expected_detected and (
            not isinstance(expected_keyword, str)
            or normalized_keyword(str(result["keyword"])) != normalized_keyword(expected_keyword)
        ):
            raise ValueError(f"KWS sample_output keyword disagrees with fixture for {key}")


def promote_generated_audio(artifacts: Path, raw_path: str) -> str:
    declared = Path(raw_path)
    candidates: list[Path] = []
    if declared.is_absolute():
        if declared.parts[:2] == ("/", "validation"):
            candidates.append(artifacts / "adapter_validation" / Path(*declared.parts[2:]))
        candidates.append(declared)
    else:
        candidates.extend([artifacts / "adapter_validation" / declared, artifacts / declared])
    source_candidate = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source_candidate is None or source_candidate.is_symlink():
        raise ValueError(
            "audio sample_output.json must reference a real generated file; write it below "
            "SURE_VALIDATE_ARTIFACTS_DIR/outputs"
        )
    source = source_candidate.resolve()
    if (artifacts / "adapter_validation" / "outputs").is_symlink() or (artifacts / "outputs").is_symlink():
        raise ValueError("generated audio output roots must not be symlinks")
    validation_outputs = (artifacts / "adapter_validation" / "outputs").resolve()
    artifact_outputs = (artifacts / "outputs").resolve()
    if not source.is_relative_to(validation_outputs) and not source.is_relative_to(artifact_outputs):
        raise ValueError("generated audio must stay below artifacts/adapter_validation/outputs")
    relative = source.relative_to(validation_outputs if source.is_relative_to(validation_outputs) else artifact_outputs)
    destination = artifact_outputs / relative
    ensure_safe_bundle_parent(artifacts, destination)
    if source != destination:
        shutil.copy2(source, destination)
    return (Path("artifacts") / "outputs" / relative).as_posix()


def promote_sample_output(run_dir: Path) -> None:
    artifacts = run_dir / "artifacts"
    candidates = [
        artifacts / "adapter_validation" / "sample_output.json",
        artifacts / "sample_output.json",
    ]
    source = next((path for path in candidates if path.is_file() and not path.is_symlink()), None)
    if source is None:
        raise ValueError(
            "sample_output.json is missing; the infer stage must write "
            "artifacts/adapter_validation/sample_output.json"
        )
    sample = read_object(source)
    adapter = read_object(artifacts / "adapter_manifest.json")
    contract = adapter.get("io_contract") if isinstance(adapter.get("io_contract"), dict) else {}
    if contract.get("output_type") == "keyword_detection":
        validate_kws_sample_output(
            sample,
            contract,
            read_object(artifacts / "fixture_manifest.json"),
        )
        write_identical(artifacts / "sample_output.json", json_bytes(sample))
        return
    required = contract.get("required_fields") if isinstance(contract.get("required_fields"), list) else []
    nonempty = contract.get("nonempty_fields") if isinstance(contract.get("nonempty_fields"), list) else []
    primary = contract.get("primary_field")
    fields = [field for field in (*required, *nonempty, primary) if isinstance(field, str) and field]
    for field in dict.fromkeys(fields):
        if field not in sample:
            raise ValueError(f"sample_output.json is missing io_contract field: {field}")
        if field in nonempty or field == primary:
            if not sample_value_is_nonempty(sample[field]):
                raise ValueError(f"sample_output.json io_contract field is empty: {field}")
    if contract.get("output_type") == "audio":
        audio_path = sample.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError("audio sample_output.json must contain a non-empty audio_path")
        sample["audio_path"] = promote_generated_audio(artifacts, audio_path)
    write_identical(artifacts / "sample_output.json", json_bytes(sample))


def portableize_value(value: object, replacements: list[tuple[str, str]]) -> object:
    if isinstance(value, dict):
        return {key: portableize_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [portableize_value(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    portable = value
    for source, replacement in replacements:
        portable = portable.replace(source, replacement)
    return portable


def project_mcp_result(
    source: Path,
    destination: Path,
    *,
    run_dir: Path,
    model_dir: Path,
    resolved: dict,
) -> None:
    payload = read_object(source)
    roots: list[tuple[str, str]] = [
        (str(run_dir.resolve()), "<run_dir>"),
        (str(model_dir.resolve()), "<model_dir>"),
        (str(REPO_ROOT.resolve()), "<repo_root>"),
    ]
    for field in (
        "build_context",
        "model_path",
        "inference_entrypoint",
        "fixture_path",
        "model_dir",
    ):
        raw = str(resolved.get(field) or "").strip()
        if raw and Path(raw).is_absolute():
            roots.append((str(Path(raw).expanduser().resolve()), f"<{field}>"))
    replacements = sorted(set(roots), key=lambda item: len(item[0]), reverse=True)
    projected = portableize_value(payload, replacements)
    if not isinstance(projected, dict):
        raise ValueError("portable MCP result projection must remain an object")
    projected["path_policy"] = {
        "portable_projection": True,
        "host_roots_replaced": sorted(replacement for _, replacement in replacements),
    }
    content = json_bytes(projected)
    if HOST_SHARED_PATH.search(content.decode("utf-8")):
        raise ValueError("portable MCP result still contains a host shared-storage path")
    write_identical(destination, content)


def stage_artifacts(run_dir: Path, model_dir: Path) -> dict[str, str]:
    artifacts = run_dir / "artifacts"
    model_artifacts = model_dir / "artifacts"
    resolved = read_object(artifacts / "trans_input_resolved.json")
    hashes: dict[str, str] = {}
    for name in REQUIRED_ARTIFACTS:
        source = artifacts / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"required artifact missing: {source}")
        destination = model_artifacts / name
        ensure_safe_bundle_parent(model_dir, destination)
        if name == "mcp_result.json":
            project_mcp_result(
                source,
                destination,
                run_dir=run_dir,
                model_dir=model_dir,
                resolved=resolved,
            )
        else:
            shutil.copy2(source, destination)
    for name in ("validation.log",):
        source = artifacts / name
        if source.is_symlink():
            raise ValueError(f"optional validation artifact must not be a symlink: {source}")
        if source.is_file():
            destination = model_artifacts / name
            ensure_safe_bundle_parent(model_dir, destination)
            shutil.copy2(source, destination)
    outputs = artifacts / "outputs"
    if outputs.is_symlink():
        raise ValueError(f"generated outputs directory must not be a symlink: {outputs}")
    if outputs.is_dir():
        unsafe = next((path for path in outputs.rglob("*") if path.is_symlink()), None)
        if unsafe is not None:
            raise ValueError(f"generated outputs must not contain symlinks: {unsafe}")
        destination = model_artifacts / "outputs"
        if destination.is_symlink():
            raise ValueError(f"model artifact outputs directory must not be a symlink: {destination}")
        if destination.exists():
            shutil.rmtree(destination)
        ensure_safe_bundle_parent(model_dir, destination)
        shutil.copytree(outputs, destination)
    return hashes


def write_package_gate(run_dir: Path, model_dir: Path, registry: dict) -> dict:
    adapter_image = read_object(run_dir / "artifacts" / "adapter_image_result.json")
    package = {
        "schema": "sure.onboard.package_gate.v2",
        "generated_at": now_iso(),
        "status": "passed",
        "package_profile": "docker-registry",
        "model_name": str(model_dir.name),
        "model_dir": ".",
        "artifact_manifest_path": "artifacts/artifact_manifest.json",
        "readiness": {
            "local_ready": True,
            "docker_ready": True,
            "registry_ready": registry.get("pull_verified") is True,
            "bundle_ready": True,
        },
        "docker": {
            "dockerfile_path": "Dockerfile.sure",
            "dockerfile_sha256": sha256(model_dir / "Dockerfile.sure"),
            "base_image": adapter_image.get("source_image"),
            "target_image": registry.get("target_image"),
            "target_image_digest": registry.get("target_image_digest"),
            "target_image_ref": registry.get("target_image_ref"),
            "build_result_path": "artifacts/adapter_image_result.json",
            "validation_result_path": "artifacts/contract_result.json",
            "registry_result_path": "artifacts/docker_registry_result.json",
        },
        "notes": "Container-only Eval binding produced by /sure_trans.",
    }
    content = json_bytes(package)
    write_identical(run_dir / "artifacts" / "package_gate.json", content)
    model_output = model_dir / "artifacts" / "package_gate.json"
    ensure_safe_bundle_parent(model_dir, model_output)
    write_identical(model_output, content)
    return package


def write_artifact_manifest(run_dir: Path, model_dir: Path, resolved: dict) -> dict:
    required = {
        name.replace(".", "_").replace("-", "_"): {
            "path": name,
            "description": f"Required model bundle file: {name}.",
        }
        for name in CORE_MANIFEST_FILES
    }
    required.update({
        name.replace(".", "_").replace("-", "_"): {
            "path": f"artifacts/{name}",
            "description": f"Finalized trans artifact: {name}.",
        }
        for name in (*REQUIRED_ARTIFACTS, *TERMINAL_FILES)
    })
    fixture_root = model_dir / "fixture"
    if fixture_root.is_dir():
        for path in sorted(item for item in fixture_root.rglob("*") if item.is_file()):
            relative = path.relative_to(model_dir).as_posix()
            key = f"file:{relative}"
            if key in required:
                raise ValueError(f"duplicate finalized artifact manifest key: {key}")
            required[key] = {
                "path": relative,
                "description": f"Finalized smoke fixture file: {relative}.",
            }
    payload = read_object(model_dir / "artifacts" / "model_payload_manifest.json")
    payload_files = payload.get("files")
    if not isinstance(payload_files, dict) or not payload_files:
        raise ValueError("model payload manifest must list files before finalization")
    for raw_path in sorted(payload_files):
        relative = Path(str(raw_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"model payload path is not portable: {raw_path}")
        target = (model_dir / relative).resolve()
        if not target.is_file() or not target.is_relative_to(model_dir.resolve()):
            raise ValueError(f"model payload file is missing during finalization: {raw_path}")
        path_value = relative.as_posix()
        key = f"file:{path_value}"
        if key in required:
            raise ValueError(f"model payload conflicts with another required bundle file: {path_value}")
        required[key] = {
            "path": path_value,
            "description": f"Staged model payload file: {path_value}.",
        }
    outputs_root = model_dir / "artifacts" / "outputs"
    if outputs_root.is_dir():
        for path in sorted(item for item in outputs_root.rglob("*") if item.is_file()):
            relative = path.relative_to(model_dir).as_posix()
            key = f"file:{relative}"
            if key in required:
                raise ValueError(f"generated output conflicts with another required bundle file: {relative}")
            required[key] = {
                "path": relative,
                "description": f"Generated smoke output file: {relative}.",
            }
    manifest = {
        "schema": "sure.onboard.artifact_manifest.v1",
        "model_dir": ".",
        "model_id": resolved["model_name"],
        "model_name": resolved["model_name"],
        "phase": "deployment_ready",
        "status": "finalized",
        "generated_at": now_iso(),
        "timestamp": now_iso(),
        "artifacts": {"required": required, "conditional": {}, "optional": {}},
    }
    content = json_bytes(manifest)
    path = model_dir / "artifacts" / "artifact_manifest.json"
    ensure_safe_bundle_parent(model_dir, path)
    write_identical(path, content)
    write_identical(run_dir / "artifacts" / "artifact_manifest.json", content)
    return manifest


def finalized_hashes(model_dir: Path) -> dict[str, str]:
    manifest = read_object(model_dir / "artifacts" / "artifact_manifest.json")
    required = manifest.get("artifacts", {}).get("required") if isinstance(manifest.get("artifacts"), dict) else {}
    if not isinstance(required, dict) or not required:
        raise ValueError("finalized artifact manifest has no required entries")
    hashes: dict[str, str] = {}
    for entry in required.values():
        if not isinstance(entry, dict):
            raise ValueError("finalized artifact manifest entry is invalid")
        raw_path = str(entry.get("path") or "")
        relative = Path(raw_path)
        if not raw_path or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"finalized artifact path is not portable: {raw_path}")
        if relative.as_posix() == "artifacts/deployment_ready.json":
            continue
        path = (model_dir / relative).resolve()
        if not path.is_file() or not path.is_relative_to(model_dir.resolve()):
            raise ValueError(f"finalized required artifact is missing: {raw_path}")
        hashes[relative.as_posix()] = sha256(path)
    return hashes


def regenerate_terminal_evidence(run_dir: Path) -> None:
    """Rewrite the inventory and verdict so the terminal timeline is ordered.

    The bundle they describe is only complete once the wrapper, fixture and
    payload are staged, so evidence written before that has an earlier
    timestamp than the manifest it is supposed to follow. Rerun the writers
    and their gates on the sealed bundle instead of trusting the run copies.
    """
    scripts = Path(__file__).resolve().parent
    prior_inventory = read_object(run_dir / "artifacts" / "runtime_inventory.json")
    container = prior_inventory.get("container_runtime") if isinstance(prior_inventory.get("container_runtime"), dict) else {}
    inventory_command = [
        sys.executable,
        str(scripts / "write_runtime_inventory.py"),
        "--run-dir",
        str(run_dir),
        "--gpu-required" if container.get("gpu_required") is True else "--no-gpu-required",
    ]
    commands = [
        [sys.executable, str(scripts / "check_artifact.py"), "--run-dir", str(run_dir), "--produces", str(run_dir / "artifacts" / "model_payload_manifest.json"), "--kind", "model_payload"],
        inventory_command,
        [sys.executable, str(scripts / "check_artifact.py"), "--run-dir", str(run_dir), "--produces", str(run_dir / "artifacts" / "runtime_inventory.json"), "--kind", "runtime_inventory"],
        [sys.executable, str(scripts / "write_verdict.py"), "--run-dir", str(run_dir)],
        [sys.executable, str(scripts / "check_artifact.py"), "--run-dir", str(run_dir), "--produces", str(run_dir / "artifacts" / "verdict.json"), "--kind", "verdict"],
    ]
    for command in commands:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"terminal evidence regeneration failed: {detail}")


def build_deployment_ready(run_dir: Path, model_dir: Path, resolved: dict, registry: dict, package: dict) -> dict:
    inventory = read_object(model_dir / "artifacts" / "runtime_inventory.json")
    verdict = read_object(model_dir / "artifacts" / "verdict.json")
    if verdict.get("status") not in {"passed", "success", "PASS", "PASSED", "pass"}:
        raise ValueError("verdict must be terminal-success before finalizing")
    if inventory.get("status") != "ready":
        raise ValueError("runtime inventory must be ready before finalizing")
    if registry.get("status") != "passed" or registry.get("pull_verified") is not True:
        raise ValueError("docker-registry bundle requires passed registry push and digest pull verification")

    hashes = finalized_hashes(model_dir)
    bundle_identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    deployment = {
        "schema": "sure.onboard.deployment_ready.v1",
        "integrity_profile": "manifest-complete-v1",
        # A trans payload is always copied into the bundle, so every weight
        # file is covered by required_artifact_sha256.
        "weights_integrity": "bundled",
        "generated_at": now_iso(),
        "status": "ready",
        "model_name": str(resolved["model_name"]),
        "package_profile": str(package.get("package_profile") or "docker-registry"),
        "target_image": registry.get("target_image"),
        "target_image_digest": registry.get("target_image_digest"),
        "target_image_ref": registry.get("target_image_ref"),
        "runtime_inventory": "artifacts/runtime_inventory.json",
        "package_gate": "artifacts/package_gate.json",
        "verdict": "artifacts/verdict.json",
        "artifact_manifest": "artifacts/artifact_manifest.json",
        "required_artifact_sha256": hashes,
        "bundle_identity_sha256": bundle_identity,
        "execution_policy": {
            "container_only": True,
            "nfs_models_read_only": True,
            "host_python_fallback": False,
            "approved_image_override": False,
        },
    }
    harness = inventory.get("harness_runtime") if isinstance(inventory.get("harness_runtime"), dict) else {}
    if harness.get("required") is True:
        deployment["harness_runtime"] = {
            key: harness.get(key)
            for key in (
                "schema",
                "runtime_id",
                "runtime_type",
                "python_executable",
                "python_version",
                "python_abi",
                "lock_sha256",
                "manifest_path",
                "runtime_root",
                "materialization",
            )
            if harness.get(key) is not None
        }
    return deployment


def build_blocked_marker(run_dir: Path, resolved: dict, reason: str) -> dict:
    """Terminal marker for a run that stopped before the bundle was sealed.

    Hashes whatever terminal evidence does exist so the marker still identifies
    the run, and pins execution_policy.container_only to False so nothing
    downstream can read it as an Eval-ready bundle.
    """
    artifacts = run_dir / "artifacts"
    hashes = {
        f"artifacts/{name}": sha256(artifacts / name)
        for name in TERMINAL_FILES
        if name != "deployment_ready.json" and (artifacts / name).is_file()
    }
    return {
        "schema": "sure.onboard.deployment_ready.v1",
        "integrity_profile": "partial-run-v1",
        "generated_at": now_iso(),
        "status": "blocked",
        "blocked_reason": reason,
        "model_name": str(resolved["model_name"]),
        "package_profile": "docker-registry",
        "required_artifact_sha256": hashes,
        "bundle_identity_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "execution_policy": {
            "container_only": False,
            "nfs_models_read_only": True,
            "host_python_fallback": False,
            "approved_image_override": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--blocked",
        metavar="REASON",
        help="seal the run as blocked instead of ready, recording REASON",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    artifacts = run_dir / "artifacts"
    resolved = read_object(artifacts / "trans_input_resolved.json")
    if args.blocked:
        marker = build_blocked_marker(run_dir, resolved, args.blocked)
        write_identical(artifacts / "deployment_ready.json", json_bytes(marker))
        print(artifacts / "deployment_ready.json")
        return 0
    registry = read_object(artifacts / "docker_registry_result.json")
    model_dir = resolve_model_dir(resolved)
    model_artifacts = model_dir / "artifacts"
    ensure_safe_bundle_parent(model_dir, model_artifacts / ".artifact-placeholder")

    stage_wrapper(run_dir / "adapter", model_dir)
    stage_fixture(run_dir, model_dir, resolved)
    promote_sample_output(run_dir)
    stage_artifacts(run_dir, model_dir)
    write_artifact_manifest(run_dir, model_dir, resolved)
    package = write_package_gate(run_dir, model_dir, registry)
    regenerate_terminal_evidence(run_dir)
    stage_artifacts(run_dir, model_dir)
    deployment = build_deployment_ready(run_dir, model_dir, resolved, registry, package)
    content = json_bytes(deployment)
    model_deployment = model_artifacts / "deployment_ready.json"
    ensure_safe_bundle_parent(model_dir, model_deployment)
    write_identical(model_deployment, content)
    write_identical(artifacts / "deployment_ready.json", content)
    print(artifacts / "deployment_ready.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
