#!/usr/bin/env python3
"""Stage run artifacts into the model-local artifacts directory.

The Sure run directory is transient; the onboarded model directory is the
durable product. This helper is intentionally narrow: it copies already-created
run artifacts into ``sure/models/<model_name>/artifacts/`` and writes the
preferred ``artifact_manifest.json`` shape. It does not create wrappers, specs,
validation results, weights, or verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from structured_segments import is_structured_task, validate_structured_rows
from tse_contract import validate_output_object as validate_tse_output_object

CORE_FILES = ["model.spec.yaml", "model.py", "server.py", "__init__.py", "validate.py", "config.yaml"]
REQUIRED_RUN_ARTIFACTS = [
    "model_input_resolved.json",
    "context_selection.json",
    "repo_summary.json",
    "classification.json",
    "backend_choice.json",
    "build_plan.json",
    "spec_validation.json",
    "fixture_manifest.json",
    "build_env_result.json",
    "weights_manifest.json",
    "env_compat_result.json",
    "import_result.json",
    "load_result.json",
    "infer_result.json",
    "contract_result.json",
    "wrapper_manifest.json",
]
OPTIONAL_RUN_ARTIFACTS = [
    "build.log",
    "validation.log",
    "sample_output.json",
    "sample_outputs.jsonl",
    "local_env.json",
    "requirements.lock",
    "uv.lock",
    "pixi.lock",
    "patch_report.json",
    "failure_classification.json",
    "retry_recommendation.json",
    "escalation.json",
	"docker_build_result.json",
	"docker_validation.json",
	"docker_registry_result.json",
    "package_gate.json",
    "verdict.json",
]
MODEL_RUNTIME_ARTIFACT = "model_runtime_manifest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object.")
    return data


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return False


def copy_artifact(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and same_file(source, dest):
        return
    shutil.copy2(source, dest)


def canonical_task(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"speech_enhancement", "acoustic_noise_suppression"}:
        return "se"
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
        return "tse"
    return normalized


def validated_output_files(root: Path, *, require_pcm_wav: bool) -> list[Path]:
    if root.is_symlink():
        raise ValueError(f"generated outputs root must not be a symlink: {root}")
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"generated outputs root must be a directory: {root}")
    resolved_root = root.resolve()
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"generated outputs must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"generated outputs contain an unsafe entry: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"generated output is empty: {path}")
        if require_pcm_wav:
            try:
                with wave.open(str(path), "rb") as handle:
                    if (
                        handle.getcomptype() != "NONE"
                        or handle.getnchannels() < 1
                        or handle.getsampwidth() not in {1, 2, 3, 4}
                        or handle.getframerate() < 1
                        or handle.getnframes() < 1
                    ):
                        raise ValueError(f"SE generated output must be a non-empty PCM WAV: {path}")
            except (EOFError, OSError, wave.Error) as error:
                raise ValueError(f"SE generated output must be a readable PCM WAV: {path}: {error}") from error
        files.append(path)
    return files


def stage_output_tree(run_artifacts: Path, model_artifacts: Path, *, task: str) -> list[str]:
    source = run_artifacts / "outputs"
    destination = model_artifacts / "outputs"
    require_pcm_wav = task in {"se", "tse"}
    if source.exists() or source.is_symlink():
        validated_output_files(source, require_pcm_wav=require_pcm_wav)
        if not same_file(source, destination):
            if destination.is_symlink():
                raise ValueError(f"model outputs destination must not be a symlink: {destination}")
            if destination.exists():
                if not destination.is_dir() or not destination.resolve().is_relative_to(model_artifacts.resolve()):
                    raise ValueError(f"model outputs destination is unsafe: {destination}")
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
    output_files = validated_output_files(destination, require_pcm_wav=require_pcm_wav)
    return [path.relative_to(model_artifacts).as_posix() for path in output_files]


def validate_se_sample_evidence(model_artifacts: Path) -> None:
    output_root_path = model_artifacts / "outputs"
    if output_root_path.is_symlink() or not output_root_path.is_dir():
        raise ValueError("SE bundled outputs must be a real directory")
    output_root = output_root_path.resolve()
    evidence_paths = [model_artifacts / "sample_output.json", model_artifacts / "sample_outputs.jsonl"]
    for evidence_path in evidence_paths:
        if not evidence_path.is_file():
            continue
        if evidence_path.name.endswith(".jsonl"):
            documents = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            documents = [read_json(evidence_path)]
        for document in documents:
            output = document.get("output") if isinstance(document.get("output"), dict) else document
            value = output.get("audio_path") if isinstance(output, dict) else None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SE sample evidence requires audio_path: {evidence_path}")
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("artifacts", "outputs"):
                raise ValueError(f"SE sample evidence audio_path must be portable: {value}")
            resolved = (model_artifacts.parent / relative).resolve()
            if not resolved.is_file() or not resolved.is_relative_to(output_root):
                raise ValueError(f"SE sample evidence audio_path is missing from model outputs: {value}")


def validate_structured_sample_evidence(model_artifacts: Path, *, task: str) -> None:
    sample_output_path = model_artifacts / "sample_output.json"
    sample_outputs_path = model_artifacts / "sample_outputs.jsonl"
    fixture_manifest_path = model_artifacts / "fixture_manifest.json"
    for path in (sample_output_path, sample_outputs_path, fixture_manifest_path):
        if not path.is_file():
            raise ValueError(f"SD/SA-ASR staging requires {path.name}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(sample_outputs_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{sample_outputs_path}:{line_no} must be a JSON object")
        rows.append(row)
    manifest = read_json(fixture_manifest_path)
    violations: list[str] = []
    if canonical_task(manifest.get("task_type")) != task:
        violations.append("fixture manifest task_type disagrees with SD/SA-ASR staging task")
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
    first_output = read_json(sample_output_path)
    if rows and rows[0].get("output") != first_output:
        violations.append("sample_output.json must equal the first structured output row")
    if violations:
        raise ValueError("invalid SD/SA-ASR sample evidence: " + "; ".join(violations))


def validate_tse_sample_evidence(model_artifacts: Path) -> None:
    output_root_path = model_artifacts / "outputs"
    sample_output_path = model_artifacts / "sample_output.json"
    sample_outputs_path = model_artifacts / "sample_outputs.jsonl"
    fixture_manifest_path = model_artifacts / "fixture_manifest.json"
    for path in (sample_output_path, sample_outputs_path, fixture_manifest_path):
        if not path.is_file():
            raise ValueError(f"TSE staging requires {path.name}")
    manifest = read_json(fixture_manifest_path)
    if canonical_task(manifest.get("task_type")) != "tse":
        raise ValueError("fixture manifest task_type disagrees with TSE staging task")
    fixture_samples = manifest.get("samples") if isinstance(manifest.get("samples"), list) else []
    expected = [
        str(item.get("key") or item.get("sample_id") or "")
        for item in fixture_samples
        if isinstance(item, dict)
    ]
    if not 1 <= len(expected) <= 5 or len(expected) != len(fixture_samples):
        raise ValueError("TSE fixture manifest must contain 1 to 5 object samples")
    rows = [
        json.loads(line)
        for line in sample_outputs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_output = read_json(sample_output_path)
    if sample_output.get("rows") != rows:
        raise ValueError("TSE sample_output.json must exactly mirror sample_outputs.jsonl")
    observed: list[str] = []
    referenced_outputs: set[Path] = set()
    output_root = output_root_path.resolve()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{sample_outputs_path}:{index} must be an object")
        unknown_row_fields = sorted(str(field) for field in row if field not in {"key", "sample_id", "result"})
        if unknown_row_fields:
            raise ValueError(
                f"TSE sample row {index} contains unapproved field(s): "
                + ", ".join(unknown_row_fields)
            )
        key = str(row.get("key") or row.get("sample_id") or "").strip()
        if (
            not key
            or key in observed
            or "/" in key
            or "\\" in key
            or any(ord(character) < 32 or character.isspace() for character in key)
        ):
            raise ValueError(f"TSE sample row {index} has an unsafe or duplicate key")
        observed.append(key)
        result = row.get("result")
        canonical = validate_tse_output_object(result, sample_id=key)
        if result != canonical:
            raise ValueError(f"TSE sample output {key} is not canonical")
        relative = Path(str(canonical["prediction_audio"]))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("artifacts", "outputs"):
            raise ValueError(f"TSE sample output {key} prediction_audio must be portable")
        expected_name = f"{index:02d}-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}.wav"
        if relative != Path("artifacts", "outputs", expected_name):
            raise ValueError(f"TSE sample output {key} prediction_audio must use the harness-assigned path")
        candidate = model_artifacts.parent / relative
        generated = candidate.resolve()
        if candidate.is_symlink() or not generated.is_file() or not generated.is_relative_to(output_root):
            raise ValueError(f"TSE sample output {key} prediction_audio is missing from model outputs")
        current = output_root
        for part in generated.relative_to(output_root).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"TSE sample output {key} prediction_audio traverses a symlink")
        try:
            with wave.open(str(generated), "rb") as handle:
                if (
                    handle.getcomptype() != "NONE"
                    or handle.getnchannels() < 1
                    or handle.getsampwidth() not in {1, 2, 3, 4}
                    or handle.getframerate() < 1
                    or handle.getnframes() < 1
                ):
                    raise ValueError(f"TSE sample output {key} must be a non-empty PCM WAV")
        except (EOFError, OSError, wave.Error) as error:
            raise ValueError(f"TSE sample output {key} is not a readable PCM WAV: {error}") from error
        referenced_outputs.add(generated)
    if observed != expected:
        raise ValueError(f"TSE sample output keys mismatch: expected={expected}, observed={observed}")
    output_entries = list(output_root_path.rglob("*"))
    output_symlinks = [path for path in output_entries if path.is_symlink()]
    if output_symlinks:
        raise ValueError(f"TSE output tree must not contain symlinks: {output_symlinks[0]}")
    actual_outputs = {
        path.resolve()
        for path in output_entries
        if path.is_file()
    }
    if actual_outputs != referenced_outputs:
        extras = sorted(path.name for path in actual_outputs - referenced_outputs)
        missing = sorted(path.name for path in referenced_outputs - actual_outputs)
        raise ValueError(
            f"TSE output tree must exactly match referenced predictions: extras={extras}, missing={missing}"
        )


def artifact_entry(path: str, description: str) -> dict[str, Any]:
    return {"path": path, "description": description}


def infer_model_dir(run_artifacts: Path, explicit_model_dir: str | None) -> tuple[Path, dict[str, Any]]:
    resolved_path = run_artifacts / "model_input_resolved.json"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{resolved_path} is required; run materialize_onboard_inputs.py during LOAD_MODEL_INPUT first."
        )
    resolved = read_json(resolved_path)
    raw_model_dir = explicit_model_dir or resolved.get("model_dir")
    if not raw_model_dir:
        raise ValueError("model_dir is required, either via --model-dir or model_input_resolved.json.")
    return Path(str(raw_model_dir)).expanduser(), resolved


def build_manifest(
    *,
    model_dir: Path,
    resolved: dict[str, Any],
    copied_required: list[str],
    copied_optional: list[str],
    copied_outputs: list[str],
) -> dict[str, Any]:
    required: dict[str, Any] = {
        "spec": artifact_entry("model.spec.yaml", "Model specification."),
        "model_py": artifact_entry("model.py", "SURE ModelWrapper implementation."),
        "server_py": artifact_entry("server.py", "Local serving surface."),
        "package_init": artifact_entry("__init__.py", "Import package marker."),
        "validate_py": artifact_entry("validate.py", "Model-local validation runner."),
        "config": artifact_entry("config.yaml", "Model-local runtime/server config."),
        "manifest": artifact_entry("artifacts/artifact_manifest.json", "This model artifact manifest."),
    }
    for name in copied_required:
        key = name.replace(".", "_").replace("-", "_")
        required[key] = artifact_entry(f"artifacts/{name}", f"Required /sure_onboard run artifact: {name}.")
    for relative in copied_outputs:
        path = f"artifacts/{relative}"
        required[f"file:{path}"] = artifact_entry(path, f"Generated model output: {path}.")

    optional: dict[str, Any] = {}
    for name in copied_optional:
        key = name.replace(".", "_").replace("-", "_")
        optional[key] = artifact_entry(f"artifacts/{name}", f"Optional /sure_onboard run artifact: {name}.")

    return {
        "$schema": "./artifact_manifest.schema.json",
        "model_dir": str(model_dir),
        "instance_id": f"{resolved.get('model_name', model_dir.name)}-onboard",
        "timestamp": now_iso(),
        "model_id": resolved.get("model_id", ""),
        "model_name": resolved.get("model_name", model_dir.name),
        "phase": "local_onboard",
        "status": "staged",
        "artifacts": {
            "required": required,
            "conditional": {},
            "optional": optional,
        },
        "staging": {
            "source": "stage_model_artifacts.py",
            "copied_required_run_artifacts": copied_required,
            "copied_optional_run_artifacts": copied_optional,
            "copied_output_files": copied_outputs,
        },
    }


def main_with_args(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--produces", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument(
        "--allow-missing-run-artifacts",
        action="store_true",
        help="Stage only present run artifacts. Use only for diagnostics; normal SAVE_ARTIFACTS should be strict.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_artifacts = run_dir / "artifacts"
    produces = Path(args.produces).expanduser().resolve()
    try:
        model_dir, resolved = infer_model_dir(run_artifacts, args.model_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"stage_model_artifacts failed: {exc}", file=sys.stderr)
        return 1

    model_artifacts = model_dir / "artifacts"
    missing_core = [name for name in CORE_FILES if not (model_dir / name).exists()]
    if missing_core:
        print(
            "stage_model_artifacts failed: model_dir is missing core local deployment files: "
            + ", ".join(missing_core),
            file=sys.stderr,
        )
        return 1

    task = canonical_task(resolved.get("task_type"))
    required_run_artifacts = list(REQUIRED_RUN_ARTIFACTS)
    if is_structured_task(task) or task == "tse":
        required_run_artifacts.extend(("sample_output.json", "sample_outputs.jsonl"))
    if resolved.get("deployment_type") == "local" and resolved.get("package_profile") == "none":
        required_run_artifacts.append(MODEL_RUNTIME_ARTIFACT)
    missing_required = [name for name in required_run_artifacts if not (run_artifacts / name).exists()]
    if missing_required and not args.allow_missing_run_artifacts:
        print(
            "stage_model_artifacts failed: run artifacts missing required state-machine outputs: "
            + ", ".join(missing_required),
            file=sys.stderr,
        )
        return 1

    copied_required: list[str] = []
    copied_optional: list[str] = []
    for name in required_run_artifacts:
        source = run_artifacts / name
        if source.exists():
            copy_artifact(source, model_artifacts / name)
            copied_required.append(name)
    for name in OPTIONAL_RUN_ARTIFACTS:
        if name in required_run_artifacts:
            continue
        if resolved.get("package_profile") == "none" and name.startswith("docker_"):
            continue
        source = run_artifacts / name
        if source.exists():
            copy_artifact(source, model_artifacts / name)
            copied_optional.append(name)

    try:
        copied_outputs = stage_output_tree(
            run_artifacts,
            model_artifacts,
            task=task,
        )
        if task == "se":
            if not copied_outputs:
                raise ValueError("SE staging requires at least one generated output")
            validate_se_sample_evidence(model_artifacts)
        elif task == "tse":
            if not copied_outputs:
                raise ValueError("TSE staging requires at least one generated output")
            validate_tse_sample_evidence(model_artifacts)
        elif is_structured_task(task) and (
            not args.allow_missing_run_artifacts
            or all(
                (model_artifacts / name).is_file()
                for name in ("sample_output.json", "sample_outputs.jsonl", "fixture_manifest.json")
            )
        ):
            validate_structured_sample_evidence(model_artifacts, task=task)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"stage_model_artifacts failed: {exc}", file=sys.stderr)
        return 1

    manifest = build_manifest(
        model_dir=model_dir,
        resolved=resolved,
        copied_required=copied_required,
        copied_optional=copied_optional,
        copied_outputs=copied_outputs,
    )
    model_manifest = model_artifacts / "artifact_manifest.json"
    model_manifest.parent.mkdir(parents=True, exist_ok=True)
    model_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    produces.parent.mkdir(parents=True, exist_ok=True)
    produces.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "stage_model_artifacts OK: "
        f"model_dir={model_dir}, required={len(copied_required)}, optional={len(copied_optional)}, "
        f"manifest={produces}"
    )
    return 0


def main() -> int:
    return main_with_args()


if __name__ == "__main__":
    sys.exit(main())
