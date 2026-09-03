from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
SPEECH_UNDERSTANDING_SUBTASK_ORDER = ("asr", "s2tt", "slu", "ser", "gr", "sd", "sa_asr")
KWS_POSITIVE_VALUES = {"detect", "detected", "positive", "true", "1", "yes"}
KWS_NEGATIVE_VALUES = {"reject", "rejected", "negative", "false", "0", "no"}


def normalize_task(task: str) -> str:
    value = (
        str(task or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if value in {
        "sa_asr",
        "saasr",
        "speaker_attributed_asr",
        "speaker_aware_asr",
        "transcribe_diarize",
    }:
        return "sa_asr"
    if value in {"sd", "speaker_diarization", "speaker_diarisation", "diarization", "diarisation"}:
        return "sd"
    if value in {"speech_enhancement", "acoustic_noise_suppression"}:
        return "se"
    if value in {
        "speech_activity_detection",
        "voice_activity_detection",
    }:
        return "vad"
    if value in {
        "speech_emotion_recognition",
        "speaker_emotion_recognition",
        "emotion_recognition",
    }:
        return "ser"
    if value in {"gender_recognition", "speaker_gender"}:
        return "gr"
    if value == "spoken_language_understanding":
        return "slu"
    if value in {
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
    return value


def _kws_expected_detected(sample: dict[str, Any]) -> bool:
    declared: list[tuple[str, bool]] = []
    key = sample.get("key") or sample.get("sample_id") or sample.get("id") or "<unknown>"
    for field in ("expected", "label", "expected_detected"):
        if field not in sample:
            continue
        value = sample[field]
        if isinstance(value, bool):
            parsed = value
        else:
            normalized = str(value or "").strip().lower()
            if normalized in KWS_POSITIVE_VALUES:
                parsed = True
            elif normalized in KWS_NEGATIVE_VALUES:
                parsed = False
            else:
                raise ValueError(
                    f"KWS fixture {key!r} has unsupported {field} value {value!r}"
                )
        declared.append((field, parsed))
    if not declared:
        raise ValueError(
            f"KWS fixture {key!r} must declare expected, label, or expected_detected"
        )
    if len({parsed for _field, parsed in declared}) != 1:
        fields = ", ".join(f"{field}={sample[field]!r}" for field, _parsed in declared)
        raise ValueError(f"KWS fixture {key!r} has conflicting polarity fields: {fields}")
    return declared[0][1]


def find_repo_root(start: Path | None = None) -> Path:
    starts = [start] if start else []
    starts.extend([Path.cwd(), Path(__file__).resolve()])
    for item in starts:
        if item is None:
            continue
        base = item if item.is_dir() else item.parent
        for path in (base, *base.parents):
            if (path / "fixtures" / "tasks").is_dir():
                return path
    return Path.cwd()


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
        if len(rows) >= limit:
            break
    return rows


def _read_json(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)][:limit]
    if isinstance(value, dict):
        rows = value.get("samples") or value.get("items") or []
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)][:limit]
    return []


def _path_from_row(row: dict[str, Any], sample_dir: Path, repo_root: Path) -> str | None:
    value = row.get("audio") or row.get("wav") or row.get("audio_path") or row.get("source_audio")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = sample_dir / path
    return _rel(path, repo_root)


def _named_audio_path(
    row: dict[str, Any], field: str, sample_dir: Path, repo_root: Path
) -> str | None:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = sample_dir / path
    return _rel(path, repo_root)


def _safe_fixture_audio(value: Any, sample_dir: Path, repo_root: Path) -> str | None:
    """Return a fixture-relative audio path without exposing host paths."""

    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    relative = Path(raw)
    if "\\" in raw or URI_PREFIX.match(raw) or ".." in relative.parts:
        return None
    sample_root = sample_dir.expanduser().absolute()
    if sample_root.is_symlink():
        return None
    candidate = relative if relative.is_absolute() else sample_root / relative
    try:
        candidate_relative = candidate.absolute().relative_to(sample_root)
    except ValueError:
        candidate_relative = None
    if candidate_relative is not None:
        current = sample_root
        for part in candidate_relative.parts:
            current = current / part
            if current.is_symlink():
                return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        resolved.relative_to(sample_root.resolve())
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    try:
        return resolved.relative_to(repo_root.expanduser().resolve()).as_posix()
    except ValueError:
        return resolved.relative_to(sample_root.resolve()).as_posix()


def _first_safe_fixture_audio(
    row: dict[str, Any], fields: tuple[str, ...], sample_dir: Path, repo_root: Path
) -> str | None:
    for field in fields:
        normalized = _safe_fixture_audio(row.get(field), sample_dir, repo_root)
        if normalized:
            return normalized
    return None


def _compact_sample(
    row: dict[str, Any], sample_dir: Path, repo_root: Path, task: str | None = None
) -> dict[str, Any]:
    task = normalize_task(task or row.get("task") or row.get("task_type") or "")
    sample: dict[str, Any] = {}
    fields = (
        "id",
        "key",
        "sample_id",
        "task",
        "language",
        "target_language",
        "dataset",
        "ground_truth",
        "text",
        "label",
        "expected",
        "expected_detected",
        "expected_keyword",
        "prompt",
        "choices",
        "options",
        "mixture_audio",
        "mixture_audio_path",
        "enrollment_audio",
        "enrollment_audio_path",
        "mixed_audio",
        "target_audio",
        "target_audio_path",
        "keywords",
        "threshold",
        "duration",
        "num_speakers",
        "speakers",
        "annotation_format",
        "metric_format",
        "segments",
        "speech_segments",
        "rttm",
        "stm",
        "uem",
        "description",
    )
    if task == "tse":
        # TSE roles are projected below from validated aliases. Keeping raw
        # *_path values here would leak source-host paths into model input.
        fields = tuple(
            key
            for key in fields
            if key
            not in {
                "mixture_audio",
                "mixture_audio_path",
                "enrollment_audio",
                "enrollment_audio_path",
                "mixed_audio",
                "reference_audio",
                "reference_audio_path",
                "target_audio",
                "target_audio_path",
            }
        )
    for key in fields:
        if key in row:
            sample[key] = row[key]
    if sample.get("sample_id") and not sample.get("key"):
        sample["key"] = sample["sample_id"]
    if task == "tse":
        if "task" in sample:
            sample["task"] = "tse"
        mixture = _first_safe_fixture_audio(
            row,
            ("mixture_audio", "mixture_audio_path", "mixed_audio", "audio", "audio_path", "path"),
            sample_dir,
            repo_root,
        )
        enrollment = _first_safe_fixture_audio(
            row,
            ("enrollment_audio", "enrollment_audio_path", "enrollment", "speaker_audio", "enroll_audio"),
            sample_dir,
            repo_root,
        )
        reference = _first_safe_fixture_audio(
            row,
            ("reference_audio", "reference_audio_path", "target_audio", "target_audio_path"),
            sample_dir,
            repo_root,
        )
        if mixture:
            sample["audio"] = mixture
            sample["mixture_audio"] = mixture
        if enrollment:
            sample["enrollment_audio"] = enrollment
        if reference:
            sample["reference_audio"] = reference
        return sample
    audio = _path_from_row(row, sample_dir, repo_root)
    if audio:
        sample["audio"] = audio
    reference_audio = _named_audio_path(row, "reference_audio", sample_dir, repo_root)
    if reference_audio:
        sample["reference_audio"] = reference_audio
    reference_audio_path = _named_audio_path(row, "reference_audio_path", sample_dir, repo_root)
    if reference_audio_path:
        sample["reference_audio_path"] = reference_audio_path
    for field in ("mixture_audio", "enrollment_audio", "target_audio"):
        resolved = _named_audio_path(row, field, sample_dir, repo_root)
        if resolved:
            sample[field] = resolved
    return sample


def _candidate_text(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""
    fields: list[str] = []
    for key in ("model_id", "repo", "description", "model_card_text", "readme", "pipeline_tag"):
        value = candidate.get(key)
        if value:
            fields.append(str(value))
    fields.extend(str(item) for item in candidate.get("tasks") or [])
    fields.extend(str(item) for item in candidate.get("tags") or [])
    return "\n".join(fields).lower()


def _prefer_gt_files(task: str, gt_files: list[Path], candidate_text: str) -> list[Path]:
    if task != "asr":
        return gt_files
    zh_hint = any(term in candidate_text for term in ("chinese", "mandarin", "cantonese", "zh", "中文", "普通话"))
    en_hint = any(term in candidate_text for term in ("english", "en", "librispeech"))

    def score(path: Path) -> tuple[int, str]:
        text = path.as_posix().lower()
        if zh_hint and ("asr_zh" in text or "/zh" in text):
            return (0, text)
        if en_hint and ("asr_en" in text or "/en" in text):
            return (0, text)
        if "asr_en" in text:
            return (1, text)
        if "asr_zh" in text:
            return (2, text)
        return (3, text)

    return sorted(gt_files, key=score)


def io_contract_for_task(task: str) -> dict[str, Any]:
    normalized = normalize_task(task)
    if normalized == "vad":
        return {
            "input_type": "audio_path",
            "output_type": "voice_activity_detection",
            "input": {"audio_path": "string"},
            "output": {
                "speech_segments": "array<{start:number,end:number}>",
                "frame_scores": "optional array<{start:number,end:number,score:number}>",
            },
            "primary_field": "speech_segments",
            "required_fields": ["speech_segments"],
            "nonempty_fields": [],
            "allow_empty_primary": True,
            "json_serializable": True,
            "approved_output_fields": ["frame_scores", "speech_segments"],
            "segment_schema": {
                "type": "object",
                "required": ["start", "end"],
                "properties": {
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "exclusiveMinimum": 0},
                },
                "additionalProperties": False,
            },
            "frame_score_schema": {
                "type": "object",
                "required": ["start", "end", "score"],
                "properties": {
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "exclusiveMinimum": 0},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "additionalProperties": False,
            },
        }
    if normalized in {"sd", "sa_asr"}:
        sa_asr = normalized == "sa_asr"
        item_required = ["speaker", "start", "end", *(("text",) if sa_asr else ())]
        segment_schema = {
            "type": "object",
            "required": item_required,
            "properties": {
                "speaker": {"type": "string", "minLength": 1},
                "start": {"type": "number", "minimum": 0},
                "end": {"type": "number", "exclusiveMinimum": 0},
                "duration": {"type": "number", "exclusiveMinimum": 0},
                **({"text": {"type": "string", "minLength": 1}} if sa_asr else {}),
            },
            "additionalProperties": False,
        }
        return {
            "input_type": "audio_path",
            "output_type": "structured_segments",
            "input": {"audio_path": "string"},
            "output": {
                "segments": (
                    "array<{speaker:string,start:number,end:number,text:string}>"
                    if sa_asr
                    else "array<{speaker:string,start:number,end:number}>"
                ),
                "num_speakers": "optional integer",
            },
            "primary_field": "segments",
            "required_fields": ["segments"],
            "nonempty_fields": ["segments"] if sa_asr else [],
            "allow_empty_primary": not sa_asr,
            "json_serializable": True,
            "allow_empty_segments": False if sa_asr else "silence_only",
            "approved_output_fields": ["num_speakers", "segments"],
            "segment_schema": segment_schema,
        }
    if normalized in {"tts", "vc", "se"}:
        return {
            "input_type": (
                "text_with_reference_audio"
                if normalized == "tts"
                else "audio_pair" if normalized == "vc" else "audio_path"
            ),
            "output_type": "audio" if normalized == "se" else "json",
            "primary_field": "audio_path",
            "required_fields": ["audio_path"],
            "nonempty_fields": ["audio_path"],
            "json_serializable": True,
        }
    if normalized in {"ser", "gr"}:
        label_spec = "ser_default" if normalized == "ser" else "gr_default"
        return {
            "input_type": "audio_path",
            "output_type": "classification",
            "input": {"audio_path": "string", "language": "optional string"},
            "output": {"label": "string", "score": "optional number"},
            "primary_field": "label",
            "required_fields": ["label"],
            "nonempty_fields": ["label"],
            "approved_output_fields": ["label", "score"],
            "label_spec": label_spec,
            "json_serializable": True,
        }
    if normalized == "slu":
        return {
            "input_type": "audio_with_prompt",
            "output_type": "classification_answer",
            "input": {
                "audio_path": "string",
                "language": "optional string",
                "prompt": "string",
                "choices": "optional object|array",
            },
            "output": {"answer": "string", "label": "optional string"},
            "primary_field": "answer",
            "required_fields": ["answer"],
            "nonempty_fields": ["answer"],
            "approved_output_fields": ["answer", "label"],
            "json_serializable": True,
        }
    if normalized == "tse":
        return {
            "input_type": "audio_pair",
            "output_type": "audio",
            "input": {
                "mixture_audio_path": "string",
                "enrollment_audio_path": "string",
                "output_path": "string",
            },
            "output": {"prediction_audio": "string", "sample_id": "optional string"},
            "primary_field": "prediction_audio",
            "required_fields": ["prediction_audio"],
            "nonempty_fields": ["prediction_audio"],
            "approved_output_fields": ["prediction_audio", "sample_id"],
            "json_serializable": True,
        }
    if normalized == "kws":
        return {
            "input_type": "audio_path",
            "output_type": "keyword_detection",
            "primary_field": "detected",
            "required_fields": ["detected", "keyword", "score"],
            "nonempty_fields": ["detected"],
            "json_serializable": True,
        }
    if normalized == "s2tt":
        primary = "translation"
    else:
        primary = "text"
    return {
        "input_type": "audio_path",
        "output_type": "json",
        "primary_field": primary,
        "required_fields": [primary],
        "nonempty_fields": [primary],
        "json_serializable": True,
    }


def _apply_task_specific_fields(task: str, fixture: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    if not samples:
        return
    first = samples[0]
    first_audio = first.get("audio")
    if isinstance(first_audio, str) and first_audio:
        fixture["audio"] = first_audio
    if task == "tts":
        if isinstance(first.get("text"), str):
            fixture["text"] = first["text"]
        if isinstance(first_audio, str):
            fixture["reference_audio"] = first_audio
    elif task == "vc":
        audios = [sample.get("audio") for sample in samples if isinstance(sample.get("audio"), str)]
        if audios:
            fixture["audio"] = audios[0]
            fixture["source_audio"] = audios[0]
        if len(audios) > 1:
            fixture["reference_audio"] = audios[1]
        elif audios:
            fixture["reference_audio"] = audios[0]
    elif task == "se":
        reference_audio = first.get("reference_audio")
        if isinstance(reference_audio, str) and reference_audio:
            fixture["reference_audio"] = reference_audio
    elif task == "tse":
        for field in ("mixture_audio", "enrollment_audio", "reference_audio"):
            value = first.get(field)
            if isinstance(value, str) and value:
                fixture[field] = value
        fixture["audio"] = fixture.get("mixture_audio", fixture.get("audio", ""))
    elif task == "kws":
        polarities = [(sample, _kws_expected_detected(sample)) for sample in samples]
        positives = [sample for sample, expected in polarities if expected]
        negatives = [sample for sample, expected in polarities if not expected]
        if positives and positives[0].get("audio"):
            fixture["audio"] = positives[0]["audio"]
            fixture["positive_audio"] = positives[0]["audio"]
        if negatives and negatives[0].get("audio"):
            fixture["negative_audio"] = negatives[0]["audio"]
        if first.get("keywords"):
            fixture["keywords"] = first["keywords"]
    elif task == "slu" and first.get("prompt"):
        fixture["prompt"] = first["prompt"]
    if first.get("language"):
        fixture["language"] = first["language"]
    if first.get("target_language"):
        fixture["target_language"] = first["target_language"]


def select_atomic_fixture(
    task: str,
    candidate: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    max_samples: int = 3,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    normalized = normalize_task(task)
    root = repo_root or find_repo_root()
    task_dir = root / "fixtures" / "tasks" / normalized
    index_path = task_dir / "README.md"
    issues: list[str] = []
    if not index_path.exists():
        return None, io_contract_for_task(normalized), [f"missing:fixture.index.{normalized}"]

    text = _candidate_text(candidate)
    gt_files = _prefer_gt_files(normalized, sorted(task_dir.glob("**/gt.jsonl")), text)
    manifest_files = sorted(task_dir.glob("**/manifest.json"))
    samples: list[dict[str, Any]] = []
    fixture_root: Path | None = None
    gt_path: Path | None = None
    manifest_path: Path | None = None
    if gt_files:
        gt_path = gt_files[0]
        fixture_root = gt_path.parent
        rows = _read_jsonl(gt_path, max_samples)
        samples = [_compact_sample(row, fixture_root, root, normalized) for row in rows]
    elif manifest_files:
        manifest_path = manifest_files[0]
        fixture_root = manifest_path.parent
        rows = _read_json(manifest_path, max_samples)
        samples = [_compact_sample(row, fixture_root, root, normalized) for row in rows]
    else:
        audio_files = sorted(path for path in task_dir.glob("**/*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
        if audio_files:
            fixture_root = audio_files[0].parent
            samples = [{"audio": _rel(path, root), "key": path.stem} for path in audio_files[:max_samples]]

    if not fixture_root or not samples:
        issues.append(f"missing:fixture.samples.{normalized}")
        return None, io_contract_for_task(normalized), issues

    if normalized == "kws":
        positive = any(_kws_expected_detected(sample) is True for sample in samples)
        negative = any(_kws_expected_detected(sample) is False for sample in samples)
        if not positive:
            issues.append("missing:fixture.kws.positive")
        if not negative:
            issues.append("missing:fixture.kws.negative")
        if issues:
            return None, io_contract_for_task(normalized), issues
    if normalized == "se" and any(
        not isinstance(sample.get("reference_audio"), str)
        or not str(sample["reference_audio"]).strip()
        for sample in samples
    ):
        return None, io_contract_for_task(normalized), ["missing:fixture.se.reference_audio"]
    if normalized == "tse":
        required_roles = ("mixture_audio", "enrollment_audio", "reference_audio")
        missing_roles = [
            role
            for role in required_roles
            if any(
                not isinstance(sample.get(role), str) or not sample[role].strip()
                for sample in samples
            )
        ]
        if missing_roles:
            return None, io_contract_for_task(normalized), [
                f"missing:fixture.tse.{role}" for role in missing_roles
            ]

    fixture: dict[str, Any] = {
        "fixture_id": f"{normalized}/{_rel(fixture_root, root).removeprefix(f'fixtures/tasks/{normalized}/')}",
        "fixture_source": "task_registry",
        "fixture_index": _rel(index_path, root),
        "fixture_root": _rel(fixture_root, root),
        "task_specific": True,
        "fallback_allowed": False,
        "sample_count": len(samples),
        "samples": samples,
    }
    if gt_path:
        fixture["gt"] = _rel(gt_path, root)
    if manifest_path:
        fixture["manifest"] = _rel(manifest_path, root)
        if normalized in {"sd", "sa_asr"}:
            fixture["requires_meeteval_annotation"] = True
    _apply_task_specific_fields(normalized, fixture, samples)
    return fixture, io_contract_for_task(normalized), issues


def infer_speech_understanding_subtasks(candidate: dict[str, Any] | None) -> list[str]:
    text = _candidate_text(candidate)
    subtasks: list[str] = []

    def add(task: str, patterns: tuple[str, ...]) -> None:
        if task not in subtasks and any(re.search(pattern, text) for pattern in patterns):
            subtasks.append(task)

    add("asr", (r"\basr\b", r"automatic speech recognition", r"speech[- ]to[- ]text", r"transcrib", r"transcription"))
    add("s2tt", (r"\bs2tt\b", r"speech translation", r"speech[- ]to[- ]text translation", r"translate"))
    add("slu", (r"\bslu\b", r"spoken language understanding", r"speech understanding", r"reasoning", r"question answering"))
    add("ser", (r"\bser\b", r"speech emotion", r"emotion recognition"))
    add("gr", (r"gender recognition", r"speaker gender"))
    add("sd", (r"speaker diarization", r"\bdiari[sz]ation\b", r"\bdiari[sz]e\b"))
    add("sa_asr", (r"speaker[- ]attributed", r"speaker[- ]aware asr", r"\bsa[-_ ]asr\b"))
    return [task for task in SPEECH_UNDERSTANDING_SUBTASK_ORDER if task in subtasks]


def select_fixture_for_task(
    task: str,
    candidate: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    max_samples: int = 3,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str], list[dict[str, Any]]]:
    normalized = normalize_task(task)
    root = repo_root or find_repo_root()
    evidence: list[dict[str, Any]] = []
    if normalized != "speech_understanding":
        fixture, io_contract, issues = select_atomic_fixture(normalized, candidate, root, max_samples)
        if fixture:
            evidence.extend(_fixture_evidence(fixture, io_contract))
        return fixture, io_contract, issues, evidence

    composite_index = root / "fixtures" / "tasks" / "speech_understanding" / "README.md"
    if not composite_index.exists():
        return None, io_contract_for_task("asr"), ["missing:fixture.index.speech_understanding"], evidence
    subtasks = infer_speech_understanding_subtasks(candidate)
    if not subtasks:
        return (
            None,
            io_contract_for_task("asr"),
            ["missing:fixture.speech_understanding_subtask"],
            [
                {
                    "source": "local",
                    "field": "fixture_registry.composite_index",
                    "value": _rel(composite_index, root),
                    "strength": "medium",
                    "model_input_field": "fixture",
                }
            ],
        )

    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    issues: list[str] = []
    for subtask in subtasks:
        sub_fixture, sub_contract, sub_issues = select_atomic_fixture(subtask, candidate, root, max_samples)
        if sub_fixture:
            selected.append((subtask, sub_fixture, sub_contract))
        issues.extend(sub_issues)
    if not selected:
        return None, io_contract_for_task("asr"), issues or ["missing:fixture.samples.speech_understanding"], evidence

    primary_task, primary_fixture, primary_contract = selected[0]
    fixture = dict(primary_fixture)
    fixture["fixture_id"] = f"speech_understanding/{primary_fixture['fixture_id']}"
    fixture["fixture_index"] = _rel(composite_index, root)
    fixture["atomic_fixture_index"] = primary_fixture.get("fixture_index")
    fixture["selected_subtasks"] = subtasks
    fixture["primary_subtask"] = primary_task
    fixture["subtask_fixtures"] = [
        {
            "task_type": subtask,
            "fixture_index": sub_fixture.get("fixture_index"),
            "fixture_root": sub_fixture.get("fixture_root"),
            "gt": sub_fixture.get("gt"),
            "manifest": sub_fixture.get("manifest"),
            "sample_count": sub_fixture.get("sample_count"),
        }
        for subtask, sub_fixture, _sub_contract in selected
    ]
    evidence.append(
        {
            "source": "local",
            "field": "fixture_registry.composite_index",
            "value": _rel(composite_index, root),
            "strength": "strong",
            "model_input_field": "fixture",
        }
    )
    evidence.extend(_fixture_evidence(fixture, primary_contract))
    return fixture, primary_contract, issues, evidence


def _fixture_evidence(fixture: dict[str, Any], io_contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source": "local",
            "field": "fixture_registry.index",
            "value": fixture.get("fixture_index"),
            "strength": "strong",
            "model_input_field": "fixture",
        },
        {
            "source": "local",
            "field": "fixture_registry.samples",
            "value": {
                "fixture_root": fixture.get("fixture_root"),
                "sample_count": fixture.get("sample_count"),
                "first_audio": fixture.get("audio"),
            },
            "strength": "strong",
            "model_input_field": "fixture",
        },
        {
            "source": "local",
            "field": "fixture_registry.io_contract",
            "value": io_contract,
            "strength": "strong",
            "model_input_field": "io_contract",
        },
    ]
