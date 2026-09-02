#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import finalize_model_bundle
import materialize_onboard_inputs
import prepare_fixture
import run_validate
import stage_model_artifacts
from structured_segments import (
    pcm_wav_info,
    structured_task_contract,
    validate_segments,
    validate_structured_output,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[3]
TEMPLATE = SCRIPTS_DIR / "templates" / "validate.py"


def load_validate_template():
    spec = importlib.util.spec_from_file_location("structured_validate_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load validate template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StructuredSegmentsOnboardingTest(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path, *, duration_sec: float = 1.0, silence: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = b"\x00\x00" if silence else b"\x01\x00"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(frame * round(16000 * duration_sec))

    @staticmethod
    def _write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_model_input_materializes_canonical_sd_and_sa_asr_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "MODEL_INPUT.yaml"
            input_path.write_text("task_type: sd\n", encoding="utf-8")
            for requested, expected_task, method in (
                ("sd", "sd", "diarize"),
                ("sa-asr", "sa_asr", "transcribe_with_speakers"),
            ):
                resolved = materialize_onboard_inputs.make_model_input_resolved(
                    {
                        "model_id": f"example/{expected_task}",
                        "task_type": requested,
                        "deployment_type": "local",
                        "repo": {"url": f"https://example.invalid/{expected_task}"},
                    },
                    model_input_path=input_path,
                    repo_root=root,
                    package_profile="none",
                    weights_link_policy="auto",
                    device="cpu",
                    force_repair=False,
                    skip_download=True,
                    max_retries=3,
                    cpu_fallback_after_cuda_failures=3,
                    cuda_repair_attempts_before_cpu=3,
                    raw_args="",
                    existing_model_dir=None,
                    image_version=None,
                )
                expected = structured_task_contract(expected_task)
                self.assertEqual(resolved["task_type"], expected_task)
                self.assertEqual(resolved["task_contract"], expected)
                self.assertEqual(resolved["normalized_model_input"]["tool_name"], method)
                self.assertEqual(resolved["normalized_model_input"]["predict_method"], method)
                self.assertEqual(
                    resolved["normalized_model_input"]["io_contract"],
                    expected["io_contract"],
                )
                self.assertEqual(expected["io_contract"]["output_type"], "structured_segments")
                self.assertEqual(expected["io_contract"]["segment_schema"]["type"], "object")
                self.assertIs(
                    expected["io_contract"]["allow_empty_primary"],
                    expected_task == "sd",
                )

            with self.assertRaisesRegex(ValueError, "io_contract conflicts"):
                materialize_onboard_inputs.make_model_input_resolved(
                    {
                        "model_id": "example/conflict",
                        "task_type": "sd",
                        "deployment_type": "local",
                        "repo": {"url": "https://example.invalid/conflict"},
                        "io_contract": {
                            "input_type": "audio_path",
                            "output_type": "json",
                            "primary_field": "segments",
                            "required_fields": ["segments"],
                            "nonempty_fields": ["segments"],
                            "json_serializable": True,
                        },
                    },
                    model_input_path=input_path,
                    repo_root=root,
                    package_profile="none",
                    weights_link_policy="auto",
                    device="cpu",
                    force_repair=False,
                    skip_download=True,
                    max_retries=3,
                    cpu_fallback_after_cuda_failures=3,
                    cuda_repair_attempts_before_cpu=3,
                    raw_args="",
                    existing_model_dir=None,
                    image_version=None,
                )

    def test_prepare_and_check_accept_all_shared_sd_and_sa_asr_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in ("sd", "sa_asr"):
                run_dir = root / ".sure" / "runs" / task
                artifacts = run_dir / "artifacts"
                artifacts.mkdir(parents=True)
                model_dir = root / "sure" / "models" / f"example__{task}"
                (artifacts / "model_input_resolved.json").write_text(
                    json.dumps(
                        {
                            "model_id": f"example/{task}",
                            "model_name": f"example__{task}",
                            "model_dir": str(model_dir),
                            "task_type": task,
                        }
                    ),
                    encoding="utf-8",
                )
                manifest = artifacts / "fixture_manifest.json"
                source = REPO_ROOT / "fixtures" / "tasks" / task / "librispeech_2spk_smoke"
                prepared = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "prepare_fixture.py"),
                        "--run-dir",
                        str(run_dir),
                        "--produces",
                        str(manifest),
                        "--source-dir",
                        str(source),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(prepared.returncode, 0, prepared.stderr)
                checked = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "check_fixture.py"),
                        "--run-dir",
                        str(run_dir),
                        "--produces",
                        str(manifest),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(checked.returncode, 0, checked.stderr)
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(payload["sample_count"], 3)
                self.assertEqual(
                    payload["validation_protocol_env"],
                    "SURE_VALIDATE_PROTOCOL_JSON",
                )
                self.assertEqual(len({sample["key"] for sample in payload["samples"]}), 3)
                for sample in payload["samples"]:
                    self.assertGreater(sample["duration_sec"], 0)
                    self.assertEqual(sample["sample_rate"], 16000)
                    self.assertIs(sample["audio_is_silence"], False)
                    self.assertNotIn("segments", sample)
                if task == "sd":
                    sidecar = Path(payload["staged_dir"]) / "unreferenced-private.txt"
                    sidecar.write_text("private\n", encoding="utf-8")
                    rejected = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPTS_DIR / "check_fixture.py"),
                            "--run-dir",
                            str(run_dir),
                            "--produces",
                            str(manifest),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("unreferenced sidecar", rejected.stderr)
                    sidecar.unlink()

    def test_reference_validation_rejects_malformed_and_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            self._write_wav(fixture / "one.wav")
            (fixture / "gt.jsonl").write_text("{not-json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                prepare_fixture.load_samples(fixture, "sd")

            rows = [
                {
                    "key": "duplicate",
                    "audio": "one.wav",
                    "segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}],
                },
                {
                    "key": "duplicate",
                    "audio": "one.wav",
                    "segments": [{"speaker": "spk1", "start": 0.5, "end": 0.75}],
                },
            ]
            self._write_rows(fixture / "gt.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "duplicates fixture key"):
                prepare_fixture.load_samples(fixture, "sd")

    def test_structured_staging_rejects_symlinks_and_omits_unreferenced_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            self._write_wav(source / "sample.wav")
            self._write_rows(
                source / "gt.jsonl",
                [
                    {
                        "key": "sample",
                        "audio": "sample.wav",
                        "segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}],
                    }
                ],
            )
            (source / "private-sidecar.txt").write_text("must not be copied\n", encoding="utf-8")
            samples = prepare_fixture.load_samples(source, "sd")
            staged = root / "staged"
            prepare_fixture.replace_structured_tree(source, staged, samples)
            self.assertTrue((staged / "gt.jsonl").is_file())
            self.assertTrue((staged / "sample.wav").is_file())
            self.assertFalse((staged / "private-sidecar.txt").exists())

            outside = root / "outside.txt"
            outside.write_text("private\n", encoding="utf-8")
            (source / "unreferenced-link").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                prepare_fixture.load_samples(source, "sd")
            (source / "unreferenced-link").unlink()
            alias = root / "source-alias"
            alias.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
                prepare_fixture.load_samples(alias, "sd")

    def test_truncated_wav_is_rejected_by_shared_and_model_local_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "truncated.wav"
            self._write_wav(audio)
            audio.write_bytes(audio.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "PCM data is truncated"):
                pcm_wav_info(audio)
            module = load_validate_template()
            with self.assertRaisesRegex(ValueError, "PCM data is truncated"):
                module.structured_wav_info(audio)

    def test_segment_contract_rejects_all_invalid_time_and_role_shapes(self) -> None:
        cases = (
            ({"speaker": "spk1", "start": 0.0, "end": 0.5}, "segments must be an array"),
            ([{"speaker": "spk1", "start": float("nan"), "end": 0.5}], "start must be a finite number"),
            ([{"speaker": "spk1", "start": -0.1, "end": 0.5}], "start must be >= 0"),
            ([{"speaker": "spk1", "start": 0.5, "end": 0.5}], "end must be greater than start"),
            ([{"speaker": "spk1", "start": 0.0, "end": float("inf")}], "end must be a finite number"),
            ([{"speaker": "", "start": 0.0, "end": 0.5}], "speaker must be a non-empty string"),
            ([{"speaker": "spk1", "start": 0.0, "end": 1.1}], "exceeds WAV duration"),
        )
        for segments, message in cases:
            with self.subTest(message=message):
                self.assertTrue(
                    any(
                        message in violation
                        for violation in validate_segments(
                            segments,
                            task="sd",
                            duration_sec=1.0,
                            audio_is_silence=False,
                        )
                    )
                )
        self.assertIn(
            "segment 1 text must be a non-empty string for SA-ASR",
            validate_segments(
                [{"speaker": "spk1", "start": 0.0, "end": 0.5}],
                task="sa_asr",
                duration_sec=1.0,
                audio_is_silence=False,
            ),
        )

    def test_structured_output_schema_is_closed_and_path_free(self) -> None:
        module = load_validate_template()
        valid_segment = {"speaker": "spk1", "start": 0.0, "end": 0.5}
        self.assertEqual(
            validate_structured_output(
                {"segments": [valid_segment], "num_speakers": 1},
                task="sd",
                duration_sec=1.0,
                audio_is_silence=False,
            ),
            [],
        )
        cases = (
            (
                {"segments": [valid_segment], "raw": {"debug": "secret"}},
                "unapproved field(s): raw",
            ),
            (
                {"segments": [{**valid_segment, "debug": "secret"}]},
                "segment 1 contains unapproved field(s): debug",
            ),
            (
                {"segments": [{"speaker": "/nonpublic/reference", "start": 0.0, "end": 0.5}]},
                "absolute path or URI",
            ),
            (
                {
                    "segments": [
                        {"speaker": "C:\\company\\private", "start": 0.0, "end": 0.5}
                    ]
                },
                "absolute path or URI",
            ),
            (
                {
                    "segments": [
                        {
                            **valid_segment,
                            "text": "https://restricted.example.invalid/reference",
                        }
                    ]
                },
                "absolute path or URI",
            ),
            (
                {"segments": [valid_segment], "num_speakers": 2},
                "num_speakers must equal",
            ),
        )
        for output, message in cases:
            with self.subTest(message=message):
                task = "sa_asr" if "text" in output.get("segments", [{}])[0] else "sd"
                violations = validate_structured_output(
                    output,
                    task=task,
                    duration_sec=1.0,
                    audio_is_silence=False,
                )
                self.assertTrue(any(message in violation for violation in violations), violations)
                template_violations = module.validate_structured_output(
                    output,
                    task=task,
                    duration_sec=1.0,
                    audio_is_silence=False,
                )
                self.assertTrue(
                    any(message in violation for violation in template_violations),
                    template_violations,
                )

    def test_only_pure_silence_allows_empty_sd_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            self._write_wav(fixture / "silence.wav", silence=True)
            self._write_rows(
                fixture / "gt.jsonl",
                [{"key": "silence", "audio": "silence.wav", "segments": []}],
            )
            samples = prepare_fixture.load_samples(fixture, "sd")
            self.assertIs(samples[0]["audio_is_silence"], True)
            self.assertEqual(
                validate_segments(
                    [],
                    task="sd",
                    duration_sec=float(samples[0]["duration_sec"]),
                    audio_is_silence=True,
                ),
                [],
            )

            self._write_wav(fixture / "speech.wav", silence=False)
            self._write_rows(
                fixture / "gt.jsonl",
                [{"key": "speech", "audio": "speech.wav", "segments": []}],
            )
            with self.assertRaisesRegex(ValueError, "pure-silence"):
                prepare_fixture.load_samples(fixture, "sd")

    def test_model_input_never_contains_reference_and_uses_task_method(self) -> None:
        module = load_validate_template()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            self._write_wav(audio)
            for task, method in (("sd", "diarize"), ("sa_asr", "transcribe_with_speakers")):
                module.TASK_TYPE = task
                reference = [{"speaker": "spk1", "start": 0.0, "end": 0.5}]
                if task == "sa_asr":
                    reference[0]["text"] = "reference answer"
                explicit = {
                    "key": task,
                    "audio_path": str(audio),
                    "segments": reference,
                    "ground_truth": "reference answer",
                    "reference_text": "reference answer",
                }
                with mock.patch.dict(
                    os.environ,
                    {
                        "SURE_VALIDATE_INPUT_JSON": json.dumps(explicit),
                        "SURE_VALIDATE_PROTOCOL_JSON": json.dumps(
                            {"language": "en", "num_speakers": 1}
                        ),
                    },
                    clear=False,
                ):
                    fixture = module.fixture_payloads()[0]
                self.assertEqual(
                    set(fixture["input"]),
                    {"audio_path", "language", "num_speakers"},
                )
                self.assertFalse(
                    {"segments", "ground_truth", "reference_text"} & set(fixture["input"])
                )

                calls: list[tuple[str, dict]] = []

                class Wrapper:
                    def diarize(self, audio_path: str, **kwargs):
                        calls.append((audio_path, kwargs))
                        return {"segments": reference}

                    def transcribe_with_speakers(self, audio_path: str, **kwargs):
                        calls.append((audio_path, kwargs))
                        return {"segments": reference}

                    def predict(self, _payload):
                        raise AssertionError("structured validation must not use generic predict")

                output = module.run_predict(Wrapper(), fixture["input"])
                self.assertEqual(output, {"segments": reference})
                self.assertEqual(calls, [(str(audio), {"language": "en", "num_speakers": 1})])
                self.assertTrue(hasattr(Wrapper(), method))

            module.TASK_TYPE = "sd"
            with mock.patch.dict(
                os.environ,
                {
                    "SURE_VALIDATE_INPUT_JSON": json.dumps(
                        {
                            "audio_path": str(audio),
                            "segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}],
                            "num_speakers": 1,
                        }
                    ),
                    "SURE_VALIDATE_PROTOCOL_JSON": "",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "must be supplied through"):
                    module.fixture_payloads()

    def test_fixture_oracle_parameters_are_not_forwarded(self) -> None:
        module = load_validate_template()
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            fixture = model_dir / "fixture" / "sd" / "oracle"
            fixture.mkdir(parents=True)
            self._write_wav(fixture / "sample.wav")
            self._write_rows(
                fixture / "gt.jsonl",
                [
                    {
                        "key": "sample",
                        "task": "SD",
                        "audio": "sample.wav",
                        "language": "en",
                        "num_speakers": 2,
                        "min_speakers": 2,
                        "max_speakers": 2,
                        "segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}],
                    }
                ],
            )
            module.MODEL_DIR = model_dir
            module.TASK_TYPE = "sd"
            with mock.patch.dict(
                os.environ,
                {
                    "SURE_VALIDATE_INPUT_JSON": "",
                    "SURE_VALIDATE_PROTOCOL_JSON": json.dumps({"vad_threshold": 0.4}),
                },
                clear=False,
            ):
                payload = module.fixture_payloads()[0]["input"]
            self.assertEqual(
                payload,
                {
                    "audio_path": str((fixture / "sample.wav").resolve()),
                    "vad_threshold": 0.4,
                },
            )
            self.assertFalse(
                {"language", "num_speakers", "min_speakers", "max_speakers"} & set(payload)
            )

            row = json.loads((fixture / "gt.jsonl").read_text(encoding="utf-8"))
            row["inference_params"] = {"num_speakers": 2}
            self._write_rows(fixture / "gt.jsonl", [row])
            with mock.patch.dict(
                os.environ,
                {"SURE_VALIDATE_INPUT_JSON": "", "SURE_VALIDATE_PROTOCOL_JSON": ""},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "must not declare inference_params"):
                    module.fixture_payloads()

    def test_contract_checks_every_row_and_rejects_reference_leakage(self) -> None:
        module = load_validate_template()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module.ARTIFACTS_DIR = root / "artifacts"
            module.VALIDATION_LOG = module.ARTIFACTS_DIR / "validation.log"
            module.SAMPLE_OUTPUT = module.ARTIFACTS_DIR / "sample_output.json"
            module.SAMPLE_OUTPUTS = module.ARTIFACTS_DIR / "sample_outputs.jsonl"
            module.TASK_TYPE = "sd"
            module.IO_CONTRACT = structured_task_contract("sd")["io_contract"]
            fixtures = []
            for index in range(1, 6):
                audio = root / f"sample-{index}.wav"
                self._write_wav(audio)
                info = pcm_wav_info(audio)
                fixtures.append(
                    {
                        "input": {"audio_path": str(audio)},
                        "fixture": {
                            "key": f"sample-{index}",
                            "audio": audio.name,
                            "dataset": "test",
                            **info,
                        },
                    }
                )
            module.load_wrapper = lambda: object()
            module.fixture_payloads = lambda: fixtures
            module.run_predict = lambda _wrapper, _payload: {
                "segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}]
            }
            self.assertTrue(module.stage_infer())
            infer_result = json.loads(
                (module.ARTIFACTS_DIR / "infer_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(infer_result["protocol_arguments"], {})
            self.assertTrue(module.stage_contract())

            rows = [
                json.loads(line)
                for line in module.SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
            ]
            module.write_jsonl(module.SAMPLE_OUTPUTS, rows[:1])
            self.assertFalse(module.stage_contract())
            missing_result = json.loads(
                (module.ARTIFACTS_DIR / "contract_result.json").read_text(encoding="utf-8")
            )
            self.assertIn("preserve every fixture key", missing_result["error"])

            rows[-1]["output"] = {
                "segments": [{"speaker": "spk1", "start": -0.1, "end": 0.5}],
                "reference_segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}],
            }
            module.write_jsonl(module.SAMPLE_OUTPUTS, rows)
            self.assertFalse(module.stage_contract())
            result = json.loads(
                (module.ARTIFACTS_DIR / "contract_result.json").read_text(encoding="utf-8")
            )
            self.assertIn("structured output row 5", result["error"])
            self.assertIn("reference or path field", result["error"])
            self.assertIn("start must be >= 0", result["error"])

    def test_silent_sd_output_is_valid_and_evidence_omits_reference(self) -> None:
        module = load_validate_template()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            fixture = model_dir / "fixture" / "sd" / "silence"
            fixture.mkdir(parents=True)
            self._write_wav(fixture / "silence.wav", silence=True)
            self._write_rows(
                fixture / "gt.jsonl",
                [
                    {
                        "key": "silence",
                        "audio": "silence.wav",
                        "ground_truth": "must-not-leak",
                        "segments": [],
                    }
                ],
            )
            module.MODEL_DIR = model_dir
            module.ARTIFACTS_DIR = model_dir / "artifacts"
            module.VALIDATION_LOG = module.ARTIFACTS_DIR / "validation.log"
            module.SAMPLE_OUTPUT = module.ARTIFACTS_DIR / "sample_output.json"
            module.SAMPLE_OUTPUTS = module.ARTIFACTS_DIR / "sample_outputs.jsonl"
            module.TASK_TYPE = "sd"
            module.IO_CONTRACT = structured_task_contract("sd")["io_contract"]

            class Wrapper:
                def diarize(self, _audio_path: str, **_kwargs):
                    return {"segments": []}

            module.load_wrapper = Wrapper
            self.assertTrue(module.stage_infer())
            self.assertTrue(module.stage_contract())
            row = json.loads(module.SAMPLE_OUTPUTS.read_text(encoding="utf-8"))
            self.assertEqual(row["output"], {"segments": []})
            self.assertEqual(set(row), module.STRUCTURED_EVIDENCE_FIELDS)
            self.assertNotIn("ground_truth", row)

    def test_run_gate_rechecks_manifest_timing_against_the_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "sd-gate"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            model_dir = root / "sure" / "models" / "example__sd"
            fixture_dir = model_dir / "fixture" / "sd" / "smoke"
            fixture_dir.mkdir(parents=True)
            (model_dir / "model.py").write_text("# test\n", encoding="utf-8")
            audio = fixture_dir / "sample.wav"
            self._write_wav(audio)
            info = pcm_wav_info(audio)
            contract = structured_task_contract("sd")["io_contract"]
            (artifacts / "model_input_resolved.json").write_text(
                json.dumps(
                    {
                        "task_type": "sd",
                        "model_dir": str(model_dir),
                        "normalized_model_input": {"io_contract": contract},
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "key": "sample",
                "audio": "sample.wav",
                "audio_path": str(audio),
                "annotation_fields": ["segments"],
                **info,
            }
            manifest = {
                "task_type": "sd",
                "staged_dir": str(fixture_dir),
                "samples": [sample],
            }
            (artifacts / "fixture_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = {"segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}]}
            row = {
                "id": 1,
                "key": "sample",
                "audio": "sample.wav",
                "language": None,
                "dataset": "test",
                "duration_sec": info["duration_sec"],
                "sample_rate": info["sample_rate"],
                "audio_is_silence": info["audio_is_silence"],
                "output": output,
            }
            (artifacts / "sample_output.json").write_text(json.dumps(output), encoding="utf-8")
            (artifacts / "sample_outputs.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            gate_data = {
                "model_dir": str(model_dir),
                "io_contract": contract,
                "sample_output_path": str(artifacts / "sample_output.json"),
                "sample_outputs_path": str(artifacts / "sample_outputs.jsonl"),
            }
            self.assertEqual(run_validate.validate_structured_evidence(gate_data, run_dir), [])

            manifest["samples"][0]["raw"] = {"reference": "/private/reference.rttm"}
            (artifacts / "fixture_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            violations = run_validate.validate_structured_evidence(gate_data, run_dir)
            self.assertTrue(any("exposes unapproved field(s): raw" in item for item in violations))
            manifest["samples"][0].pop("raw")

            manifest["samples"][0]["duration_sec"] = 2.0
            row["duration_sec"] = 2.0
            (artifacts / "fixture_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (artifacts / "sample_outputs.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            violations = run_validate.validate_structured_evidence(gate_data, run_dir)
            self.assertTrue(any("duration_sec disagrees with the WAV" in item for item in violations))

    def test_stage_and_final_manifest_require_portable_structured_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "sd-stage"
            run_artifacts = run_dir / "artifacts"
            run_artifacts.mkdir(parents=True)
            model_dir = root / "sure" / "models" / "example__sd"
            model_dir.mkdir(parents=True)
            for name in stage_model_artifacts.CORE_FILES:
                (model_dir / name).write_text("# test\n", encoding="utf-8")
            fixture_dir = model_dir / "fixture" / "sd" / "smoke"
            fixture_dir.mkdir(parents=True)
            audio = fixture_dir / "sample.wav"
            self._write_wav(audio)
            self._write_rows(
                fixture_dir / "gt.jsonl",
                [
                    {
                        "key": "sample",
                        "audio": "sample.wav",
                        "segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}],
                    }
                ],
            )
            info = pcm_wav_info(audio)
            resolved = {
                "model_id": "example/sd",
                "model_name": "example__sd",
                "model_dir": str(model_dir),
                "task_type": "sd",
                "deployment_type": "api",
                "package_profile": "none",
            }
            (run_artifacts / "model_input_resolved.json").write_text(
                json.dumps(resolved), encoding="utf-8"
            )
            fixture_manifest = {
                "model_id": "example/sd",
                "model_name": "example__sd",
                "model_dir": str(model_dir),
                "task_type": "sd",
                "source_dir": str(fixture_dir),
                "staged_dir": str(fixture_dir),
                "gt_jsonl": str(fixture_dir / "gt.jsonl"),
                "sample_count": 1,
                "link_policy": "copy",
                "validation_protocol_env": "SURE_VALIDATE_PROTOCOL_JSON",
                "samples": [
                    {
                        "key": "sample",
                        "audio": "sample.wav",
                        "audio_path": str(audio),
                        "annotation_fields": ["segments"],
                        **info,
                    }
                ],
            }
            (run_artifacts / "fixture_manifest.json").write_text(
                json.dumps(fixture_manifest), encoding="utf-8"
            )
            output = {"segments": [{"speaker": "spk1", "start": 0.0, "end": 0.5}]}
            row = {
                "id": 1,
                "key": "sample",
                "audio": "sample.wav",
                "language": None,
                "dataset": "test",
                "duration_sec": info["duration_sec"],
                "sample_rate": info["sample_rate"],
                "audio_is_silence": info["audio_is_silence"],
                "output": output,
            }
            (run_artifacts / "sample_output.json").write_text(json.dumps(output), encoding="utf-8")
            (run_artifacts / "sample_outputs.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            manifest_path = run_artifacts / "artifact_manifest.json"
            status = stage_model_artifacts.main_with_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                    "--allow-missing-run-artifacts",
                ]
            )
            self.assertEqual(status, 0)
            staged = json.loads(manifest_path.read_text(encoding="utf-8"))
            staged_paths = {
                entry["path"] for entry in staged["artifacts"]["required"].values()
            }
            self.assertIn("artifacts/sample_output.json", staged_paths)
            self.assertIn("artifacts/sample_outputs.jsonl", staged_paths)

            finalized = finalize_model_bundle.update_manifest(model_dir, resolved)
            finalized_paths = {
                entry["path"] for entry in finalized["artifacts"]["required"].values()
            }
            self.assertIn("artifacts/sample_output.json", finalized_paths)
            self.assertIn("artifacts/sample_outputs.jsonl", finalized_paths)
            bundled_row = json.loads(
                (model_dir / "artifacts" / "sample_outputs.jsonl").read_text(encoding="utf-8")
            )
            self.assertNotIn("ground_truth", bundled_row)
            self.assertFalse(Path(bundled_row["audio"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
