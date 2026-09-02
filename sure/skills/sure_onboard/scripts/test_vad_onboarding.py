#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
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
VAD_FIXTURE = REPO_ROOT / "fixtures" / "tasks" / "vad" / "librispeech_vad_smoke"
VALIDATE_TEMPLATE = SCRIPTS_DIR / "templates" / "validate.py"


def load_validate_template():
    spec = importlib.util.spec_from_file_location("vad_validate_template", VALIDATE_TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load validate template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VADOnboardingTest(unittest.TestCase):
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
            "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _configure_template(module, model_dir: Path) -> None:
        module.MODEL_DIR = model_dir
        module.ARTIFACTS_DIR = model_dir / "artifacts"
        module.VALIDATION_LOG = module.ARTIFACTS_DIR / "validation.log"
        module.SAMPLE_OUTPUT = module.ARTIFACTS_DIR / "sample_output.json"
        module.SAMPLE_OUTPUTS = module.ARTIFACTS_DIR / "sample_outputs.jsonl"
        module.TASK_TYPE = "vad"
        module.IO_CONTRACT = structured_task_contract("vad")["io_contract"]

    def test_model_input_materializes_vad_contract_and_playbook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "MODEL_INPUT.yaml"
            input_path.write_text("task_type: voice-activity-detection\n", encoding="utf-8")
            resolved = materialize_onboard_inputs.make_model_input_resolved(
                {
                    "model_id": "snakers4/silero-vad",
                    "task_type": "voice-activity-detection",
                    "deployment_type": "local",
                    "repo": {"url": "https://github.com/snakers4/silero-vad"},
                    "environment_hint": {"preferred_backend": "uv"},
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
            context = materialize_onboard_inputs.make_context_selection(
                resolved, resolved["normalized_model_input"]
            )

        expected = structured_task_contract("vad")
        self.assertEqual(resolved["task_type"], "vad")
        self.assertEqual(resolved["task_contract"], expected)
        self.assertEqual(resolved["normalized_model_input"]["tool_name"], "detect_speech")
        self.assertEqual(resolved["normalized_model_input"]["predict_method"], "detect_speech")
        self.assertEqual(
            context["selected_references"]["task_playbooks"],
            ["references/task_playbooks/VAD.md"],
        )

    def test_repository_fixture_stages_only_referenced_pcm_and_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "vad"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            model_dir = root / "sure" / "models" / "snakers4__silero-vad"
            (artifacts / "model_input_resolved.json").write_text(
                json.dumps(
                    {
                        "model_id": "snakers4/silero-vad",
                        "model_name": "snakers4__silero-vad",
                        "model_dir": str(model_dir),
                        "task_type": "vad",
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = artifacts / "fixture_manifest.json"
            prepared = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_fixture.py"),
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                    "--source-dir",
                    str(VAD_FIXTURE),
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
                    str(manifest_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            staged = Path(manifest["staged_dir"])
            staged_files = {
                path.relative_to(staged).as_posix()
                for path in staged.rglob("*")
                if path.is_file()
            }

            self.assertEqual(manifest["sample_count"], 3)
            self.assertEqual(manifest["validation_protocol_env"], "SURE_VALIDATE_PROTOCOL_JSON")
            self.assertNotIn("provenance.json", staged_files)
            self.assertEqual(
                staged_files,
                {
                    "gt.jsonl",
                    "librispeech_vad_001.wav",
                    "librispeech_vad_002.wav",
                    "librispeech_vad_silence.wav",
                },
            )
            self.assertEqual(
                [sample["annotation_fields"] for sample in manifest["samples"]],
                [["speech_segments"]] * 3,
            )
            self.assertNotIn("speech_segments", manifest["samples"][0])
            self.assertIs(manifest["samples"][-1]["audio_is_silence"], True)

            (staged / "private-sidecar.txt").write_text("not portable\n", encoding="utf-8")
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "check_fixture.py"),
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("unreferenced sidecar", rejected.stderr)

    def test_fixture_loader_enforces_row_count_paths_pcm_and_declared_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture"
            fixture.mkdir()
            rows: list[dict] = []
            for index in range(1, 7):
                audio = f"sample-{index}.wav"
                self._write_wav(fixture / audio)
                rows.append(
                    {
                        "key": f"sample-{index}",
                        "task": "VAD",
                        "audio": audio,
                        "duration_sec": 1.0,
                        "sample_rate": 16000,
                        "speech_segments": [{"start": 0.1, "end": 0.9}],
                    }
                )

            self._write_rows(fixture / "gt.jsonl", rows[:1])
            self.assertEqual(len(prepare_fixture.load_samples(fixture, "vad")), 1)
            self._write_rows(fixture / "gt.jsonl", rows[:5])
            self.assertEqual(len(prepare_fixture.load_samples(fixture, "vad")), 5)
            self._write_rows(fixture / "gt.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "at most 5"):
                prepare_fixture.load_samples(fixture, "vad")
            (fixture / "gt.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No samples"):
                prepare_fixture.load_samples(fixture, "vad")

            invalid_cases = (
                ({**rows[0], "task": "ASR"}, "declares task"),
                ({**rows[0], "audio": "../sample-1.wav"}, "must be relative"),
                (
                    {
                        **{
                            key: value
                            for key, value in rows[0].items()
                            if key != "audio"
                        },
                        "reference_audio": "sample-1.wav",
                    },
                    "must contain a non-empty relative audio/wav field",
                ),
                ({**rows[0], "duration_sec": 2.0}, "duration_sec disagrees"),
                ({**rows[0], "sample_rate": 8000}, "sample_rate disagrees"),
                ({key: value for key, value in rows[0].items() if key != "speech_segments"}, "requires speech_segments"),
            )
            for row, message in invalid_cases:
                with self.subTest(message=message):
                    self._write_rows(fixture / "gt.jsonl", [row])
                    with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                        prepare_fixture.load_samples(fixture, "vad")

            self._write_rows(fixture / "gt.jsonl", rows[:1])
            audio_path = fixture / "sample-1.wav"
            audio_path.write_bytes(audio_path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "PCM data is truncated"):
                prepare_fixture.load_samples(fixture, "vad")

    def test_fixture_loader_rejects_all_symlink_surfaces(self) -> None:
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
                        "task": "VAD",
                        "audio": "sample.wav",
                        "speech_segments": [{"start": 0.1, "end": 0.9}],
                    }
                ],
            )
            outside = root / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")
            (source / "sidecar-link").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                prepare_fixture.load_samples(source, "vad")
            (source / "sidecar-link").unlink()
            alias = root / "source-alias"
            alias.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
                prepare_fixture.load_samples(alias, "vad")

            alias.unlink()
            for relative in (Path("gt.jsonl"), Path("sample.wav")):
                with self.subTest(hard_link=relative):
                    hard_link = root / f"hard-link-{relative.name}"
                    os.link(source / relative, hard_link)
                    with self.assertRaisesRegex(ValueError, "must not be hard-linked"):
                        prepare_fixture.load_samples(source, "vad")
                    hard_link.unlink()

    def test_fixture_gate_binds_audio_path_and_rejects_post_stage_hard_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "vad"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            model_dir = root / "sure" / "models" / "example__vad"
            source = root / "source"
            source.mkdir()
            rows = []
            for name in ("a.wav", "b.wav"):
                self._write_wav(source / name)
                rows.append(
                    {
                        "key": name.removesuffix(".wav"),
                        "task": "VAD",
                        "audio": name,
                        "speech_segments": [{"start": 0.1, "end": 0.9}],
                    }
                )
            self._write_rows(source / "gt.jsonl", rows)
            (artifacts / "model_input_resolved.json").write_text(
                json.dumps(
                    {
                        "model_id": "example/vad",
                        "model_name": "example__vad",
                        "model_dir": str(model_dir),
                        "task_type": "vad",
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = artifacts / "fixture_manifest.json"
            prepared = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_fixture.py"),
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                    "--source-dir",
                    str(source),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            staged = Path(manifest["staged_dir"])
            manifest["samples"][0]["audio_path"] = str(staged / "b.wav")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            swapped = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "check_fixture.py"),
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(swapped.returncode, 0)
            self.assertIn("audio_path must resolve", swapped.stderr)

            manifest["samples"][0]["audio_path"] = str(staged / "a.wav")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            for relative in (Path("gt.jsonl"), Path("a.wav")):
                with self.subTest(staged_hard_link=relative):
                    target = staged / relative
                    external = root / f"external-{relative.name}"
                    external.write_bytes(target.read_bytes())
                    target.unlink()
                    os.link(external, target)
                    rejected = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPTS_DIR / "check_fixture.py"),
                            "--run-dir",
                            str(run_dir),
                            "--produces",
                            str(manifest_path),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("must not be hard-linked", rejected.stderr)
                    target.unlink()
                    target.write_bytes(external.read_bytes())

    def test_vad_output_is_closed_finite_ordered_and_pcm_bounded(self) -> None:
        module = load_validate_template()
        valid = {
            "speech_segments": [
                {"start": 0.1, "end": 0.4},
                {"start": 0.6, "end": 0.9},
            ],
            "frame_scores": [
                {"start": 0.0, "end": 0.5, "score": 0.8},
                {"start": 0.5, "end": 1.0, "score": 0.2},
            ],
        }
        self.assertEqual(
            validate_structured_output(
                valid, task="vad", duration_sec=1.0, audio_is_silence=False
            ),
            [],
        )
        self.assertEqual(
            module.validate_structured_output(
                valid, task="vad", duration_sec=1.0, audio_is_silence=False
            ),
            [],
        )

        cases = (
            ({"speech_segments": [{"start": float("nan"), "end": 0.5}]}, "finite number"),
            ({"speech_segments": [{"start": -0.1, "end": 0.5}]}, "start must be >= 0"),
            ({"speech_segments": [{"start": 0.5, "end": 0.5}]}, "greater than start"),
            ({"speech_segments": [{"start": 0.5, "end": float("inf")}]}, "finite number"),
            ({"speech_segments": [{"start": 0.5, "end": 1.1}]}, "exceeds WAV duration"),
            (
                {"speech_segments": [{"start": 0.1, "end": 0.6}, {"start": 0.5, "end": 0.8}]},
                "overlaps the previous interval",
            ),
            (
                {"speech_segments": [{"start": 0.6, "end": 0.8}, {"start": 0.1, "end": 0.2}]},
                "not ordered by start time",
            ),
            ({"speech_segments": [{"start": 0.1, "end": 0.5, "speaker": "spk"}]}, "unapproved field"),
            ({"speech_segments": [{"start": 0.1, "end": 0.5}], "raw": {"debug": "x"}}, "unapproved field"),
            ({"speech_segments": [{"start": 0.1, "end": 0.5}], "reference_path": "/restricted/ref"}, "reference or path"),
            ({"speech_segments": [{"start": 0.1, "end": 0.5}], "frame_scores": []}, "must be non-empty"),
            (
                {"speech_segments": [{"start": 0.1, "end": 0.5}], "frame_scores": [{"start": 0.1, "end": 1.0, "score": 0.5}]},
                "must start at 0",
            ),
            (
                {"speech_segments": [{"start": 0.1, "end": 0.5}], "frame_scores": [{"start": 0.0, "end": 0.4, "score": 0.5}, {"start": 0.5, "end": 1.0, "score": 0.5}]},
                "leaves a gap",
            ),
            (
                {"speech_segments": [{"start": 0.1, "end": 0.5}], "frame_scores": [{"start": 0.0, "end": 0.5, "score": 1.1}, {"start": 0.5, "end": 1.0, "score": 0.0}]},
                "score must be within",
            ),
            (
                {"speech_segments": [{"start": 0.1, "end": 0.5}], "frame_scores": [{"start": 0.0, "end": 0.5, "score": 0.5}]},
                "must end at WAV duration",
            ),
        )
        for output, message in cases:
            with self.subTest(message=message):
                violations = validate_structured_output(
                    output, task="vad", duration_sec=1.0, audio_is_silence=False
                )
                self.assertTrue(any(message in item for item in violations), violations)
                template_violations = module.validate_structured_output(
                    output, task="vad", duration_sec=1.0, audio_is_silence=False
                )
                self.assertTrue(
                    any(message in item for item in template_violations),
                    template_violations,
                )

        self.assertEqual(
            validate_segments(
                [], task="vad", duration_sec=1.0, audio_is_silence=True
            ),
            [],
        )
        self.assertIn(
            "pure-silence",
            "; ".join(
                validate_segments(
                    [], task="vad", duration_sec=1.0, audio_is_silence=False
                )
            ),
        )

    def test_reference_fields_never_enter_vad_wrapper_arguments(self) -> None:
        module = load_validate_template()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            self._write_wav(audio)
            module.TASK_TYPE = "vad"
            explicit = {
                "key": "sample",
                "audio": "sample.wav",
                "audio_path": str(audio),
                "duration_sec": 1.0,
                "speech_segments": [{"start": 0.1, "end": 0.9}],
                "ground_truth": "must not leak",
                "reference_path": "/restricted/reference.json",
            }
            with mock.patch.dict(
                os.environ,
                {
                    "SURE_VALIDATE_INPUT_JSON": json.dumps(explicit),
                    "SURE_VALIDATE_PROTOCOL_JSON": json.dumps({"vad_threshold": 0.5}),
                },
                clear=False,
            ):
                fixture = module.fixture_payloads()[0]
            self.assertEqual(
                set(fixture["input"]), {"audio_path", "vad_threshold"}
            )
            calls: list[tuple[str, dict]] = []

            class Wrapper:
                def detect_speech(self, audio_path: str, **kwargs):
                    calls.append((audio_path, kwargs))
                    return {"speech_segments": [{"start": 0.1, "end": 0.9}]}

            output = module.run_predict(Wrapper(), fixture["input"])
            self.assertEqual(output, {"speech_segments": [{"start": 0.1, "end": 0.9}]})
            self.assertEqual(calls, [(str(audio), {"vad_threshold": 0.5})])

            with mock.patch.dict(
                os.environ,
                {"SURE_VALIDATE_PROTOCOL_JSON": json.dumps({"num_speakers": 1})},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "unsupported public inference"):
                    module.structured_protocol_arguments()

    def test_template_validates_every_fixture_row_and_preserves_order(self) -> None:
        module = load_validate_template()
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            fixture_dir = model_dir / "fixture" / "vad" / "smoke"
            fixture_dir.mkdir(parents=True)
            rows = []
            for index in range(1, 6):
                audio = f"sample-{index}.wav"
                self._write_wav(fixture_dir / audio)
                rows.append(
                    {
                        "key": f"sample-{index}",
                        "task": "VAD",
                        "audio": audio,
                        "duration_sec": 1.0,
                        "sample_rate": 16000,
                        "speech_segments": [{"start": 0.1, "end": 0.9}],
                    }
                )
            self._write_rows(fixture_dir / "gt.jsonl", rows)
            self._configure_template(module, model_dir)

            class Wrapper:
                def detect_speech(self, _audio_path: str, **_kwargs):
                    return {
                        "speech_segments": [{"start": 0.1, "end": 0.9}],
                        "frame_scores": [
                            {"start": 0.0, "end": 1.0, "score": 0.75}
                        ],
                    }

            module.load_wrapper = Wrapper
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertTrue(module.stage_infer())
                self.assertTrue(module.stage_contract())
            output_rows = [
                json.loads(line)
                for line in module.SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["key"] for row in output_rows], [row["key"] for row in rows])
            self.assertTrue(all(set(row) == module.STRUCTURED_EVIDENCE_FIELDS for row in output_rows))
            module.write_jsonl(module.SAMPLE_OUTPUTS, output_rows[:-1])
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(module.stage_contract())
            result = json.loads(
                (module.ARTIFACTS_DIR / "contract_result.json").read_text(encoding="utf-8")
            )
            self.assertIn("preserve every fixture key", result["error"])

    def test_run_gate_rechecks_outputs_against_actual_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "vad-gate"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            model_dir = root / "sure" / "models" / "example__vad"
            fixture_dir = model_dir / "fixture" / "vad" / "smoke"
            fixture_dir.mkdir(parents=True)
            (model_dir / "model.py").write_text("# test\n", encoding="utf-8")
            audio = fixture_dir / "sample.wav"
            self._write_wav(audio)
            self._write_rows(
                fixture_dir / "gt.jsonl",
                [
                    {
                        "key": "sample",
                        "task": "VAD",
                        "audio": "sample.wav",
                        "speech_segments": [{"start": 0.1, "end": 0.9}],
                    }
                ],
            )
            info = pcm_wav_info(audio)
            contract = structured_task_contract("vad")["io_contract"]
            (artifacts / "model_input_resolved.json").write_text(
                json.dumps(
                    {
                        "task_type": "vad",
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
                "annotation_fields": ["speech_segments"],
                **info,
            }
            manifest = {
                "task_type": "vad",
                "staged_dir": str(fixture_dir),
                "samples": [sample],
            }
            (artifacts / "fixture_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = {
                "speech_segments": [{"start": 0.1, "end": 0.9}],
                "frame_scores": [{"start": 0.0, "end": 1.0, "score": 0.8}],
            }
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
            (artifacts / "sample_output.json").write_text(
                json.dumps(output), encoding="utf-8"
            )
            (artifacts / "sample_outputs.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            gate_data = {
                "model_dir": str(model_dir),
                "io_contract": contract,
                "sample_output_path": str(artifacts / "sample_output.json"),
                "sample_outputs_path": str(artifacts / "sample_outputs.jsonl"),
            }
            self.assertEqual(
                run_validate.validate_structured_evidence(gate_data, run_dir), []
            )
            stage_model_artifacts.validate_structured_sample_evidence(
                artifacts, task="vad"
            )
            model_artifacts = model_dir / "artifacts"
            model_artifacts.mkdir()
            for name in (
                "fixture_manifest.json",
                "sample_output.json",
                "sample_outputs.jsonl",
            ):
                (model_artifacts / name).write_bytes((artifacts / name).read_bytes())
            finalize_model_bundle.validate_structured_sample_outputs(
                model_dir, task="vad"
            )

            row["duration_sec"] = 2.0
            (artifacts / "sample_outputs.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            violations = run_validate.validate_structured_evidence(gate_data, run_dir)
            self.assertTrue(
                any("duration evidence disagrees" in item for item in violations),
                violations,
            )


if __name__ == "__main__":
    unittest.main()
