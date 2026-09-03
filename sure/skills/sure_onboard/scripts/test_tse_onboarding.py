from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import run_validate
from tse_contract import task_contract


TEMPLATE = Path(__file__).parent / "templates" / "validate.py"


def load_template():
    spec = importlib.util.spec_from_file_location("sure_onboard_tse_validate", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load validate template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stage_model_artifacts():
    path = Path(__file__).with_name("stage_model_artifacts.py")
    spec = importlib.util.spec_from_file_location("sure_onboard_stage_model_artifacts_tse", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stage_model_artifacts")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, *, frames: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)


class TSEOnboardingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.model_dir = root / "model"
        self.fixture_dir = self.model_dir / "fixture" / "tse" / "smoke"
        self.fixture_dir.mkdir(parents=True)
        for name in ("mixture.wav", "enrollment.wav", "reference.wav"):
            write_wav(self.fixture_dir / name)
        (self.fixture_dir / "gt.jsonl").write_text(
            json.dumps(
                {
                    "key": "sample-1",
                    "sample_id": "sample-1",
                    "task_type": "tse",
                    "audio": "mixture.wav",
                    "mixture_audio": "mixture.wav",
                    "enrollment_audio": "enrollment.wav",
                    "reference_audio": "reference.wav",
                    "reference_text": "target",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.module = load_template()
        artifacts = self.model_dir / "artifacts"
        self.module.MODEL_DIR = self.model_dir
        self.module.ARTIFACTS_DIR = artifacts
        self.module.VALIDATION_LOG = artifacts / "validation.log"
        self.module.SAMPLE_OUTPUT = artifacts / "sample_output.json"
        self.module.SAMPLE_OUTPUTS = artifacts / "sample_outputs.jsonl"
        self.module.TASK_TYPE = "target-speaker-extraction"
        self.module.IO_CONTRACT = {
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

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fixture_and_inference_keep_reference_out_of_model_input(self) -> None:
        fixtures = self.module.fixture_payloads()
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(
            set(fixtures[0]["input"]),
            {"mixture_audio_path", "enrollment_audio_path"},
        )
        self.assertNotIn("reference_audio_path", fixtures[0]["input"])
        calls: list[dict[str, str]] = []

        class Wrapper:
            def predict(self, payload: dict[str, str]) -> dict[str, str]:
                calls.append(dict(payload))
                write_wav(Path(payload["output_path"]))
                return {"prediction_audio": payload["output_path"]}

        self.module.load_wrapper = lambda: Wrapper()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(self.module.stage_infer())
            self.assertTrue(self.module.stage_contract())
        self.assertEqual(
            set(calls[0]),
            {"mixture_audio_path", "enrollment_audio_path", "output_path"},
        )
        document = json.loads(self.module.SAMPLE_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(document["rows"][0]["result"]["prediction_audio"].split("/", 1)[0], "artifacts")
        self.assertNotIn("reference_audio", document["rows"][0])

    def test_output_alias_and_reference_fields_are_rejected(self) -> None:
        fixtures = self.module.fixture_payloads()
        reference = Path(fixtures[0]["fixture"]["reference_audio_path"])
        with self.assertRaisesRegex(ValueError, "unapproved field"):
            self.module.validate_tse_output_object(
                {"prediction_audio": "out.wav", "reference_audio": str(reference)}
            )

        class AliasingWrapper:
            def predict(self, payload: dict[str, str]) -> dict[str, str]:
                return {"prediction_audio": payload["mixture_audio_path"]}

        self.module.load_wrapper = lambda: AliasingWrapper()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.module.stage_infer())

    def test_explicit_input_ignores_reference_and_output_fields(self) -> None:
        mixture = self.model_dir / "mixture.wav"
        enrollment = self.model_dir / "enrollment.wav"
        reference = self.model_dir / "reference.wav"
        write_wav(mixture)
        write_wav(enrollment)
        write_wav(reference)
        with mock.patch.dict(
            os.environ,
            {
                "SURE_VALIDATE_INPUT_JSON": json.dumps(
                    {
                        "mixture_audio_path": str(mixture),
                        "enrollment_audio_path": str(enrollment),
                        "reference_audio": str(reference),
                        "output_path": str(self.model_dir / "answer.wav"),
                    }
                )
            },
            clear=True,
        ):
            fixture = self.module.fixture_payloads()[0]
        self.assertEqual(
            fixture["input"],
            {
                "mixture_audio_path": str(mixture),
                "enrollment_audio_path": str(enrollment),
            },
        )

    def test_outer_gate_rejects_unreferenced_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "tse"
            artifacts = run_dir / "artifacts"
            fixture = root / "sure" / "models" / "example__tse" / "fixture" / "tse" / "smoke"
            fixture.mkdir(parents=True)
            for name in ("mixture.wav", "enrollment.wav", "reference.wav"):
                write_wav(fixture / name)
            (fixture / "gt.jsonl").write_text(
                json.dumps(
                    {
                        "key": "sample-1",
                        "sample_id": "sample-1",
                        "task_type": "tse",
                        "audio": "mixture.wav",
                        "mixture_audio": "mixture.wav",
                        "enrollment_audio": "enrollment.wav",
                        "reference_audio": "reference.wav",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_root = artifacts / "outputs"
            output_root.mkdir(parents=True)
            output_name = f"01-{hashlib.sha256(b'sample-1').hexdigest()[:12]}.wav"
            write_wav(output_root / output_name)
            (output_root / "stale.wav").write_bytes(b"stale")
            manifest = {
                "task_type": "tse",
                "staged_dir": str(fixture),
                "samples": [
                    {
                        "key": "sample-1",
                        "sample_id": "sample-1",
                        "mixture_audio": "mixture.wav",
                        "enrollment_audio": "enrollment.wav",
                        "reference_audio": "reference.wav",
                    }
                ],
            }
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            row = {
                "key": "sample-1",
                "sample_id": "sample-1",
                "result": {
                    "prediction_audio": f"artifacts/outputs/{output_name}",
                    "sample_id": "sample-1",
                },
            }
            (artifacts / "sample_outputs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            data = {
                "task_type": "tse",
                "model_dir": str(root / "sure" / "models" / "example__tse"),
                "io_contract": task_contract()["io_contract"],
            }
            violations = run_validate.validate_tse_evidence(data, run_dir)
            self.assertTrue(any("unreferenced" in item for item in violations), violations)

    def test_stage_model_artifacts_accepts_valid_tse_sample_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "model" / "artifacts"
            outputs = artifacts / "outputs"
            outputs.mkdir(parents=True)
            key = "sample-1"
            output_name = f"01-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}.wav"
            write_wav(outputs / output_name)
            row = {
                "key": key,
                "sample_id": key,
                "result": {
                    "prediction_audio": f"artifacts/outputs/{output_name}",
                    "sample_id": key,
                },
            }
            (artifacts / "sample_outputs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (artifacts / "sample_output.json").write_text(json.dumps({"rows": [row]}), encoding="utf-8")
            (artifacts / "fixture_manifest.json").write_text(
                json.dumps({
                    "task_type": "tse",
                    "samples": [{"key": key, "sample_id": key}],
                }),
                encoding="utf-8",
            )
            load_stage_model_artifacts().validate_tse_sample_evidence(artifacts)


if __name__ == "__main__":
    unittest.main()
