import importlib.util
import json
import os
import shutil
import tempfile
import unittest
import wave
from unittest import mock
from pathlib import Path

import run_validate
import check_fixture
import prepare_fixture


TEMPLATE = Path(__file__).parent / "templates" / "validate.py"
STAGE_MODEL_ARTIFACTS = Path(__file__).parent / "stage_model_artifacts.py"
SE_FIXTURE = Path(__file__).resolve().parents[4] / "fixtures" / "tasks" / "se" / "fleurs_noise_smoke"


def load_template():
    spec = importlib.util.spec_from_file_location("sure_onboard_validate_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load validate template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stage_model_artifacts():
    spec = importlib.util.spec_from_file_location("sure_onboard_stage_model_artifacts", STAGE_MODEL_ARTIFACTS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stage_model_artifacts")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidateTemplateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_template()
        self.temp_dir = tempfile.TemporaryDirectory()
        artifacts = Path(self.temp_dir.name) / "artifacts"
        self.module.ARTIFACTS_DIR = artifacts
        self.module.VALIDATION_LOG = artifacts / "validation.log"
        self.module.SAMPLE_OUTPUT = artifacts / "sample_output.json"
        self.module.SAMPLE_OUTPUTS = artifacts / "sample_outputs.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_json_preserves_arabic_text(self):
        path = self.module.ARTIFACTS_DIR / "arabic.json"
        self.module.write_json(path, {"text": "مرحبا بالعالم"})
        self.assertIn("مرحبا بالعالم", path.read_text(encoding="utf-8"))

    def test_classification_reference_normalizers_reject_non_scalar_values(self):
        normalizers = (
            prepare_fixture.normalize_classification_answer,
            check_fixture.normalized_classification_answer,
        )
        for normalize in normalizers:
            for value in ({"answer": "A"}, ["A"], True, float("nan"), float("inf")):
                with self.subTest(normalize=normalize.__module__, value=value):
                    with self.assertRaises(ValueError):
                        normalize(value)
        label_normalizers = (
            prepare_fixture.normalize_classification_label,
            check_fixture.normalized_classification_label,
        )
        for normalize in label_normalizers:
            for value in ({"label": "neu"}, ["neu"], True, float("nan"), float("inf")):
                with self.subTest(normalize=normalize.__module__, value=value):
                    with self.assertRaises(ValueError):
                        normalize("ser", value)

    def test_output_summary_is_complete_json(self):
        outputs = [{"text": "مرحبا " * 200} for _ in range(3)]
        summary = self.module.output_summary(outputs)
        parsed = json.loads(summary)
        self.assertEqual(parsed["sample_count"], 3)
        self.assertIsInstance(parsed["first_output"], dict)

    def test_infer_runs_every_fixture_and_writes_jsonl(self):
        fixtures = [
            {
                "input": {"audio_path": f"sample_{index}.wav", "language": "ar"},
                "fixture": {
                    "key": f"key-{index}",
                    "audio": f"sample_{index}.wav",
                    "dataset": "arabic-test",
                    "ground_truth": f"مرجع {index}",
                },
            }
            for index in range(1, 4)
        ]
        calls = []
        self.module.load_wrapper = lambda: object()
        self.module.fixture_payloads = lambda: fixtures

        def predict(_wrapper, payload):
            calls.append(payload)
            return {"text": f"نص {len(calls)}", "language": "ar"}

        self.module.run_predict = predict
        self.assertTrue(self.module.stage_infer())
        self.assertEqual(calls, [fixture["input"] for fixture in fixtures])
        rows = [json.loads(line) for line in self.module.SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["key"], "key-3")
        self.assertEqual(rows[2]["ground_truth"], "مرجع 3")
        self.assertEqual(rows[2]["dataset"], "arabic-test")
        self.assertEqual(rows[2]["output"]["text"], "نص 3")
        self.assertEqual(json.loads(self.module.SAMPLE_OUTPUT.read_text(encoding="utf-8"))["text"], "نص 1")
        infer_result = json.loads((self.module.ARTIFACTS_DIR / "infer_result.json").read_text(encoding="utf-8"))
        self.assertEqual(infer_result["sample_outputs_path"], "artifacts/sample_outputs.jsonl")

    def test_classification_output_rows_are_closed_and_reference_safe(self):
        self.module.TASK_TYPE = "ser"
        fixtures = [
            {
                "input": {"audio_path": "sample.wav"},
                "fixture": {
                    "key": "sample-1",
                    "audio": "sample.wav",
                    "dataset": "classification-demo",
                    "ground_truth": "neu",
                },
            }
        ]
        self.module.classification_fixture_payloads = lambda: fixtures
        valid = {
            "id": 1,
            "key": "sample-1",
            "task": "ser",
            "audio": "sample.wav",
            "dataset": "classification-demo",
            "ground_truth": "neu",
            "result": {"label": "neu"},
        }
        self.assertEqual(
            self.module.validate_classification_output_document({"rows": [valid]}, "ser"),
            [],
        )
        for field, value in (
            ("reference_audio", "/company/private.wav"),
            ("company_path", "/company/private"),
            ("debug", {"target": "private-reference"}),
        ):
            with self.subTest(field=field):
                bad = {**valid, field: value}
                violations = self.module.validate_classification_output_document(
                    {"rows": [bad]}, "ser"
                )
                self.assertTrue(violations)
                self.assertTrue(
                    any("unapproved" in item or "reference/path" in item for item in violations),
                    violations,
                )

    def test_classification_jsonl_mirror_rejects_row_metadata_leakage(self):
        self.module.TASK_TYPE = "ser"
        self.module.IO_CONTRACT = {
            "primary_field": "label",
            "required_fields": ["label"],
            "nonempty_fields": ["label"],
            "json_serializable": True,
        }
        fixtures = [
            {
                "input": {"audio_path": "sample.wav"},
                "fixture": {
                    "key": "sample-1",
                    "audio": "sample.wav",
                    "dataset": "classification-demo",
                    "ground_truth": "neu",
                },
            }
        ]
        self.module.classification_fixture_payloads = lambda: fixtures
        row = {
            "id": 1,
            "key": "sample-1",
            "task": "ser",
            "audio": "sample.wav",
            "dataset": "classification-demo",
            "ground_truth": "neu",
            "result": {"label": "neu"},
        }
        self.module.write_json(self.module.SAMPLE_OUTPUT, {"rows": [row]})
        self.module.write_jsonl(self.module.SAMPLE_OUTPUTS, [row])
        self.assertTrue(self.module.stage_contract())

        leaked = {**row, "company_path": "/company/private"}
        self.module.write_jsonl(self.module.SAMPLE_OUTPUTS, [leaked])
        self.assertFalse(self.module.stage_contract())
        contract_result = json.loads(
            (self.module.ARTIFACTS_DIR / "contract_result.json").read_text(encoding="utf-8")
        )
        self.assertIn("reference/path", contract_result["error"])

    def test_outer_gate_rechecks_classification_evidence_and_mirror(self):
        root = Path(self.temp_dir.name)
        run_dir = root / ".sure" / "runs" / "ser-gate"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True)
        model_dir = root / "sure" / "models" / "example__ser"
        fixture_dir = model_dir / "fixture" / "ser" / "smoke"
        fixture_dir.mkdir(parents=True)
        (model_dir / "model.py").write_text("# test\n", encoding="utf-8")
        audio = fixture_dir / "sample.wav"
        audio.write_bytes(b"fixture")
        gt = {
            "key": "sample-1",
            "task_type": "ser",
            "audio": "sample.wav",
            "ground_truth": "neu",
            "dataset": "classification-demo",
        }
        (fixture_dir / "gt.jsonl").write_text(json.dumps(gt) + "\n", encoding="utf-8")
        manifest = {
            "model_dir": str(model_dir),
            "task_type": "ser",
            "staged_dir": str(fixture_dir),
            "gt_jsonl": str(fixture_dir / "gt.jsonl"),
            "sample_count": 1,
            "samples": [
                {
                    "key": "sample-1",
                    "audio": "sample.wav",
                    "audio_path": str(audio),
                    "annotation_fields": ["ground_truth"],
                }
            ],
        }
        (artifacts / "fixture_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        contract = {
            "input_type": "audio_path",
            "output_type": "json",
            "primary_field": "label",
            "required_fields": ["label"],
            "nonempty_fields": ["label"],
            "json_serializable": True,
        }
        (artifacts / "model_input_resolved.json").write_text(
            json.dumps(
                {
                    "task_type": "ser",
                    "model_dir": str(model_dir),
                    "normalized_model_input": {"io_contract": contract},
                }
            ),
            encoding="utf-8",
        )
        row = {
            "id": 1,
            "key": "sample-1",
            "task": "ser",
            "audio": "sample.wav",
            "dataset": "classification-demo",
            "ground_truth": "neu",
            "result": {"label": "neu"},
        }
        sample_output = artifacts / "sample_output.json"
        sample_outputs = artifacts / "sample_outputs.jsonl"
        sample_output.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
        sample_outputs.write_text(json.dumps(row) + "\n", encoding="utf-8")
        gate_data = {
            "model_dir": str(model_dir),
            "io_contract": contract,
            "sample_output_path": str(sample_output),
            "sample_outputs_path": str(sample_outputs),
        }
        self.assertEqual(run_validate.validate_classification_evidence(gate_data, run_dir), [])

        leaked = {**row, "company_path": "/company/private"}
        sample_outputs.write_text(json.dumps(leaked) + "\n", encoding="utf-8")
        violations = run_validate.validate_classification_evidence(gate_data, run_dir)
        self.assertTrue(any("reference/path" in item or "unapproved" in item for item in violations))

    def test_fixture_payloads_use_first_selected_set_and_preserve_metadata(self):
        model_dir = Path(self.temp_dir.name) / "model"
        first = model_dir / "fixture" / "asr" / "a-selected"
        second = model_dir / "fixture" / "asr" / "z-other"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / "one.wav").write_bytes(b"wav")
        (second / "other.wav").write_bytes(b"wav")
        (first / "gt.jsonl").write_text(
            json.dumps(
                {
                    "key": "arabic-1",
                    "audio": "one.wav",
                    "language": "ar",
                    "dataset": "arabic-test",
                    "ground_truth": "النص المرجعي",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (second / "gt.jsonl").write_text(
            json.dumps({"key": "other", "audio": "other.wav", "ground_truth": "other"}) + "\n",
            encoding="utf-8",
        )
        self.module.MODEL_DIR = model_dir
        fixtures = self.module.fixture_payloads()
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0]["fixture"]["key"], "arabic-1")
        self.assertEqual(fixtures[0]["fixture"]["ground_truth"], "النص المرجعي")

    def test_kws_fixture_payloads_preserve_keywords_and_require_both_polarities(self):
        model_dir = Path(self.temp_dir.name) / "model"
        fixture_dir = model_dir / "fixture" / "kws" / "smoke"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "positive.wav").write_bytes(b"wav")
        (fixture_dir / "negative.wav").write_bytes(b"wav")
        rows = [
            {
                "key": "positive",
                "audio": "positive.wav",
                "keywords": ["你好问问", "嗨小问"],
                "expected": "detect",
                "label": "positive",
                "expected_detected": True,
                "text": "嗨小问",
            },
            {
                "key": "negative",
                "audio": "negative.wav",
                "keywords": ["你好问问", "嗨小问"],
                "expected": "reject",
                "label": "negative",
                "expected_detected": False,
            },
        ]
        (fixture_dir / "gt.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.module.MODEL_DIR = model_dir
        self.module.TASK_TYPE = "kws"

        fixtures = self.module.fixture_payloads()

        self.assertEqual(len(fixtures), 2)
        self.assertEqual(fixtures[0]["input"]["keywords"], ["你好问问", "嗨小问"])
        self.assertEqual(fixtures[0]["fixture"]["expected"], "detect")
        self.assertEqual(fixtures[1]["fixture"]["expected"], "reject")

        rows[0]["label"] = "negative"
        (fixture_dir / "gt.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "conflicting KWS polarity fields"):
            self.module.fixture_payloads()

        rows[0]["label"] = "maybe"
        (fixture_dir / "gt.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "unsupported label value"):
            self.module.fixture_payloads()

        rows[0]["label"] = "positive"
        for row in rows:
            row.pop("keywords")
        (fixture_dir / "gt.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        fixed_keyword_fixtures = self.module.fixture_payloads()
        self.assertNotIn("keywords", fixed_keyword_fixtures[0]["input"])

    def test_kws_contract_checks_every_positive_and_negative_output(self):
        contract = {
            "output_type": "keyword_detection",
            "primary_field": "detected",
            "required_fields": ["detected", "keyword", "score"],
            "nonempty_fields": ["detected"],
            "json_serializable": True,
        }
        fixtures = [
            {
                "input": {"audio_path": "positive.wav", "keywords": ["嗨小问"]},
                "fixture": {
                    "key": "positive",
                    "audio": "positive.wav",
                    "expected": "detect",
                    "text": "嗨小问",
                },
            },
            {
                "input": {"audio_path": "negative.wav", "keywords": ["嗨小问"]},
                "fixture": {
                    "key": "negative",
                    "audio": "negative.wav",
                    "expected": "reject",
                },
            },
        ]
        calls = []
        self.module.TASK_TYPE = "kws"
        self.module.IO_CONTRACT = contract
        self.module.load_wrapper = lambda: object()
        self.module.fixture_payloads = lambda: fixtures

        def predict(_wrapper, payload):
            calls.append(payload)
            if payload["audio_path"] == "positive.wav":
                return {"detected": True, "keyword": "嗨小问", "score": 0.9}
            return {"detected": False, "keyword": None, "score": None}

        self.module.run_predict = predict
        self.assertTrue(self.module.stage_infer())
        self.assertTrue(self.module.stage_contract())
        self.assertEqual([call["keywords"] for call in calls], [["嗨小问"], ["嗨小问"]])

        output_rows = [
            json.loads(line)
            for line in self.module.SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
        ]
        output_rows[1]["output"] = {"detected": True, "keyword": "嗨小问", "score": 0.8}
        self.module.write_jsonl(self.module.SAMPLE_OUTPUTS, output_rows)
        self.assertFalse(self.module.stage_contract())
        result = json.loads((self.module.ARTIFACTS_DIR / "contract_result.json").read_text(encoding="utf-8"))
        self.assertIn("must reject the negative fixture", result["error"])

    def test_kws_single_output_contract_accepts_false_and_rejects_non_finite_score(self):
        self.assertEqual(
            self.module.validate_kws_output(
                {"detected": False, "keyword": None, "score": None}
            ),
            [],
        )
        violations = self.module.validate_kws_output(
            {"detected": False, "keyword": None, "score": float("nan")}
        )
        self.assertIn("score must be a finite number or null", violations)
        self.assertIn(
            "detected=true requires score >= 0.5",
            self.module.validate_kws_output(
                {"detected": True, "keyword": "嗨小问", "score": 0.4}
            ),
        )
        self.assertIn(
            "detected=false requires score < 0.5",
            self.module.validate_kws_output(
                {"detected": False, "keyword": None, "score": 0.6}
            ),
        )
        positive_without_score = self.module.validate_kws_output(
            {"detected": True, "keyword": "嗨小问", "score": None}
        )
        self.assertIn("detected=true requires a finite numeric score", positive_without_score)

    def test_kws_single_input_preserves_optional_keywords(self):
        self.module.TASK_TYPE = "kws"
        with mock.patch.dict(
            os.environ,
            {
                "SURE_VALIDATE_INPUT_JSON": json.dumps(
                    {
                        "audio_path": "negative.wav",
                        "keywords": ["嗨小问"],
                    },
                    ensure_ascii=False,
                )
            },
            clear=False,
        ):
            fixtures = self.module.fixture_payloads()
        self.assertEqual(fixtures[0]["input"]["keywords"], ["嗨小问"])

        with mock.patch.dict(
            os.environ,
            {"SURE_VALIDATE_INPUT_JSON": json.dumps({"audio_path": "negative.wav"})},
            clear=False,
        ):
            fixtures = self.module.fixture_payloads()
        self.assertNotIn("keywords", fixtures[0]["input"])

        with mock.patch.dict(
            os.environ,
            {
                "SURE_VALIDATE_INPUT_JSON": json.dumps(
                    {"audio_path": "negative.wav", "keywords": []}
                )
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "keywords"):
                self.module.fixture_payloads()

    def test_se_validation_uses_only_noisy_input_and_checks_real_output_audio(self):
        model_dir = Path(self.temp_dir.name) / "model"
        fixture_dir = model_dir / "fixture" / "se" / "fleurs_noise_smoke"
        shutil.copytree(SE_FIXTURE, fixture_dir)
        self.module.MODEL_DIR = model_dir
        self.module.TASK_TYPE = "speech-enhancement"
        self.module.IO_CONTRACT = {
            "input_type": "audio_path",
            "output_type": "audio",
            "primary_field": "audio_path",
            "required_fields": ["audio_path"],
            "nonempty_fields": ["audio_path"],
            "json_serializable": True,
        }

        fixtures = self.module.fixture_payloads()
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(set(fixtures[0]["input"]), {"audio_path"})
        self.assertTrue(fixtures[0]["input"]["audio_path"].endswith("noisy.wav"))
        self.assertEqual(fixtures[0]["fixture"]["reference_audio"], "clean.wav")

        calls = []
        enhanced = self.module.se_output_path("fleurs_ar_eg_1993_noise_10db", 1)
        self.module.load_wrapper = lambda: object()

        def predict(_wrapper, payload):
            calls.append(dict(payload))
            enhanced.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(enhanced), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 160)
            return {"audio_path": str(enhanced)}

        self.module.run_predict = predict
        self.assertTrue(self.module.stage_infer())
        self.assertTrue(self.module.stage_contract())
        self.assertEqual(calls[0]["audio_path"], fixtures[0]["input"]["audio_path"])
        self.assertEqual(calls[0]["output_path"], str(enhanced))
        self.assertNotIn("reference_audio_path", calls[0])
        rows = [
            json.loads(line)
            for line in self.module.SAMPLE_OUTPUTS.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(rows[0]["audio"], "noisy.wav")
        self.assertEqual(rows[0]["reference_audio"], "clean.wav")

        enhanced.write_bytes(b"")
        self.assertFalse(self.module.stage_contract())
        result = json.loads(
            (self.module.ARTIFACTS_DIR / "contract_result.json").read_text(encoding="utf-8")
        )
        self.assertIn("SE output audio_path is empty", result["error"])

    def test_se_explicit_input_does_not_forward_reference_audio(self):
        self.module.TASK_TYPE = "acoustic-noise-suppression"
        with mock.patch.dict(
            os.environ,
            {
                "SURE_VALIDATE_INPUT_JSON": json.dumps(
                    {
                        "audio_path": "noisy.wav",
                        "reference_audio": "clean.wav",
                        "output_path": "enhanced.wav",
                    }
                )
            },
            clear=False,
        ):
            fixtures = self.module.fixture_payloads()
        self.assertEqual(
            fixtures[0]["input"],
            {"audio_path": "noisy.wav"},
        )
        self.assertEqual(fixtures[0]["fixture"]["reference_audio"], "clean.wav")

    def test_se_rejects_input_alias_external_path_symlink_and_non_wav(self):
        model_dir = Path(self.temp_dir.name) / "model"
        fixture_dir = model_dir / "fixture" / "se" / "fleurs_noise_smoke"
        shutil.copytree(SE_FIXTURE, fixture_dir)
        self.module.MODEL_DIR = model_dir
        self.module.TASK_TYPE = "se"
        self.module.IO_CONTRACT = {
            "input_type": "audio_path",
            "output_type": "audio",
            "primary_field": "audio_path",
            "required_fields": ["audio_path"],
            "nonempty_fields": ["audio_path"],
            "json_serializable": True,
        }
        self.module.load_wrapper = lambda: object()

        self.module.run_predict = lambda _wrapper, payload: {
            "audio_path": payload["audio_path"]
        }
        self.assertFalse(self.module.stage_infer())

        def non_wav(_wrapper, payload):
            output = Path(payload["output_path"])
            output.write_bytes(b"not-a-wave")
            return {"audio_path": str(output)}

        self.module.run_predict = non_wav
        self.assertFalse(self.module.stage_infer())

        external = Path(self.temp_dir.name) / "outside.wav"
        with wave.open(str(external), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 8)
        self.module.run_predict = lambda _wrapper, _payload: {
            "audio_path": str(external)
        }
        self.assertFalse(self.module.stage_infer())

        target = Path(self.temp_dir.name) / "target.wav"
        shutil.copy2(external, target)

        def symlink_output(_wrapper, payload):
            output = Path(payload["output_path"])
            output.symlink_to(target)
            return {"audio_path": str(output)}

        self.module.run_predict = symlink_output
        self.assertFalse(self.module.stage_infer())

        def hardlink_input(_wrapper, payload):
            output = Path(payload["output_path"])
            os.link(payload["audio_path"], output)
            return {"audio_path": str(output)}

        self.module.run_predict = hardlink_input
        self.assertFalse(self.module.stage_infer())

    def test_sample_outputs_jsonl_is_staged(self):
        stage_model_artifacts = load_stage_model_artifacts()
        self.assertIn("sample_outputs.jsonl", stage_model_artifacts.OPTIONAL_RUN_ARTIFACTS)


if __name__ == "__main__":
    unittest.main()
