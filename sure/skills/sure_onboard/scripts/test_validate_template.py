import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path


TEMPLATE = Path(__file__).parent / "templates" / "validate.py"
STAGE_MODEL_ARTIFACTS = Path(__file__).parent / "stage_model_artifacts.py"


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

    def test_sample_outputs_jsonl_is_staged(self):
        stage_model_artifacts = load_stage_model_artifacts()
        self.assertIn("sample_outputs.jsonl", stage_model_artifacts.OPTIONAL_RUN_ARTIFACTS)


if __name__ == "__main__":
    unittest.main()
