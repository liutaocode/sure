#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_predictions as ep  # noqa: E402
import validate_prediction_files as vp  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
ENGINE_SRC = REPO_ROOT / "sure" / "external" / "sure-evaluation" / "src"
KWS_PIPELINE = {
    "task": "kws",
    "language": "n/a",
    "metric": "accuracy",
    "pipeline_id": "kws.any.accuracy.conversion_kws_sure_json_to_samples_v1.wekws_det_v1",
    "required_roles": ["reference_jsonl", "sample_output"],
    "run_args": {
        "reference_jsonl": None,
        "sample_output": None,
        "output_dir": None,
    },
    "nodes": [{"node_id": "scoring/wekws_det"}],
}


class FakeDatasetManager:
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path

    def normalize_dataset_name(self, name: str) -> str:
        return name

    def get_jsonl_path(self, _: str) -> Path:
        return self.jsonl_path

    def download_and_convert(self, _: str) -> Path:
        raise AssertionError("test dataset must already exist")


class FakeSOTAManager:
    def __init__(self, baseline: object | None = None) -> None:
        self.baseline = baseline
        self.calculate_calls = 0

    def get_metric(self, _: str, fallback_names: object = ()) -> None:
        return None

    def get_baseline(self, _: str, fallback_names: object = ()) -> object | None:
        return self.baseline

    def calculate_rps(
        self,
        dataset: str,
        score: object,
        fallback_names: object = (),
    ) -> float | dict[str, object]:
        self.calculate_calls += 1
        if self.baseline is not None:
            return 0.75
        return {
            "status": "missing_baseline",
            "dataset": dataset,
            "score": score,
        }


class FakeBaseline:
    def __init__(self, metric: str) -> None:
        self.metric = metric


class FakeRecordDatabase:
    def __init__(self) -> None:
        self.records: list[object] = []

    def add_record(self, record: object) -> None:
        self.records.append(record)


class FakeRecordManager:
    def __init__(self) -> None:
        self.database = FakeRecordDatabase()
        self.evaluate_calls = 0

    def evaluate_and_record(self, **kwargs: object) -> object:
        self.evaluate_calls += 1
        return kwargs


def _structured_row(key: str, prediction: dict[str, object]) -> dict[str, object]:
    projection = json.dumps(prediction, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {
        "key": key,
        "dataset": "demo",
        "task": "KWS",
        "language": "zh",
        "prediction": prediction,
        "normalized_prediction": projection,
        "raw_response": prediction,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class KWSRoleFileTests(unittest.TestCase):
    def test_writes_jsonl_reference_and_json_array_output(self) -> None:
        samples = [
            {
                "key": "pos",
                "task": "KWS",
                "expected_detected": True,
                "expected_keyword": "wake",
                "duration": 1.0,
                "audio": "pos.wav",
            },
            {
                "key": "neg",
                "task": "KWS",
                "expected_detected": False,
                "expected_keyword": None,
                "duration": 2.0,
                "audio": "neg.wav",
            },
        ]
        predictions = {
            "pos": _structured_row(
                "pos",
                {
                    "detected": True,
                    "keyword": "wake",
                    "score": 0.9,
                    "events": [{"keyword": "wake", "score": 0.9}],
                },
            ),
            "neg": _structured_row(
                "neg",
                {"detected": False, "keyword": None, "score": None},
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            reference_path, sample_output_path = ep._write_external_kws_role_files(
                samples=samples,
                structured_predictions=predictions,
                output_dir=Path(tmp),
            )
            references = ep.load_jsonl(reference_path)
            outputs = json.loads(sample_output_path.read_text(encoding="utf-8"))

        self.assertEqual([row["key"] for row in references], ["pos", "neg"])
        self.assertEqual(references[1]["expected_detected"], False)
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs[0]["result"]["events"], [{"keyword": "wake", "score": 0.9}])
        self.assertEqual(
            outputs[1],
            {
                "key": "neg",
                "result": {"detected": False, "keyword": None, "score": None},
            },
        )

    def test_missing_reference_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "missing an expected label"
        ):
            ep._write_external_kws_role_files(
                samples=[{"key": "unknown", "task": "KWS", "path": "unknown.wav"}],
                structured_predictions={
                    "unknown": _structured_row(
                        "unknown", {"detected": False, "keyword": None, "score": None}
                    )
                },
                output_dir=Path(tmp),
            )

    def test_invalid_or_conflicting_reference_labels_are_rejected(self) -> None:
        invalid_samples = [
            {"key": "typo", "task": "KWS", "expected": "detcet"},
            {
                "key": "conflict",
                "task": "KWS",
                "expected": True,
                "label": "reject",
                "expected_detected": True,
            },
        ]
        for sample in invalid_samples:
            key = str(sample["key"])
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "invalid|conflicting"):
                    ep._write_external_kws_role_files(
                        samples=[sample],
                        structured_predictions={
                            key: _structured_row(
                                key, {"detected": False, "keyword": None, "score": None}
                            )
                        },
                        output_dir=Path(tmp),
                    )

    def test_equivalent_negative_reference_labels_are_allowed(self) -> None:
        sample = {
            "key": "neg",
            "task": "KWS",
            "expected": "reject",
            "label": "0",
            "expected_detected": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            reference_path, _ = ep._write_external_kws_role_files(
                samples=[sample],
                structured_predictions={
                    "neg": _structured_row(
                        "neg", {"detected": False, "keyword": None, "score": 0.1}
                    )
                },
                output_dir=Path(tmp),
            )
            reference = ep.load_jsonl(reference_path)[0]
        self.assertEqual(reference["expected"], "reject")
        self.assertEqual(reference["label"], "0")
        self.assertIs(reference["expected_detected"], False)

    def test_canonical_path_populates_audio_without_overriding_audio_or_wav(self) -> None:
        samples = [
            {
                "key": "path-only",
                "task": "KWS",
                "expected_detected": False,
                "path": "/canonical/path.wav",
            },
            {
                "key": "audio-first",
                "task": "KWS",
                "expected_detected": False,
                "audio": "/explicit/audio.wav",
                "path": "/canonical/ignored.wav",
            },
        ]
        predictions = {
            sample["key"]: _structured_row(
                str(sample["key"]), {"detected": False, "keyword": None, "score": None}
            )
            for sample in samples
        }
        with tempfile.TemporaryDirectory() as tmp:
            reference_path, _ = ep._write_external_kws_role_files(
                samples=samples,
                structured_predictions=predictions,
                output_dir=Path(tmp),
            )
            references = ep.load_jsonl(reference_path)
        self.assertEqual(references[0]["audio"], "/canonical/path.wav")
        self.assertEqual(references[1]["audio"], "/explicit/audio.wav")

    def test_malformed_prediction_types_are_rejected(self) -> None:
        invalid_predictions = [
            {"detected": "false", "keyword": None, "score": 0.1},
            {"detected": False, "keyword": 3, "score": 0.1},
            {"detected": False, "keyword": None, "score": True},
            {"detected": False, "keyword": None, "score": float("inf")},
            {"detected": False, "keyword": None, "score": 0.1, "events": {}},
            {"events": []},
            {"detected": True, "keyword": None, "score": 0.9},
            {"detected": True, "keyword": "", "score": 0.9},
            {"detected": True, "keyword": "wake", "score": None},
            {"detected": False, "keyword": "wake", "score": 0.1},
            {"detected": True, "keyword": "wake", "score": 0.49},
            {"detected": False, "keyword": None, "score": 0.5},
            {"detected": False, "keyword": None, "score": -0.1},
            {"detected": True, "keyword": "wake", "score": 1.1},
        ]
        sample = {"key": "neg", "task": "KWS", "expected_detected": False}
        for prediction in invalid_predictions:
            with self.subTest(prediction=prediction), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    ep._write_external_kws_role_files(
                        samples=[sample],
                        structured_predictions={"neg": _structured_row("neg", prediction)},
                        output_dir=Path(tmp),
                    )

    def test_macro_recall_role_requires_rejected_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "requires a score"
        ):
            ep._write_external_kws_role_files(
                samples=[{"key": "neg", "task": "KWS", "expected_detected": False}],
                structured_predictions={
                    "neg": _structured_row(
                        "neg", {"detected": False, "keyword": None, "score": None}
                    )
                },
                output_dir=Path(tmp),
                require_scores=True,
            )


@unittest.skipUnless(ENGINE_SRC.is_dir(), "sure-evaluation submodule is not checked out")
class KWSStandaloneSemanticsTests(unittest.TestCase):
    def _evaluate(
        self,
        samples: list[dict[str, object]],
        predictions: dict[str, dict[str, object]],
        *,
        metric: str = "accuracy",
    ) -> dict[str, object]:
        code = """
import json
import sys
from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_files

report = evaluate_kws_files(
    reference_jsonl=sys.argv[1],
    sample_output=sys.argv[2],
    metric=sys.argv[3],
    thresholds=[0.5],
)
print(json.dumps({"score": report.score, "rows": report.details["rows"]}))
"""
        with tempfile.TemporaryDirectory() as tmp:
            reference_path, sample_output_path = ep._write_external_kws_role_files(
                samples=samples,
                structured_predictions={
                    key: _structured_row(key, prediction) for key, prediction in predictions.items()
                },
                output_dir=Path(tmp),
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ENGINE_SRC)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-c", code, str(reference_path), str(sample_output_path), metric],
                cwd=REPO_ROOT / "sure" / "external" / "sure-evaluation",
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_positive_and_negative_samples_score_correctly(self) -> None:
        payload = self._evaluate(
            [
                {"key": "pos", "expected_detected": True, "expected_keyword": "wake", "duration": 1.0},
                {"key": "neg", "expected_detected": False, "duration": 2.0},
            ],
            {
                "pos": {"detected": True, "keyword": "wake", "score": 0.9},
                "neg": {"detected": False, "keyword": None, "score": None},
            },
        )
        self.assertEqual(payload["score"], 1.0)
        self.assertEqual([row["error_type"] for row in payload["rows"]], [None, None])

    def test_all_negative_set_keeps_zero_macro_recall(self) -> None:
        payload = self._evaluate(
            [{"key": "neg", "expected_detected": False, "duration": 2.0}],
            {"neg": {"detected": False, "keyword": None, "score": 0}},
            metric="macro_recall",
        )
        self.assertEqual(payload["score"], 0.0)
        self.assertIsNone(payload["rows"][0]["error_type"])

    def test_wrong_keyword_is_a_false_reject(self) -> None:
        payload = self._evaluate(
            [{"key": "pos", "expected_detected": True, "expected_keyword": "wake"}],
            {"pos": {"detected": True, "keyword": "other", "score": 0.9}},
        )
        self.assertEqual(payload["score"], 0.0)
        self.assertEqual(payload["rows"][0]["error_type"], "wrong_keyword")


class KWSValidationTests(unittest.TestCase):
    def test_valid_negative_is_nonempty_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            prediction_path = root / "demo.txt"
            structured_path = root / "demo.jsonl"
            sample = {"key": "neg", "task": "KWS", "expected_detected": False}
            prediction = {"detected": False, "keyword": None, "score": None}
            row = _structured_row("neg", prediction)
            _write_jsonl(dataset_path, [sample])
            prediction_path.write_text(
                f"neg\t{row['normalized_prediction']}\n",
                encoding="utf-8",
            )
            _write_jsonl(structured_path, [row])

            result = vp.validate_prediction_file(
                FakeDatasetManager(dataset_path),
                "demo",
                prediction_path,
                require_nonempty=True,
            )

        self.assertTrue(result["is_valid"], result)
        self.assertEqual(result["empty_prediction_keys"], [])
        self.assertEqual(result["contract_violation_keys"], [])

    def test_events_do_not_replace_direct_fields(self) -> None:
        samples = [{"key": "sample", "task": "KWS", "expected_detected": True}]
        structured = {
            "sample": {
                "key": "sample",
                "task": "KWS",
                "prediction": {"events": [{"keyword": "wake", "score": 0.9}]},
                "normalized_prediction": "{}",
            }
        }
        self.assertEqual(vp._task_contract_violations(samples, structured), ["sample"])

    def test_invalid_or_conflicting_reference_labels_are_contract_violations(self) -> None:
        invalid_samples = [
            {"key": "sample", "task": "KWS", "expected": "typo"},
            {
                "key": "sample",
                "task": "KWS",
                "expected": True,
                "expected_detected": False,
            },
        ]
        structured = {
            "sample": _structured_row(
                "sample", {"detected": False, "keyword": None, "score": None}
            )
        }
        for sample in invalid_samples:
            with self.subTest(sample=sample):
                self.assertEqual(vp._task_contract_violations([sample], structured), ["sample"])

    def test_malformed_types_are_contract_violations(self) -> None:
        samples = [{"key": "sample", "task": "KWS", "expected_detected": False}]
        invalid_predictions = [
            {"detected": "false", "keyword": None, "score": 0.1},
            {"detected": False, "keyword": [], "score": 0.1},
            {"detected": False, "keyword": None, "score": "0.1"},
            {"detected": False, "keyword": None, "score": float("nan")},
            {"detected": False, "keyword": None, "score": 0.1, "events": {}},
            {"detected": True, "keyword": None, "score": 0.9},
            {"detected": True, "keyword": "  ", "score": 0.9},
            {"detected": True, "keyword": "wake", "score": None},
            {"detected": False, "keyword": "wake", "score": 0.1},
            {"detected": True, "keyword": "wake", "score": 0.49},
            {"detected": False, "keyword": None, "score": 0.5},
            {"detected": False, "keyword": None, "score": -0.1},
            {"detected": True, "keyword": "wake", "score": 1.1},
        ]
        for prediction in invalid_predictions:
            with self.subTest(prediction=prediction):
                structured = {
                    "sample": {
                        "key": "sample",
                        "task": "KWS",
                        "prediction": prediction,
                        "normalized_prediction": "invalid",
                    }
                }
                self.assertEqual(vp._task_contract_violations(samples, structured), ["sample"])

    def test_macro_recall_validation_requires_rejected_score(self) -> None:
        samples = [{"key": "neg", "task": "KWS", "expected_detected": False}]
        structured = {
            "neg": _structured_row(
                "neg", {"detected": False, "keyword": None, "score": None}
            )
        }
        self.assertEqual(
            vp._task_contract_violations(samples, structured, kws_require_score=True),
            ["neg"],
        )


class KWSSampleReportTests(unittest.TestCase):
    def test_sample_report_uses_structured_prediction_and_metric_rows(self) -> None:
        sample = {
            "key": "pos",
            "task": "KWS",
            "expected_detected": True,
            "expected_keyword": "wake",
            "duration": 1.0,
            "audio": "pos.wav",
        }
        prediction = {"detected": True, "keyword": "other", "score": 0.9}
        result = {
            "dataset": "fixture-kws",
            "task": "KWS",
            "metric": "accuracy",
            "details": {
                "report": {
                    "details": {
                        "rows": [
                            {
                                "key": "pos",
                                "correct": False,
                                "error_type": "wrong_keyword",
                            }
                        ]
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample-report.jsonl"
            ep._write_sample_report(
                output_path=output,
                samples=[sample],
                predictions={"pos": "compact TSV must not become the report prediction"},
                result=result,
                structured_predictions={"pos": _structured_row("pos", prediction)},
            )
            row = ep.load_jsonl(output)[0]

        self.assertEqual(row["prediction"], prediction)
        self.assertEqual(
            row["reference"],
            {
                "expected_detected": True,
                "expected_keyword": "wake",
                "duration": 1.0,
                "audio": "pos.wav",
            },
        )
        self.assertEqual(row["metric_details"]["error_type"], "wrong_keyword")

    def test_run_artifacts_load_the_structured_prediction_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir(parents=True)
            dataset_path = root / "dataset.jsonl"
            prediction_path = predictions_dir / "fixture-kws.txt"
            sample = {
                "key": "pos",
                "task": "KWS",
                "language": "zh",
                "expected_detected": True,
                "expected_keyword": "wake",
                "path": "/canonical/pos.wav",
            }
            prediction = {"detected": True, "keyword": "wake", "score": 0.9}
            structured = _structured_row("pos", prediction)
            _write_jsonl(dataset_path, [sample])
            prediction_path.write_text("pos\tcompact TSV\n", encoding="utf-8")
            _write_jsonl(prediction_path.with_suffix(".jsonl"), [structured])
            result = {
                "dataset": "fixture-kws",
                "jsonl_path": str(dataset_path),
                "prediction_path": str(prediction_path),
                "task": "KWS",
                "language": "zh",
                "metric": "accuracy",
                "score": 1.0,
                "rps": None,
                "rps_status": {
                    "status": "missing_metric_baseline",
                    "metric": "accuracy",
                    "available_baseline_metric": "macro_recall",
                },
                "num_samples": 1,
                "evaluation_backend": "external",
                "evaluator_version": "sure-evaluation",
                "pipeline_id": KWS_PIPELINE["pipeline_id"],
                "evaluation_context": {"nodes": ["scoring/wekws_det"]},
                "details": {
                    "summary": {},
                    "pipeline": dict(KWS_PIPELINE),
                    "report": {
                        "details": {
                            "rows": [{"key": "pos", "correct": True, "error_type": None}]
                        }
                    },
                },
            }
            payload = ep._evaluation_payload_v2(
                evaluation_backend="external",
                external_engine=None,
                results=[result],
            )
            with mock.patch.object(ep, "_write_protocol_yaml"):
                ep._write_run_artifacts(
                    run_dir=run_dir,
                    tool_name="kws_predict",
                    protocol_id="standard_system",
                    model_dir=None,
                    payload=payload,
                    results=[result],
                )
            row = ep.load_jsonl(
                run_dir / "sample_reports" / "fixture-kws" / "accuracy.jsonl"
            )[0]
            payload_row = json.loads(
                (run_dir / "evaluation_payload.json").read_text(encoding="utf-8")
            )["results"][0]
            report_row = ep.load_jsonl(run_dir / "report.jsonl")[0]

        self.assertEqual(row["prediction"], prediction)
        self.assertEqual(row["reference"]["expected_keyword"], "wake")
        self.assertEqual(row["reference"]["audio"], "/canonical/pos.wav")
        self.assertTrue(row["metric_details"]["correct"])
        self.assertIsNone(payload_row["rps"])
        self.assertEqual(payload_row["rps_status"]["status"], "missing_metric_baseline")
        self.assertIsNone(report_row["rps"])
        self.assertEqual(report_row["rps_status"], payload_row["rps_status"])


class KWSRPSRecordTests(unittest.TestCase):
    @staticmethod
    def _result(
        *,
        task: str = "KWS",
        rps: object = None,
        rps_status: object = None,
    ) -> dict[str, object]:
        return {
            "dataset": "fixture-kws",
            "task": task,
            "metric": "macro_recall",
            "score": 0.5,
            "rps": rps,
            "rps_status": rps_status,
            "num_samples": 2,
            "prediction_path": "/tmp/fixture-kws.txt",
            "details": {},
        }

    def test_macro_recall_does_not_use_accuracy_baseline(self) -> None:
        sota = FakeSOTAManager(FakeBaseline("accuracy"))
        rps, status = ep._calculate_metric_rps(sota, "fixture-kws", "macro_recall", 0.5)
        self.assertIsNone(rps)
        self.assertEqual(status["status"], "missing_metric_baseline")
        self.assertEqual(status["available_baseline_metric"], "accuracy")
        self.assertEqual(sota.calculate_calls, 0)

    def test_matching_metric_baseline_calculates_rps(self) -> None:
        sota = FakeSOTAManager(FakeBaseline("macro-recall"))
        rps, status = ep._calculate_metric_rps(sota, "fixture-kws", "macro_recall", 0.5)
        self.assertEqual(rps, 0.75)
        self.assertIsNone(status)
        self.assertEqual(sota.calculate_calls, 1)

    def test_kws_fraction_metrics_use_fraction_unit(self) -> None:
        for metric in ("accuracy", "precision", "recall", "macro_recall", "f1"):
            with self.subTest(metric=metric):
                self.assertEqual(ep._metric_unit(metric), "fraction")

    def test_standard_report_rps_fields_match_report_row_schema(self) -> None:
        status = {"status": "missing_metric_baseline", "metric": "macro_recall"}
        payload_row = ep._dataset_metric_row(self._result(rps_status=status))
        report_row = ep._standard_report_row_v1(
            row=payload_row,
            validation={},
            run_id="run",
            protocol_id="standard_system",
            model_dir=None,
            tool_name="kws_predict",
        )
        schema = json.loads(
            (REPO_ROOT / "sure" / "skills" / "sure_eval" / "schemas" / "report_row.schema.json")
            .read_text(encoding="utf-8")
        )
        self.assertTrue(set(schema["required"]).issubset(report_row))
        self.assertEqual(schema["properties"]["rps"]["type"], ["number", "null"])
        self.assertEqual(
            schema["properties"]["rps_status"]["type"], ["object", "null"]
        )
        self.assertIsNone(report_row["rps"])
        self.assertEqual(report_row["rps_status"], status)

    def test_record_keeps_missing_metric_baseline_without_fabricating_rps(self) -> None:
        manager = FakeRecordManager()
        status = {
            "status": "missing_metric_baseline",
            "metric": "macro_recall",
            "available_baseline_metric": "accuracy",
        }
        record = ep._record_evaluation_result(
            manager,
            tool_name="kws_predict",
            result=self._result(rps=None, rps_status=status),
        )
        self.assertIsNotNone(record)
        self.assertIsNone(record.rps)
        self.assertEqual(record.metadata["rps_status"], status)
        self.assertEqual(manager.evaluate_calls, 0)
        self.assertEqual(manager.database.records, [record])

    def test_non_kws_record_keeps_existing_manager_path(self) -> None:
        manager = FakeRecordManager()
        ep._record_evaluation_result(
            manager,
            tool_name="asr_predict",
            result=self._result(task="ASR", rps=0.5),
        )
        self.assertEqual(manager.evaluate_calls, 1)
        self.assertEqual(manager.database.records, [])


class ExternalRunDirectorySafetyTests(unittest.TestCase):
    def test_dot_components_are_rejected(self) -> None:
        for component in (".", ".."):
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, "unsafe"):
                ep._safe_path_component(component)

    def test_existing_symlink_cannot_escape_external_runs_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external = root / "external"
            outside = root / "outside"
            external.mkdir()
            outside.mkdir()
            (external / "demo").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                ep._external_run_dir(external, "demo", "accuracy")

    def test_normal_run_directory_is_resolved_below_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            run_dir = ep._external_run_dir(root, "demo", "accuracy")
            self.assertEqual(run_dir, root.resolve() / "demo" / "accuracy")


class KWSExternalEvaluationTests(unittest.TestCase):
    def test_bridge_uses_structured_jsonl_as_the_prediction_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            prediction_path = root / "demo.txt"
            structured_path = root / "demo.jsonl"
            external_runs_dir = root / "external"
            _write_jsonl(
                dataset_path,
                [
                    {
                        "key": "pos",
                        "task": "KWS",
                        "language": "zh",
                        "expected_detected": True,
                        "expected_keyword": "wake",
                    },
                    {
                        "key": "neg",
                        "task": "KWS",
                        "language": "zh",
                        "expected_detected": False,
                    },
                ],
            )
            prediction_path.write_text("pos\twrong scalar\nneg\twrong scalar\n", encoding="utf-8")
            _write_jsonl(
                structured_path,
                [
                    _structured_row("pos", {"detected": True, "keyword": "wake", "score": 0.9}),
                    _structured_row("neg", {"detected": False, "keyword": None, "score": None}),
                ],
            )
            captured: dict[str, object] = {}

            def fake_run(*, engine_root: Path, request: dict[str, object], timeout: int) -> dict[str, object]:
                captured.update(request)
                outputs = json.loads(Path(str(request["sample_output"])).read_text(encoding="utf-8"))
                self.assertTrue(outputs[0]["result"]["detected"])
                self.assertIsNone(outputs[1]["result"]["score"])
                return {
                    "pipeline": dict(KWS_PIPELINE),
                    "summary": {
                        "metric": "accuracy",
                        "score": 1.0,
                        "pipeline_id": KWS_PIPELINE["pipeline_id"],
                        "language": "n/a",
                        "output_dir": request["output_dir"],
                        "node_config_paths": [],
                    },
                    "report": {"score": 1.0},
                }

            with (
                mock.patch.object(ep, "_describe_external_pipeline", return_value=dict(KWS_PIPELINE)),
                mock.patch.object(ep, "_run_external_pipeline", side_effect=fake_run),
                mock.patch.object(ep, "_evaluation_runtime_binding", return_value={"runtime_id": "test"}),
            ):
                result = ep.evaluate_kws_prediction_file_external(
                    FakeDatasetManager(dataset_path),
                    FakeSOTAManager(),
                    "demo",
                    prediction_path,
                    engine_source="test",
                    engine_root=root / "engine",
                    external_runs_dir=external_runs_dir,
                    device="cpu",
                    cache_dir=None,
                    timeout=30,
                    metric_override="accuracy",
                )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["prediction_jsonl_path"], str(structured_path))
        self.assertTrue(str(captured["reference_jsonl"]).endswith("reference.jsonl"))
        self.assertTrue(str(captured["sample_output"]).endswith("sample_output.json"))

    def test_macro_recall_rejects_null_rejected_score_before_external_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            prediction_path = root / "demo.txt"
            _write_jsonl(
                dataset_path,
                [{"key": "neg", "task": "KWS", "expected_detected": False}],
            )
            prediction_path.write_text("neg\t{}\n", encoding="utf-8")
            _write_jsonl(
                prediction_path.with_suffix(".jsonl"),
                [_structured_row("neg", {"detected": False, "keyword": None, "score": None})],
            )
            pipeline = dict(KWS_PIPELINE)
            pipeline["metric"] = "macro_recall"
            pipeline["pipeline_id"] = (
                "kws.any.macro_recall.conversion_kws_sure_json_to_samples_v1.wekws_det_v1"
            )
            with (
                mock.patch.object(ep, "_describe_external_pipeline", return_value=pipeline),
                mock.patch.object(ep, "_run_external_pipeline") as external_run,
                self.assertRaisesRegex(ValueError, "requires a score"),
            ):
                ep.evaluate_kws_prediction_file_external(
                    FakeDatasetManager(dataset_path),
                    FakeSOTAManager(),
                    "demo",
                    prediction_path,
                    engine_source="test",
                    engine_root=root / "engine",
                    external_runs_dir=root / "external",
                    device="cpu",
                    cache_dir=None,
                    timeout=30,
                    metric_override="macro_recall",
                )
            external_run.assert_not_called()

    def test_main_dispatches_kws_to_the_structured_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            prediction_path = root / "demo.txt"
            _write_jsonl(
                dataset_path,
                [{"key": "neg", "task": "KWS", "language": "zh", "expected_detected": False}],
            )
            prediction_path.write_text("neg\t{}\n", encoding="utf-8")
            result = {
                "dataset": "demo",
                "jsonl_path": str(dataset_path),
                "prediction_path": str(prediction_path),
                "task": "KWS",
                "language": "n/a",
                "metric": "accuracy",
                "score": 1.0,
                "rps": None,
                "rps_status": {"status": "missing_baseline", "dataset": "demo", "score": 1.0},
                "rps_is_unbounded": False,
                "num_samples": 1,
                "expected_samples": 1,
                "provided_predictions": 1,
                "evaluation_backend": "external",
                "evaluator_version": "sure-evaluation",
                "pipeline_id": KWS_PIPELINE["pipeline_id"],
                "evaluation_context": {"nodes": ["scoring/wekws_det"]},
                "details": {"summary": {}, "report": {}, "pipeline": dict(KWS_PIPELINE)},
            }
            dedicated = mock.Mock(return_value=result)
            text_bridge = mock.Mock(side_effect=AssertionError("text bridge must not run for KWS"))
            record_manager = FakeRecordManager()
            argv = [
                "evaluate_predictions.py",
                "--dataset",
                "demo",
                "--pred",
                "demo",
                str(prediction_path),
                "--evaluation-backend",
                "external",
                "--evaluation-engine-root",
                str(root / "engine"),
                "--evaluation-metric",
                "accuracy",
                "--record",
                "--tool-name",
                "kws_predict",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(ep.Config, "from_env", return_value=object()),
                mock.patch.object(ep, "DatasetManager", return_value=FakeDatasetManager(dataset_path)),
                mock.patch.object(ep, "RPSManager", return_value=record_manager),
                mock.patch.object(ep, "SOTAManager", return_value=FakeSOTAManager()),
                mock.patch.object(ep, "resolve_engine_root", return_value=("test", root / "engine")),
                mock.patch.object(ep, "_peek_dataset_task_language", return_value=("KWS", "zh")),
                mock.patch.object(ep, "_external_metric_applies_to_task_language", return_value=True),
                mock.patch.object(ep, "_describe_external_pipeline", return_value=dict(KWS_PIPELINE)),
                mock.patch.object(ep, "evaluate_kws_prediction_file_external", dedicated),
                mock.patch.object(ep, "evaluate_prediction_file_external", text_bridge),
                mock.patch.object(ep, "_evaluation_runtime_binding", return_value={"runtime_id": "test"}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return_code = ep.main()

        self.assertEqual(return_code, 0)
        dedicated.assert_called_once()
        text_bridge.assert_not_called()
        self.assertEqual(record_manager.evaluate_calls, 0)
        self.assertEqual(len(record_manager.database.records), 1)
        self.assertIsNone(record_manager.database.records[0].rps)


if __name__ == "__main__":
    unittest.main()
