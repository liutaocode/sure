#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_predictions as ep  # noqa: E402
import generate_predictions_via_server as gp  # noqa: E402
import materialize_predictions_template as mt  # noqa: E402
import model_wrapper_mcp_server as mw  # noqa: E402
import resolve_eval_input as rei  # noqa: E402
import validate_prediction_files as vpf  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
ENGINE_ROOT = REPO_ROOT / "sure" / "external" / "sure-evaluation"


def _meeteval_runtime() -> dict:
    runtime_root = REPO_ROOT / "sure" / ".runtime" / "evaluation"
    runtime_spec = json.loads(
        (REPO_ROOT / "sure" / "runtime" / "evaluation" / "runtime.json").read_text(
            encoding="utf-8"
        )
    )
    for manifest_path in sorted(runtime_root.glob("*/runtime-manifest.json"), reverse=True):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            Path(str(payload.get("engine_root") or "")).resolve() == ENGINE_ROOT.resolve()
            and payload.get("engine_commit") == runtime_spec.get("engine_commit")
            and "meeteval" in payload.get("required_imports", [])
            and Path(str(payload.get("python_executable") or "")).is_file()
        ):
            return payload
    return {}


MEETEVAL_RUNTIME = _meeteval_runtime()


class _DatasetManager:
    def __init__(self, path: Path) -> None:
        self.path = path

    def normalize_dataset_name(self, value: str) -> str:
        return value

    def get_jsonl_path(self, _value: str) -> Path:
        return self.path

    def download_and_convert(self, _value: str) -> Path:
        raise AssertionError("fixture JSONL already exists")


class _SotaManager:
    def get_metric(self, _dataset: str, *, fallback_names=()):
        return None

    def get_baseline(self, _dataset: str, *, fallback_names=()):
        return None

    def calculate_rps(self, _dataset: str, _score: float, *, fallback_names=()):
        return None


def _sample(task: str = "SD") -> dict:
    row = {
        "key": "meeting-1",
        "task": task,
        "language": "en" if task != "SD" else "n/a",
        "duration_sec": 4.0,
        "segments": [
            {"speaker": "spk1", "start": 0.0, "end": 1.25},
            {"speaker": "spk2", "start": 1.5, "end": 3.75},
        ],
    }
    if task != "SD":
        row["segments"] = [
            {**segment, "text": f"utterance {index}"}
            for index, segment in enumerate(row["segments"], 1)
        ]
    return row


def _structured(sample: dict) -> dict[str, dict]:
    segments = json.loads(json.dumps(sample["segments"]))
    return {
        sample["key"]: {
            "key": sample["key"],
            "task": sample["task"],
            "prediction": {"segments": segments},
            "normalized_prediction": json.dumps(segments),
        }
    }


class GenerationAndValidationTests(unittest.TestCase):
    def test_sa_asr_aliases_keep_metric_and_tool_contracts_aligned(self) -> None:
        self.assertEqual(rei._normalize_task("sa_asr"), "SA-ASR")
        self.assertEqual(rei._normalize_task("sa-asr"), "SA-ASR")
        self.assertEqual(rei._fallback_default_metrics("SA_ASR", "en"), ["cpwer"])
        self.assertEqual(mt._default_metric("SA_ASR", "en"), "cpwer")
        config = {"model": {"task": "SA_ASR"}}
        self.assertEqual(mw._model_task(config), "SA-ASR")
        self.assertEqual(mw._tool_names(config), ["transcribe_with_speakers"])
        self.assertEqual(mw._tool_names({"model": {"task": "SD"}}), ["diarize"])

    def test_sa_asr_aliases_share_one_generation_contract(self) -> None:
        self.assertEqual(gp._normalize_task("SA_ASR"), "SA-ASR")
        self.assertEqual(gp._normalize_task("sa-asr"), "SA-ASR")
        segments = [{"speaker": "spk1", "start": 0.0, "end": 1.0, "text": "hello"}]
        for task in ("SA_ASR", "SA-ASR"):
            projection, normalized = gp._normalize_prediction_payload(
                {"segments": segments}, task=task
            )
            self.assertEqual(json.loads(projection), segments)
            self.assertEqual(normalized, {"segments": segments})

    def test_sd_allows_silence_but_sa_asr_requires_transcript_segments(self) -> None:
        self.assertTrue(vpf._valid_annotation_segments([], task="SD"))
        self.assertFalse(vpf._valid_annotation_segments([], task="SA-ASR"))
        self.assertFalse(
            vpf._valid_annotation_segments(
                [{"speaker": "spk1", "start": 0.0, "end": 1.0}],
                task="SA_ASR",
            )
        )

    def test_prediction_contract_rejects_malformed_segments(self) -> None:
        invalid = [
            None,
            {},
            [{"speaker": "", "start": 0.0, "end": 1.0}],
            [{"speaker": "two words", "start": 0.0, "end": 1.0}],
            [{"speaker": "spk", "start": -0.1, "end": 1.0}],
            [{"speaker": "spk", "start": 1.0, "end": 1.0}],
            [{"speaker": "spk", "start": True, "end": 1.0}],
            [{"speaker": "spk", "start": 0.0, "end": float("nan")}],
        ]
        for segments in invalid:
            with self.subTest(segments=segments):
                self.assertFalse(vpf._valid_annotation_segments(segments, task="SD"))

        duplicate = [
            {"speaker": "spk", "start": 0.0, "end": 1.0},
            {"speaker": "spk", "start": 0.0, "end": 1.0},
        ]
        self.assertFalse(vpf._valid_annotation_segments(duplicate, task="SD"))
        self.assertFalse(
            vpf._valid_annotation_segments(
                [{"speaker": "spk", "start": 0.0, "end": 4.1}],
                task="SD",
                duration=4.0,
            )
        )

    def test_generation_rejects_annotation_paths_and_malformed_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object with segments"):
            gp._normalize_prediction_payload("/untrusted/hyp.rttm", task="SD")
        with self.assertRaisesRegex(ValueError, "unapproved field"):
            gp._normalize_prediction_payload({"annotation_path": "/untrusted/hyp.rttm"}, task="SD")
        invalid = [
            [{"speaker": "spk", "start": 0.0, "end": float("nan")}],
            [{"speaker": "spk", "start": 1.0, "end": 0.0}],
            [{"speaker": "two words", "start": 0.0, "end": 1.0}],
            [{"speaker": ";comment", "start": 0.0, "end": 1.0}],
            [{"speaker": "spk", "start": 0.0, "end": 1.0, "debug": "x"}],
            [{"speaker": "spk", "start": 0.0, "end": 1.0, "duration": 0.5}],
        ]
        for segments in invalid:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                gp._normalize_prediction_payload({"segments": segments}, task="SD")
        with self.assertRaisesRegex(ValueError, "unapproved field"):
            gp._normalize_prediction_payload(
                {"segments": [{"speaker": "spk", "start": 0.0, "end": 1.0}], "raw": {}},
                task="SD",
            )
        with self.assertRaisesRegex(ValueError, "num_speakers"):
            gp._normalize_prediction_payload(
                {
                    "segments": [{"speaker": "spk", "start": 0.0, "end": 1.0}],
                    "num_speakers": 2,
                },
                task="SD",
            )
        with self.assertRaisesRegex(ValueError, "envelope contains unapproved field"):
            gp._normalize_prediction_payload(
                {
                    "prediction": {
                        "segments": [{"speaker": "spk", "start": 0.0, "end": 1.0}]
                    },
                    "debug_path": "/private/model.log",
                },
                task="SD",
            )

    def test_strict_task_compatibility_separates_sd_and_sa_asr(self) -> None:
        for model_task, dataset_task in (("SD", "SA-ASR"), ("SA_ASR", "SD")):
            with self.subTest(model_task=model_task, dataset_task=dataset_task):
                with self.assertRaisesRegex(rei.EvalInputError, "Task mismatch"):
                    rei._check_task_compatibility(
                        {"name": "model", "declared_task": model_task},
                        [{"name": "dataset", "task": dataset_task}],
                    )

    def test_structured_tasks_require_jsonl_sidecar(self) -> None:
        sample = _sample("SD")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            prediction = root / "prediction.txt"
            prediction.write_text(
                f"{sample['key']}\t{json.dumps(sample['segments'])}\n",
                encoding="utf-8",
            )
            result = vpf.validate_prediction_file(
                _DatasetManager(dataset),
                "fixture",
                prediction,
                require_nonempty=True,
            )
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["structured_required"])
        self.assertEqual(result["structured_missing_keys"], [sample["key"]])

    def test_evaluator_rejects_duplicate_and_partial_prediction_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structured = root / "prediction.jsonl"
            row = next(iter(_structured(_sample("SD")).values()))
            structured.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate structured prediction key"):
                ep.load_structured_prediction_map(structured)

            text = root / "prediction.txt"
            text.write_text("one\tx\none\ty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prediction key"):
                ep.load_prediction_map(text)

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            ep._samples_with_predictions(
                [{"key": "one"}, {"key": "two"}],
                {"one"},
                dataset_name="fixture",
            )

    def test_file_validator_reports_non_object_structured_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structured = Path(temporary) / "prediction.jsonl"
            structured.write_text("[]\n", encoding="utf-8")
            predictions, duplicate_keys, invalid_rows = vpf._load_structured_predictions(
                structured
            )

        self.assertEqual(predictions, {})
        self.assertEqual(duplicate_keys, [])
        self.assertEqual(invalid_rows, ["line:1:not_object"])

    def test_fallback_server_dispatches_structured_task_methods(self) -> None:
        calls = []

        class Wrapper:
            def predict(self, payload):
                raise AssertionError("generic predict must not run")

            def diarize(self, audio_path, **kwargs):
                calls.append(("sd", audio_path, kwargs))
                return {"segments": []}

            def transcribe_with_speakers(self, audio_path, **kwargs):
                calls.append(("sa", audio_path, kwargs))
                return {"segments": []}

        wrapper = Wrapper()
        self.assertEqual(
            mw._call_model(wrapper, "diarize", {"audio_path": "/audio.wav"}),
            {"segments": []},
        )
        self.assertEqual(
            mw._call_model(
                wrapper,
                "transcribe_with_speakers",
                {"audio_path": "/audio.wav", "language": "en"},
            ),
            {"segments": []},
        )
        self.assertEqual(
            calls,
            [("sd", "/audio.wav", {}), ("sa", "/audio.wav", {"language": "en"})],
        )

    def test_file_validator_checks_reference_duration_and_silence_semantics(self) -> None:
        sample = _sample("SD")
        valid = _structured(sample)
        self.assertEqual(
            vpf._task_contract_violations([sample], valid),
            [],
        )

        overlong = _structured(sample)
        overlong[sample["key"]]["prediction"]["segments"][0]["end"] = 4.5
        self.assertEqual(
            vpf._task_contract_violations([sample], overlong),
            [sample["key"]],
        )

        malformed_reference = {**sample, "segments": [{"speaker": "spk", "start": 1, "end": 1}]}
        self.assertEqual(
            vpf._task_contract_violations([malformed_reference], valid),
            [sample["key"]],
        )

        silence = {**sample, "segments": []}
        false_alarm = _structured(silence)
        false_alarm[sample["key"]]["prediction"]["segments"] = [
            {"speaker": "spk", "start": 0.0, "end": 1.0}
        ]
        self.assertEqual(
            vpf._task_contract_violations([silence], false_alarm),
            [],
        )

    def test_file_validator_rejects_structured_task_spoof_but_accepts_alias(self) -> None:
        sample = _sample("SD")
        spoofed = {
            sample["key"]: {
                "key": sample["key"],
                "task": "ASR",
                "prediction": {"text": "reference-derived text"},
                "normalized_prediction": "reference-derived text",
            }
        }
        self.assertEqual(
            vpf._task_contract_violations([sample], spoofed),
            [sample["key"]],
        )

        sa_asr = _sample("SA-ASR")
        alias = _structured(sa_asr)
        alias[sa_asr["key"]]["task"] = "SA_ASR"
        self.assertEqual(vpf._task_contract_violations([sa_asr], alias), [])


class AnnotationConversionTests(unittest.TestCase):
    def test_sd_segments_are_written_as_deterministic_rttm(self) -> None:
        sample = _sample("SD")
        with tempfile.TemporaryDirectory() as temporary:
            ref, hyp, manifest = ep._write_external_annotation_role_files(
                task="SD",
                samples=[sample],
                structured_predictions=_structured(sample),
                output_dir=Path(temporary),
            )
            expected = (
                "SPEAKER meeting-1 1 0.000000 1.250000 <NA> <NA> spk1 <NA> <NA>\n"
                "SPEAKER meeting-1 1 1.500000 2.250000 <NA> <NA> spk2 <NA> <NA>\n"
            )
            self.assertEqual(ref.read_text(encoding="utf-8"), expected)
            self.assertEqual(hyp.read_text(encoding="utf-8"), expected)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], "RTTM")
        self.assertEqual(payload["roles"], {"ref": "reference.rttm", "hyp": "hypothesis.rttm"})

    def test_sa_asr_segments_are_written_as_six_field_stm(self) -> None:
        sample = _sample("SA-ASR")
        with tempfile.TemporaryDirectory() as temporary:
            ref, hyp, manifest = ep._write_external_annotation_role_files(
                task="SA_ASR",
                samples=[sample],
                structured_predictions=_structured(sample),
                output_dir=Path(temporary),
            )
            expected = (
                "meeting-1 1 spk1 0.000000 1.250000 utterance 1\n"
                "meeting-1 1 spk2 1.500000 3.750000 utterance 2\n"
            )
            self.assertEqual(ref.read_text(encoding="utf-8"), expected)
            self.assertEqual(hyp.read_text(encoding="utf-8"), expected)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["task"], "SA-ASR")
        self.assertEqual(payload["format"], "STM")

    def test_sd_silence_materializes_empty_annotation_files(self) -> None:
        sample = {**_sample("SD"), "segments": []}
        with tempfile.TemporaryDirectory() as temporary:
            ref, hyp, manifest = ep._write_external_annotation_role_files(
                task="SD",
                samples=[sample],
                structured_predictions=_structured(sample),
                output_dir=Path(temporary),
            )
            self.assertEqual(ref.read_text(encoding="utf-8"), "")
            self.assertEqual(hyp.read_text(encoding="utf-8"), "")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["reference_line_count"], 0)
        self.assertEqual(payload["hypothesis_line_count"], 0)

    def test_conversion_rejects_injection_duration_and_text_errors(self) -> None:
        cases = [
            ("SD", {**_sample("SD"), "key": "bad key"}, "annotation token"),
            ("SD", {**_sample("SD"), "key": ";comment"}, "annotation token"),
            (
                "SD",
                {**_sample("SD"), "segments": [{"speaker": "spk", "start": 0, "end": 5}]},
                "exceeds sample duration",
            ),
            (
                "SA-ASR",
                {
                    **_sample("SA-ASR"),
                    "segments": [{"speaker": "spk", "start": 0, "end": 1, "text": "bad\nrow"}],
                },
                "text must be non-empty",
            ),
            ("SD", {**_sample("SD"), "duration_sec": float("nan")}, "duration field"),
        ]
        for task, sample, message in cases:
            with self.subTest(task=task, message=message), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, message):
                    ep._write_external_annotation_role_files(
                        task=task,
                        samples=[sample],
                        structured_predictions=_structured(sample),
                        output_dir=Path(temporary),
                    )


class ExternalBridgeTests(unittest.TestCase):
    def _run(self, task: str) -> tuple[dict, dict]:
        sample = _sample(task)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            prediction = root / "predictions" / "fixture.txt"
            prediction.parent.mkdir()
            projection = json.dumps(sample["segments"], separators=(",", ":"))
            prediction.write_text(f"{sample['key']}\t{projection}\n", encoding="utf-8")
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(next(iter(_structured(sample).values()))) + "\n",
                encoding="utf-8",
            )
            observed: dict = {}

            def fake_run(*, engine_root: Path, request: dict, timeout: int) -> dict:
                observed.update(request)
                self.assertTrue(Path(request["ref_file"]).is_file())
                self.assertTrue(Path(request["hyp_file"]).is_file())
                metric = "der" if task == "SD" else "cpwer"
                pipeline_id = (
                    "sd.any.der.meeteval_v1"
                    if task == "SD"
                    else "sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1."
                    "whisper_norm_english_v1.meeteval_v1"
                )
                report = {"score": 0.0, "metric": metric, "pipeline_id": pipeline_id}
                return {
                    "pipeline": request["pipeline"],
                    "summary": {
                        "metric": metric,
                        "score": 0.0,
                        "pipeline_id": pipeline_id,
                        "language": sample["language"],
                        "output_dir": request["output_dir"],
                    },
                    "report": report,
                }

            metric = "der" if task == "SD" else "cpwer"
            pipeline = {
                "task": task,
                "metric": metric,
                "required_roles": ["hyp", "ref"],
                "nodes": [{"node_id": "scoring/meeteval"}],
            }
            with (
                mock.patch.object(ep, "_describe_external_pipeline", return_value=pipeline),
                mock.patch.object(ep, "_run_external_pipeline", side_effect=fake_run),
                mock.patch.object(ep, "_evaluation_runtime_binding", return_value={"status": "ready"}),
            ):
                result = ep.evaluate_annotation_prediction_file_external(
                    _DatasetManager(dataset),
                    _SotaManager(),
                    "fixture",
                    prediction,
                    engine_source="test",
                    engine_root=root,
                    external_runs_dir=root / "runs",
                    device="cpu",
                    cache_dir=None,
                    timeout=30,
                    metric_override=metric,
                    task_override=task,
                )
        return result, observed

    def test_sd_bridge_preserves_annotation_roles(self) -> None:
        result, observed = self._run("SD")
        self.assertEqual(result["pipeline_id"], "sd.any.der.meeteval_v1")
        self.assertTrue(observed["ref_file"].endswith(".rttm"))
        self.assertTrue(observed["hyp_file"].endswith(".rttm"))
        self.assertEqual(result["details"]["annotation_conversion"]["format"], "RTTM")

    def test_sa_asr_bridge_preserves_annotation_roles(self) -> None:
        result, observed = self._run("SA-ASR")
        self.assertEqual(result["metric"], "cpwer")
        self.assertTrue(observed["ref_file"].endswith(".stm"))
        self.assertTrue(observed["hyp_file"].endswith(".stm"))
        primary = ep._primary_result("cpwer", 0.0)
        self.assertEqual(primary["score_key"], "cpwer")
        self.assertEqual(primary["cpwer"], 0.0)

    def test_sample_report_includes_meeteval_per_session_details(self) -> None:
        sample = _sample("SD")
        metric_details = {
            "error_rate": 0.25,
            "missed_speaker_time": 0.5,
            "falarm_speaker_time": 0.0,
            "speaker_error_time": 0.0,
        }
        result = {
            "dataset": "fixture",
            "task": "SD",
            "metric": "der",
            "details": {
                "report": {
                    "pipeline_trace": [
                        {
                            "node_id": "scoring/meeteval",
                            "details": {"result": {"per_session": {sample["key"]: metric_details}}},
                        }
                    ]
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "samples.jsonl"
            ep._write_sample_report(
                output_path=output,
                samples=[sample],
                predictions={sample["key"]: json.dumps(sample["segments"])},
                result=result,
                structured_predictions=_structured(sample),
            )
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(row["metric_details"], metric_details)

    def test_bounded_run_artifacts_write_only_the_evaluated_sample_scope(self) -> None:
        first = _sample("SD")
        second = {**_sample("SD"), "key": "meeting-2"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir(parents=True)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                "".join(json.dumps(sample) + "\n" for sample in (first, second)),
                encoding="utf-8",
            )
            prediction = predictions_dir / "fixture.txt"
            projection = json.dumps(first["segments"], separators=(",", ":"))
            prediction.write_text(f"{first['key']}\t{projection}\n", encoding="utf-8")
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(next(iter(_structured(first).values()))) + "\n",
                encoding="utf-8",
            )
            result = {
                "dataset": "fixture",
                "jsonl_path": str(dataset),
                "prediction_path": str(prediction),
                "task": "SD",
                "language": "n/a",
                "metric": "der",
                "score": 0.0,
                "rps": None,
                "num_samples": 1,
                "expected_samples": 1,
                "total_dataset_samples": 2,
                "evaluation_max_samples": 1,
                "evaluation_backend": "external",
                "evaluator_version": "sure-evaluation",
                "pipeline_id": "sd.any.der.meeteval_v1",
                "evaluation_context": {"nodes": ["scoring/meeteval"]},
                "details": {
                    "summary": {},
                    "pipeline": {},
                    "report": {"score": 0.0},
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
                    tool_name="diarize",
                    protocol_id="standard_system",
                    model_dir=None,
                    payload=payload,
                    results=[result],
                )
            rows = ep.load_jsonl(
                run_dir / "sample_reports" / "fixture" / "der.jsonl"
            )

        self.assertEqual([row["key"] for row in rows], [first["key"]])


@unittest.skipUnless(
    ENGINE_ROOT.is_dir() and bool(MEETEVAL_RUNTIME),
    "locked Evaluation Runtime with MeetEval is required",
)
class RealMeetEvalTests(unittest.TestCase):
    def _run(
        self,
        task: str,
        sample: dict,
        prediction_segments: list[dict] | None = None,
    ) -> dict:
        structured = _structured(sample)
        if prediction_segments is not None:
            structured[sample["key"]]["prediction"]["segments"] = prediction_segments
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref_file, hyp_file, _ = ep._write_external_annotation_role_files(
                task=task,
                samples=[sample],
                structured_predictions=structured,
                output_dir=root,
            )
            request = root / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "task": task,
                        "language": sample["language"],
                        "ref_file": str(ref_file),
                        "hyp_file": str(hyp_file),
                        "output_dir": str(root / "output"),
                    }
                ),
                encoding="utf-8",
            )
            code = (
                "import json,sys;from pathlib import Path;"
                "from sure_eval.evaluation.cli_adapters import build_pipeline_spec,run_pipeline_spec;"
                "r=json.loads(Path(sys.argv[1]).read_text());"
                "p=build_pipeline_spec(r['task'],language=r['language']);"
                "s=run_pipeline_spec(p,ref_file=r['ref_file'],hyp_file=r['hyp_file'],"
                "output_dir=r['output_dir']);print(json.dumps(s))"
            )
            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env["SURE_HARNESS_RUNTIME_ROOT"] = str(MEETEVAL_RUNTIME["harness_runtime_root"])
            env["PYTHONPATH"] = str(ENGINE_ROOT / "src")
            completed = subprocess.run(
                [str(MEETEVAL_RUNTIME["python_executable"]), "-c", code, str(request)],
                cwd=ENGINE_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return json.loads(
                [line for line in completed.stdout.splitlines() if line.strip()][-1]
            )

    def test_identical_sd_annotations_score_zero_der(self) -> None:
        summary = self._run("SD", _sample("SD"))
        self.assertEqual(summary["pipeline_id"], "sd.any.der.meeteval_v1")
        self.assertEqual(summary["metric"], "der")
        self.assertEqual(float(summary["score"]), 0.0)

    @unittest.skip(
        "the pinned standalone evaluator does not define empty-session DER semantics"
    )
    def test_identical_silent_sd_annotations_score_zero_der(self) -> None:
        summary = self._run("SD", {**_sample("SD"), "segments": []})
        self.assertEqual(summary["pipeline_id"], "sd.any.der.meeteval_v1")
        self.assertEqual(float(summary["score"]), 0.0)

    @unittest.skip(
        "the pinned standalone evaluator does not define empty-session DER semantics"
    )
    def test_silent_reference_with_false_alarm_scores_one_der(self) -> None:
        summary = self._run(
            "SD",
            {**_sample("SD"), "segments": []},
            prediction_segments=[{"speaker": "spk1", "start": 0.0, "end": 1.0}],
        )
        self.assertEqual(float(summary["score"]), 1.0)

    @unittest.skip(
        "the pinned standalone evaluator does not define empty-session DER semantics"
    )
    def test_speech_reference_with_empty_prediction_scores_one_der(self) -> None:
        summary = self._run("SD", _sample("SD"), prediction_segments=[])
        self.assertEqual(summary["pipeline_id"], "sd.any.der.meeteval_v1")
        self.assertEqual(float(summary["score"]), 1.0)

    def test_identical_sa_asr_annotations_score_zero_cpwer(self) -> None:
        summary = self._run("SA-ASR", _sample("SA-ASR"))
        self.assertEqual(
            summary["pipeline_id"],
            "sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1."
            "whisper_norm_english_v1.meeteval_v1",
        )
        self.assertEqual(summary["metric"], "cpwer")
        self.assertEqual(float(summary["score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
