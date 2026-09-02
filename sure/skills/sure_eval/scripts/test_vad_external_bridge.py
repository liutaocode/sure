#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_predictions as ep  # noqa: E402
import evaluation_capabilities as ec  # noqa: E402
import generate_predictions_via_server as gp  # noqa: E402
import materialize_predictions_template as mt  # noqa: E402
import model_wrapper_mcp_server as mw  # noqa: E402
import resolve_eval_input as rei  # noqa: E402
import validate_prediction_files as vpf  # noqa: E402
from sure_eval.tools.mcp_client import MCPToolClient, MCPToolConfig  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
ENGINE_ROOT = REPO_ROOT / "sure" / "external" / "sure-evaluation"


def _locked_evaluation_runtime_available() -> bool:
    required_env = {
        "HARNESS_PYTHON_BIN",
        "SURE_HARNESS_RUNTIME_ID",
        "SURE_HARNESS_LOCK_SHA256",
        "SURE_HARNESS_MANIFEST_PATH",
        "SURE_HARNESS_RUNTIME_ROOT",
    }
    if any(not os.environ.get(name) for name in required_env):
        return False
    spec = json.loads(
        (REPO_ROOT / "sure/runtime/evaluation/runtime.json").read_text(encoding="utf-8")
    )
    runtime_root = REPO_ROOT / "sure/.runtime/evaluation"
    for manifest_path in runtime_root.glob("*/runtime-manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("engine_commit") == spec.get("engine_commit")
            and Path(str(payload.get("python_executable") or "")).is_file()
        ):
            return True
    return False


def _sample(
    key: str = "vad-1",
    *,
    segments: list[dict[str, float]] | None = None,
    duration: float = 1.0,
) -> dict:
    return {
        "key": key,
        "task": "VAD",
        "language": "n/a",
        "duration_sec": duration,
        "path": f"{key}.wav",
        "speech_segments": (
            [{"start": 0.2, "end": 0.6}] if segments is None else segments
        ),
    }


def _prediction(
    *,
    segments: list[dict[str, float]] | None = None,
    frame_scores: list[dict[str, float]] | None = None,
) -> dict:
    value: dict = {
        "speech_segments": (
            [{"start": 0.2, "end": 0.6}] if segments is None else segments
        )
    }
    if frame_scores is not None:
        value["frame_scores"] = frame_scores
    return value


def _structured(key: str = "vad-1", **kwargs) -> dict:
    prediction = _prediction(**kwargs)
    projection = json.dumps(
        prediction,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "key": key,
        "dataset": "vad-fixture",
        "task": "VAD",
        "language": "n/a",
        "prediction": prediction,
        "normalized_prediction": projection,
        "raw_response": prediction,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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

    def calculate_rps(self, _dataset: str, _score, *, fallback_names=()):
        return None


class GenerationContractTests(unittest.TestCase):
    def test_vad_aliases_are_canonical_across_eval_entrypoints(self) -> None:
        aliases = (
            "vad",
            "voice activity detection",
            "voice-activity-detection",
            "voice_activity_detection",
            "speech activity detection",
            "speech-activity-detection",
            "speech_activity_detection",
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(ep._normalize_task(alias), "VAD")
                self.assertEqual(gp._normalize_task(alias), "VAD")
                self.assertEqual(rei._normalize_task(alias), "VAD")
                self.assertEqual(vpf._normalize_task(alias), "VAD")
                self.assertEqual(ec.normalize_engine_task(alias), "vad")
                self.assertEqual(mt._default_metric(alias, None), "f1")
                config = {"model": {"task": alias}}
                self.assertEqual(mw._model_task(config), "VAD")
                self.assertEqual(mw._tool_names(config), ["detect_speech"])

    def test_capability_metric_and_tool_contracts(self) -> None:
        self.assertEqual(ec.normalize_engine_task("VAD"), "vad")
        self.assertEqual(mt._default_metric("vad", None), "f1")
        self.assertEqual(
            rei._default_metrics("VAD", "n/a", ENGINE_ROOT),
            ["f1", "p_fa", "p_miss", "dcf_nist"],
        )
        config = {"model": {"task": "VAD"}}
        self.assertEqual(mw._tool_names(config), ["detect_speech"])
        self.assertFalse(mw._tool_schema("VAD")["additionalProperties"])

    def test_fallback_server_dispatches_detect_speech(self) -> None:
        calls = []

        class Wrapper:
            def predict(self, _payload):
                raise AssertionError("generic predict must not run")

            def detect_speech(self, audio_path, **kwargs):
                calls.append((audio_path, kwargs))
                return {"speech_segments": []}

        result = mw._call_model(
            Wrapper(),
            "detect_speech",
            {"audio_path": "/fixture/input.wav"},
        )
        self.assertEqual(result, {"speech_segments": []})
        self.assertEqual(calls, [("/fixture/input.wav", {})])

    def test_legacy_tool_resolver_prefers_canonical_name(self) -> None:
        client = MCPToolClient(MCPToolConfig("vad", ["unused"], "."))
        client._tools = [{"name": "vad_predict"}, {"name": "detect_speech"}]
        self.assertEqual(client.resolve_tool_name("VAD"), "detect_speech")

    def test_vad_arguments_do_not_add_language_or_reference_fields(self) -> None:
        arguments = gp._build_tool_arguments(
            repo_root=Path("/repo"),
            sample={"key": "vad-1", "speech_segments": [{"start": 0, "end": 1}]},
            task="VAD",
            language="en",
            argument_name="audio_path",
            audio_path=Path("/fixture/input.wav"),
            output_audio_dir=Path("/unused"),
            tool_args=None,
        )
        self.assertEqual(arguments, {"audio_path": "/fixture/input.wav"})

    def test_normalizer_preserves_silence_and_frame_scores(self) -> None:
        frame_scores = [
            {"start": 0.0, "end": 0.5, "score": 0.1},
            {"start": 0.5, "end": 1.0, "score": 0.9},
        ]
        projection, normalized = gp._normalize_prediction_payload(
            {"speech_segments": [], "frame_scores": frame_scores},
            task="VAD",
            vad_duration=1.0,
        )
        self.assertEqual(normalized["speech_segments"], [])
        self.assertEqual(normalized["frame_scores"], frame_scores)
        self.assertEqual(json.loads(projection), normalized)

    def test_normalizer_rejects_malformed_or_leaky_payloads(self) -> None:
        invalid = [
            "0.2 0.6",
            {},
            {"speech_segments": "0.2-0.6"},
            {"speech_segments": [{"start": 0.6, "end": 0.2}]},
            {"speech_segments": [{"start": 0.0, "end": math.nan}]},
            {"speech_segments": [{"start": 0.0, "end": 1.1}]},
            {
                "speech_segments": [
                    {"start": 0.4, "end": 0.8},
                    {"start": 0.2, "end": 0.3},
                ]
            },
            {"speech_segments": [], "frame_scores": []},
            {
                "speech_segments": [],
                "frame_scores": [
                    {"start": 0.0, "end": 0.4, "score": 0.1},
                    {"start": 0.5, "end": 1.0, "score": 0.9},
                ],
            },
            {
                "speech_segments": [],
                "frame_scores": [{"start": 0.0, "end": 1.0, "score": 1.1}],
            },
            {"speech_segments": [], "debug_path": "/private/output.json"},
            {"prediction": {"speech_segments": []}, "raw": {"secret": "x"}},
            {"prediction": []},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                gp._normalize_prediction_payload(
                    payload,
                    task="VAD",
                    vad_duration=1.0,
                )


class ValidationContractTests(unittest.TestCase):
    def test_valid_silence_and_scores_pass(self) -> None:
        sample = _sample(segments=[])
        structured = {
            sample["key"]: _structured(
                segments=[],
                frame_scores=[{"start": 0.0, "end": 1.0, "score": 0.0}],
            )
        }
        self.assertEqual(vpf._task_contract_violations([sample], structured), [])

    def test_reference_and_prediction_bounds_are_enforced(self) -> None:
        cases = [
            (
                {**_sample(), "duration": float("nan")},
                _structured(),
            ),
            (
                {**_sample(), "speech_segments": [{"start": -0.1, "end": 0.2}]},
                _structured(),
            ),
            (
                _sample(),
                _structured(segments=[{"start": 0.0, "end": 1.1}]),
            ),
            (
                _sample(),
                _structured(
                    frame_scores=[{"start": 0.0, "end": 1.0, "score": -0.1}]
                ),
            ),
        ]
        for sample, row in cases:
            with self.subTest(sample=sample, row=row):
                self.assertEqual(
                    vpf._task_contract_violations([sample], {sample["key"]: row}),
                    [sample["key"]],
                )

    def test_conflicting_duration_aliases_are_rejected(self) -> None:
        sample = {**_sample(), "duration": 2.0}
        self.assertEqual(
            vpf._task_contract_violations(
                [sample],
                {sample["key"]: _structured()},
            ),
            [sample["key"]],
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "duration fields disagree"):
                ep._write_external_vad_role_files(
                    samples=[sample],
                    structured_predictions={sample["key"]: _structured()},
                    output_dir=Path(temporary),
                )

    def test_prediction_cannot_spoof_its_task(self) -> None:
        sample = _sample()
        row = {
            **_structured(),
            "task": "ASR",
            "prediction": {"text": "not a VAD result"},
            "normalized_prediction": "not a VAD result",
        }
        self.assertEqual(
            vpf._task_contract_violations([sample], {sample["key"]: row}),
            [sample["key"]],
        )

    def test_structured_sidecar_is_mandatory(self) -> None:
        sample = _sample(segments=[])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            prediction = root / "prediction.txt"
            prediction.write_text('vad-1\t{"speech_segments":[]}\n', encoding="utf-8")
            result = vpf.validate_prediction_file(
                _DatasetManager(dataset),
                "vad-fixture",
                prediction,
                require_nonempty=True,
            )
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["structured_required"])
        self.assertEqual(result["structured_missing_keys"], ["vad-1"])

    def test_full_silence_file_validation_is_nonempty(self) -> None:
        sample = _sample(segments=[])
        row = _structured(segments=[])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            prediction = root / "prediction.txt"
            prediction.write_text(
                f"vad-1\t{row['normalized_prediction']}\n",
                encoding="utf-8",
            )
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
            result = vpf.validate_prediction_file(
                _DatasetManager(dataset),
                "vad-fixture",
                prediction,
                require_nonempty=True,
            )
        self.assertTrue(result["is_valid"], result)


class RoleBridgeTests(unittest.TestCase):
    def test_role_files_keep_only_standalone_fields(self) -> None:
        frame_scores = [
            {"start": 0.0, "end": 0.2, "score": 0.1},
            {"start": 0.2, "end": 0.6, "score": 0.9},
            {"start": 0.6, "end": 1.0, "score": 0.1},
        ]
        row = _structured(frame_scores=frame_scores)
        row["raw_response"] = {"debug": "must-not-cross-role-bridge"}
        with tempfile.TemporaryDirectory() as temporary:
            reference, output, manifest = ep._write_external_vad_role_files(
                samples=[_sample()],
                structured_predictions={"vad-1": row},
                output_dir=Path(temporary),
            )
            reference_rows = _read_jsonl(reference)
            output_rows = _read_jsonl(output)
            conversion = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            reference_rows,
            [
                {
                    "key": "vad-1",
                    "duration": 1.0,
                    "speech_segments": [{"start": 0.2, "end": 0.6}],
                }
            ],
        )
        self.assertEqual(set(output_rows[0]), {"key", "speech_segments", "frame_scores"})
        self.assertNotIn("debug", json.dumps(output_rows))
        self.assertEqual(
            conversion["roles"],
            {"reference_jsonl": "reference.jsonl", "sample_output": "sample_output.jsonl"},
        )

    def test_role_files_preserve_empty_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference, output, _ = ep._write_external_vad_role_files(
                samples=[_sample(segments=[])],
                structured_predictions={"vad-1": _structured(segments=[])},
                output_dir=Path(temporary),
            )
            self.assertEqual(_read_jsonl(reference)[0]["speech_segments"], [])
            self.assertEqual(_read_jsonl(output)[0]["speech_segments"], [])

    def test_role_files_reject_bad_reference_or_prediction(self) -> None:
        cases = [
            ({**_sample(), "duration": 0}, _structured(), "duration"),
            (
                _sample(),
                _structured(segments=[{"start": 0.5, "end": 1.1}]),
                "exceeds duration",
            ),
        ]
        for sample, row, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, message):
                    ep._write_external_vad_role_files(
                        samples=[sample],
                        structured_predictions={"vad-1": row},
                        output_dir=Path(temporary),
                    )

    def test_role_files_reject_task_or_dataset_spoofing(self) -> None:
        for field, value, message in (
            ("task", "ASR", "task mismatch"),
            ("dataset", "other-dataset", "dataset mismatch"),
        ):
            row = {**_structured(), field: value}
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, message):
                    ep._write_external_vad_role_files(
                        samples=[_sample()],
                        structured_predictions={"vad-1": row},
                        output_dir=Path(temporary),
                        dataset_name="vad-fixture",
                    )


class RoutingTests(unittest.TestCase):
    def test_default_suite_is_the_four_detection_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "vad.txt"
            prediction.write_text("vad-1\tx\n", encoding="utf-8")
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(
                    _structured(
                        frame_scores=[{"start": 0.0, "end": 1.0, "score": 0.5}]
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            requests = ep._dataset_evaluation_requests(
                task="VAD",
                prediction_path=prediction,
                evaluation_requests=[(None, None)],
                explicit_metric_requested=False,
                pipeline_overrides=[],
                samples=[_sample()],
            )
            self.assertEqual(
                [metric for metric, _ in requests],
                list(ep.VAD_DETECTION_METRICS),
            )
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(_structured()) + "\n",
                encoding="utf-8",
            )
            requests = ep._dataset_evaluation_requests(
                task="VAD",
                prediction_path=prediction,
                evaluation_requests=[(None, None)],
                explicit_metric_requested=False,
                pipeline_overrides=[],
                samples=[_sample()],
            )
        self.assertEqual([metric for metric, _ in requests], list(ep.VAD_DETECTION_METRICS))

    def test_explicit_auc_requires_complete_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "vad.txt"
            prediction.write_text("vad-1\tx\n", encoding="utf-8")
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(_structured()) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete frame_scores"):
                ep._dataset_evaluation_requests(
                    task="VAD",
                    prediction_path=prediction,
                    evaluation_requests=[("auc_roc", None)],
                    explicit_metric_requested=True,
                    pipeline_overrides=[],
                    samples=[_sample()],
                )
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(
                    _structured(
                        frame_scores=[{"start": 0.0, "end": 1.0, "score": 0.5}]
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            requests = ep._dataset_evaluation_requests(
                task="VAD",
                prediction_path=prediction,
                evaluation_requests=[("auc_roc", None)],
                explicit_metric_requested=True,
                pipeline_overrides=[],
                samples=[_sample()],
            )
        self.assertEqual(requests, [("auc_roc", None)])

    def test_auc_pipeline_id_also_requires_complete_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prediction = Path(temporary) / "vad.txt"
            prediction.write_text("vad-1\tx\n", encoding="utf-8")
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(_structured()) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete frame_scores"):
                ep._dataset_evaluation_requests(
                    task="VAD",
                    prediction_path=prediction,
                    evaluation_requests=[
                        (
                            None,
                            "vad.any.auc_roc.vad_contract_v1."
                            "vad_timebase_strict_v1.vad_auc_roc_v1",
                        )
                    ],
                    explicit_metric_requested=False,
                    pipeline_overrides=["vad.any.auc_roc.test"],
                    samples=[_sample()],
                )

    def test_explicit_single_metric_is_not_expanded(self) -> None:
        requests = [("f1", None)]
        self.assertIs(
            ep._dataset_evaluation_requests(
                task="VAD",
                prediction_path=Path("missing.txt"),
                evaluation_requests=requests,
                explicit_metric_requested=True,
                pipeline_overrides=[],
                samples=[_sample()],
            ),
            requests,
        )

    def test_external_bridge_uses_exact_roles_and_bounded_scope(self) -> None:
        first = _sample()
        second = _sample("vad-2")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            prediction = root / "prediction.txt"
            row = _structured()
            prediction.write_text(
                f"vad-1\t{row['normalized_prediction']}\n",
                encoding="utf-8",
            )
            prediction.with_suffix(".jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
            pipeline = {
                "task": "VAD",
                "metric": "f1",
                "pipeline_id": "vad.test.f1",
                "route_id": "vad-test",
                "required_roles": ["reference_jsonl", "sample_output"],
                "nodes": [{"node_id": "validation/vad_contract"}],
            }
            observed = {}

            def fake_run(*, engine_root, request, timeout):
                observed.update(request)
                self.assertEqual(len(_read_jsonl(Path(request["reference_jsonl"]))), 1)
                self.assertEqual(len(_read_jsonl(Path(request["sample_output"]))), 1)
                return {
                    "pipeline": pipeline,
                    "summary": {
                        "metric": "f1",
                        "score": 1.0,
                        "pipeline_id": "vad.test.f1",
                        "language": "n/a",
                        "output_dir": request["output_dir"],
                    },
                    "report": {"score": 1.0},
                }

            with (
                mock.patch.object(ep, "_describe_external_pipeline", return_value=pipeline),
                mock.patch.object(ep, "_run_external_pipeline", side_effect=fake_run),
                mock.patch.object(ep, "_evaluation_runtime_binding", return_value={"status": "ready"}),
            ):
                explicit_result = ep.evaluate_vad_prediction_file_external(
                    _DatasetManager(dataset),
                    _SotaManager(),
                    "vad-fixture",
                    prediction,
                    engine_source="test",
                    engine_root=ENGINE_ROOT,
                    external_runs_dir=root / "runs",
                    device="cpu",
                    cache_dir=None,
                    timeout=30,
                    metric_override="f1",
                    max_samples=1,
                )
                result = ep.evaluate_vad_prediction_file_external(
                    _DatasetManager(dataset),
                    _SotaManager(),
                    "vad-fixture",
                    prediction,
                    engine_source="test",
                    engine_root=ENGINE_ROOT,
                    external_runs_dir=root / "runs",
                    device="cpu",
                    cache_dir=None,
                    timeout=30,
                    metric_override="f1",
                    max_samples=1,
                    requested_metric_source_override="task_default_suite",
                )
        self.assertEqual(result["expected_samples"], 1)
        self.assertEqual(result["total_dataset_samples"], 2)
        self.assertEqual(
            explicit_result["evaluation_context"]["requested_metric_source"],
            "cli_override",
        )
        self.assertEqual(
            result["evaluation_context"]["requested_metric_source"],
            "task_default_suite",
        )
        self.assertIn("reference_jsonl", observed)
        self.assertIn("sample_output", observed)

    def test_sample_report_keeps_structures_and_metric_details(self) -> None:
        result = {
            "dataset": "vad-fixture",
            "task": "VAD",
            "metric": "f1",
            "details": {
                "report": {
                    "details": {
                        "rows": [{"key": "vad-1", "tp_sec": 0.4, "fp_sec": 0.0}]
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "samples.jsonl"
            ep._write_sample_report(
                output_path=output,
                samples=[_sample()],
                predictions={"vad-1": "projection"},
                result=result,
                structured_predictions={"vad-1": _structured()},
            )
            row = _read_jsonl(output)[0]
        self.assertEqual(row["prediction"], _prediction())
        self.assertEqual(row["reference"]["duration"], 1.0)
        self.assertEqual(row["metric_details"]["tp_sec"], 0.4)


@unittest.skipUnless(
    ENGINE_ROOT.is_dir() and _locked_evaluation_runtime_available(),
    "locked standalone Evaluation Runtime is required",
)
class RealStandaloneTests(unittest.TestCase):
    def test_all_vad_routes_score_exact_predictions(self) -> None:
        frame_scores = [
            {"start": 0.0, "end": 0.2, "score": 0.1},
            {"start": 0.2, "end": 0.6, "score": 0.9},
            {"start": 0.6, "end": 1.0, "score": 0.1},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, output, _ = ep._write_external_vad_role_files(
                samples=[_sample()],
                structured_predictions={
                    "vad-1": _structured(frame_scores=frame_scores)
                },
                output_dir=root / "roles",
            )
            scores = {}
            for metric in (*ep.VAD_DETECTION_METRICS, "auc_roc"):
                pipeline = ep._describe_external_pipeline(
                    engine_root=ENGINE_ROOT,
                    task="VAD",
                    language="n/a",
                    metric=metric,
                    timeout=30,
                )
                payload = ep._run_external_pipeline(
                    engine_root=ENGINE_ROOT,
                    request={
                        "task": "VAD",
                        "language": None,
                        "metric": metric,
                        "pipeline_id": None,
                        "output_dir": str(root / metric),
                        "reference_jsonl": str(reference),
                        "sample_output": str(output),
                        "pipeline": pipeline,
                        "device": "cpu",
                        "cache_dir": None,
                    },
                    timeout=30,
                )
                scores[metric] = payload["summary"]["score"]
        self.assertEqual(
            scores,
            {"f1": 1.0, "p_fa": 0.0, "p_miss": 0.0, "dcf_nist": 0.0, "auc_roc": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
