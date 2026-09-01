#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_predictions as ep  # noqa: E402
import generate_predictions_via_server as gp  # noqa: E402
import materialize_predictions_template as mt  # noqa: E402
import resolve_eval_input as rei  # noqa: E402
import validate_prediction_files as vp  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
ENGINE_ROOT = REPO_ROOT / "sure" / "external" / "sure-evaluation"
MODEL_WRAPPER_SERVER = Path(__file__).resolve().parent / "model_wrapper_mcp_server.py"
SE_FIXTURE_ROOT = REPO_ROOT / "fixtures" / "tasks" / "se" / "fleurs_noise_smoke"
SE_PIPELINE = {
    "task": "se",
    "language": "n/a",
    "metric": "si_sdr",
    "metrics": ["si-sdr"],
    "pipeline_id": "se.any.si_sdr.si_sdr_v1",
    "required_roles": ["samples_jsonl"],
    "run_args": {"samples_jsonl": None, "output_dir": None},
    "nodes": [{"node_id": "scoring/si_sdr"}],
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
        self, dataset: str, score: object, fallback_names: object = ()
    ) -> float | dict[str, object]:
        self.calculate_calls += 1
        if self.baseline is not None:
            return 0.75
        return {"status": "missing_baseline", "dataset": dataset, "score": score}


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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_pcm_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    return path


def _evaluation_runtime_contract() -> dict[str, object]:
    runtime_root = REPO_ROOT / "sure" / ".runtime" / "evaluation"
    for manifest_path in runtime_root.glob("*/runtime-manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(str(payload.get("engine_root") or "")).resolve() == ENGINE_ROOT.resolve():
            return payload
    return {}


EVALUATION_RUNTIME = _evaluation_runtime_contract()


def _structured(key: str, enhanced_audio: Path) -> dict[str, object]:
    resolved = str(enhanced_audio.resolve())
    return {
        "key": key,
        "dataset": "fixture-se",
        "task": "SE",
        "language": "n/a",
        "prediction": {"audio_path": resolved, "enhanced_audio": resolved},
        "normalized_prediction": resolved,
        "raw_response": {"enhanced_audio": resolved},
    }


class SEDefaultAndPayloadTests(unittest.TestCase):
    def test_si_sdr_is_the_harness_default(self) -> None:
        self.assertEqual(rei._fallback_default_metrics("SE", "n/a"), ["si-sdr"])
        self.assertEqual(rei._default_metrics("SE", "n/a", ENGINE_ROOT), ["si-sdr"])
        self.assertEqual(mt._default_metric("SE", "n/a"), "si-sdr")

    def test_validation_requires_matching_nonempty_audio_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_dir = root / "predictions"
            enhanced = _write_pcm_wav(predictions_dir / "audio" / "enhanced.wav")
            samples = [{"key": "utt", "task": "SE"}]
            valid = {"utt": _structured("utt", enhanced)}
            self.assertEqual(
                vp._task_contract_violations(
                    samples,
                    valid,
                    base_dir=predictions_dir,
                ),
                [],
            )
            invalid_rows = [
                {"prediction": {"audio_path": str(enhanced)}},
                {
                    "prediction": {
                        "audio_path": str(enhanced),
                        "enhanced_audio": str(root / "other.wav"),
                    }
                },
            ]
            for row in invalid_rows:
                row.update({"key": "utt", "task": "SE", "normalized_prediction": "x"})
                with self.subTest(row=row):
                    self.assertEqual(
                        vp._task_contract_violations(
                            samples,
                            {"utt": row},
                            base_dir=predictions_dir,
                        ),
                        ["utt"],
                    )

    def test_validation_rejects_outside_symlink_and_non_pcm_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_dir = root / "predictions"
            audio_dir = predictions_dir / "audio"
            outside = _write_pcm_wav(root / "outside.wav")
            target = _write_pcm_wav(root / "target.wav")
            audio_dir.mkdir(parents=True)
            nested = audio_dir / "nested"
            nested.mkdir()
            lexical_target = _write_pcm_wav(predictions_dir / "lexical-target.wav")
            lexical_escape = nested / ".." / ".." / lexical_target.name
            symlink = audio_dir / "linked.wav"
            symlink.symlink_to(target)
            non_pcm = audio_dir / "not-pcm.wav"
            non_pcm.write_bytes(b"not a PCM WAV")
            samples = [{"key": "utt", "task": "SE"}]

            for candidate in (outside, lexical_escape, symlink, non_pcm):
                row = {
                    "key": "utt",
                    "task": "SE",
                    "prediction": {
                        "audio_path": str(candidate),
                        "enhanced_audio": str(candidate),
                    },
                    "normalized_prediction": str(candidate),
                }
                with self.subTest(candidate=candidate):
                    self.assertEqual(
                        vp._task_contract_violations(
                            samples,
                            {"utt": row},
                            base_dir=predictions_dir,
                        ),
                        ["utt"],
                    )


class SEBridgeTests(unittest.TestCase):
    def test_bridge_rejects_lexical_audio_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "audio" / "nested"
            nested.mkdir(parents=True)
            escaped = _write_pcm_wav(root / "escaped.wav")
            lexical_escape = nested / ".." / ".." / escaped.name
            structured = {
                "key": "utt",
                "dataset": "fixture-se",
                "task": "SE",
                "prediction": {
                    "audio_path": str(lexical_escape),
                    "enhanced_audio": str(lexical_escape),
                },
                "normalized_prediction": str(lexical_escape),
            }
            with self.assertRaisesRegex(ValueError, "parent component"):
                ep._write_external_audio_samples_jsonl(
                    task="SE",
                    dataset_jsonl_path=root / "dataset.jsonl",
                    samples=[{"key": "utt", "task": "SE"}],
                    structured_predictions={"utt": structured},
                    structured_prediction_path=root / "predictions.jsonl",
                    output_path=root / "samples.jsonl",
                    required_roles={"enhanced_audio"},
                )

    def test_samples_jsonl_maps_noisy_clean_and_enhanced_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            enhanced = _write_pcm_wav(root / "audio" / "enhanced.wav")
            sample = json.loads(
                (SE_FIXTURE_ROOT / "gt.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            output = root / "samples.jsonl"
            ep._write_external_audio_samples_jsonl(
                task="SE",
                dataset_jsonl_path=SE_FIXTURE_ROOT / "gt.jsonl",
                samples=[sample],
                structured_predictions={sample["key"]: _structured(sample["key"], enhanced)},
                structured_prediction_path=root / "predictions.jsonl",
                output_path=output,
                required_roles={"enhanced_audio", "noisy_audio", "reference_audio"},
            )
            row = ep.load_jsonl(output)[0]
        self.assertEqual(row["sample_id"], sample["key"])
        self.assertEqual(row["enhanced_audio"], str(enhanced.resolve()))
        self.assertEqual(row["noisy_audio"], str((SE_FIXTURE_ROOT / "noisy.wav").resolve()))
        self.assertEqual(row["reference_audio"], str((SE_FIXTURE_ROOT / "clean.wav").resolve()))
        self.assertNotEqual(row["reference_audio"], row["noisy_audio"])

    def test_full_reference_route_never_falls_back_to_noisy_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            noisy = _write_pcm_wav(root / "noisy.wav")
            enhanced = _write_pcm_wav(root / "audio" / "enhanced.wav")
            with self.assertRaisesRegex(ValueError, "reference_audio"):
                ep._write_external_audio_samples_jsonl(
                    task="SE",
                    dataset_jsonl_path=root / "dataset.jsonl",
                    samples=[{"key": "utt", "task": "SE", "path": str(noisy)}],
                    structured_predictions={"utt": _structured("utt", enhanced)},
                    structured_prediction_path=root / "predictions.jsonl",
                    output_path=root / "samples.jsonl",
                    required_roles={"enhanced_audio", "noisy_audio", "reference_audio"},
                )

    def test_noisy_and_clean_reference_must_not_resolve_to_the_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            noisy = _write_pcm_wav(root / "noisy.wav")
            enhanced = _write_pcm_wav(root / "audio" / "enhanced.wav")
            with self.assertRaisesRegex(ValueError, "noisy and clean reference audio must differ"):
                ep._write_external_audio_samples_jsonl(
                    task="SE",
                    dataset_jsonl_path=root / "dataset.jsonl",
                    samples=[
                        {
                            "key": "utt",
                            "task": "SE",
                            "path": str(noisy),
                            "reference_audio": str(noisy),
                        }
                    ],
                    structured_predictions={"utt": _structured("utt", enhanced)},
                    structured_prediction_path=root / "predictions.jsonl",
                    output_path=root / "samples.jsonl",
                    required_roles={"enhanced_audio", "noisy_audio", "reference_audio"},
                )

    def test_external_evaluator_defaults_to_atomic_si_sdr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            noisy = _write_pcm_wav(root / "noisy.wav")
            clean = _write_pcm_wav(root / "clean.wav")
            predictions_dir = root / "predictions"
            enhanced = _write_pcm_wav(
                predictions_dir / "audio" / "fixture-se" / "enhanced.wav"
            )
            dataset = root / "dataset.jsonl"
            prediction = predictions_dir / "fixture-se.txt"
            _write_jsonl(
                dataset,
                [
                    {
                        "key": "utt",
                        "task": "SE",
                        "path": str(noisy),
                        "reference_audio": str(clean),
                    }
                ],
            )
            prediction.write_text(f"utt\t{enhanced}\n", encoding="utf-8")
            _write_jsonl(prediction.with_suffix(".jsonl"), [_structured("utt", enhanced)])
            captured: dict[str, object] = {}

            def fake_run(*, engine_root: Path, request: dict[str, object], timeout: int):
                captured.update(request)
                row = ep.load_jsonl(Path(str(request["samples_jsonl"])))[0]
                self.assertEqual(row["reference_audio"], str(clean))
                return {
                    "pipeline": dict(SE_PIPELINE),
                    "summary": {
                        "metric": "si_sdr",
                        "score": 1.0,
                        "pipeline_id": SE_PIPELINE["pipeline_id"],
                        "language": "n/a",
                        "output_dir": request["output_dir"],
                        "node_config_paths": [],
                    },
                    "report": {"score": 1.0},
                }

            with (
                mock.patch.object(ep, "_describe_external_pipeline", return_value=dict(SE_PIPELINE)) as describe,
                mock.patch.object(ep, "_run_external_pipeline", side_effect=fake_run),
                mock.patch.object(ep, "_evaluation_runtime_binding", return_value={}),
            ):
                result = ep.evaluate_audio_prediction_file_external(
                    FakeDatasetManager(dataset),
                    FakeSOTAManager(),
                    "fixture-se",
                    prediction,
                    engine_source="test",
                    engine_root=root / "engine",
                    external_runs_dir=root / "external",
                    device="cpu",
                    cache_dir=None,
                    timeout=30,
                    metric_override=None,
                    task_override="SE",
                )
        self.assertEqual(describe.call_args.kwargs["metric"], "si-sdr")
        self.assertEqual(result["metric"], "si_sdr")
        self.assertIsNone(result["rps"])
        self.assertEqual(
            result["rps_status"]["status"],
            "rps_undefined_for_db_metric",
        )
        self.assertTrue(str(captured["samples_jsonl"]).endswith("samples.jsonl"))


class SESampleReportTests(unittest.TestCase):
    def test_sample_report_uses_structured_prediction_reference_roles_and_sample_id_rows(
        self,
    ) -> None:
        sample = {
            "key": "utt",
            "task": "SE",
            "noisy_audio": "noisy.wav",
            "reference_audio": "clean.wav",
        }
        prediction = {
            "audio_path": "/run/predictions/audio/fixture-se/enhanced.wav",
            "enhanced_audio": "/run/predictions/audio/fixture-se/enhanced.wav",
        }
        result = {
            "dataset": "fixture-se",
            "task": "SE",
            "metric": "si_sdr",
            "details": {
                "report": {
                    "details": {
                        "rows": [
                            {
                                "sample_id": "utt",
                                "full_reference": {"si_sdr": {"si_sdr": 8.0}},
                            }
                        ]
                    }
                }
            },
        }
        structured = {
            "key": "utt",
            "task": "SE",
            "prediction": prediction,
            "normalized_prediction": prediction["audio_path"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sample-report.jsonl"
            ep._write_sample_report(
                output_path=output,
                samples=[sample],
                predictions={"utt": "TSV projection must not become the report prediction"},
                result=result,
                structured_predictions={"utt": structured},
            )
            row = ep.load_jsonl(output)[0]

        self.assertEqual(row["prediction"], prediction)
        self.assertEqual(
            row["reference"],
            {"noisy_audio": "noisy.wav", "reference_audio": "clean.wav"},
        )
        self.assertEqual(row["metric_details"]["sample_id"], "utt")
        self.assertEqual(
            row["metric_details"]["full_reference"]["si_sdr"]["si_sdr"],
            8.0,
        )


class SERPSAndReportTests(unittest.TestCase):
    @staticmethod
    def _result(
        *,
        metric: str = "si_sdr",
        rps: object = None,
        rps_status: object = None,
    ) -> dict[str, object]:
        return {
            "dataset": "fixture-se",
            "jsonl_path": "/tmp/fixture-se.jsonl",
            "prediction_path": "/tmp/fixture-se.txt",
            "task": "SE",
            "language": "n/a",
            "metric": metric,
            "score": 8.0,
            "rps": rps,
            "rps_status": rps_status,
            "num_samples": 1,
            "evaluation_backend": "external",
            "evaluator_version": "sure-evaluation",
            "pipeline_id": SE_PIPELINE["pipeline_id"],
            "evaluation_context": {"nodes": ["scoring/si_sdr"]},
            "details": {"report": {}, "pipeline": dict(SE_PIPELINE)},
        }

    def test_si_sdr_has_db_unit_and_undefined_rps_status(self) -> None:
        sota = FakeSOTAManager(FakeBaseline("si_sdr"))
        rps, status = ep._calculate_se_rps(sota, "fixture-se", "si-sdr", 8.0)
        self.assertIsNone(rps)
        self.assertEqual(status["status"], "rps_undefined_for_db_metric")
        self.assertEqual(status["unit"], "dB")
        self.assertEqual(sota.calculate_calls, 0)
        for metric in ("si_sdr", "si-sdr", "sisdr"):
            with self.subTest(metric=metric):
                self.assertEqual(ep._metric_unit(metric), "dB")

    def test_non_db_se_metric_rejects_a_different_metric_baseline(self) -> None:
        sota = FakeSOTAManager(FakeBaseline("pesq"))
        rps, status = ep._calculate_se_rps(sota, "fixture-se", "stoi", 0.9)
        self.assertIsNone(rps)
        self.assertEqual(status["status"], "missing_metric_baseline")
        self.assertEqual(status["available_baseline_metric"], "pesq")
        self.assertEqual(sota.calculate_calls, 0)

    def test_record_keeps_si_sdr_status_without_recalculating_rps(self) -> None:
        manager = FakeRecordManager()
        status = {"status": "rps_undefined_for_db_metric", "unit": "dB"}
        record = ep._record_evaluation_result(
            manager,
            tool_name="enhance_speech",
            result=self._result(rps_status=status),
        )
        self.assertIsNotNone(record)
        self.assertIsNone(record.rps)
        self.assertEqual(record.metadata["rps_status"], status)
        self.assertEqual(manager.evaluate_calls, 0)
        self.assertEqual(manager.database.records, [record])

    def test_payload_report_and_merge_preserve_si_sdr_rps_status(self) -> None:
        status = {"status": "rps_undefined_for_db_metric", "unit": "dB"}
        payload_row = ep._dataset_metric_row(self._result(rps_status=status))
        report_row = ep._standard_report_row_v1(
            row=payload_row,
            validation={},
            run_id="run",
            protocol_id="standard_system",
            model_dir=None,
            tool_name="enhance_speech",
        )
        schema = json.loads(
            (
                REPO_ROOT
                / "sure"
                / "skills"
                / "sure_eval"
                / "schemas"
                / "report_row.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(set(schema["required"]).issubset(report_row))
        self.assertEqual(report_row["metric"]["unit"], "dB")
        self.assertIsNone(report_row["rps"])
        self.assertEqual(report_row["rps_status"], status)

        with tempfile.TemporaryDirectory() as temporary:
            payload_path = Path(temporary) / "evaluation_payload.json"
            payload_path.write_text(
                json.dumps({"results": [payload_row]}),
                encoding="utf-8",
            )
            merged = ep.merge_payload_results([payload_path])
        self.assertIsNone(merged[0]["rps"])
        self.assertEqual(merged[0]["rps_status"], status)


@unittest.skipUnless(
    ENGINE_ROOT.is_dir() and SE_FIXTURE_ROOT.is_dir() and bool(EVALUATION_RUNTIME),
    "sure-evaluation, SE fixture, and locked Evaluation Runtime are required",
)
class SEFauxMcpStandaloneE2ETests(unittest.TestCase):
    def test_faux_mcp_output_runs_through_standalone_si_sdr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.yaml").write_text(
                "model:\n  task: SE\ntools:\n  - name: enhance_speech\n",
                encoding="utf-8",
            )
            (model_dir / "model.py").write_text(
                "from pathlib import Path\n"
                "import shutil\n"
                "class ModelWrapper:\n"
                "    def __init__(self, *args): pass\n"
                "    def predict(self, args):\n"
                "        output = Path(args['output_path'])\n"
                "        output.parent.mkdir(parents=True, exist_ok=True)\n"
                "        shutil.copyfile(args['audio_path'], output)\n"
                "        return {'enhanced_audio': str(output)}\n",
                encoding="utf-8",
            )
            clean = (SE_FIXTURE_ROOT / "clean.wav").resolve()
            noisy = (SE_FIXTURE_ROOT / "noisy.wav").resolve()
            sample = json.loads(
                (SE_FIXTURE_ROOT / "gt.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            output_dir = root / "run" / "predictions" / "audio" / "fixture-se"
            arguments = gp._build_tool_arguments(
                repo_root=root,
                sample=sample,
                task="SE",
                language="n/a",
                argument_name="audio_path",
                audio_path=noisy,
                output_audio_dir=output_dir,
            )
            process = subprocess.Popen(
                [sys.executable, str(MODEL_WRAPPER_SERVER), "--model-dir", str(model_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                gp._send_request(
                    process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                )
                response = gp._send_request(
                    process,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "enhance_speech", "arguments": arguments},
                    },
                )
                raw = gp._extract_response_payload(response)
                projection, normalized = gp._normalize_prediction_payload(
                    raw,
                    task="SE",
                    expected_audio_output=arguments["output_path"],
                )
            finally:
                try:
                    gp._send_request(
                        process,
                        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
                    )
                except Exception:
                    process.terminate()
                process.wait(timeout=10)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

            samples_jsonl = root / "samples.jsonl"
            ep._write_external_audio_samples_jsonl(
                task="SE",
                dataset_jsonl_path=SE_FIXTURE_ROOT / "gt.jsonl",
                samples=[sample],
                structured_predictions={
                    sample["key"]: {
                        "key": sample["key"],
                        "dataset": "fixture-se",
                        "task": "SE",
                        "prediction": normalized,
                        "normalized_prediction": projection,
                    }
                },
                structured_prediction_path=(
                    root / "run" / "predictions" / "fixture-se.jsonl"
                ),
                output_path=samples_jsonl,
                required_roles={"enhanced_audio", "noisy_audio", "reference_audio"},
            )
            code = (
                "import json,sys;"
                "from sure_eval.evaluation.cli_adapters import build_pipeline_spec,run_pipeline_spec;"
                "p=build_pipeline_spec('se',metric='si-sdr');"
                "s=run_pipeline_spec(p,samples_jsonl=sys.argv[1],output_dir=sys.argv[2],device='cpu');"
                "print(json.dumps(s))"
            )
            env = os.environ.copy()
            for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
                env.pop(key, None)
            env["SURE_HARNESS_RUNTIME_ROOT"] = str(EVALUATION_RUNTIME["harness_runtime_root"])
            env["PYTHONPATH"] = str(ENGINE_ROOT / "src")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    str(EVALUATION_RUNTIME["python_executable"]),
                    "-c",
                    code,
                    str(samples_jsonl),
                    str(root / "metric"),
                ],
                cwd=ENGINE_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])
        self.assertEqual(summary["pipeline_id"], "se.any.si_sdr.si_sdr_v1")
        self.assertTrue(math.isfinite(float(summary["score"])))


if __name__ == "__main__":
    unittest.main()
