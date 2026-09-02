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
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[3]
KWS_FIXTURE = REPO_ROOT / "fixtures" / "tasks" / "kws" / "wenwen_smoke" / "kws"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_artifact  # noqa: E402
import finalize_trans_bundle  # noqa: E402
import materialize_trans_inputs  # noqa: E402
import mcp_smoke  # noqa: E402
import prepare_fixture  # noqa: E402
import run_trans_validate  # noqa: E402
import scaffold_adapter  # noqa: E402


KWS_CONTRACT = scaffold_adapter.io_contract_for("kws")


def write_resolved(run_dir: Path, fixture: Path) -> None:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "trans_input_resolved.json").write_text(
        json.dumps(
            {
                "model_name": "example__kws",
                "task_type": "kws",
                "build_context": str(fixture.parent),
                "fixture_path": str(fixture),
            }
        ),
        encoding="utf-8",
    )


def prepare_shared_fixture(run_dir: Path) -> dict:
    write_resolved(run_dir, KWS_FIXTURE)
    with mock.patch.object(sys, "argv", ["prepare_fixture.py", "--run-dir", str(run_dir)]):
        assert prepare_fixture.main() == 0
    return json.loads((run_dir / "artifacts" / "fixture_manifest.json").read_text(encoding="utf-8"))


def render_validate(model_dir: Path) -> Path:
    template = (SCRIPTS_DIR / "templates" / "validate.py").read_text(encoding="utf-8")
    rendered = (
        template.replace("__MODEL_NAME__", "example__kws")
        .replace("__TASK_TYPE__", "KWS")
        .replace(
            "__IO_CONTRACT_JSON__",
            json.dumps(KWS_CONTRACT, ensure_ascii=True, separators=(",", ":")),
        )
    )
    validate_path = model_dir / "validate.py"
    validate_path.write_text(rendered, encoding="utf-8")
    (model_dir / "model.py").write_text(
        "from pathlib import Path\n"
        "class ModelWrapper:\n"
        "    def load(self): self.model = object()\n"
        "    def predict(self, payload):\n"
        "        if 'positive' in Path(payload['audio_path']).name:\n"
        "            return {'detected': True, 'keyword': '嗨小问', 'score': 1.0}\n"
        "        return {'detected': False, 'keyword': None, 'score': None}\n",
        encoding="utf-8",
    )
    return validate_path


class KwsTransTests(unittest.TestCase):
    def test_task_and_adapter_contract_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "infer.py"
            entrypoint.write_text("def kws_predict(audio_path): pass\n", encoding="utf-8")
            self.assertEqual(
                materialize_trans_inputs.resolve_task_type(None, entrypoint, Path("wakeword-model")),
                "kws",
            )
        tool_name, input_schema = scaffold_adapter.tool_contract("kws")
        self.assertEqual(tool_name, "kws_predict")
        self.assertEqual(input_schema["required"], ["audio_path"])
        self.assertEqual(KWS_CONTRACT["primary_field"], "detected")
        self.assertEqual(KWS_CONTRACT["required_fields"], ["detected", "keyword", "score"])
        self.assertTrue(mcp_smoke.output_is_nonempty("detected", False))

    def test_asr_hotword_support_is_not_silently_classified_as_kws(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "infer.py"
            entrypoint.write_text(
                "def transcribe_audio(audio_path, hotwords=None): return model.transcribe(audio_path)\n",
                encoding="utf-8",
            )
            self.assertEqual(
                materialize_trans_inputs.resolve_task_type(None, entrypoint, Path("asr-model")),
                "asr",
            )

    def test_prepare_and_gate_preserve_positive_negative_nested_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            manifest = prepare_shared_fixture(run_dir)
            check_artifact.validate_fixture_manifest(manifest)
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(
                {sample["expected_detected"] for sample in manifest["samples"]},
                {False, True},
            )
            self.assertTrue((run_dir / "fixture" / "kws" / "audio" / "positive_nihao_wenwen.wav").is_file())
            self.assertTrue((run_dir / "fixture" / "kws" / "audio" / "negative_mobvoi.wav").is_file())

    def test_prepare_rejects_single_polarity_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture"
            source.mkdir()
            audio = source / "positive.wav"
            shutil.copy2(KWS_FIXTURE / "audio" / "positive_nihao_wenwen.wav", audio)
            row = {
                "key": "positive",
                "audio": "positive.wav",
                "keywords": ["嗨小问"],
                "expected_detected": True,
                "expected_keyword": "嗨小问",
                "text": "嗨小问",
            }
            (source / "gt.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for _ in range(2)) + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                prepare_fixture.prepare_kws_fixture(
                    {"model_name": "example__kws"}, source, run_dir
                )

            second_positive = {**row, "key": "positive-2"}
            (source / "gt.jsonl").write_text(
                json.dumps(row, ensure_ascii=False)
                + "\n"
                + json.dumps(second_positive, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "positive and one negative"):
                prepare_fixture.prepare_kws_fixture(
                    {"model_name": "example__kws"}, source, run_dir
                )

            escaped = {**row, "key": "escaped", "audio": "../outside.wav"}
            negative = {
                "key": "negative",
                "audio": "positive.wav",
                "keywords": ["嗨小问"],
                "expected_detected": False,
                "expected_keyword": None,
                "label": "negative",
            }
            (source / "gt.jsonl").write_text(
                json.dumps(escaped, ensure_ascii=False)
                + "\n"
                + json.dumps(negative, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "stay inside"):
                prepare_fixture.prepare_kws_fixture(
                    {"model_name": "example__kws"}, source, run_dir
                )

            positive_with_threshold = {**row, "threshold": 0.4}
            (source / "gt.jsonl").write_text(
                json.dumps(positive_with_threshold, ensure_ascii=False)
                + "\n"
                + json.dumps(negative, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "threshold must equal 0.5"):
                prepare_fixture.prepare_kws_fixture(
                    {"model_name": "example__kws"}, source, run_dir
                )

    def test_kws_output_contract_rejects_malformed_and_wrong_keyword_results(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "sure_trans_validate_template", SCRIPTS_DIR / "templates" / "validate.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        positive = {
            "expected_detected": True,
            "expected_keyword": "嗨小问",
        }
        negative = {
            "expected_detected": False,
            "expected_keyword": None,
        }
        self.assertEqual(
            module.validate_kws_result(
                {"detected": False, "keyword": None, "score": None}, negative
            ),
            [],
        )
        malformed = module.validate_kws_result(
            {"detected": "false", "keyword": None, "score": float("nan")}, negative
        )
        self.assertIn("detected must be a boolean", malformed)
        self.assertIn("score must be a finite number or null", malformed)
        wrong_keyword = module.validate_kws_result(
            {"detected": True, "keyword": "你好问问", "score": 0.9}, positive
        )
        self.assertTrue(any("keyword disagrees" in item for item in wrong_keyword))
        self.assertIn(
            "score must be within [0, 1]",
            module.validate_kws_result(
                {"detected": True, "keyword": "嗨小问", "score": 1.1}, positive
            ),
        )
        self.assertIn(
            "detected=true requires score >= 0.5",
            module.validate_kws_result(
                {"detected": True, "keyword": "嗨小问", "score": 0.4}, positive
            ),
        )
        self.assertIn(
            "detected=false requires score < 0.5",
            module.validate_kws_result(
                {"detected": False, "keyword": None, "score": 0.6}, negative
            ),
        )

    def test_generated_validate_runs_every_kws_fixture_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            shutil.copytree(KWS_FIXTURE, model_dir / "fixture" / "kws")
            rows = []
            gt_path = model_dir / "fixture" / "kws" / "gt.jsonl"
            for row in prepare_fixture.read_jsonl(gt_path):
                detected = prepare_fixture.kws_expected_detected(row, key=str(row["key"]))
                row["expected_detected"] = detected
                row["expected_keyword"] = row.get("text") if detected else None
                rows.append(row)
            gt_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            validate_path = render_validate(model_dir)
            artifacts = root / "artifacts"
            env = {**os.environ, "SURE_VALIDATE_ARTIFACTS_DIR": str(artifacts)}
            for stage in ("infer", "contract"):
                completed = subprocess.run(
                    [sys.executable, str(validate_path), "--stage", stage],
                    cwd=model_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            sample = json.loads((artifacts / "sample_output.json").read_text(encoding="utf-8"))
            self.assertEqual(len(sample["rows"]), 2)
            results = {row["result"]["detected"] for row in sample["rows"]}
            self.assertEqual(results, {False, True})

    def test_mcp_smoke_checks_positive_and_negative_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            shutil.copytree(KWS_FIXTURE, fixture)
            server = root / "server.py"
            server.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "for line in sys.stdin:\n"
                " req=json.loads(line); method=req.get('method'); rid=req.get('id')\n"
                " if method=='initialize': result={'protocolVersion':'2024-11-05'}\n"
                " elif method=='tools/list': result={'tools':[{'name':'kws_predict'}]}\n"
                " elif method=='tools/call':\n"
                "  audio=Path(req['params']['arguments']['audio_path']).name\n"
                "  value={'detected': True, 'keyword': '嗨小问', 'score': 1.0} if 'positive' in audio else {'detected': False, 'keyword': None, 'score': None}\n"
                "  result={'content':[{'type':'text','text':json.dumps(value, ensure_ascii=False)}]}\n"
                " else: result={}\n"
                " print(json.dumps({'jsonrpc':'2.0','id':rid,'result':result}, ensure_ascii=False), flush=True)\n"
                " if method=='shutdown': break\n",
                encoding="utf-8",
            )
            evidence = root / "mcp_smoke.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "mcp_smoke.py"),
                    "--fixture-gt-jsonl",
                    str(fixture / "gt.jsonl"),
                    "--tool",
                    "kws_predict",
                    "--server-command",
                    sys.executable,
                    str(server),
                    "--produces",
                    str(evidence),
                    "--timeout",
                    "30",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["tools_call"]["num_samples"], 2)
            self.assertEqual(payload["fixture_gt_jsonl"], "gt.jsonl")
            self.assertNotIn(str(root), json.dumps(payload, ensure_ascii=False))
            self.assertTrue(payload["fixture_gt_sha256"])
            self.assertEqual(
                {row["result"]["detected"] for row in payload["tools_call"]["samples"]},
                {False, True},
            )
            self.assertIsNone(run_trans_validate.validate_mcp_evidence(evidence, "kws_predict"))

    def test_equivalence_compares_all_kws_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "trans_input_resolved.json").write_text(
                json.dumps({"task_type": "kws"}), encoding="utf-8"
            )
            value = {
                "rows": [
                    {"key": "positive", "result": {"detected": True, "keyword": "嗨小问", "score": 0.9}},
                    {"key": "negative", "result": {"detected": False, "keyword": None, "score": None}},
                ]
            }
            baseline = run_dir / "baseline.json"
            adapter = run_dir / "adapter.json"
            baseline.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            adapter.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNone(error)
            self.assertTrue(evidence["match"])
            value["rows"][0]["result"]["score"] = 0.8
            adapter.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNotNone(error)
            self.assertEqual(evidence["mismatches"]["positive"]["fields"], ["score"])

    def test_finalizer_preserves_fixture_tree_and_keyed_sample_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            manifest = prepare_shared_fixture(run_dir)
            model_dir = root / "models" / "example__kws"
            model_dir.mkdir(parents=True)
            resolved = {"task_type": "kws", "model_name": "example__kws"}
            finalize_trans_bundle.stage_fixture(run_dir, model_dir, resolved)
            finalized = json.loads(
                (run_dir / "artifacts" / "fixture_manifest.json").read_text(encoding="utf-8")
            )
            check_artifact.validate_fixture_manifest(finalized)
            self.assertTrue((model_dir / "fixture" / "kws" / "audio" / "negative_mobvoi.wav").is_file())

            (run_dir / "artifacts" / "adapter_manifest.json").write_text(
                json.dumps({"io_contract": KWS_CONTRACT}), encoding="utf-8"
            )
            output = {
                "rows": [
                    {
                        "key": sample["key"],
                        "result": (
                            {"detected": True, "keyword": sample["expected_keyword"], "score": 1.0}
                            if sample["expected_detected"]
                            else {"detected": False, "keyword": None, "score": None}
                        ),
                    }
                    for sample in manifest["samples"]
                ]
            }
            validation_dir = run_dir / "artifacts" / "adapter_validation"
            validation_dir.mkdir()
            (validation_dir / "sample_output.json").write_text(
                json.dumps(output, ensure_ascii=False), encoding="utf-8"
            )
            finalize_trans_bundle.promote_sample_output(run_dir)
            promoted = json.loads(
                (run_dir / "artifacts" / "sample_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(promoted, output)

    def test_finalized_mcp_projection_replaces_host_paths_and_rejects_unknown_shared_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            model_dir = root / "model"
            run_dir.mkdir()
            model_dir.mkdir()
            source = root / "mcp_result.json"
            destination = root / "portable.json"
            source.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "run_command": [
                            "python",
                            str(run_dir / "adapter" / "mcp_smoke.py"),
                            "--audio",
                            str(run_dir / "fixture" / "kws" / "positive.wav"),
                        ],
                        "cwd": str(run_dir),
                        "protocol": {"server_command": ["python", "/opt/sure_trans/server.py"]},
                    }
                ),
                encoding="utf-8",
            )
            finalize_trans_bundle.project_mcp_result(
                source,
                destination,
                run_dir=run_dir,
                model_dir=model_dir,
                resolved={},
            )
            projected_text = destination.read_text(encoding="utf-8")
            self.assertNotIn(str(run_dir), projected_text)
            self.assertIn("<run_dir>", projected_text)
            self.assertIn("/opt/sure_trans/server.py", projected_text)

            source.write_text(
                json.dumps({"status": "passed", "log_path": "/shared-storage/private/log.txt"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-portable"):
                finalize_trans_bundle.project_mcp_result(
                    source,
                    destination,
                    run_dir=run_dir,
                    model_dir=model_dir,
                    resolved={},
                )


if __name__ == "__main__":
    unittest.main()
