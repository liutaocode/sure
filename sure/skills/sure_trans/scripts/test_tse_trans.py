#!/usr/bin/env python3
"""Regression tests for the structured TSE Trans contract."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import check_artifact  # noqa: E402
import classification_contract  # noqa: E402
import mcp_smoke  # noqa: E402
import prepare_fixture  # noqa: E402
import run_trans_validate  # noqa: E402
import scaffold_adapter  # noqa: E402


def write_wav(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))


class TSETransTest(unittest.TestCase):
    def test_slu_schema_accepts_scalar_choice_arrays(self) -> None:
        schema = classification_contract.input_schema_for("slu")
        choices = schema["properties"]["choices"]["oneOf"]
        self.assertIn({"type": "array", "minItems": 1}, choices)

    def test_model_template_normalizes_classification_aliases(self) -> None:
        template = runpy.run_path(str(SCRIPTS_DIR / "templates" / "validate.py"))
        for alias, expected in (
            ("speech-emotion-recognition", "ser"),
            ("speaker-gender", "gr"),
            ("spoken-language-understanding", "slu"),
        ):
            normalize = template["normalized_task_type"]
            normalize.__globals__["TASK_TYPE"] = alias
            self.assertEqual(normalize(), expected)

    def test_all_supported_tse_aliases_are_canonical(self) -> None:
        aliases = (
            "tse",
            "target_speaker",
            "target-speaker",
            "target speaker",
            "target_speaker_extraction",
            "target-speaker-extraction",
            "target speaker extraction",
            "target_speaker_extraction_model",
            "target-speaker-extraction-model",
            "target speaker extraction model",
            "target_voice_separation",
            "target-voice-separation",
            "target voice separation",
        )
        template = runpy.run_path(
            str(SCRIPTS_DIR / "templates" / "validate.py"),
            run_name="sure_trans_validate_template_test",
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(prepare_fixture.canonical_tse_task(alias), "tse")
                self.assertEqual(scaffold_adapter.tool_contract(alias)[0], "extract_target_speaker")
                template["normalized_task_type"].__globals__["TASK_TYPE"] = alias
                self.assertEqual(template["normalized_task_type"](), "tse")

    def make_fixture(self, root: Path) -> Path:
        fixture = root / "source"
        fixture.mkdir()
        write_wav(fixture / "mixture.wav", [1, 2, 3, 4])
        write_wav(fixture / "enrollment.wav", [5, 6, 7, 8])
        write_wav(fixture / "reference.wav", [9, 10, 11, 12])
        (fixture / "gt.jsonl").write_text(
            json.dumps(
                {
                    "sample_id": "utt-1",
                    "mixture_audio": "mixture.wav",
                    "enrollment_audio": "enrollment.wav",
                    "reference_audio": "reference.wav",
                    "reference_text": "hello",
                    "language": "en",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return fixture

    def prepare(self, root: Path) -> tuple[Path, dict]:
        fixture = self.make_fixture(root)
        run_dir = root / "run"
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "artifacts" / "trans_input_resolved.json").write_text(
            json.dumps({"model_name": "example__tse", "task_type": "tse", "fixture_path": str(fixture)}),
            encoding="utf-8",
        )
        payload = prepare_fixture.prepare_tse_fixture(
            {"model_name": "example__tse"}, fixture, run_dir
        )
        (run_dir / "artifacts" / "fixture_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return run_dir, payload

    def test_contract_aliases_and_fixture_roles(self) -> None:
        self.assertEqual(scaffold_adapter.tool_contract("target-speaker-extraction")[0], "extract_target_speaker")
        tool, schema = scaffold_adapter.tool_contract("tse")
        self.assertEqual(tool, "extract_target_speaker")
        self.assertEqual(
            schema["required"], ["mixture_audio_path", "enrollment_audio_path", "output_path"]
        )
        self.assertTrue(schema["additionalProperties"] is False)
        contract = scaffold_adapter.io_contract_for("tse")
        self.assertEqual(contract["primary_field"], "prediction_audio")
        self.assertEqual(contract["output"], {"prediction_audio": "string", "sample_id": "optional string"})

        with tempfile.TemporaryDirectory() as temporary:
            run_dir, payload = self.prepare(Path(temporary))
            check_artifact.validate_fixture_manifest(payload)
            sample = payload["samples"][0]
            self.assertEqual(sample["sample_id"], "utt-1")
            self.assertEqual(sample["mixture_audio"], "mixture.wav")
            self.assertEqual(sample["enrollment_audio"], "enrollment.wav")
            self.assertEqual(sample["reference_audio"], "reference.wav")
            self.assertTrue(Path(sample["reference_audio_path"]).is_file())
            self.assertEqual(
                json.loads((run_dir / "fixture" / "tse" / "gt.jsonl").read_text())["audio"],
                "mixture.wav",
            )

    def test_prepare_rejects_missing_enrollment_and_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            row = {
                "sample_id": "bad",
                "mixture_audio": "../mixture.wav",
                "enrollment_audio": "enrollment.wav",
                "reference_audio": "reference.wav",
            }
            (fixture / "gt.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative and contained"):
                prepare_fixture.prepare_tse_fixture({"model_name": "example__tse"}, fixture, root / "run")
            row["mixture_audio"] = "mixture.wav"
            row.pop("enrollment_audio")
            (fixture / "gt.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "enrollment_audio"):
                prepare_fixture.prepare_tse_fixture({"model_name": "example__tse"}, fixture, root / "run")

    def test_mcp_smoke_uses_only_mixture_enrollment_and_preserves_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, payload = self.prepare(root)
            server = root / "server.py"
            server.write_text(
                "import json, shutil, sys\n"
                "for line in sys.stdin:\n"
                " req=json.loads(line); rid=req.get('id'); method=req.get('method')\n"
                " if method=='initialize': result={'protocolVersion':'2024-11-05'}\n"
                " elif method=='tools/list': result={'tools':[{'name':'extract_target_speaker'}]}\n"
                " elif method=='tools/call':\n"
                "  args=req['params']['arguments']; assert set(args)=={'mixture_audio_path','enrollment_audio_path','output_path'}\n"
                "  shutil.copyfile(args['mixture_audio_path'], args['output_path']); result={'content':[{'type':'text','text':json.dumps({'prediction_audio':args['output_path']})}]}\n"
                " elif method=='shutdown': result={}; print(json.dumps({'jsonrpc':'2.0','id':rid,'result':result}), flush=True); break\n"
                " else: result={}\n"
                " print(json.dumps({'jsonrpc':'2.0','id':rid,'result':result}), flush=True)\n",
                encoding="utf-8",
            )
            evidence = root / "mcp.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "mcp_smoke.py"),
                    "--fixture-gt-jsonl",
                    str(run_dir / "fixture" / "tse" / "gt.jsonl"),
                    "--tool",
                    "extract_target_speaker",
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
            data = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "passed")
            row = data["tools_call"]["samples"][0]
            self.assertEqual(row["reference_audio"], "reference.wav")
            self.assertEqual(row["enrollment_audio"], "enrollment.wav")
            self.assertEqual(row["result"]["prediction_audio"].split("/")[0], "outputs")
            self.assertNotIn(str(root), json.dumps(data))
            self.assertIsNone(run_trans_validate.validate_mcp_evidence(evidence, "extract_target_speaker"))

    def test_tse_output_validator_rejects_reference_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = root / "outputs"
            outputs.mkdir()
            mixture = root / "mixture.wav"
            enrollment = root / "enrollment.wav"
            reference = root / "reference.wav"
            expected = outputs / "assigned.wav"
            for path, values in ((mixture, [1]), (enrollment, [2]), (reference, [3]), (expected, [4])):
                write_wav(path, values)
            violations, _ = mcp_smoke.validate_tse_output(
                {"prediction_audio": str(expected), "reference_audio": str(reference)},
                key="utt-1",
                outputs_root=outputs,
                expected_path=expected,
                forbidden_inputs=(mixture, enrollment, reference),
            )
            self.assertTrue(violations)
            violations, _ = mcp_smoke.validate_tse_output(
                {"prediction_audio": str(reference)},
                key="utt-1",
                outputs_root=outputs,
                expected_path=expected,
                forbidden_inputs=(mixture, enrollment, reference),
            )
            self.assertTrue(violations)

    def test_equivalence_compares_prediction_audio_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            baseline_dir = run_dir / "original_output"
            adapter_dir = run_dir / "artifacts" / "adapter_validation" / "outputs"
            baseline_dir.mkdir(parents=True)
            adapter_dir.mkdir(parents=True)
            baseline_audio = baseline_dir / "a.wav"
            adapter_audio = adapter_dir / "b.wav"
            write_wav(baseline_audio, [1, 2, 3])
            write_wav(adapter_audio, [1, 2, 3])
            baseline = run_dir / "baseline.json"
            adapter = run_dir / "adapter.json"
            baseline.write_text(json.dumps({"rows": [{"sample_id": "utt-1", "result": {"prediction_audio": str(baseline_audio)}}]}), encoding="utf-8")
            adapter.write_text(json.dumps({"rows": [{"sample_id": "utt-1", "result": {"prediction_audio": str(adapter_audio)}}]}), encoding="utf-8")
            evidence, error = run_trans_validate.compare_tse_equivalence(run_dir, baseline, adapter, "exact")
            self.assertIsNone(error)
            self.assertTrue(evidence["match"])
            self.assertEqual(evidence["primary_field"], "prediction_audio")


if __name__ == "__main__":
    unittest.main()
