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
import wave
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[3]
SE_FIXTURE = REPO_ROOT / "fixtures" / "tasks" / "se" / "fleurs_noise_smoke"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_artifact  # noqa: E402
import finalize_trans_bundle  # noqa: E402
import materialize_trans_inputs  # noqa: E402
import mcp_smoke  # noqa: E402
import prepare_fixture  # noqa: E402
import run_trans_validate  # noqa: E402
import scaffold_adapter  # noqa: E402


SE_CONTRACT = scaffold_adapter.io_contract_for("se")


def write_resolved(run_dir: Path, fixture: Path) -> None:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "trans_input_resolved.json").write_text(
        json.dumps(
            {
                "model_name": "example__se",
                "task_type": "se",
                "build_context": str(fixture.parent),
                "fixture_path": str(fixture),
            }
        ),
        encoding="utf-8",
    )


def prepare_shared_fixture(run_dir: Path) -> dict:
    write_resolved(run_dir, SE_FIXTURE)
    with mock.patch.object(sys, "argv", ["prepare_fixture.py", "--run-dir", str(run_dir)]):
        assert prepare_fixture.main() == 0
    return json.loads((run_dir / "artifacts" / "fixture_manifest.json").read_text(encoding="utf-8"))


def render_validate(model_dir: Path) -> Path:
    template = (SCRIPTS_DIR / "templates" / "validate.py").read_text(encoding="utf-8")
    rendered = (
        template.replace("__MODEL_NAME__", "example__se")
        .replace("__TASK_TYPE__", "SE")
        .replace(
            "__IO_CONTRACT_JSON__",
            json.dumps(SE_CONTRACT, ensure_ascii=True, separators=(",", ":")),
        )
    )
    validate_path = model_dir / "validate.py"
    validate_path.write_text(rendered, encoding="utf-8")
    (model_dir / "model.py").write_text(
        "import shutil\n"
        "class ModelWrapper:\n"
        "    def load(self): self.model = object()\n"
        "    def predict(self, payload):\n"
        "        shutil.copyfile(payload['audio_path'], payload['output_path'])\n"
        "        return {'audio_path': payload['output_path']}\n",
        encoding="utf-8",
    )
    return validate_path


def write_pcm(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))


class SETransTests(unittest.TestCase):
    def test_task_aliases_canonicalize_only_at_input_and_adapter_is_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "infer.py"
            entrypoint.write_text("def enhance_speech(audio_path): return audio_path\n", encoding="utf-8")
            for alias in (
                "se",
                "speech-enhancement",
                "speech_enhancement",
                "speech enhancement",
                "acoustic-noise-suppression",
                "acoustic_noise_suppression",
                "acoustic noise suppression",
            ):
                self.assertEqual(
                    materialize_trans_inputs.resolve_task_type(alias, entrypoint, Path("enhancer")),
                    "se",
                )
            self.assertEqual(
                materialize_trans_inputs.resolve_task_type(None, entrypoint, Path("enhancer")),
                "se",
            )
        tool_name, input_schema = scaffold_adapter.tool_contract("se")
        self.assertEqual(tool_name, "enhance_speech")
        self.assertEqual(input_schema["required"], ["audio_path"])
        self.assertIn("output_path", input_schema["properties"])
        self.assertEqual(SE_CONTRACT["input_type"], "audio_path")
        self.assertEqual(SE_CONTRACT["output_type"], "audio")
        self.assertEqual(SE_CONTRACT["primary_field"], "audio_path")

        input_schema_document = json.loads(
            (SCRIPTS_DIR.parent / "schemas" / "trans_input_resolved.schema.json").read_text(
                encoding="utf-8"
            )
        )
        fixture_schema_document = json.loads(
            (SCRIPTS_DIR.parent / "schemas" / "fixture_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("se", input_schema_document["properties"]["task_type"]["enum"])
        self.assertIn("se", fixture_schema_document["properties"]["task_type"]["enum"])

    def test_prepare_gate_and_finalizer_preserve_noisy_clean_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            manifest = prepare_shared_fixture(run_dir)
            check_artifact.validate_fixture_manifest(manifest)
            self.assertEqual(manifest["sample_count"], 1)
            sample = manifest["samples"][0]
            self.assertEqual(sample["noisy_audio"], "noisy.wav")
            self.assertEqual(sample["reference_audio"], "clean.wav")
            self.assertTrue((run_dir / "fixture" / "se" / "noisy.wav").is_file())
            self.assertTrue((run_dir / "fixture" / "se" / "clean.wav").is_file())

            model_dir = root / "models" / "example__se"
            model_dir.mkdir(parents=True)
            finalize_trans_bundle.stage_fixture(
                run_dir,
                model_dir,
                {"task_type": "se", "model_name": "example__se"},
            )
            finalized = json.loads(
                (run_dir / "artifacts" / "fixture_manifest.json").read_text(encoding="utf-8")
            )
            check_artifact.validate_fixture_manifest(finalized)
            self.assertTrue((model_dir / "fixture" / "se" / "noisy.wav").is_file())
            self.assertTrue((model_dir / "fixture" / "se" / "clean.wav").is_file())

    def test_prepare_rejects_escape_missing_clean_and_unbounded_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            fixture.mkdir()
            shutil.copy2(SE_FIXTURE / "noisy.wav", fixture / "noisy.wav")
            shutil.copy2(SE_FIXTURE / "clean.wav", fixture / "clean.wav")
            escaped = {
                "key": "escaped",
                "audio": "../outside.wav",
                "reference_audio": "clean.wav",
            }
            (fixture / "gt.jsonl").write_text(json.dumps(escaped) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stay inside"):
                prepare_fixture.prepare_se_fixture(
                    {"model_name": "example__se"}, fixture, root / "run"
                )

            missing = {"key": "missing", "audio": "noisy.wav", "reference_audio": "absent.wav"}
            (fixture / "gt.jsonl").write_text(json.dumps(missing) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                prepare_fixture.prepare_se_fixture(
                    {"model_name": "example__se"}, fixture, root / "run"
                )

            hardlinked_clean = fixture / "hardlinked-clean.wav"
            os.link(fixture / "noisy.wav", hardlinked_clean)
            hardlinked = {
                "key": "hardlinked",
                "audio": "noisy.wav",
                "reference_audio": hardlinked_clean.name,
            }
            (fixture / "gt.jsonl").write_text(
                json.dumps(hardlinked) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "independent files"):
                prepare_fixture.prepare_se_fixture(
                    {"model_name": "example__se"}, fixture, root / "run"
                )

            rows = [
                {"key": f"row-{index}", "audio": "noisy.wav", "reference_audio": "clean.wav"}
                for index in range(6)
            ]
            (fixture / "gt.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "1 to 5"):
                prepare_fixture.prepare_se_fixture(
                    {"model_name": "example__se"}, fixture, root / "run"
                )

    def test_generated_validate_writes_every_output_below_validation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            prepare_shared_fixture(run_dir)
            model_dir = root / "model"
            shutil.copytree(run_dir / "fixture" / "se", model_dir / "fixture" / "se")
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
            self.assertEqual(len(sample["rows"]), 1)
            generated = Path(sample["rows"][0]["result"]["audio_path"])
            self.assertTrue(generated.is_file())
            self.assertTrue(generated.resolve().is_relative_to((artifacts / "outputs").resolve()))

            outside = root / "outside.wav"
            shutil.copy2(SE_FIXTURE / "noisy.wav", outside)
            spec = importlib.util.spec_from_file_location(
                "sure_trans_se_validate_template", SCRIPTS_DIR / "templates" / "validate.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.ARTIFACTS_DIR = artifacts
            with self.assertRaisesRegex(ValueError, "validation outputs"):
                module.resolve_se_generated_audio(str(outside), key="outside")
            real_dir = artifacts / "outputs" / "real"
            real_dir.mkdir()
            symlink_dir = artifacts / "outputs" / "linked"
            symlink_dir.symlink_to(real_dir, target_is_directory=True)
            linked_output = symlink_dir / "enhanced.wav"
            shutil.copy2(SE_FIXTURE / "noisy.wav", linked_output)
            with self.assertRaisesRegex(ValueError, "symlink"):
                module.resolve_se_generated_audio(str(linked_output), key="linked")

            expected = module.se_output_path("assigned", 1)
            alternate = artifacts / "outputs" / "alternate.wav"
            shutil.copy2(SE_FIXTURE / "noisy.wav", alternate)
            with self.assertRaisesRegex(ValueError, "harness-assigned output_path"):
                module.resolve_se_generated_audio(
                    str(alternate),
                    key="assigned",
                    expected_path=expected,
                )

            expected.write_bytes(b"not-a-wave")
            with self.assertRaisesRegex(ValueError, "readable PCM WAV"):
                module.resolve_se_generated_audio(
                    str(expected),
                    key="assigned",
                    expected_path=expected,
                )

            expected.unlink()
            local_noisy = root / "local-noisy.wav"
            shutil.copy2(SE_FIXTURE / "noisy.wav", local_noisy)
            os.link(local_noisy, expected)
            with self.assertRaisesRegex(ValueError, "must not alias"):
                module.resolve_se_generated_audio(
                    str(expected),
                    key="assigned",
                    expected_path=expected,
                    forbidden_inputs=(local_noisy,),
                )

    def test_mcp_smoke_proves_generated_output_and_clean_reference_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            shutil.copytree(SE_FIXTURE, fixture)
            server = root / "server.py"
            server.write_text(
                "import json, shutil, sys\n"
                "for line in sys.stdin:\n"
                " req=json.loads(line); method=req.get('method'); rid=req.get('id')\n"
                " if method=='initialize': result={'protocolVersion':'2024-11-05'}\n"
                " elif method=='tools/list': result={'tools':[{'name':'enhance_speech'}]}\n"
                " elif method=='tools/call':\n"
                "  args=req['params']['arguments']; shutil.copyfile(args['audio_path'], args['output_path'])\n"
                "  result={'content':[{'type':'text','text':json.dumps({'audio_path':args['output_path']})}]}\n"
                " else: result={}\n"
                " print(json.dumps({'jsonrpc':'2.0','id':rid,'result':result}), flush=True)\n"
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
                    "enhance_speech",
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
            self.assertEqual(payload["tools_call"]["num_samples"], 1)
            row = payload["tools_call"]["samples"][0]
            self.assertEqual(row["reference_audio"], "clean.wav")
            self.assertTrue(row["reference_audio_sha256"])
            self.assertTrue(row["result"]["audio_path"].startswith("outputs/"))
            self.assertTrue(row["result"]["audio_sha256"])
            self.assertNotIn(str(root), json.dumps(payload))
            self.assertIsNone(run_trans_validate.validate_mcp_evidence(evidence, "enhance_speech"))

    def test_mcp_smoke_binds_pcm_output_and_rejects_input_inode_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = root / "outputs"
            expected = outputs / "assigned.wav"
            noisy = root / "noisy.wav"
            clean = root / "clean.wav"
            alternate = outputs / "alternate.wav"
            for path, values in (
                (expected, [0, 1, -1]),
                (noisy, [2, 3, -3]),
                (clean, [4, 5, -5]),
                (alternate, [6, 7, -7]),
            ):
                write_pcm(path, values)

            violations, resolved = mcp_smoke.validate_se_output(
                {"audio_path": str(expected)},
                key="sample",
                outputs_root=outputs,
                expected_path=expected,
                forbidden_inputs=(noisy, clean),
            )
            self.assertEqual(violations, [])
            self.assertEqual(resolved, expected.resolve())

            violations, _ = mcp_smoke.validate_se_output(
                {"audio_path": str(alternate)},
                key="sample",
                outputs_root=outputs,
                expected_path=expected,
                forbidden_inputs=(noisy, clean),
            )
            self.assertIn("harness-assigned output_path", violations[0])

            expected.write_bytes(b"not-a-wave")
            violations, _ = mcp_smoke.validate_se_output(
                {"audio_path": str(expected)},
                key="sample",
                outputs_root=outputs,
                expected_path=expected,
                forbidden_inputs=(noisy, clean),
            )
            self.assertIn("readable PCM WAV", violations[0])

            expected.unlink()
            os.link(noisy, expected)
            violations, _ = mcp_smoke.validate_se_output(
                {"audio_path": str(expected)},
                key="sample",
                outputs_root=outputs,
                expected_path=expected,
                forbidden_inputs=(noisy, clean),
            )
            self.assertIn("must not alias", violations[0])

    def test_equivalence_compares_pcm_content_not_path_with_one_lsb_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "trans_input_resolved.json").write_text(
                json.dumps({"task_type": "se"}), encoding="utf-8"
            )
            baseline_dir = run_dir / "original_output"
            adapter_dir = artifacts / "adapter_validation" / "outputs"
            baseline_dir.mkdir()
            adapter_dir.mkdir(parents=True)
            baseline_audio = baseline_dir / "baseline.wav"
            adapter_audio = adapter_dir / "different-name.wav"
            write_pcm(baseline_audio, [0, 100, -100, 300])
            write_pcm(adapter_audio, [1, 99, -99, 301])
            baseline = run_dir / "baseline.json"
            adapter = run_dir / "adapter.json"
            baseline.write_text(
                json.dumps({"rows": [{"key": "sample", "result": {"audio_path": str(baseline_audio)}}]}),
                encoding="utf-8",
            )
            adapter.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "key": "sample",
                                "result": {"audio_path": "/validation/outputs/different-name.wav"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNone(error)
            self.assertTrue(evidence["match"])
            self.assertFalse(evidence["path_strings_compared"])
            self.assertEqual(evidence["rows"]["sample"]["pcm_integer_lsb_tolerance"], 1)
            self.assertNotIn(str(run_dir), json.dumps(evidence))

            write_pcm(adapter_audio, [2, 100, -100, 300])
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNotNone(error)
            self.assertEqual(evidence["mismatched_keys"], ["sample"])

            baseline_bytes = baseline_dir / "baseline.audio"
            adapter_bytes = adapter_dir / "adapter.audio"
            baseline_bytes.write_bytes(b"deterministic-enhanced-audio")
            adapter_bytes.write_bytes(b"deterministic-enhanced-audio")
            baseline.write_text(
                json.dumps({"rows": [{"key": "sample", "result": {"audio_path": str(baseline_bytes)}}]}),
                encoding="utf-8",
            )
            adapter.write_text(
                json.dumps({"rows": [{"key": "sample", "result": {"audio_path": str(adapter_bytes)}}]}),
                encoding="utf-8",
            )
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNone(error)
            self.assertEqual(
                evidence["rows"]["sample"]["method"],
                "exact_content_sha256_fallback",
            )

    def test_equivalence_rejects_external_same_file_and_parent_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "trans_input_resolved.json").write_text(
                json.dumps({"task_type": "se"}), encoding="utf-8"
            )
            baseline_document = run_dir / "baseline.json"
            adapter_document = run_dir / "adapter.json"
            outside = root / "outside.wav"
            write_pcm(outside, [0, 1, -1])
            document = {"rows": [{"key": "sample", "result": {"audio_path": str(outside)}}]}
            baseline_document.write_text(json.dumps(document), encoding="utf-8")
            adapter_document.write_text(json.dumps(document), encoding="utf-8")
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {
                    "baseline_output": str(baseline_document),
                    "adapter_output": str(adapter_document),
                },
            )
            self.assertIsNone(evidence)
            self.assertIn("baseline audio_path", error)

            outside_dir = root / "outside-dir"
            outside_dir.mkdir()
            escaped_audio = outside_dir / "escaped.wav"
            write_pcm(escaped_audio, [0, 1, -1])
            baseline_root = run_dir / "original_output"
            baseline_root.mkdir()
            (baseline_root / "linked").symlink_to(outside_dir, target_is_directory=True)
            adapter_root = artifacts / "adapter_validation" / "outputs"
            adapter_root.mkdir(parents=True)
            adapter_audio = adapter_root / "adapter.wav"
            write_pcm(adapter_audio, [0, 1, -1])
            baseline_document.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "key": "sample",
                                "result": {
                                    "audio_path": str(baseline_root / "linked" / "escaped.wav")
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            adapter_document.write_text(
                json.dumps(
                    {"rows": [{"key": "sample", "result": {"audio_path": str(adapter_audio)}}]}
                ),
                encoding="utf-8",
            )
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {
                    "baseline_output": str(baseline_document),
                    "adapter_output": str(adapter_document),
                },
            )
            self.assertIsNone(evidence)
            self.assertIn("parent symlink", error)

            independent_baseline = baseline_root / "baseline.wav"
            write_pcm(independent_baseline, [4, 5, 6])
            linked_adapter = adapter_root / "hardlink.wav"
            os.link(independent_baseline, linked_adapter)
            baseline_document.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"key": "sample", "result": {"audio_path": str(independent_baseline)}}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            adapter_document.write_text(
                json.dumps(
                    {"rows": [{"key": "sample", "result": {"audio_path": str(linked_adapter)}}]}
                ),
                encoding="utf-8",
            )
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {
                    "baseline_output": str(baseline_document),
                    "adapter_output": str(adapter_document),
                },
            )
            self.assertIsNotNone(error)
            self.assertEqual(
                evidence["rows"]["sample"]["method"],
                "rejected_same_recorded_file",
            )

    def test_finalizer_promotes_keyed_enhanced_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            manifest = prepare_shared_fixture(run_dir)
            artifacts = run_dir / "artifacts"
            (artifacts / "adapter_manifest.json").write_text(
                json.dumps({"io_contract": SE_CONTRACT}), encoding="utf-8"
            )
            validation_outputs = artifacts / "adapter_validation" / "outputs"
            validation_outputs.mkdir(parents=True)
            enhanced = finalize_trans_bundle.expected_se_validation_output(
                artifacts,
                manifest["samples"][0]["key"],
                1,
            )
            shutil.copy2(SE_FIXTURE / "noisy.wav", enhanced)
            (artifacts / "adapter_validation" / "sample_output.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "key": manifest["samples"][0]["key"],
                                "result": {"audio_path": str(enhanced)},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            finalize_trans_bundle.promote_sample_output(run_dir)
            promoted = json.loads((artifacts / "sample_output.json").read_text(encoding="utf-8"))
            relative = promoted["rows"][0]["result"]["audio_path"]
            self.assertTrue(relative.startswith("artifacts/outputs/"))
            self.assertTrue((run_dir / relative).is_file())

    def test_finalizer_rejects_unassigned_non_pcm_and_input_alias_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            manifest = prepare_shared_fixture(run_dir)
            artifacts = run_dir / "artifacts"
            (artifacts / "adapter_manifest.json").write_text(
                json.dumps({"io_contract": SE_CONTRACT}), encoding="utf-8"
            )
            key = manifest["samples"][0]["key"]
            expected = finalize_trans_bundle.expected_se_validation_output(
                artifacts,
                key,
                1,
            )
            expected.parent.mkdir(parents=True, exist_ok=True)
            alternate = expected.parent / "alternate.wav"
            shutil.copy2(SE_FIXTURE / "noisy.wav", alternate)
            sample_output = artifacts / "adapter_validation" / "sample_output.json"

            def write_sample(audio_path: Path) -> None:
                sample_output.parent.mkdir(parents=True, exist_ok=True)
                sample_output.write_text(
                    json.dumps(
                        {
                            "rows": [
                                {"key": key, "result": {"audio_path": str(audio_path)}}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            write_sample(alternate)
            with self.assertRaisesRegex(ValueError, "harness-assigned output_path"):
                finalize_trans_bundle.promote_sample_output(run_dir)

            expected.write_bytes(b"not-a-wave")
            write_sample(expected)
            with self.assertRaisesRegex(ValueError, "readable PCM WAV"):
                finalize_trans_bundle.promote_sample_output(run_dir)

            expected.unlink()
            os.link(Path(manifest["samples"][0]["audio_path"]), expected)
            with self.assertRaisesRegex(ValueError, "must not alias"):
                finalize_trans_bundle.promote_sample_output(run_dir)


if __name__ == "__main__":
    unittest.main()
