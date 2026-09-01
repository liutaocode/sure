#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import finalize_model_bundle
import materialize_onboard_inputs
import stage_model_artifacts


SCRIPTS_DIR = Path(__file__).resolve().parent
SE_FIXTURE = SCRIPTS_DIR.parents[3] / "fixtures" / "tasks" / "se" / "fleurs_noise_smoke"


class SEOnboardingTest(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 160)

    def test_model_input_aliases_resolve_to_se_playbook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_input_path = root / "model_input.yaml"
            model_input_path.write_text("task_type: speech-enhancement\n", encoding="utf-8")
            resolved = materialize_onboard_inputs.make_model_input_resolved(
                {
                    "model_id": "example/enhancer",
                    "task_type": "speech-enhancement",
                    "deployment_type": "local",
                    "repo": {"url": "https://example.invalid/enhancer"},
                    "environment_hint": {"preferred_backend": "uv"},
                },
                model_input_path=model_input_path,
                repo_root=root,
                package_profile="none",
                weights_link_policy="auto",
                device="cpu",
                force_repair=False,
                skip_download=True,
                max_retries=3,
                cpu_fallback_after_cuda_failures=3,
                cuda_repair_attempts_before_cpu=3,
                raw_args="",
                existing_model_dir=None,
                image_version=None,
            )
            context = materialize_onboard_inputs.make_context_selection(
                resolved,
                resolved["normalized_model_input"],
            )

        self.assertEqual(resolved["task_type"], "se")
        self.assertEqual(resolved["normalized_model_input"]["task_type"], "se")
        self.assertEqual(
            context["selected_references"]["task_playbooks"],
            ["references/task_playbooks/SE.md"],
        )

    def test_prepare_and_check_preserve_noisy_and_clean_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "se-test"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            model_dir = root / "sure" / "models" / "example__enhancer"
            (artifacts / "model_input_resolved.json").write_text(
                json.dumps(
                    {
                        "model_id": "example/enhancer",
                        "model_name": "example__enhancer",
                        "model_dir": str(model_dir),
                        "task_type": "speech-enhancement",
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = artifacts / "fixture_manifest.json"
            prepared = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_fixture.py"),
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                    "--source-dir",
                    str(SE_FIXTURE),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            checked = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "check_fixture.py"),
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["task_type"], "se")
        self.assertEqual(manifest["samples"][0]["audio"], "noisy.wav")
        self.assertEqual(manifest["samples"][0]["reference_audio"], "clean.wav")
        self.assertTrue(manifest["samples"][0]["audio_path"].endswith("noisy.wav"))
        self.assertTrue(
            manifest["samples"][0]["reference_audio_path"].endswith("clean.wav")
        )

    def test_stage_and_final_manifest_keep_portable_generated_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".sure" / "runs" / "se-stage"
            run_artifacts = run_dir / "artifacts"
            run_artifacts.mkdir(parents=True)
            model_dir = root / "sure" / "models" / "example__enhancer"
            model_dir.mkdir(parents=True)
            for name in stage_model_artifacts.CORE_FILES:
                (model_dir / name).write_text("# test\n", encoding="utf-8")
            resolved = {
                "model_id": "example/enhancer",
                "model_name": "example__enhancer",
                "model_dir": str(model_dir),
                "task_type": "se",
                "deployment_type": "api",
                "package_profile": "none",
            }
            (run_artifacts / "model_input_resolved.json").write_text(
                json.dumps(resolved), encoding="utf-8"
            )
            generated = run_artifacts / "outputs" / "01-enhanced.wav"
            self._write_wav(generated)
            portable_path = "artifacts/outputs/01-enhanced.wav"
            (run_artifacts / "sample_output.json").write_text(
                json.dumps({"audio_path": portable_path}), encoding="utf-8"
            )
            (run_artifacts / "sample_outputs.jsonl").write_text(
                json.dumps(
                    {
                        "key": "sample",
                        "audio": "noisy.wav",
                        "reference_audio": "clean.wav",
                        "output": {"audio_path": portable_path},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = run_artifacts / "artifact_manifest.json"
            status = stage_model_artifacts.main_with_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--produces",
                    str(manifest_path),
                    "--allow-missing-run-artifacts",
                ]
            )
            self.assertEqual(status, 0)
            staged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            finalized_manifest = finalize_model_bundle.update_manifest(model_dir, resolved)

            self.assertTrue((model_dir / portable_path).is_file())
            staged_paths = {
                entry["path"] for entry in staged_manifest["artifacts"]["required"].values()
            }
            finalized_paths = {
                entry["path"] for entry in finalized_manifest["artifacts"]["required"].values()
            }
            self.assertIn(portable_path, staged_paths)
            self.assertIn(portable_path, finalized_paths)
            sample = json.loads(
                (model_dir / "artifacts" / "sample_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sample["audio_path"], portable_path)


if __name__ == "__main__":
    unittest.main()
