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
sys.path.insert(0, str(SCRIPTS_DIR))

import check_artifact  # noqa: E402
import finalize_trans_bundle  # noqa: E402
import materialize_trans_inputs  # noqa: E402
import mcp_smoke  # noqa: E402
import prepare_fixture  # noqa: E402
import run_trans_validate  # noqa: E402
import scaffold_adapter  # noqa: E402


def write_wav(path: Path, *, silence: bool, seconds: float = 1.0) -> None:
    frame_count = int(16000 * seconds)
    sample = 0 if silence else 1000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(sample.to_bytes(2, "little", signed=True) * frame_count)


def valid_result(*, silence: bool) -> dict:
    return {
        "speech_segments": [] if silence else [{"start": 0.1, "end": 0.6}],
        "frame_scores": [
            {"start": 0.0, "end": 0.5, "score": 0.9 if not silence else 0.01},
            {"start": 0.5, "end": 1.0, "score": 0.1 if not silence else 0.02},
        ],
    }


def make_fixture(root: Path) -> Path:
    fixture = root / "fixture"
    fixture.mkdir(parents=True)
    write_wav(fixture / "speech.wav", silence=False)
    write_wav(fixture / "silence.wav", silence=True)
    (fixture / "unused.txt").write_text("must not be staged", encoding="utf-8")
    rows = [
        {
            "key": "speech",
            "audio": "speech.wav",
            "speech_segments": [{"start": 0.1, "end": 0.6}],
        },
        {"key": "silence", "audio": "silence.wav", "speech_segments": []},
    ]
    (fixture / "gt.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return fixture


def prepare_vad_run(root: Path) -> tuple[Path, dict]:
    run_dir = root / "run"
    fixture = make_fixture(root)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "trans_input_resolved.json").write_text(
        json.dumps(
            {
                "model_name": "example__vad",
                "task_type": "vad",
                "build_context": str(root),
                "fixture_path": str(fixture),
            }
        ),
        encoding="utf-8",
    )
    with mock.patch.object(sys, "argv", ["prepare_fixture.py", "--run-dir", str(run_dir)]):
        assert prepare_fixture.main() == 0
    manifest = json.loads(
        (artifacts / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    return run_dir, manifest


def render_validate(model_dir: Path) -> Path:
    contract = scaffold_adapter.io_contract_for("vad")
    template = (SCRIPTS_DIR / "templates" / "validate.py").read_text(encoding="utf-8")
    rendered = (
        template.replace("__MODEL_NAME__", "example__vad")
        .replace("__TASK_TYPE__", "VAD")
        .replace(
            "__IO_CONTRACT_JSON__",
            json.dumps(contract, ensure_ascii=True, separators=(",", ":")),
        )
    )
    validate_path = model_dir / "validate.py"
    validate_path.write_text(rendered, encoding="utf-8")
    (model_dir / "model.py").write_text(
        "from pathlib import Path\n"
        "class ModelWrapper:\n"
        "    def load(self): self.model = object()\n"
        "    def predict(self, payload):\n"
        "        if set(payload) != {'audio_path'}:\n"
        "            raise ValueError(f'reference leakage: {sorted(payload)}')\n"
        "        silence = Path(payload['audio_path']).stem == 'silence'\n"
        "        return {\n"
        "            'speech_segments': [] if silence else [{'start': 0.1, 'end': 0.6}],\n"
        "            'frame_scores': [\n"
        "                {'start': 0.0, 'end': 0.5, 'score': 0.01 if silence else 0.9},\n"
        "                {'start': 0.5, 'end': 1.0, 'score': 0.02 if silence else 0.1},\n"
        "            ],\n"
        "        }\n",
        encoding="utf-8",
    )
    return validate_path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VadTransTests(unittest.TestCase):
    def test_task_alias_tool_contract_and_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "infer.py"
            entrypoint.write_text("def infer(audio_path): return audio_path\n", encoding="utf-8")
            for alias in (
                "vad",
                "voice activity detection",
                "voice-activity-detection",
                "voice_activity_detection",
                "speech activity detection",
            ):
                self.assertEqual(
                    materialize_trans_inputs.resolve_task_type(
                        alias, entrypoint, Path("activity-model")
                    ),
                    "vad",
                )
            entrypoint.write_text(
                "def detect_speech(audio_path): return []\n", encoding="utf-8"
            )
            self.assertEqual(
                materialize_trans_inputs.resolve_task_type(
                    None, entrypoint, Path("activity-model")
                ),
                "vad",
            )

        tool, input_schema = scaffold_adapter.tool_contract("vad")
        contract = scaffold_adapter.io_contract_for("vad")
        self.assertEqual(tool, "detect_speech")
        self.assertEqual(input_schema["properties"], {"audio_path": {"type": "string", "minLength": 1}})
        self.assertEqual(input_schema["required"], ["audio_path"])
        self.assertFalse(input_schema["additionalProperties"])
        self.assertEqual(contract["primary_field"], "speech_segments")
        self.assertEqual(contract["required_fields"], ["speech_segments"])
        self.assertEqual(contract["approved_output_fields"], ["frame_scores", "speech_segments"])
        self.assertTrue(contract["allow_empty_primary"])

        for schema_name in ("trans_input_resolved", "fixture_manifest"):
            schema = json.loads(
                (SCRIPTS_DIR.parent / "schemas" / f"{schema_name}.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("vad", schema["properties"]["task_type"]["enum"])
        output_schema = json.loads(
            (SCRIPTS_DIR.parent / "schemas" / "vad_output.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(output_schema["required"], ["speech_segments"])
        self.assertFalse(output_schema["additionalProperties"])
        self.assertFalse(
            output_schema["properties"]["speech_segments"]["items"]["additionalProperties"]
        )
        self.assertFalse(
            output_schema["properties"]["frame_scores"]["items"]["additionalProperties"]
        )

    def test_prepare_stages_only_referenced_audio_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = prepare_vad_run(Path(temporary))
            check_artifact.validate_fixture_manifest(manifest)
            self.assertEqual(manifest["task_type"], "vad")
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(
                sorted(path.name for path in (run_dir / "fixture" / "vad").iterdir()),
                ["gt.jsonl", "silence.wav", "speech.wav"],
            )
            for sample in manifest["samples"]:
                self.assertEqual(sample["annotation_fields"], ["speech_segments"])
                self.assertNotIn("speech_segments", sample)
                self.assertNotIn("frame_scores", sample)
                self.assertTrue(Path(sample["audio_path"]).is_file())

    def test_fixture_supports_one_to_five_rows_and_silence_empty_segments(self) -> None:
        for count in range(1, 6):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = root / "fixture"
                fixture.mkdir()
                write_wav(fixture / "silence.wav", silence=True)
                rows = [
                    {"key": f"silence-{index}", "audio": "silence.wav", "speech_segments": []}
                    for index in range(count)
                ]
                (fixture / "gt.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                manifest = prepare_fixture.prepare_vad_fixture(
                    {"model_name": "example__vad"}, fixture, root / "run"
                )
                self.assertEqual(manifest["sample_count"], count)
                check_artifact.validate_fixture_manifest(manifest)

    def test_fixture_rejects_empty_non_silence_duplicate_and_symlink_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            fixture.mkdir()
            write_wav(fixture / "speech.wav", silence=False)
            row = {"key": "speech", "audio": "speech.wav", "speech_segments": []}
            (fixture / "gt.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pure-silence"):
                prepare_fixture.prepare_vad_fixture(
                    {"model_name": "example__vad"}, fixture, root / "run-empty"
                )

            row["speech_segments"] = [{"start": 0.1, "end": 0.6}]
            (fixture / "gt.jsonl").write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                prepare_fixture.prepare_vad_fixture(
                    {"model_name": "example__vad"}, fixture, root / "run-duplicate"
                )

            (fixture / "gt.jsonl").write_text(
                json.dumps(
                    {
                        **row,
                        "frame_scores": [{"start": 0.0, "end": 1.0, "score": 1.0}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prediction frame_scores"):
                prepare_fixture.prepare_vad_fixture(
                    {"model_name": "example__vad"}, fixture, root / "run-frame-scores"
                )

            linked = fixture / "linked.wav"
            linked.symlink_to(fixture / "speech.wav")
            (fixture / "gt.jsonl").write_text(
                json.dumps({**row, "audio": "linked.wav"}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_fixture.prepare_vad_fixture(
                    {"model_name": "example__vad"}, fixture, root / "run-symlink"
                )

    def test_interval_validation_is_closed_finite_ordered_and_bounded(self) -> None:
        valid = [{"start": 0.1, "end": 0.4}, {"start": 0.5, "end": 0.9}]
        self.assertEqual(
            prepare_fixture.validate_vad_intervals(
                valid, label="VAD", empty_allowed=False, duration_sec=1.0
            ),
            valid,
        )
        malformed = (
            (None, "array"),
            ([{"start": True, "end": 0.4}], "start"),
            ([{"start": -0.1, "end": 0.4}], "start"),
            ([{"start": 0.1, "end": float("inf")}], "end"),
            ([{"start": 0.4, "end": 0.4}], "end"),
            ([{"start": 0.1, "end": 0.4, "debug": True}], "unapproved"),
            ([{"start": 0.1}], "missing"),
            ([{"start": 0.5, "end": 0.7}, {"start": 0.4, "end": 0.8}], "ordered"),
            ([{"start": 0.8, "end": 1.1}], "duration"),
        )
        for intervals, message in malformed:
            with self.subTest(intervals=intervals), self.assertRaisesRegex(ValueError, message):
                prepare_fixture.validate_vad_intervals(
                    intervals, label="VAD", empty_allowed=False, duration_sec=1.0
                )

    def test_frame_scores_are_finite_and_cover_the_complete_timebase(self) -> None:
        valid = [
            {"start": 0.0, "end": 0.5, "score": 0.0},
            {"start": 0.5, "end": 1.0, "score": 1.0},
        ]
        self.assertEqual(
            prepare_fixture.validate_vad_intervals(
                valid, label="VAD", frame_scores=True, duration_sec=1.0
            ),
            valid,
        )
        malformed = (
            ([], "complete audio"),
            ([{"start": 0.1, "end": 1.0, "score": 0.5}], "start at 0"),
            (
                [
                    {"start": 0.0, "end": 0.4, "score": 0.5},
                    {"start": 0.5, "end": 1.0, "score": 0.5},
                ],
                "contiguous",
            ),
            ([{"start": 0.0, "end": 0.9, "score": 0.5}], "end at WAV duration"),
            ([{"start": 0.0, "end": 1.0, "score": -0.1}], "within"),
            ([{"start": 0.0, "end": 1.0, "score": float("nan")}], "finite"),
            ([{"start": 0.0, "end": 1.0, "score": True}], "finite"),
        )
        for intervals, message in malformed:
            with self.subTest(intervals=intervals), self.assertRaisesRegex(ValueError, message):
                prepare_fixture.validate_vad_intervals(
                    intervals, label="VAD", frame_scores=True, duration_sec=1.0
                )

    def test_generated_validate_runs_every_row_without_reference_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, _ = prepare_vad_run(root)
            model_dir = root / "model"
            shutil.copytree(run_dir / "fixture" / "vad", model_dir / "fixture" / "vad")
            validate_path = render_validate(model_dir)
            artifacts = root / "validation"
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
            sample = json.loads(
                (artifacts / "sample_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual([row["key"] for row in sample["rows"]], ["speech", "silence"])
            self.assertEqual(sample["rows"][1]["result"]["speech_segments"], [])

    def test_mcp_smoke_runs_all_rows_and_preserves_optional_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = make_fixture(root)
            server = root / "server.py"
            server.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "for line in sys.stdin:\n"
                " request=json.loads(line); method=request.get('method'); rid=request.get('id')\n"
                " if method=='initialize': response={'protocolVersion':'2024-11-05'}\n"
                " elif method=='tools/list': response={'tools':[{'name':'detect_speech'}]}\n"
                " elif method=='tools/call':\n"
                "  arguments=request['params']['arguments']\n"
                "  if set(arguments) != {'audio_path'}: raise ValueError('reference leaked')\n"
                "  silence=Path(arguments['audio_path']).stem == 'silence'\n"
                "  value={'speech_segments': [] if silence else [{'start':0.1,'end':0.6}], "
                "'frame_scores':[{'start':0.0,'end':1.0,'score':0.01 if silence else 0.9}]}\n"
                "  response={'content':[{'type':'text','text':json.dumps(value)}]}\n"
                " else: response={}\n"
                " print(json.dumps({'jsonrpc':'2.0','id':rid,'result':response}), flush=True)\n"
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
                    "detect_speech",
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
            self.assertEqual(payload["tools_call"]["primary_field"], "speech_segments")
            self.assertEqual(payload["tools_call"]["num_samples"], 2)
            self.assertEqual(
                payload["tools_call"]["samples"][0]["result"]["frame_scores"][0]["score"],
                0.9,
            )
            self.assertIsNone(
                run_trans_validate.validate_mcp_evidence(evidence, "detect_speech")
            )

    def test_runtime_validators_reject_malformed_vad_results(self) -> None:
        segment_only = {"speech_segments": [{"start": 0.1, "end": 0.6}]}
        self.assertEqual(
            mcp_smoke.validate_vad_output(
                segment_only, empty_allowed=False, duration_sec=1.0
            ),
            [],
        )
        self.assertEqual(
            run_trans_validate.validate_vad_output(
                segment_only,
                label="VAD",
                empty_allowed=False,
                duration_sec=1.0,
            ),
            segment_only,
        )
        bad_results = (
            ({}, "requires speech_segments"),
            ({"speech_segments": [], "path": "/private"}, "unapproved"),
            (
                {
                    "speech_segments": [{"start": 0.1, "end": 0.6}],
                    "frame_scores": [{"start": 0.0, "end": 0.5, "score": 0.5}],
                },
                "end at WAV duration",
            ),
        )
        for result, message in bad_results:
            with self.subTest(result=result):
                violations = mcp_smoke.validate_vad_output(
                    result, empty_allowed=False, duration_sec=1.0
                )
                self.assertTrue(any(message in violation for violation in violations), violations)
                with self.assertRaisesRegex(ValueError, message):
                    run_trans_validate.validate_vad_output(
                        result,
                        label="VAD",
                        empty_allowed=False,
                        duration_sec=1.0,
                    )

    def test_equivalence_compares_complete_structure_and_rejects_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, manifest = prepare_vad_run(root)
            value = {
                "rows": [
                    {
                        "key": sample["key"],
                        "result": valid_result(silence=sample["key"] == "silence"),
                    }
                    for sample in manifest["samples"]
                ]
            }
            baseline = run_dir / "artifacts" / "original_output.json"
            adapter = run_dir / "artifacts" / "adapter_validation" / "sample_output.json"
            adapter.parent.mkdir(parents=True)
            baseline.write_text(json.dumps(value), encoding="utf-8")
            adapter.write_text(json.dumps(value), encoding="utf-8")
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNone(error)
            self.assertTrue(evidence["match"])

            changed = json.loads(json.dumps(value))
            changed["rows"][0]["result"]["frame_scores"][0]["score"] = 0.8
            adapter.write_text(json.dumps(changed), encoding="utf-8")
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNotNone(error)
            self.assertIn("speech", evidence["mismatches"])

            external = root / "external.json"
            external.write_text(json.dumps(value), encoding="utf-8")
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(external), "adapter_output": str(adapter)},
            )
            self.assertIsNone(evidence)
            self.assertIn("run-owned", error)

            adapter.unlink()
            os.link(baseline, adapter)
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {"baseline_output": str(baseline), "adapter_output": str(adapter)},
            )
            self.assertIsNone(evidence)
            self.assertIn("independent files", error)

    def test_finalizer_preserves_fixture_and_structured_sample_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, manifest = prepare_vad_run(root)
            model_dir = root / "models" / "example__vad"
            model_dir.mkdir(parents=True)
            finalize_trans_bundle.stage_fixture(
                run_dir,
                model_dir,
                {"task_type": "vad", "model_name": "example__vad"},
            )
            finalized = json.loads(
                (run_dir / "artifacts" / "fixture_manifest.json").read_text(encoding="utf-8")
            )
            check_artifact.validate_fixture_manifest(finalized)
            self.assertTrue((model_dir / "fixture" / "vad" / "silence.wav").is_file())

            (run_dir / "artifacts" / "adapter_manifest.json").write_text(
                json.dumps({"io_contract": scaffold_adapter.io_contract_for("vad")}),
                encoding="utf-8",
            )
            output = {
                "rows": [
                    {
                        "key": sample["key"],
                        "result": valid_result(silence=sample["key"] == "silence"),
                    }
                    for sample in manifest["samples"]
                ]
            }
            validation = run_dir / "artifacts" / "adapter_validation"
            validation.mkdir()
            (validation / "sample_output.json").write_text(
                json.dumps(output), encoding="utf-8"
            )
            finalize_trans_bundle.promote_sample_output(run_dir)
            promoted = json.loads(
                (run_dir / "artifacts" / "sample_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(promoted, output)

    def test_finalizer_rejects_a_manifest_that_replaces_the_run_fixture_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, manifest = prepare_vad_run(root)
            model_dir = root / "models" / "example__vad"
            injected = model_dir / "fixture" / "injected"
            shutil.copytree(run_dir / "fixture" / "vad", injected)

            manifest["model_dir"] = str(model_dir)
            manifest["staged_dir"] = str(injected)
            manifest["staged_path"] = str(injected)
            manifest["gt_jsonl"] = str(injected / "gt.jsonl")
            manifest["annotation_source"]["staged_path"] = str(injected / "gt.jsonl")
            for sample in manifest["samples"]:
                sample["audio_path"] = str(injected / sample["audio"])
            (run_dir / "artifacts" / "fixture_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            shutil.rmtree(run_dir / "fixture" / "vad")

            with self.assertRaisesRegex(ValueError, "run-owned fixture tree"):
                finalize_trans_bundle.stage_fixture(
                    run_dir,
                    model_dir,
                    {"task_type": "vad", "model_name": "example__vad"},
                )
            self.assertFalse((model_dir / "fixture" / "vad").exists())

    def test_fixture_gate_and_finalizer_reject_post_prepare_hard_links(self) -> None:
        for relative in (Path("gt.jsonl"), Path("speech.wav")):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run_dir, manifest = prepare_vad_run(root)
                staged = run_dir / "fixture" / "vad" / relative
                external = root / f"external-{relative.name}"
                shutil.copy2(staged, external)
                staged.unlink()
                os.link(external, staged)

                with self.assertRaisesRegex(ValueError, "hard-link"):
                    check_artifact.validate_fixture_manifest(manifest)

                model_dir = root / "models" / "example__vad"
                model_dir.mkdir(parents=True)
                with (
                    mock.patch.object(
                        finalize_trans_bundle,
                        "validate_fixture_manifest",
                        return_value=None,
                    ),
                    self.assertRaisesRegex(ValueError, "hard-link"),
                ):
                    finalize_trans_bundle.stage_fixture(
                        run_dir,
                        model_dir,
                        {"task_type": "vad", "model_name": "example__vad"},
                    )
                self.assertFalse((model_dir / "fixture" / "vad").exists())

    def test_generated_template_exposes_vad_validation_helpers(self) -> None:
        module = load_module(
            "trans_vad_validate_template",
            SCRIPTS_DIR / "templates" / "validate.py",
        )
        self.assertEqual(
            module.validate_vad_result(
                valid_result(silence=False),
                label="VAD",
                empty_allowed=False,
                duration_sec=1.0,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
