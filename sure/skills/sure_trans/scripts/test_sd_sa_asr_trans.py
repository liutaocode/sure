#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[3]
FIXTURES = {
    "sd": REPO_ROOT / "fixtures" / "tasks" / "sd" / "librispeech_2spk_smoke",
    "sa_asr": REPO_ROOT / "fixtures" / "tasks" / "sa_asr" / "librispeech_2spk_smoke",
}
sys.path.insert(0, str(SCRIPTS_DIR))

import check_artifact  # noqa: E402
import finalize_trans_bundle  # noqa: E402
import materialize_trans_inputs  # noqa: E402
import mcp_smoke  # noqa: E402
import prepare_fixture  # noqa: E402
import run_trans_validate  # noqa: E402
import scaffold_adapter  # noqa: E402


def write_resolved(run_dir: Path, fixture: Path, task: str) -> None:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "trans_input_resolved.json").write_text(
        json.dumps(
            {
                "model_name": f"example__{task}",
                "task_type": task,
                "build_context": str(fixture.parent),
                "fixture_path": str(fixture),
            }
        ),
        encoding="utf-8",
    )


def prepare_shared_fixture(run_dir: Path, task: str) -> dict:
    write_resolved(run_dir, FIXTURES[task], task)
    with mock.patch.object(sys, "argv", ["prepare_fixture.py", "--run-dir", str(run_dir)]):
        assert prepare_fixture.main() == 0
    return json.loads((run_dir / "artifacts" / "fixture_manifest.json").read_text(encoding="utf-8"))


def result_for(task: str, index: int = 0) -> dict:
    if task == "sd":
        return {
            "segments": [
                {
                    "speaker": "speaker-1",
                    "start": 0.0,
                    "end": 0.5 + index,
                    "duration": 0.5 + index,
                }
            ],
            "num_speakers": 1,
        }
    return {
        "segments": [
            {
                "speaker": "speaker-1",
                "start": 0.0,
                "end": 0.5 + index,
                "text": f"utterance {index}",
                "duration": 0.5 + index,
            }
        ],
        "num_speakers": 1,
    }


def write_silence_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_validate(model_dir: Path, task: str) -> Path:
    contract = scaffold_adapter.io_contract_for(task)
    template = (SCRIPTS_DIR / "templates" / "validate.py").read_text(encoding="utf-8")
    rendered = (
        template.replace("__MODEL_NAME__", f"example__{task}")
        .replace("__TASK_TYPE__", task.upper())
        .replace(
            "__IO_CONTRACT_JSON__",
            json.dumps(contract, ensure_ascii=True, separators=(",", ":")),
        )
    )
    validate_path = model_dir / "validate.py"
    validate_path.write_text(rendered, encoding="utf-8")
    outputs = [result_for(task, index) for index in range(3)]
    (model_dir / "model.py").write_text(
        "from pathlib import Path\n"
        f"OUTPUTS = {outputs!r}\n"
        "class ModelWrapper:\n"
        "    def load(self): self.model = object()\n"
        "    def predict(self, payload):\n"
        "        if set(payload) != {'audio_path'}:\n"
        "            raise ValueError(f'reference leakage: {sorted(payload)}')\n"
        "        index = int(Path(payload['audio_path']).stem.rsplit('_', 1)[-1]) - 1\n"
        "        return OUTPUTS[index]\n",
        encoding="utf-8",
    )
    return validate_path


class SdSaAsrTransTests(unittest.TestCase):
    def test_task_aliases_schemas_and_tool_contracts(self) -> None:
        aliases = {
            "sd": (
                "sd",
                "speaker-diarization",
                "speaker_diarization",
                "diarization",
                "diarisation",
            ),
            "sa_asr": (
                "sa_asr",
                "sa-asr",
                "speaker-attributed-asr",
                "speaker-attributed asr",
                "speaker_attributed_asr",
                "speaker-aware-asr",
                "transcribe-diarize",
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "infer.py"
            entrypoint.write_text("def infer(audio_path): return audio_path\n", encoding="utf-8")
            for expected, values in aliases.items():
                for value in values:
                    self.assertEqual(
                        materialize_trans_inputs.resolve_task_type(
                            value, entrypoint, Path("speaker-model")
                        ),
                        expected,
                    )
            entrypoint.write_text(
                "def transcribe_with_speakers(audio_path): return []\n", encoding="utf-8"
            )
            self.assertEqual(
                materialize_trans_inputs.resolve_task_type(
                    None, entrypoint, Path("speaker-model")
                ),
                "sa_asr",
            )
            entrypoint.write_text("def run(audio_path): return []\n", encoding="utf-8")
            self.assertEqual(
                materialize_trans_inputs.resolve_task_type(
                    None, entrypoint, Path("MOSS-Transcribe-Diarize")
                ),
                "sa_asr",
            )
            entrypoint.write_text("def diarize(audio_path): return []\n", encoding="utf-8")
            self.assertEqual(
                materialize_trans_inputs.resolve_task_type(
                    None, entrypoint, Path("speaker-model")
                ),
                "sd",
            )

        expected_tools = {"sd": "diarize", "sa_asr": "transcribe_with_speakers"}
        for task, tool in expected_tools.items():
            tool_name, input_schema = scaffold_adapter.tool_contract(task)
            contract = scaffold_adapter.io_contract_for(task)
            self.assertEqual(tool_name, tool)
            self.assertEqual(input_schema["required"], ["audio_path"])
            self.assertFalse(input_schema["additionalProperties"])
            self.assertEqual(contract["primary_field"], "segments")
            self.assertEqual(contract["required_fields"], ["segments"])
        self.assertTrue(scaffold_adapter.io_contract_for("sd")["allow_empty_primary"])
        self.assertFalse(scaffold_adapter.io_contract_for("sa_asr")["allow_empty_primary"])

        feed_registry = load_module(
            "feed_fixture_registry_for_trans_contract",
            REPO_ROOT
            / "sure"
            / "skills"
            / "sure_feed"
            / "scripts"
            / "sure_feed"
            / "fixture_registry.py",
        )
        onboard_segments = load_module(
            "onboard_structured_segments_for_trans_contract",
            REPO_ROOT
            / "sure"
            / "skills"
            / "sure_onboard"
            / "scripts"
            / "structured_segments.py",
        )
        for task in ("sd", "sa_asr"):
            self.assertEqual(
                scaffold_adapter.io_contract_for(task),
                feed_registry.io_contract_for_task(task),
            )
            self.assertEqual(
                scaffold_adapter.io_contract_for(task),
                onboard_segments.structured_task_contract(task)["io_contract"],
            )

        input_schema = json.loads(
            (SCRIPTS_DIR.parent / "schemas" / "trans_input_resolved.schema.json").read_text(
                encoding="utf-8"
            )
        )
        fixture_schema = json.loads(
            (SCRIPTS_DIR.parent / "schemas" / "fixture_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        for task in ("sd", "sa_asr"):
            self.assertIn(task, input_schema["properties"]["task_type"]["enum"])
            self.assertIn(task, fixture_schema["properties"]["task_type"]["enum"])
            output_schema = json.loads(
                (SCRIPTS_DIR.parent / "schemas" / f"{task}_output.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(output_schema["required"], ["segments"])
            self.assertFalse(output_schema["additionalProperties"])
            self.assertFalse(
                output_schema["properties"]["segments"]["items"]["additionalProperties"]
            )

    def test_prepare_and_gate_preserve_real_structured_fixtures_without_manifest_labels(self) -> None:
        for task in ("sd", "sa_asr"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                manifest = prepare_shared_fixture(run_dir, task)
                check_artifact.validate_fixture_manifest(manifest)
                self.assertEqual(manifest["sample_count"], 3)
                self.assertEqual(manifest["task_type"], task)
                for sample in manifest["samples"]:
                    self.assertEqual(sample["annotation_fields"], ["segments"])
                    self.assertNotIn("segments", sample)
                    self.assertTrue(Path(sample["audio_path"]).is_file())
                rows = prepare_fixture.read_jsonl(Path(manifest["gt_jsonl"]))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(row["task_type"] == task for row in rows))

    def test_prepare_accepts_every_row_count_from_one_to_five_and_sd_silence(self) -> None:
        for task in ("sd", "sa_asr"):
            for count in range(1, 6):
                with self.subTest(task=task, count=count), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    fixture = root / "fixture"
                    fixture.mkdir()
                    write_silence_wav(fixture / "sample.wav")
                    segments = (
                        []
                        if task == "sd"
                        else [
                            {
                                "speaker": "speaker-1",
                                "start": 0.0,
                                "end": 0.1,
                                "text": "speech",
                            }
                        ]
                    )
                    rows = [
                        {
                            "key": f"sample-{index}",
                            "audio": "sample.wav",
                            "segments": segments,
                        }
                        for index in range(count)
                    ]
                    (fixture / "gt.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                    )
                    manifest = prepare_fixture.prepare_speaker_fixture(
                        {"model_name": f"example__{task}"},
                        fixture,
                        root / "run",
                        task=task,
                    )
                    self.assertEqual(manifest["sample_count"], count)
                    check_artifact.validate_fixture_manifest(manifest)

        for count in (0, 6):
            with self.subTest(rejected_count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = root / "fixture"
                fixture.mkdir()
                write_silence_wav(fixture / "sample.wav")
                rows = [
                    {"key": f"sample-{index}", "audio": "sample.wav", "segments": []}
                    for index in range(count)
                ]
                (fixture / "gt.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "1 to 5"):
                    prepare_fixture.prepare_speaker_fixture(
                        {"model_name": "example__sd"}, fixture, root / "run", task="sd"
                    )

    def test_prepare_rejects_duplicate_unsafe_and_symlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            fixture.mkdir()
            audio = fixture / "sample.wav"
            shutil.copy2(FIXTURES["sd"] / "librispeech_2spk_001.wav", audio)
            row = {
                "key": "sample",
                "audio": "sample.wav",
                "segments": [{"speaker": "speaker-1", "start": 0.0, "end": 0.5}],
            }

            (fixture / "gt.jsonl").write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                prepare_fixture.prepare_speaker_fixture(
                    {"model_name": "example__sd"}, fixture, root / "run", task="sd"
                )

            for unsafe in ("../outside.wav", str(audio.resolve()), "C:\\outside.wav"):
                (fixture / "gt.jsonl").write_text(
                    json.dumps({**row, "audio": unsafe}) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "stay inside"):
                    prepare_fixture.prepare_speaker_fixture(
                        {"model_name": "example__sd"}, fixture, root / "run", task="sd"
                    )

            linked_audio = fixture / "linked.wav"
            linked_audio.symlink_to(audio)
            (fixture / "gt.jsonl").write_text(
                json.dumps({**row, "audio": linked_audio.name}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_fixture.prepare_speaker_fixture(
                    {"model_name": "example__sd"}, fixture, root / "run", task="sd"
                )

            outside_dir = root / "outside"
            outside_dir.mkdir()
            shutil.copy2(audio, outside_dir / "nested.wav")
            (fixture / "nested").symlink_to(outside_dir, target_is_directory=True)
            (fixture / "gt.jsonl").write_text(
                json.dumps({**row, "audio": "nested/nested.wav"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_fixture.prepare_speaker_fixture(
                    {"model_name": "example__sd"}, fixture, root / "run", task="sd"
                )

            real_gt = fixture / "real.jsonl"
            real_gt.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (fixture / "gt.jsonl").unlink()
            (fixture / "gt.jsonl").symlink_to(real_gt)
            with self.assertRaisesRegex(ValueError, "must contain gt.jsonl"):
                prepare_fixture.prepare_speaker_fixture(
                    {"model_name": "example__sd"}, fixture, root / "run", task="sd"
                )

            real_fixture = root / "real-fixture"
            real_fixture.mkdir()
            shutil.copy2(audio, real_fixture / "sample.wav")
            (real_fixture / "gt.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            linked_fixture = root / "linked-fixture"
            linked_fixture.symlink_to(real_fixture, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "fixture root must not be a symlink"):
                prepare_fixture.prepare_speaker_fixture(
                    {"model_name": "example__sd"},
                    linked_fixture,
                    root / "linked-run",
                    task="sd",
                )

    def test_segment_schema_rejects_malformed_values(self) -> None:
        malformed = (
            ("sd", None, "array"),
            ("sd", ["bad"], "object"),
            ("sd", [{"speaker": "", "start": 0, "end": 1}], "speaker"),
            ("sd", [{"speaker": "s", "start": -1, "end": 1}], "start"),
            ("sd", [{"speaker": "s", "start": True, "end": 1}], "start"),
            ("sd", [{"speaker": "s", "start": 0, "end": float("inf")}], "end"),
            ("sd", [{"speaker": "s", "start": 1, "end": 1}], "end"),
            (
                "sd",
                [{"speaker": "s", "start": 0, "end": 1, "duration": 0.5}],
                "duration",
            ),
            ("sa_asr", [], "must not be empty"),
            (
                "sa_asr",
                [{"speaker": "s", "start": 0, "end": 1, "text": " "}],
                "text",
            ),
        )
        for task, segments, message in malformed:
            with self.subTest(task=task, segments=segments):
                with self.assertRaisesRegex(ValueError, message):
                    prepare_fixture.validate_speaker_segments(
                        segments, task=task, label="test"
                    )
        with self.assertRaisesRegex(ValueError, "pure-silence"):
            prepare_fixture.validate_speaker_segments([], task="sd", label="speech")
        self.assertEqual(
            prepare_fixture.validate_speaker_segments(
                [], task="sd", label="silence", empty_sd_allowed=True
            ),
            [],
        )
        with self.assertRaisesRegex(ValueError, "exceeds WAV duration"):
            prepare_fixture.validate_speaker_segments(
                [{"speaker": "s", "start": 0, "end": 2}],
                task="sd",
                label="duration",
                duration_sec=1.0,
            )
        with self.assertRaisesRegex(ValueError, "safe token"):
            prepare_fixture.validate_speaker_segments(
                [{"speaker": "/home/private", "start": 0, "end": 0.5}],
                task="sd",
                label="speaker",
                duration_sec=1.0,
            )
        with self.assertRaisesRegex(ValueError, "unapproved field"):
            prepare_fixture.validate_speaker_segments(
                [{"speaker": "s", "start": 0, "end": 0.5, "debug": "x"}],
                task="sd",
                label="fields",
                duration_sec=1.0,
            )
        self.assertEqual(
            mcp_smoke.validate_speaker_output(
                {"segments": []}, tool="diarize", empty_sd_allowed=True
            ),
            [],
        )
        self.assertTrue(
            any(
                "pure-silence" in violation
                for violation in mcp_smoke.validate_speaker_output(
                    {"segments": []}, tool="diarize"
                )
            )
        )
        self.assertEqual(
            mcp_smoke.validate_speaker_output(
                {
                    "segments": [
                        {
                            "speaker": "s",
                            "start": 0,
                            "end": 1,
                            "text": "the transcript literally says /home/example",
                        }
                    ]
                },
                tool="transcribe_with_speakers",
                duration_sec=2.0,
            ),
            [],
        )
        self.assertTrue(
            any(
                "unapproved field" in violation
                for violation in mcp_smoke.validate_speaker_output(
                    {
                        "segments": [{"speaker": "s", "start": 0, "end": 1}],
                        "raw": "s3://private",
                    },
                    tool="diarize",
                    duration_sec=2.0,
                )
            )
        )

    def test_truncated_pcm_cannot_enable_silent_sd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "truncated.wav"
            declared_bytes = 32000
            path.write_bytes(
                b"RIFF"
                + struct.pack("<I", 36 + declared_bytes)
                + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
                + b"data"
                + struct.pack("<I", declared_bytes)
                + b"\x00\x00"
            )
            for reader in (
                prepare_fixture.pcm_wav_info,
                mcp_smoke.pcm_wav_info,
                run_trans_validate.pcm_wav_info,
            ):
                with self.subTest(reader=reader.__module__), self.assertRaisesRegex(
                    ValueError, "truncated"
                ):
                    reader(path)

            validate_module = load_module(
                "trans_validate_template_truncated_pcm",
                SCRIPTS_DIR / "templates" / "validate.py",
            )
            with self.assertRaisesRegex(ValueError, "truncated"):
                validate_module.pcm_wav_info(path)
        self.assertTrue(
            any(
                "num_speakers" in violation
                for violation in mcp_smoke.validate_speaker_output(
                    {
                        "segments": [{"speaker": "s", "start": 0, "end": 1}],
                        "num_speakers": 2,
                    },
                    tool="diarize",
                )
            )
        )
        self.assertTrue(
            any(
                "reference or path field" in violation
                for violation in mcp_smoke.validate_speaker_output(
                    {
                        "segments": [{"speaker": "s", "start": 0, "end": 1}],
                        "reference_segments": [],
                    },
                    tool="diarize",
                )
            )
        )

    def test_generated_validate_runs_all_rows_without_reference_leakage(self) -> None:
        for task in ("sd", "sa_asr"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run_dir = root / "run"
                prepare_shared_fixture(run_dir, task)
                model_dir = root / "model"
                shutil.copytree(run_dir / "fixture" / task, model_dir / "fixture" / task)
                validate_path = render_validate(model_dir, task)
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
                self.assertEqual(len(sample["rows"]), 3)
                self.assertEqual([row["key"] for row in sample["rows"]], [
                    "librispeech_2spk_001",
                    "librispeech_2spk_002",
                    "librispeech_2spk_003",
                ])

    def test_mcp_smoke_runs_all_rows_and_preserves_structured_fields(self) -> None:
        for task, tool in (("sd", "diarize"), ("sa_asr", "transcribe_with_speakers")):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = root / "fixture"
                shutil.copytree(FIXTURES[task], fixture)
                server = root / "server.py"
                result = result_for(task)
                server.write_text(
                    "import json, sys\n"
                    f"TOOL = {tool!r}\n"
                    f"VALUE = {result!r}\n"
                    "for line in sys.stdin:\n"
                    " request=json.loads(line); method=request.get('method'); rid=request.get('id')\n"
                    " if method=='initialize': response={'protocolVersion':'2024-11-05'}\n"
                    " elif method=='tools/list': response={'tools':[{'name':TOOL}]}\n"
                    " elif method=='tools/call':\n"
                    "  arguments=request['params']['arguments']\n"
                    "  if set(arguments) != {'audio_path'}: raise ValueError('reference leaked')\n"
                    "  print('model progress line', flush=True)\n"
                    "  response={'content':[{'type':'text','text':json.dumps(VALUE)}]}\n"
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
                        tool,
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
                self.assertEqual(payload["tools_call"]["num_samples"], 3)
                self.assertEqual(payload["stdout_junk_count"], 3)
                self.assertEqual(payload["tools_call"]["samples"][0]["result"], result)
                self.assertGreater(
                    payload["tools_call"]["samples"][0]["audio_duration_sec"],
                    0,
                )
                self.assertNotIn("server_stderr_tail", payload)
                self.assertNotIn("stdout_junk_tail", payload)
                self.assertEqual(payload["stdout_junk_summary"]["line_count"], 3)
                self.assertEqual(len(payload["stdout_junk_summary"]["sha256"]), 64)
                self.assertNotIn(str(root), json.dumps(payload))
                self.assertIsNone(run_trans_validate.validate_mcp_evidence(evidence, tool))

    def test_equivalence_compares_every_structured_field_and_allows_sd_silence(self) -> None:
        for task in ("sd", "sa_asr"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                manifest = prepare_shared_fixture(run_dir, task)
                artifacts = run_dir / "artifacts"
                value = {
                    "rows": [
                        {
                            "key": sample["key"],
                            "result": result_for(task, index),
                        }
                        for index, sample in enumerate(manifest["samples"])
                    ]
                }
                baseline = artifacts / "original_output.json"
                adapter = artifacts / "adapter_validation" / "sample_output.json"
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
                changed["rows"][0]["result"]["segments"][0]["end"] = 0.4
                changed["rows"][0]["result"]["segments"][0]["duration"] = 0.4
                adapter.write_text(json.dumps(changed), encoding="utf-8")
                evidence, error = run_trans_validate.compare_equivalence_outputs(
                    run_dir,
                    {"baseline_output": str(baseline), "adapter_output": str(adapter)},
                )
                self.assertIsNotNone(error)
                self.assertIn(manifest["samples"][0]["key"], evidence["mismatches"])

                typed_baseline = json.loads(json.dumps(value))
                typed_adapter = json.loads(json.dumps(value))
                typed_baseline["rows"][0]["result"]["segments"][0]["start"] = 0
                typed_adapter["rows"][0]["result"]["segments"][0]["start"] = 0.0
                baseline.write_text(json.dumps(typed_baseline), encoding="utf-8")
                adapter.write_text(json.dumps(typed_adapter), encoding="utf-8")
                evidence, error = run_trans_validate.compare_equivalence_outputs(
                    run_dir,
                    {"baseline_output": str(baseline), "adapter_output": str(adapter)},
                )
                self.assertIsNotNone(error)
                self.assertIn(manifest["samples"][0]["key"], evidence["mismatches"])

    def test_equivalence_rejects_external_or_samefile_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            manifest = prepare_shared_fixture(run_dir, "sd")
            value = {
                "rows": [
                    {
                        "key": sample["key"],
                        "result": result_for("sd", index),
                    }
                    for index, sample in enumerate(manifest["samples"])
                ]
            }
            external = root / "external.json"
            external.write_text(json.dumps(value), encoding="utf-8")
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {
                    "baseline_output": str(external),
                    "adapter_output": str(external),
                },
            )
            self.assertIsNone(evidence)
            self.assertIn("run-owned", error)

            baseline = run_dir / "artifacts" / "original_output.json"
            adapter = run_dir / "artifacts" / "adapter_validation" / "sample_output.json"
            baseline.write_text(json.dumps(value), encoding="utf-8")
            adapter.parent.mkdir(parents=True)
            os.link(baseline, adapter)
            evidence, error = run_trans_validate.compare_equivalence_outputs(
                run_dir,
                {
                    "baseline_output": str(baseline),
                    "adapter_output": str(adapter),
                },
            )
            self.assertIsNone(evidence)
            self.assertIn("independent files", error)

    def test_mcp_projection_drops_raw_diagnostics_and_rejects_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            model_dir = root / "model"
            run_dir.mkdir()
            model_dir.mkdir()
            source = root / "mcp.json"
            destination = root / "projected.json"
            source.write_text(
                json.dumps(
                    {
                        "server_stderr_tail": ["HF_TOKEN=hf_private"],
                        "stdout_junk_tail": ["cache=/home/user/.cache"],
                        "server_stderr_summary": {
                            "line_count": 1,
                            "utf8_bytes": 8,
                            "sha256": "a" * 64,
                        },
                        "server_command": ["python", "/opt/sure_trans/server.py"],
                        "portable_paths": [
                            "/fixture/sample.wav",
                            "/models/example",
                            "/validation/outputs/sample.json",
                            "/workspace/project/config.yaml",
                        ],
                        "tools_call": {
                            "samples": [
                                {
                                    "result": {
                                        "segments": [
                                            {
                                                "speaker": "spk1",
                                                "start": 0,
                                                "end": 1,
                                                "text": "literal /home/example transcript",
                                            }
                                        ]
                                    }
                                }
                            ]
                        },
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
            projected = json.loads(destination.read_text(encoding="utf-8"))
            self.assertNotIn("server_stderr_tail", projected)
            self.assertNotIn("stdout_junk_tail", projected)
            self.assertIn("literal /home/example transcript", json.dumps(projected))
            self.assertEqual(
                projected["portable_paths"],
                [
                    "/fixture/sample.wav",
                    "/models/example",
                    "/validation/outputs/sample.json",
                    "/workspace/project/config.yaml",
                ],
            )

            for unsafe_path in (
                "/srv/private/cache",
                "/optical/private/cache",
                "Z:\\private\\cache",
                "\\\\example-host\\private\\cache",
                "s3://restricted.example.invalid/object",
            ):
                with self.subTest(unsafe_path=unsafe_path):
                    source.write_text(
                        json.dumps({"artifact": unsafe_path}),
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

            source.write_text(
                json.dumps({"error": "HF_TOKEN=hf_private"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sensitive"):
                finalize_trans_bundle.project_mcp_result(
                    source,
                    destination,
                    run_dir=run_dir,
                    model_dir=model_dir,
                    resolved={},
                )

    def test_finalizer_preserves_structured_fixture_and_sample_output(self) -> None:
        for task in ("sd", "sa_asr"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run_dir = root / "run"
                manifest = prepare_shared_fixture(run_dir, task)
                model_dir = root / "models" / f"example__{task}"
                model_dir.mkdir(parents=True)
                finalize_trans_bundle.stage_fixture(
                    run_dir,
                    model_dir,
                    {"task_type": task, "model_name": f"example__{task}"},
                )
                finalized = json.loads(
                    (run_dir / "artifacts" / "fixture_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                check_artifact.validate_fixture_manifest(finalized)
                self.assertEqual(
                    prepare_fixture.read_jsonl(model_dir / "fixture" / task / "gt.jsonl"),
                    prepare_fixture.read_jsonl(Path(finalized["gt_jsonl"])),
                )

                (run_dir / "artifacts" / "adapter_manifest.json").write_text(
                    json.dumps({"io_contract": scaffold_adapter.io_contract_for(task)}),
                    encoding="utf-8",
                )
                output = {
                    "rows": [
                        {"key": sample["key"], "result": result_for(task, index)}
                        for index, sample in enumerate(manifest["samples"])
                    ]
                }
                validation = run_dir / "artifacts" / "adapter_validation"
                validation.mkdir()
                (validation / "sample_output.json").write_text(
                    json.dumps(output), encoding="utf-8"
                )
                finalize_trans_bundle.promote_sample_output(run_dir)
                promoted = json.loads(
                    (run_dir / "artifacts" / "sample_output.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(promoted, output)


if __name__ == "__main__":
    unittest.main()
