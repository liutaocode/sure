#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sure_feed.bridge import emit_sure_model_agent_handoff  # noqa: E402


class StructuredSpeakerBridgeTest(unittest.TestCase):
    def test_handoff_generates_structured_sd_and_sa_asr_tools(self) -> None:
        cases = (
            (
                "speaker-diarization",
                "sd",
                "SD",
                "diarize",
                ["speaker", "start", "end"],
            ),
            (
                "speaker-attributed-asr",
                "sa_asr",
                "SA-ASR",
                "transcribe_with_speakers",
                ["speaker", "start", "end", "text"],
            ),
        )
        for task_alias, canonical_task, display_task, tool_name, segment_fields in cases:
            with self.subTest(task=canonical_task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model_dir = root / "sure" / "models" / f"owner__{canonical_task}"
                emit_sure_model_agent_handoff(
                    {
                        "resource_type": "model",
                        "model_name": model_dir.name,
                        "task_type": task_alias,
                        "source": {
                            "provider": "local",
                            "id": str(root / "checkpoint"),
                        },
                    },
                    root / "model-manifest.json",
                    root / "handoff.json",
                    model_dir,
                )

                spec = yaml.safe_load(
                    (model_dir / "model.spec.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(spec["task_type"], canonical_task)
                self.assertEqual(spec["io_contract"]["input_type"], "audio_path")
                self.assertEqual(
                    spec["io_contract"]["output_type"],
                    "structured_segments",
                )
                self.assertEqual(spec["io_contract"]["primary_field"], "segments")
                self.assertEqual(spec["io_contract"]["required_fields"], ["segments"])
                self.assertEqual(
                    spec["io_contract"]["nonempty_fields"],
                    [] if canonical_task == "sd" else ["segments"],
                )
                self.assertEqual(
                    spec["io_contract"]["allow_empty_segments"],
                    "silence_only" if canonical_task == "sd" else False,
                )
                self.assertEqual(
                    spec["io_contract"]["segment_schema"]["required"],
                    segment_fields,
                )
                self.assertFalse(
                    spec["io_contract"]["segment_schema"]["additionalProperties"]
                )

                config = yaml.safe_load(
                    (model_dir / "config.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(config["task"], display_task)
                tool = config["tools"][0]
                self.assertEqual(tool["name"], tool_name)
                self.assertEqual(tool["input_schema"]["required"], ["audio_path"])

                model_source = (model_dir / "model.py").read_text(encoding="utf-8")
                self.assertIn(
                    "segments: list[dict[str, Any]] = field(default_factory=list)",
                    model_source,
                )
                self.assertNotIn('text: str = ""', model_source)
                compile(model_source, str(model_dir / "model.py"), "exec")
                compile(
                    (model_dir / "server.py").read_text(encoding="utf-8"),
                    str(model_dir / "server.py"),
                    "exec",
                )

                segments = [{"speaker": "spk1", "start": 0.0, "end": 1.0}]
                if canonical_task == "sa_asr":
                    segments[0]["text"] = "hello"
                (model_dir / "model.py").write_text(
                    "class Result:\n"
                    f"    def to_dict(self): return {{'segments': {segments!r}}}\n"
                    "class ModelWrapper:\n"
                    "    def __init__(self, config=None): pass\n"
                    "    def predict(self, arguments): return Result()\n",
                    encoding="utf-8",
                )
                requests = "".join(
                    json.dumps(request) + "\n"
                    for request in (
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/list",
                            "params": {},
                        },
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": tool_name,
                                "arguments": {"audio_path": "sample.wav"},
                            },
                        },
                    )
                )
                completed = subprocess.run(
                    [sys.executable, str(model_dir / "server.py")],
                    input=requests,
                    capture_output=True,
                    text=True,
                    cwd=model_dir,
                    check=True,
                )
                responses = [
                    json.loads(line)
                    for line in completed.stdout.splitlines()
                    if line.strip()
                ]
                self.assertEqual(responses[0]["result"]["tools"][0]["name"], tool_name)
                self.assertEqual(
                    json.loads(responses[1]["result"]["content"][0]["text"]),
                    {"segments": segments},
                )


if __name__ == "__main__":
    unittest.main()
