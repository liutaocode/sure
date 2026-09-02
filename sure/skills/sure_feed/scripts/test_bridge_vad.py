#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sure_feed.bridge import emit_sure_model_agent_handoff  # noqa: E402
from sure_feed.fixture_registry import io_contract_for_task  # noqa: E402


def load_trans_scaffold():
    path = Path(__file__).resolve().parents[2] / "sure_trans" / "scripts" / "scaffold_adapter.py"
    spec = importlib.util.spec_from_file_location("sure_trans_scaffold_adapter_for_vad_feed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sure_trans scaffold_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_onboard_structured_contract():
    path = (
        Path(__file__).resolve().parents[2]
        / "sure_onboard"
        / "scripts"
        / "structured_segments.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sure_onboard_structured_segments_for_vad_feed", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sure_onboard structured_segments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VADBridgeTest(unittest.TestCase):
    def test_handoff_generates_canonical_vad_tool_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "sure" / "models" / "snakers4__silero-vad"
            emit_sure_model_agent_handoff(
                {
                    "resource_type": "model",
                    "model_name": model_dir.name,
                    "task_type": "voice-activity-detection",
                    "source": {"provider": "local", "id": str(root / "checkpoint")},
                },
                root / "model-manifest.json",
                root / "handoff.json",
                model_dir,
            )

            spec = yaml.safe_load((model_dir / "model.spec.yaml").read_text(encoding="utf-8"))
            contract = io_contract_for_task("vad")
            self.assertEqual(spec["task_type"], "vad")
            self.assertEqual(spec["io_contract"], contract)
            config = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
            tool = config["tools"][0]
            self.assertEqual(config["task"], "VAD")
            self.assertEqual(tool["name"], "detect_speech")
            self.assertEqual(tool["input_schema"]["required"], ["audio_path"])

            trans_scaffold = load_trans_scaffold()
            trans_tool, trans_schema = trans_scaffold.tool_contract("vad")
            self.assertEqual((trans_tool, trans_schema), (tool["name"], tool["input_schema"]))
            self.assertEqual(trans_scaffold.io_contract_for("vad"), contract)
            self.assertEqual(
                load_onboard_structured_contract().structured_task_contract("vad")[
                    "io_contract"
                ],
                contract,
            )

            model_source = (model_dir / "model.py").read_text(encoding="utf-8")
            self.assertIn("speech_segments: list[dict[str, Any]]", model_source)
            self.assertIn("frame_scores: list[dict[str, Any]] | None = None", model_source)
            self.assertNotIn("raw: dict", model_source)
            compile(model_source, str(model_dir / "model.py"), "exec")
            namespace: dict[str, object] = {}
            exec(model_source, namespace)
            prediction_type = namespace["PredictionResult"]
            prediction = prediction_type(
                speech_segments=[{"start": 0.1, "end": 0.9}]
            )
            self.assertEqual(
                prediction.to_dict(),
                {"speech_segments": [{"start": 0.1, "end": 0.9}]},
            )
            prediction_with_scores = prediction_type(
                speech_segments=[{"start": 0.1, "end": 0.9}],
                frame_scores=[{"start": 0.0, "end": 1.0, "score": 0.8}],
            )
            self.assertIn("frame_scores", prediction_with_scores.to_dict())
            compile(
                (model_dir / "server.py").read_text(encoding="utf-8"),
                str(model_dir / "server.py"),
                "exec",
            )

            (model_dir / "model.py").write_text(
                "class Result:\n"
                "    def __init__(self, frame_scores): self.frame_scores = frame_scores\n"
                "    def to_dict(self): return {'speech_segments': [], 'frame_scores': self.frame_scores}\n"
                "class ModelWrapper:\n"
                "    def __init__(self, config=None): pass\n"
                "    def predict(self, arguments):\n"
                "        return Result([] if arguments['audio_path'] == 'empty.wav' else None)\n",
                encoding="utf-8",
            )
            requests = "".join(
                json.dumps(request) + "\n"
                for request in (
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "detect_speech",
                            "arguments": {"audio_path": "none.wav"},
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "detect_speech",
                            "arguments": {"audio_path": "empty.wav"},
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
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            responses = [json.loads(line) for line in completed.stdout.splitlines() if line]
            self.assertEqual(responses[0]["result"]["tools"][0]["name"], "detect_speech")
            self.assertEqual(
                json.loads(responses[1]["result"]["content"][0]["text"]),
                {"speech_segments": []},
            )
            self.assertTrue(responses[2]["result"]["isError"])
            self.assertIn("frame_scores must be a non-empty list", responses[2]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
