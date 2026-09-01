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


def load_trans_scaffold():
    path = Path(__file__).resolve().parents[2] / "sure_trans" / "scripts" / "scaffold_adapter.py"
    spec = importlib.util.spec_from_file_location("sure_trans_scaffold_adapter_for_se_feed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sure_trans scaffold_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SEBridgeTest(unittest.TestCase):
    def test_handoff_generates_se_tool_and_canonical_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "sure" / "models" / "owner__enhancer"
            emit_sure_model_agent_handoff(
                {
                    "resource_type": "model",
                    "model_name": "owner__enhancer",
                    "task_type": "speech_enhancement",
                    "source": {"provider": "local", "id": str(root / "checkpoint")},
                },
                root / "model-manifest.json",
                root / "handoff.json",
                model_dir,
            )

            spec = yaml.safe_load((model_dir / "model.spec.yaml").read_text(encoding="utf-8"))
            self.assertEqual(spec["task_type"], "se")
            self.assertEqual(
                spec["io_contract"],
                {
                    "input_type": "audio_path",
                    "output_type": "audio",
                    "primary_field": "audio_path",
                    "required_fields": ["audio_path"],
                    "nonempty_fields": ["audio_path"],
                    "json_serializable": True,
                },
            )
            config = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
            tool = config["tools"][0]
            self.assertEqual(tool["name"], "enhance_speech")
            self.assertEqual(tool["input_schema"]["required"], ["audio_path"])
            self.assertEqual(
                set(tool["input_schema"]["properties"]),
                {"audio_path", "output_path"},
            )
            trans_scaffold = load_trans_scaffold()
            trans_tool, trans_schema = trans_scaffold.tool_contract("se")
            self.assertEqual((trans_tool, trans_schema), (tool["name"], tool["input_schema"]))
            trans_contract = trans_scaffold.io_contract_for("se")
            for field in (
                "input_type",
                "output_type",
                "primary_field",
                "required_fields",
                "nonempty_fields",
                "json_serializable",
            ):
                self.assertEqual(spec["io_contract"][field], trans_contract[field])

            model_source = (model_dir / "model.py").read_text(encoding="utf-8")
            self.assertIn("output_path must be a non-empty string when provided", model_source)
            self.assertIn("speech enhancement inference", model_source)
            compile(model_source, str(model_dir / "model.py"), "exec")
            compile(
                (model_dir / "server.py").read_text(encoding="utf-8"),
                str(model_dir / "server.py"),
                "exec",
            )

            (model_dir / "model.py").write_text(
                "class Result:\n"
                "    def to_dict(self): return {'audio_path': 'enhanced.wav'}\n"
                "class ModelWrapper:\n"
                "    def __init__(self, config=None): pass\n"
                "    def predict(self, arguments): return Result()\n",
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
                            "name": "enhance_speech",
                            "arguments": {
                                "audio_path": "noisy.wav",
                                "output_path": "enhanced.wav",
                            },
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
            responses = [json.loads(line) for line in completed.stdout.splitlines() if line]
            self.assertEqual(responses[0]["result"]["tools"][0]["name"], "enhance_speech")
            self.assertEqual(
                json.loads(responses[1]["result"]["content"][0]["text"]),
                {"audio_path": "enhanced.wav"},
            )


if __name__ == "__main__":
    unittest.main()
