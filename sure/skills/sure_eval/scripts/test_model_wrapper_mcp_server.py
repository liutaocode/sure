#!/usr/bin/env python3
"""Regression tests for the model-wrapper MCP classification boundary."""

from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_wrapper_mcp_server as server  # noqa: E402


class ClassificationMcpBoundaryTests(unittest.TestCase):
    def test_classification_tool_schemas_are_closed(self) -> None:
        for task in ("SER", "GR", "SLU"):
            with self.subTest(task=task):
                self.assertFalse(server._tool_schema(task).get("additionalProperties", True))

        self.assertEqual(server._tool_schema("SER")["required"], ["audio_path"])
        self.assertEqual(server._tool_schema("SLU")["required"], ["audio_path", "prompt"])

    def test_known_classification_tool_rejects_reference_arguments_before_model(self) -> None:
        calls: list[dict] = []

        class Wrapper:
            def predict(self, arguments):
                calls.append(arguments)
                return {"label": "neu"}

        invalid = (
            {"audio_path": "/audio.wav", "ground_truth": "neu"},
            {"audio_path": "/audio.wav", "reference_audio_path": "/secret.wav"},
            {"audio_path": "/audio.wav", "debug": {"target": "secret"}},
            {"audio_path": "/audio.wav", "language": math.nan},
            {"audio_path": "/audio.wav", "language": None},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                server._call_model(Wrapper(), "emotion_recognize", arguments)
        self.assertEqual(calls, [])

    def test_slu_requires_prompt_and_rejects_nested_choice_references(self) -> None:
        class Wrapper:
            def predict(self, _arguments):
                return {"answer": "A"}

        with self.assertRaisesRegex(ValueError, "prompt"):
            server._call_model(Wrapper(), "slu_understand", {"audio_path": "/audio.wav"})
        with self.assertRaisesRegex(ValueError, "reference/path"):
            server._call_model(
                Wrapper(),
                "slu_understand",
                {
                    "audio_path": "/audio.wav",
                    "prompt": "Choose one",
                    "choices": {"A": {"reference_text": "secret"}},
                },
            )

    def test_custom_classification_tool_accepts_canonical_arguments(self) -> None:
        calls: list[dict] = []

        class Wrapper:
            def predict(self, arguments):
                calls.append(arguments)
                return {"label": "neu"}

        result = server._call_model(
            Wrapper(),
            "custom_classifier",
            {"audio_path": "/audio.wav", "language": "en"},
            task="SER",
        )
        self.assertEqual(result, {"label": "neu"})
        self.assertEqual(calls, [{"audio_path": "/audio.wav", "language": "en"}])
        with self.assertRaisesRegex(ValueError, "unapproved"):
            server._call_model(
                Wrapper(),
                "custom_classifier",
                {"audio_path": "/audio.wav", "ground_truth": "secret"},
                task="SER",
            )

    def test_custom_tse_tool_still_requires_pair_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixture = root / "mixture.wav"
            enrollment = root / "enrollment.wav"
            output = root / "output.wav"
            mixture.write_bytes(b"mixture")
            enrollment.write_bytes(b"enrollment")
            calls: list[dict] = []

            class Wrapper:
                def predict(self, arguments):
                    calls.append(arguments)
                    return {"prediction_audio": str(output)}

            result = server._call_model(
                Wrapper(),
                "custom_extractor",
                {
                    "mixture_audio_path": str(mixture),
                    "enrollment_audio_path": str(enrollment),
                    "output_path": str(output),
                },
                task="TSE",
            )
            self.assertEqual(result, {"prediction_audio": str(output)})
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, "requires exactly"):
                server._call_model(
                    Wrapper(),
                    "custom_extractor",
                    {"audio_path": str(mixture)},
                    task="TSE",
                )

    def test_jsonrpc_server_rejects_leaked_classification_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            marker = model_dir / "called"
            (model_dir / "config.yaml").write_text("task: SER\n", encoding="utf-8")
            (model_dir / "model.py").write_text(
                "class ModelWrapper:\n"
                "    def predict(self, arguments):\n"
                "        open('called', 'w').write('1')\n"
                "        return {'label': 'neu'}\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, str(Path(server.__file__).resolve()), "--model-dir", str(model_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            def request(request_id: int, method: str, params: dict) -> dict:
                assert process.stdin is not None
                assert process.stdout is not None
                process.stdin.write(
                    json.dumps(
                        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
                    )
                    + "\n"
                )
                process.stdin.flush()
                line = process.stdout.readline()
                self.assertTrue(line)
                return json.loads(line)

            try:
                request(1, "initialize", {})
                listed = request(2, "tools/list", {})
                tool = listed["result"]["tools"][0]
                self.assertEqual(tool["name"], "emotion_recognize")
                self.assertFalse(tool["inputSchema"]["additionalProperties"])
                rejected = request(
                    3,
                    "tools/call",
                    {
                        "name": "emotion_recognize",
                        "arguments": {"audio_path": "/audio.wav", "ground_truth": "neu"},
                    },
                )
                self.assertIn("error", rejected)
                self.assertFalse(marker.exists())
                accepted = request(
                    4,
                    "tools/call",
                    {"name": "emotion_recognize", "arguments": {"audio_path": "/audio.wav"}},
                )
                self.assertIn("result", accepted)
                self.assertTrue(marker.exists())
                request(5, "shutdown", {})
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()


if __name__ == "__main__":
    unittest.main()
