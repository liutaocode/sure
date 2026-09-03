#!/usr/bin/env python3
"""Regression tests for SER/GR/SLU structured prediction bridging."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import classification_contract as contract  # noqa: E402
import evaluate_predictions as evaluate  # noqa: E402
import evaluation_capabilities as capabilities  # noqa: E402
import generate_predictions_via_server as generate  # noqa: E402
import materialize_predictions_template as templates  # noqa: E402
import validate_prediction_files as validate  # noqa: E402
import model_wrapper_mcp_server as model_server  # noqa: E402
import resolve_eval_input as resolve_input  # noqa: E402


class ClassificationBridgeTests(unittest.TestCase):
    def test_task_aliases_are_canonical(self) -> None:
        for alias in (
            "SER",
            "speech emotion recognition",
            "emotion-recognition",
            "GR",
            "gender recognition",
            "SLU",
            "spoken-language-understanding",
        ):
            with self.subTest(alias=alias):
                expected = "SER" if "emotion" in alias.lower() or alias == "SER" else (
                    "GR" if "gender" in alias.lower() or alias == "GR" else "SLU"
                )
                self.assertEqual(generate._normalize_task(alias), expected)
                self.assertEqual(validate._normalize_task(alias), expected)
                self.assertEqual(evaluate._normalize_task(alias), expected)

    def test_composite_speech_understanding_is_not_collapsed_to_slu(self) -> None:
        self.assertEqual(contract.canonical_task("speech_understanding"), "SPEECH_UNDERSTANDING")
        self.assertEqual(generate._normalize_task("speech_understanding"), "SPEECH_UNDERSTANDING")
        self.assertEqual(validate._normalize_task("speech_understanding"), "SPEECH_UNDERSTANDING")
        self.assertEqual(evaluate._normalize_task("speech_understanding"), "SPEECH_UNDERSTANDING")
        self.assertEqual(resolve_input._normalize_task("speech_understanding"), "SPEECH_UNDERSTANDING")
        self.assertEqual(model_server._model_task({"task": "speech_understanding"}), "SPEECH_UNDERSTANDING")
        with self.assertRaises(ValueError):
            capabilities.normalize_engine_task("speech_understanding")
        self.assertNotEqual(templates._default_metric("speech_understanding", "en"), "accuracy")

    def test_inference_arguments_exclude_reference_annotations(self) -> None:
        ser = generate._build_tool_arguments(
            repo_root=Path("/repo"),
            sample={
                "key": "ser-1",
                "language": "en",
                "ground_truth": "neu",
                "target": "must-not-leak",
            },
            task="SER",
            language="en",
            argument_name="audio_path",
            audio_path=Path("/fixture/ser.wav"),
            output_audio_dir=Path("/unused"),
            tool_args=None,
        )
        self.assertEqual(ser, {"audio_path": "/fixture/ser.wav", "language": "en"})

        slu = generate._build_tool_arguments(
            repo_root=Path("/repo"),
            sample={
                "key": "slu-1",
                "language": "en",
                "prompt": "Choose one.",
                "choices": {"A": "yes", "B": "no"},
                "ground_truth": "A",
            },
            task="SLU",
            language="en",
            argument_name="audio_path",
            audio_path=Path("/fixture/slu.wav"),
            output_audio_dir=Path("/unused"),
            tool_args=None,
        )
        self.assertEqual(
            slu,
            {
                "audio_path": "/fixture/slu.wav",
                "language": "en",
                "prompt": "Choose one.",
                "choices": {"A": "yes", "B": "no"},
            },
        )

    def test_tool_args_cannot_override_classification_inputs(self) -> None:
        common = {
            "repo_root": Path("/repo"),
            "sample": {
                "key": "slu-1",
                "prompt": "Choose one.",
                "choices": {"A": "yes", "B": "no"},
            },
            "task": "SLU",
            "language": "en",
            "argument_name": "audio_path",
            "audio_path": Path("/fixture/slu.wav"),
            "output_audio_dir": Path("/unused"),
        }
        arguments = generate._build_tool_arguments(
            **common,
            tool_args={"temperature": 0, "do_sample": False, "language": "zh"},
        )
        self.assertEqual(arguments["audio_path"], "/fixture/slu.wav")
        self.assertEqual(arguments["prompt"], "Choose one.")
        self.assertEqual(arguments["choices"], {"A": "yes", "B": "no"})
        self.assertEqual(arguments["temperature"], 0)
        self.assertEqual(arguments["do_sample"], False)
        self.assertEqual(arguments["language"], "zh")

        for tool_args, message in (
            ({"audio_path": "/private/other.wav"}, "cannot be overridden"),
            ({"prompt": "reference text"}, "cannot be overridden"),
            ({"choices": {"A": "other"}}, "cannot be overridden"),
            ({"nested": {"ground_truth": "secret"}}, "reference/path"),
            ({"temperature": {"value": 0}}, "must be a scalar"),
        ):
            with self.subTest(tool_args=tool_args), self.assertRaisesRegex(ValueError, message):
                generate._build_tool_arguments(**common, tool_args=tool_args)

    def test_prediction_normalization_is_closed_and_canonical(self) -> None:
        self.assertEqual(
            generate._normalize_prediction_payload(
                {"label": "neutral", "score": 0.75}, task="SER"
            ),
            ("neu", {"label": "neu", "score": 0.75}),
        )
        self.assertEqual(
            generate._normalize_prediction_payload({"label": "female"}, task="GR"),
            ("woman", {"label": "woman"}),
        )
        self.assertEqual(
            generate._normalize_prediction_payload(
                {"answer": "The answer is A."}, task="SLU"
            ),
            ("A", {"answer": "A"}),
        )
        for payload in (
            {"label": "neu", "ground_truth": "neu"},
            {"label": "neu", "audio_path": "/private/audio.wav"},
            {"label": "unknown"},
            {"label": "neu", "score": 1.5},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                generate._normalize_prediction_payload(payload, task="SER")

    def test_structured_validator_checks_dataset_and_reference_isolation(self) -> None:
        samples = [
            {
                "key": "ser-1",
                "task": "SER",
                "ground_truth": "neu",
            }
        ]
        row = {
            "key": "ser-1",
            "dataset": "demo",
            "task": "SER",
            "prediction": {"label": "neu", "score": 0.8},
            "normalized_prediction": "neu",
            "raw_response": {"label": "neu", "score": 0.8},
        }
        self.assertEqual(
            validate._task_contract_violations(
                samples, {"ser-1": row}, dataset_name="demo"
            ),
            [],
        )
        for bad_row in (
            {**row, "dataset": "other"},
            {**row, "ground_truth": "neu"},
            {**row, "prediction": {"label": "neu", "reference": "neu"}},
        ):
            with self.subTest(bad_row=bad_row):
                self.assertEqual(
                    validate._task_contract_violations(
                        samples, {"ser-1": bad_row}, dataset_name="demo"
                    ),
                    ["ser-1"],
                )

    def test_role_projection_does_not_copy_answers_or_paths(self) -> None:
        with tempfile.TemporaryDirectory():
            label_path = Path(evaluate._write_classification_label_spec("GR"))
            self.addCleanup(label_path.unlink, missing_ok=True)
            label_payload = json.loads(label_path.read_text(encoding="utf-8"))
            self.assertEqual(label_payload["id"], "gr_default")
            self.assertNotIn("ground_truth", label_payload)

            prompt_path = Path(
                evaluate._write_slu_prompt_file(
                    [
                        {
                            "key": "slu-1",
                            "prompt": "Choose one.",
                            "choices": {"A": "yes", "B": "no"},
                            "ground_truth": "A",
                        }
                    ]
                )
            )
            self.addCleanup(prompt_path.unlink, missing_ok=True)
            prompt_payload = [
                json.loads(line)
                for line in prompt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                prompt_payload,
                [{"key": "slu-1", "prompt": "Choose one.", "choices": {"A": "yes", "B": "no"}}],
            )
            prompt_text = prompt_path.read_text(encoding="utf-8")
            self.assertNotIn("ground_truth", prompt_text)
            self.assertNotIn("/", prompt_text)

    def test_contract_helpers_reject_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            contract.normalize_label("SER", "fearful")
        self.assertEqual(contract.normalize_label("SER", 0), "neu")
        self.assertEqual(contract.normalize_label("GR", "1"), "woman")
        self.assertEqual(contract.normalize_answer(0), "0")
        with self.assertRaises(ValueError):
            contract.normalize_prediction("GR", {"label": "man", "score": True})
        with self.assertRaises(ValueError):
            contract.prompt_payload({"ground_truth": "A"})
        with self.assertRaises(ValueError):
            contract.normalize_answer("A\nB")
        with self.assertRaises(ValueError):
            contract.prompt_payload({"prompt": "Choose", "choices": {"ground_truth": "A"}})
        with self.assertRaises(ValueError):
            contract.normalize_answer({"ground_truth": "secret"})
        with self.assertRaises(ValueError):
            contract.normalize_answer(float("nan"))


if __name__ == "__main__":
    unittest.main()
