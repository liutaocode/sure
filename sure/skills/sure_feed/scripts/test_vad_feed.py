#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_model_input as check_model_input_module  # noqa: E402
from sure_feed.fixture_registry import (  # noqa: E402
    io_contract_for_task,
    normalize_task,
    select_fixture_for_task,
)
from sure_feed.providers.base import (  # noqa: E402
    canonical_task,
    infer_task,
    synthesize_model_input,
    task_defaults,
)


class VADFeedTest(unittest.TestCase):
    def test_aliases_and_known_models_classify_as_vad(self) -> None:
        for alias in (
            "vad",
            "voice-activity-detection",
            "voice activity detection",
            "speech_activity_detection",
        ):
            with self.subTest(alias=alias):
                self.assertEqual(canonical_task(alias), "vad")
                self.assertEqual(normalize_task(alias), "vad")

        candidates = (
            {
                "source": "github",
                "model_id": "snakers4/silero-vad",
                "repo": "https://github.com/snakers4/silero-vad",
                "tags": ["voice-activity-detection", "onnx"],
                "model_card_text": "Silero VAD returns speech timestamps.",
            },
            {
                "source": "modelscope",
                "model_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "repo": "https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "tasks": ["speech-activity-detection"],
                "model_card_text": "FSMN voice activity detection for streaming audio.",
            },
            {
                "source": "huggingface",
                "model_id": "owner/standalone-silero-vad",
                "pipeline_tag": "audio-to-audio",
                "tasks": ["audio-to-audio"],
                "tags": ["silero-vad"],
            },
        )
        for candidate in candidates:
            with self.subTest(model_id=candidate["model_id"]):
                matched, task, score, evidence, source = infer_task(candidate, "auto")
                self.assertTrue(matched)
                self.assertEqual(task, "vad")
                self.assertGreaterEqual(score, 0.9)
                self.assertIn(source, {"tasks", "research_narrowing"})
                self.assertTrue(evidence)

    def test_vad_frontends_do_not_override_primary_model_tasks(self) -> None:
        cases = (
            (
                {
                    "source": "huggingface",
                    "model_id": "owner/asr-with-vad-frontend",
                    "pipeline_tag": "automatic-speech-recognition",
                    "tags": ["asr"],
                    "model_card_text": "ASR transcription with a VAD frontend.",
                },
                "asr",
            ),
            (
                {
                    "source": "github",
                    "model_id": "owner/kws-with-vad-frontend",
                    "tags": ["keyword-spotting"],
                    "model_card_text": "Wake word detection with an internal voice activity detector.",
                },
                "kws",
            ),
            (
                {
                    "source": "huggingface",
                    "model_id": "owner/diarization-with-vad",
                    "tags": ["speaker-diarization", "voice-activity-detection"],
                },
                "sd",
            ),
        )
        for candidate, expected in cases:
            with self.subTest(expected=expected):
                matched, task, *_ = infer_task(candidate, "auto")
                self.assertTrue(matched)
                self.assertEqual(task, expected)

        matched, task, *_ = infer_task(
            {"source": "github", "model_id": "owner/invader-audio-model"},
            "auto",
        )
        self.assertFalse(matched)
        self.assertEqual(task, "auto")

        readme_only = {
            "source": "github",
            "model_id": "owner/generic-audio-model",
            "model_card_text": (
                "The main model uses an internal Silero-VAD frontend and "
                "get_speech_timestamps before encoding."
            ),
        }
        for requested in ("auto", "vad"):
            with self.subTest(requested=requested):
                matched, task, *_ = infer_task(readme_only, requested)
                self.assertFalse(matched)
                self.assertEqual(task, requested)

        competing_metadata = {
            "source": "huggingface",
            "model_id": "owner/general-speech-model",
            "tasks": ["keyword-spotting", "voice-activity-detection"],
            "model_card_text": "Uses VAD before keyword spotting.",
        }
        matched, task, *_ = infer_task(competing_metadata, "vad")
        self.assertFalse(matched)
        self.assertEqual(task, "vad")

    def test_registry_preserves_speech_and_silence_reference_rows(self) -> None:
        fixture, contract, issues, evidence = select_fixture_for_task(
            "vad", {"description": "voice activity detection"}
        )
        self.assertEqual(issues, [])
        self.assertIsNotNone(fixture)
        assert fixture is not None
        self.assertEqual(fixture["sample_count"], 3)
        self.assertEqual(fixture["fixture_index"], "fixtures/tasks/vad/README.md")
        self.assertEqual(fixture["samples"][-1]["speech_segments"], [])
        self.assertEqual(contract, io_contract_for_task("vad"))
        self.assertEqual(contract["primary_field"], "speech_segments")
        self.assertEqual(contract["nonempty_fields"], [])
        self.assertIn("fixture", {item.get("model_input_field") for item in evidence})

    def test_synthesized_vad_input_is_complete_and_canonical(self) -> None:
        readme = """# Silero VAD

CPU voice activity detection returning speech timestamps.

```bash
pip install silero-vad
```
```python
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()
speech_segments = get_speech_timestamps(wav, model)
```
"""
        model_id = "snakers4/silero-vad"
        model_input, weak_fields, _evidence = synthesize_model_input(
            {
                "source": "github",
                "model_id": model_id,
                "repo": "https://github.com/snakers4/silero-vad",
                "commit": "abcdef0",
                "weights_source": "release_or_pypi",
                "library_name": "silero_vad",
                "model_card_text": readme,
                "load_test": "model = load_silero_vad()",
            },
            "vad",
        )
        self.assertEqual(weak_fields, [])
        self.assertEqual(model_input["task_type"], "vad")
        self.assertEqual(model_input["io_contract"], io_contract_for_task("vad"))
        self.assertEqual(
            check_model_input_module.validate_model_input(
                model_input, model_id, "model_inputs[0]"
            ),
            [],
        )
        defaults = task_defaults("vad")
        self.assertEqual(defaults["io_contract"], model_input["io_contract"])
        self.assertIn(".detect_speech(", defaults["infer_test"])


if __name__ == "__main__":
    unittest.main()
