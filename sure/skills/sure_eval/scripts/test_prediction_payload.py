#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_predictions_via_server as gp  # noqa: E402


def _write_pcm_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    return path


class AsrPayloadNormalizationTests(unittest.TestCase):
    def test_single_element_text_list_is_unwrapped(self) -> None:
        prediction, normalized = gp._normalize_prediction_payload(
            {"text": [" 二零二二年冬奥会在北京举行"]}, task="ASR"
        )
        self.assertEqual(prediction, " 二零二二年冬奥会在北京举行")
        self.assertEqual(normalized, {"text": " 二零二二年冬奥会在北京举行"})

    def test_single_element_text_tuple_is_unwrapped(self) -> None:
        prediction, normalized = gp._normalize_prediction_payload({"text": ("hello",)}, task="S2TT")
        self.assertEqual(prediction, "hello")
        self.assertEqual(normalized, {"text": "hello"})

    def test_nested_prediction_text_list_is_unwrapped(self) -> None:
        prediction, _ = gp._normalize_prediction_payload(
            {"prediction": {"text": ["nested"]}}, task="ASR"
        )
        self.assertEqual(prediction, "nested")

    def test_plain_string_text_is_untouched(self) -> None:
        prediction, normalized = gp._normalize_prediction_payload({"text": "严浩出演的电影有什么"}, task="ASR")
        self.assertEqual(prediction, "严浩出演的电影有什么")
        self.assertEqual(normalized, {"text": "严浩出演的电影有什么"})

    def test_empty_text_list_stays_empty(self) -> None:
        prediction, normalized = gp._normalize_prediction_payload({"text": []}, task="ASR")
        self.assertEqual(prediction, "")
        self.assertEqual(normalized, {"text": ""})


class KwsToolArgumentTests(unittest.TestCase):
    def _arguments(
        self,
        sample: dict[str, object],
        tool_args: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return gp._build_tool_arguments(
            repo_root=Path("/repo"),
            sample=sample,
            task="KWS",
            language="zh",
            argument_name="audio_path",
            audio_path=Path("/audio.wav"),
            output_audio_dir=Path("/unused"),
            tool_args=tool_args,
        )

    def test_forwards_sample_keywords_and_optional_threshold_without_language(self) -> None:
        arguments = self._arguments({"keywords": ["wake", "hello"], "threshold": 0.5})
        self.assertEqual(
            arguments,
            {
                "audio_path": "/audio.wav",
                "keywords": ["wake", "hello"],
                "threshold": 0.5,
            },
        )
        self.assertTrue(gp._is_dynamic_argument_key("keyword"))
        self.assertTrue(gp._is_dynamic_argument_key("keywords"))
        self.assertTrue(gp._is_dynamic_argument_key("threshold"))

    def test_fixed_keyword_model_does_not_require_sample_keywords(self) -> None:
        self.assertEqual(self._arguments({}), {"audio_path": "/audio.wav"})

    def test_invalid_sample_keywords_are_rejected(self) -> None:
        invalid_keywords = [None, "", "  ", [], ["wake", ""], ["wake", 7], 7]
        for keywords in invalid_keywords:
            with self.subTest(keywords=keywords), self.assertRaisesRegex(ValueError, "keywords"):
                self._arguments({"keywords": keywords})

    def test_invalid_sample_thresholds_are_rejected(self) -> None:
        invalid_thresholds = [None, True, "0.5", 0.4, 0.6, float("nan"), float("inf")]
        for threshold in invalid_thresholds:
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                self._arguments({"threshold": threshold})

    def test_nonstandard_tool_argument_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "formal operating threshold"):
            self._arguments({}, {"threshold": 0.4})

    def test_explicit_tool_arguments_override_sample_values(self) -> None:
        arguments = self._arguments(
            {"keywords": "sample", "threshold": 0.5},
            {"keywords": "fixed", "threshold": 0.5},
        )
        self.assertEqual(arguments["keywords"], "fixed")
        self.assertEqual(arguments["threshold"], 0.5)
        self.assertNotIn("language", arguments)

    def test_sample_kws_fields_are_recorded_as_dynamic_arguments(self) -> None:
        arguments = self._arguments({"keywords": "wake", "threshold": 0.5})
        dynamic_fields = {
            key for key in arguments if gp._is_dynamic_argument_key(key)
        }
        status: dict[str, object] = {}
        gp._update_generation_observations(
            status,
            argument_keys_seen=set(arguments),
            dynamic_argument_fields=dynamic_fields,
            raw_response_types=set(),
            raw_response_keys=set(),
        )
        policy = status["generation"]["argument_policy"]
        self.assertEqual(
            policy["dynamic_argument_fields"],
            ["audio_path", "keywords", "threshold"],
        )


class KwsPayloadNormalizationTests(unittest.TestCase):
    def test_operating_threshold_boundaries_are_valid(self) -> None:
        gp._normalize_prediction_payload(
            {"detected": True, "keyword": "wake", "score": 0.5}, task="KWS"
        )
        gp._normalize_prediction_payload(
            {"detected": False, "keyword": None, "score": 0.499}, task="KWS"
        )

    def test_direct_fields_are_preserved_without_truthiness_loss(self) -> None:
        projection, normalized = gp._normalize_prediction_payload(
            {"detected": False, "keyword": None, "score": 0},
            task="KWS",
        )
        self.assertEqual(
            normalized,
            {"detected": False, "keyword": None, "score": 0},
        )
        self.assertEqual(json.loads(projection), normalized)

    def test_nested_prediction_preserves_events_as_evidence(self) -> None:
        events = [{"keyword": "wake", "score": 0.9}]
        projection, normalized = gp._normalize_prediction_payload(
            {
                "prediction": {
                    "detected": True,
                    "keyword": "wake",
                    "score": 0.9,
                    "events": events,
                }
            },
            task="KWS",
        )
        self.assertEqual(normalized["events"], events)
        self.assertEqual(json.loads(projection), normalized)

    def test_null_negative_score_is_preserved_in_nonempty_projection(self) -> None:
        projection, normalized = gp._normalize_prediction_payload(
            {"detected": False, "keyword": None, "score": None},
            task="KWS",
        )
        self.assertTrue(projection)
        self.assertIsNone(normalized["score"])
        self.assertEqual(json.loads(projection), normalized)

    def test_event_only_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing direct field"):
            gp._normalize_prediction_payload(
                {"events": [{"keyword": "wake", "score": 0.9}]},
                task="KWS",
            )

    def test_scalar_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            gp._normalize_prediction_payload("0.9", task="KWS")

    def test_malformed_direct_field_types_are_rejected(self) -> None:
        invalid_payloads = [
            {"detected": "false", "keyword": None, "score": 0.1},
            {"detected": False, "keyword": 7, "score": 0.1},
            {"detected": False, "keyword": None, "score": True},
            {"detected": False, "keyword": None, "score": float("nan")},
            {"detected": False, "keyword": None, "score": 0.1, "events": {}},
            {"detected": True, "keyword": None, "score": 0.9},
            {"detected": True, "keyword": "  ", "score": 0.9},
            {"detected": True, "keyword": "wake", "score": None},
            {"detected": False, "keyword": "wake", "score": 0.1},
            {"detected": True, "keyword": "wake", "score": 0.49},
            {"detected": False, "keyword": None, "score": 0.5},
            {"detected": False, "keyword": None, "score": -0.1},
            {"detected": True, "keyword": "wake", "score": 1.1},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                gp._normalize_prediction_payload(payload, task="KWS")

    def test_macro_recall_generation_requires_rejected_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a score"):
            gp._normalize_prediction_payload(
                {"detected": False, "keyword": None, "score": None},
                task="KWS",
                kws_require_score=True,
            )


class SePayloadTests(unittest.TestCase):
    def test_builder_materializes_run_local_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "predictions" / "audio" / "dataset"
            arguments = gp._build_tool_arguments(
                repo_root=Path(temporary),
                sample={"key": "utt/1"},
                task="SE",
                language="en",
                argument_name="audio_path",
                audio_path=Path(temporary) / "noisy.wav",
                output_audio_dir=output_dir,
                tool_args={"output_path": "/outside.wav", "strength": 0.5},
            )
        self.assertEqual(arguments["audio_path"], str(Path(temporary) / "noisy.wav"))
        self.assertEqual(arguments["strength"], 0.5)
        self.assertEqual(arguments["output_path"], str((output_dir / "utt_1.wav").resolve()))

    def test_builder_rejects_lexical_parent_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "predictions" / "audio" / "dataset" / ".." / "escape"
            with self.assertRaisesRegex(ValueError, "must not traverse"):
                gp._se_run_output_path(output_dir, "utt")

    def test_builder_rejects_symlinked_output_root_or_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            linked_root = root / "linked-root"
            linked_root.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                gp._se_run_output_path(linked_root, "utt")

            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                gp._se_run_output_path(linked_parent / "dataset", "utt")

    def test_normalizer_preserves_audio_aliases_and_requires_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = _write_pcm_wav(Path(temporary) / "enhanced.wav")
            projection, normalized = gp._normalize_prediction_payload(
                {"enhanced_audio": str(output)},
                task="SE",
                expected_audio_output=output,
            )
        self.assertEqual(projection, str(output.resolve()))
        self.assertEqual(
            normalized,
            {"audio_path": str(output.resolve()), "enhanced_audio": str(output.resolve())},
        )

    def test_normalizer_rejects_missing_empty_non_wav_or_nonlocal_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.wav"
            empty.write_bytes(b"")
            non_wav = root / "not-a-wave.wav"
            non_wav.write_bytes(b"not a wave")
            other = root / "other.wav"
            _write_pcm_wav(other)
            cases = [
                ({}, None),
                ({"audio_path": str(empty)}, empty),
                ({"audio_path": str(non_wav)}, non_wav),
                ({"audio_path": str(other)}, root / "expected.wav"),
            ]
            for payload, expected in cases:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    gp._normalize_prediction_payload(
                        payload,
                        task="SE",
                        expected_audio_output=expected,
                    )
            with self.assertRaisesRegex(ValueError, "JSON object"):
                gp._normalize_prediction_payload(str(other), task="SE")

    def test_normalizer_rejects_lexical_alias_of_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = _write_pcm_wav(root / "audio" / "enhanced.wav")
            lexical_alias = root / "audio" / "nested" / ".." / "enhanced.wav"
            with self.assertRaisesRegex(ValueError, "differs from run-local output_path"):
                gp._normalize_prediction_payload(
                    {"audio_path": str(lexical_alias)},
                    task="SE",
                    expected_audio_output=expected,
                )

    def test_normalizer_rejects_returned_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = _write_pcm_wav(root / "target.wav")
            expected = root / "audio" / "enhanced.wav"
            expected.parent.mkdir()
            expected.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "real non-empty file"):
                gp._normalize_prediction_payload(
                    {"audio_path": str(expected)},
                    task="SE",
                    expected_audio_output=expected,
                )


if __name__ == "__main__":
    unittest.main()
