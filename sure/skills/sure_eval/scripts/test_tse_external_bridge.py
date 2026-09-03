#!/usr/bin/env python3
"""Regression tests for TSE generation and standalone-role conversion."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_predictions as ep  # noqa: E402
import generate_predictions_via_server as gp  # noqa: E402
import validate_prediction_files as vp  # noqa: E402
import materialize_predictions_template as mt  # noqa: E402
import resolve_eval_input as rei  # noqa: E402


def write_wav(path: Path, values: list[int] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = values or [1, 2, 3, 4]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(v.to_bytes(2, "little", signed=True) for v in values))
    return path


class FakeDatasetManager:
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path

    def normalize_dataset_name(self, name: str) -> str:
        return name

    def get_jsonl_path(self, _: str) -> Path:
        return self.jsonl_path

    def download_and_convert(self, _: str) -> Path:
        raise AssertionError("fixture JSONL is already present")


class FakeSota:
    def get_metric(self, *_args: object, **_kwargs: object) -> None:
        return None

    def get_baseline(self, *_args: object, **_kwargs: object) -> None:
        return None

    def calculate_rps(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {"status": "missing_baseline"}


class TSEExternalBridgeTest(unittest.TestCase):
    def test_tse_aliases_and_default_si_sdr(self) -> None:
        for alias in (
            "tse",
            "target-speaker-extraction",
            "speaker extraction",
            "target_speaker_extraction_model",
            "target speaker",
            "target voice separation",
        ):
            self.assertEqual(ep._normalize_task(alias), "TSE")
            self.assertEqual(gp._normalize_task(alias), "TSE")
        self.assertEqual(rei._fallback_default_metrics("TSE", "en"), ["si_sdr"])
        self.assertEqual(mt._default_metric("TSE", "en"), "si_sdr")
        self.assertEqual(ep._metric_task_hint(["tse_wer"]), "TSE")
        self.assertEqual(ep._effective_audio_task("TSE", "TSE"), "TSE")

    def test_generation_arguments_are_exact_and_never_use_reference_as_enrollment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixture = write_wav(root / "mixture.wav")
            enrollment = write_wav(root / "enrollment.wav", [5, 6, 7, 8])
            output_dir = root / "predictions" / "audio" / "dataset"
            args = gp._build_tool_arguments(
                repo_root=root,
                sample={
                    "key": "utt-1",
                    "path": str(mixture),
                    "enrollment_audio": str(enrollment),
                    "reference_audio": str(root / "clean-reference.wav"),
                    "ground_truth": "must-not-leak",
                },
                task="target-speaker-extraction",
                language="en",
                argument_name="audio_path",
                audio_path=mixture,
                output_audio_dir=output_dir,
            )
            self.assertEqual(
                set(args), {"mixture_audio_path", "enrollment_audio_path", "output_path"}
            )
            self.assertEqual(args["mixture_audio_path"], str(mixture))
            self.assertEqual(args["enrollment_audio_path"], str(enrollment))
            self.assertNotIn("reference_audio", args)
            with self.assertRaisesRegex(ValueError, "enrollment"):
                gp._build_tool_arguments(
                    repo_root=root,
                    sample={"key": "missing", "path": str(mixture), "reference_audio": str(enrollment)},
                    task="TSE",
                    language="en",
                    argument_name="audio_path",
                    audio_path=mixture,
                    output_audio_dir=output_dir,
                )

    def test_prediction_normalization_binds_output_and_rejects_reference_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = write_wav(root / "audio" / "dataset" / "utt.wav")
            reference = write_wav(root / "reference.wav", [5, 6])
            normalized = gp._normalize_prediction_payload(
                {"prediction_audio": str(output)},
                task="tse",
                expected_audio_output=output,
                forbidden_inputs=(reference,),
            )
            self.assertEqual(normalized[1], {"prediction_audio": str(output)})
            with self.assertRaisesRegex(ValueError, "forbidden|unapproved"):
                gp._normalize_prediction_payload(
                    {"prediction_audio": str(output), "reference_audio": str(reference)},
                    task="TSE",
                    expected_audio_output=output,
                    forbidden_inputs=(reference,),
                )
            with self.assertRaisesRegex(ValueError, "must equal"):
                gp._normalize_prediction_payload(
                    {"prediction_audio": str(reference)},
                    task="TSE",
                    expected_audio_output=output,
                    forbidden_inputs=(reference,),
                )

    def test_samples_jsonl_bridge_preserves_all_roles_and_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixture = write_wav(root / "mixture.wav")
            enrollment = write_wav(root / "enrollment.wav", [5, 6, 7])
            reference = write_wav(root / "reference.wav", [9, 10, 11])
            prediction = write_wav(root / "predictions" / "audio" / "utt.wav", [3, 4, 5])
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "key": "utt-1",
                        "task": "TSE",
                        "language": "en",
                        "mixture_audio": str(mixture),
                        "enrollment_audio": str(enrollment),
                        "reference_audio": str(reference),
                        "reference_text": "hello",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            structured_path = root / "predictions" / "predictions.jsonl"
            structured_path.parent.mkdir(parents=True, exist_ok=True)
            structured_path.write_text(
                json.dumps(
                    {
                        "key": "utt-1",
                        "sample_id": "utt-1",
                        "task": "TSE",
                        "dataset": "fixture",
                        "prediction": {"prediction_audio": str(prediction)},
                        "normalized_prediction": str(prediction),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "external" / "samples.jsonl"
            ep._write_external_audio_samples_jsonl(
                task="TSE",
                dataset_jsonl_path=dataset,
                samples=[json.loads(dataset.read_text())],
                structured_predictions={"utt-1": json.loads(structured_path.read_text())},
                structured_prediction_path=structured_path,
                output_path=output,
                required_roles={"prediction_audio", "reference_audio", "enrollment_audio"},
            )
            row = json.loads(output.read_text())
            self.assertEqual(
                set(row),
                {
                    "sample_id",
                    "prediction_audio",
                    "reference_audio",
                    "enrollment_audio",
                    "reference_text",
                    "language",
                    "mixed_audio",
                    "metadata",
                },
            )
            self.assertEqual(Path(row["prediction_audio"]).resolve(), prediction.resolve())
            self.assertEqual(Path(row["mixed_audio"]).resolve(), mixture.resolve())
            self.assertEqual(Path(row["enrollment_audio"]).resolve(), enrollment.resolve())
            self.assertEqual(Path(row["reference_audio"]).resolve(), reference.resolve())

            # No mixed/reference text is still a valid signal-only row; the
            # required model enrollment and clean target remain enforced.
            no_optional = json.loads(dataset.read_text())
            no_optional.pop("reference_text")
            no_optional.pop("mixture_audio")
            output2 = root / "external" / "samples2.jsonl"
            ep._write_external_audio_samples_jsonl(
                task="TSE",
                dataset_jsonl_path=dataset,
                samples=[no_optional],
                structured_predictions={"utt-1": json.loads(structured_path.read_text())},
                structured_prediction_path=structured_path,
                output_path=output2,
                required_roles={"prediction_audio", "reference_audio", "enrollment_audio"},
            )
            row2 = json.loads(output2.read_text())
            self.assertNotIn("mixed_audio", row2)
            self.assertNotIn("reference_text", row2)

    def test_structured_prediction_validation_rejects_leakage_and_accepts_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = write_wav(root / "predictions" / "audio" / "utt.wav")
            reference = write_wav(root / "reference.wav", [2, 3])
            sample = {
                "key": "utt-1",
                "sample_id": "utt-1",
                "task": "TSE",
                "dataset": "fixture",
                "reference_audio": str(reference),
                "enrollment_audio": str(root / "enrollment.wav"),
            }
            valid = {
                "key": "utt-1",
                "sample_id": "utt-1",
                "task": "TSE",
                "dataset": "fixture",
                "prediction": {"prediction_audio": str(prediction), "sample_id": "utt-1"},
                "normalized_prediction": str(prediction),
            }
            self.assertEqual(vp._task_contract_violations([sample], {"utt-1": valid}, base_dir=root / "predictions"), [])
            leaked = dict(valid)
            leaked["prediction"] = {"prediction_audio": str(prediction), "reference_audio": str(reference)}
            self.assertEqual(vp._task_contract_violations([sample], {"utt-1": leaked}, base_dir=root / "predictions"), ["utt-1"])

            target_alias_sample = dict(sample)
            target_alias_sample.pop("reference_audio")
            target_alias_sample["target_audio"] = str(reference)
            target_alias_leak = dict(valid)
            target_alias_leak["prediction"] = {"prediction_audio": str(reference)}
            self.assertEqual(
                vp._task_contract_violations(
                    [target_alias_sample],
                    {"utt-1": target_alias_leak},
                    base_dir=root / "predictions",
                ),
                ["utt-1"],
            )

    def test_external_bridge_allows_signal_only_rows_without_optional_enrollment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixture = write_wav(root / "mixture.wav")
            reference = write_wav(root / "reference.wav", [2, 3])
            prediction = write_wav(root / "predictions" / "audio" / "utt.wav")
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "key": "utt-1",
                        "task": "TSE",
                        "language": "en",
                        "path": str(mixture),
                        "reference_audio": str(reference),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            structured_path = root / "predictions" / "predictions.jsonl"
            structured_path.parent.mkdir(parents=True, exist_ok=True)
            structured_path.write_text(
                json.dumps(
                    {
                        "key": "utt-1",
                        "task": "TSE",
                        "prediction": {"prediction_audio": str(prediction)},
                        "normalized_prediction": str(prediction),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ep._write_external_audio_samples_jsonl(
                task="TSE",
                dataset_jsonl_path=dataset,
                samples=[json.loads(dataset.read_text())],
                structured_predictions={"utt-1": json.loads(structured_path.read_text())},
                structured_prediction_path=structured_path,
                output_path=root / "samples.jsonl",
                required_roles={"prediction_audio", "reference_audio"},
            )
            row = json.loads((root / "samples.jsonl").read_text())
            self.assertNotIn("enrollment_audio", row)

    def test_external_bridge_accepts_target_audio_reference_alias(self) -> None:
        self.assertEqual(
            ep._sample_tse_mixture_audio({"audio_path": "mixture.wav"}),
            "mixture.wav",
        )
        self.assertEqual(
            ep._sample_tse_mixture_audio({"mixed_audio_path": "mixed.wav"}),
            "mixed.wav",
        )
        self.assertEqual(
            ep._sample_tse_enrollment_audio({"enrollment_path": "enroll.wav"}),
            "enroll.wav",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixture = write_wav(root / "mixture.wav")
            target = write_wav(root / "target.wav", [2, 3])
            prediction = write_wav(root / "predictions" / "audio" / "utt.wav")
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "key": "utt-1",
                        "task": "TSE",
                        "language": "en",
                        "path": str(mixture),
                        "target_audio": str(target),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            structured_path = root / "predictions" / "predictions.jsonl"
            structured_path.parent.mkdir(parents=True, exist_ok=True)
            structured = {
                "key": "utt-1",
                "task": "TSE",
                "prediction": {"prediction_audio": str(prediction)},
                "normalized_prediction": str(prediction),
            }
            structured_path.write_text(json.dumps(structured) + "\n", encoding="utf-8")
            output = root / "samples.jsonl"
            ep._write_external_audio_samples_jsonl(
                task="TSE",
                dataset_jsonl_path=dataset,
                samples=[json.loads(dataset.read_text())],
                structured_predictions={"utt-1": structured},
                structured_prediction_path=structured_path,
                output_path=output,
                required_roles={"prediction_audio", "reference_audio"},
            )
            row = json.loads(output.read_text())
            self.assertEqual(Path(row["reference_audio"]).resolve(), target.resolve())


if __name__ == "__main__":
    unittest.main()
