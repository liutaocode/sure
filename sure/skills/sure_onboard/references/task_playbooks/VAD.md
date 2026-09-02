# VAD Model Onboarding

Use this playbook only for standalone voice or speech activity detection
models, including Silero VAD and FSMN-VAD. An ASR, KWS, or diarization model
that merely uses VAD internally keeps its primary task.

## Canonical Surface

- Task: `vad`
- MCP tool and wrapper method: `detect_speech`
- Input: `audio_path`
- Required output: `speech_segments: [{start, end}, ...]`
- Optional output: `frame_scores: [{start, end, score}, ...]`

Output objects are closed. Do not expose model-native tensors, raw/debug data,
reference annotations, filesystem paths, or URIs. Every number must be finite;
times stay within the measured PCM WAV duration. Speech segments are ordered
and non-overlapping. When `frame_scores` is present it is non-empty, begins at
zero, has neither gaps nor overlaps, and ends at the WAV duration; every score
is within `[0, 1]`.

## Fixture Boundary

Use `fixtures/tasks/vad/README.md`. Validation runs exactly 1-5 rows. The
fixture source and staged tree may not contain symlinks. Stage only `gt.jsonl`
and audio files referenced by its rows; do not stage provenance, expected
output sidecars, or unrelated files. Audio must be readable non-empty PCM WAV,
and declared duration/sample rate must match the file.

`speech_segments` in `gt.jsonl` is reference-only. Pass only `audio_path` and
approved parameters from `SURE_VALIDATE_PROTOCOL_JSON` to `detect_speech`.
Never pass reference segments, labels, duration, sample rate, or silence flags
to the wrapper. Empty reference and prediction segments are accepted only for
byte-level pure-silence smoke audio.

## Validation Checklist

- `MODEL_INPUT`, model spec, config, wrapper, and server all use `vad` and
  `detect_speech`.
- Every fixture key is predicted exactly once and in source order.
- `sample_output.json` equals the first row output in `sample_outputs.jsonl`.
- Every output is rechecked against the actual staged WAV before bundling.
- Optional frame scores meet full-timebase coverage so Eval may enable AUC-ROC.
- F1, `p_fa`, `p_miss`, and DCF remain Eval concerns; Onboard proves only the
  callable model and structured output contract.
