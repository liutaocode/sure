# Speech Enhancement Onboarding

Use this playbook for canonical task `se`. The accepted boundary aliases are
`speech_enhancement`, `speech-enhancement`, `acoustic-noise-suppression`,
`acoustic_noise_suppression`, and `acoustic noise suppression`. A generic `audio-to-audio` provider tag is not
enough to classify a model as SE; require explicit enhancement, denoising, or
noise-suppression evidence.

## Tool Contract

Expose the MCP tool `enhance_speech` with this input:

```json
{"audio_path":"fixture/se/noisy.wav","output_path":"artifacts/outputs/enhanced.wav"}
```

`audio_path` is required and `output_path` is optional. Return a JSON object
with `audio_path`. The returned path must identify a real, non-empty audio
file. Do not return an in-memory tensor, waveform summary, or clean reference
path as the enhanced result.

## Fixture Roles

Use `fixtures/tasks/se/fleurs_noise_smoke/gt.jsonl`. Every row carries two
distinct relative files:

- `audio`: noisy model input;
- `reference_audio`: clean evaluation reference.

Stage both files into the model-local fixture and preserve both roles in
`fixture_manifest.json`. Inference receives only the noisy `audio_path`; never
pass `reference_audio` to the model or copy it into the model input under a
different name.

## Validation

Run every bounded fixture row. The Harness assigns each `output_path` below
`artifacts/outputs`; a model-provided path cannot override it. `validate.py`
requires the returned path to identify that exact, non-symlink, non-empty PCM
WAV and rejects files that alias the noisy or clean input. Staging copies every
validated output into the model bundle, rewrites sample evidence to portable
`artifacts/outputs/...` paths, and includes each file in the final hash manifest.

Formal evaluation maps the three roles as `noisy_audio`, `reference_audio`, and
`enhanced_audio`. Full-reference metrics include SI-SDR, STOI, and PESQ;
no-reference quality may include DNSMOS. Onboarding validates execution and
roles only and does not claim benchmark quality.
