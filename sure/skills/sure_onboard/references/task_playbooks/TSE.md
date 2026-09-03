# Target Speaker Extraction

Use this playbook for models that take a speech mixture and an enrollment
recording and return the extracted target-speaker audio.

The inference contract is closed:

- input: `mixture_audio_path`, `enrollment_audio_path`, and harness-owned
  `output_path`;
- output: `prediction_audio` and optional matching `sample_id`;
- the model must not receive `reference_audio`, `reference_text`, or any
  ground-truth annotation.

The fixture must contain one to five keyed rows with three independent,
relative audio files: `mixture_audio`, `enrollment_audio`, and
`reference_audio`. `reference_text` is optional and is used only by semantic
evaluation. Direct and MCP runs must write distinct non-empty PCM WAV outputs
under the harness-assigned output directory.

Standalone evaluation consumes `samples_jsonl` with the roles
`prediction_audio`, `reference_audio`, optional `mixed_audio`, and optional
`enrollment_audio`. SI-SDR is the default metric; when `mixed_audio` is
present, the evaluator also reports SI-SDRi.
