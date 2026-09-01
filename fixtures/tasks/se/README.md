# Speech Enhancement Fixture Index

Use this index for speech enhancement and noise-suppression models. Copy the
bounded noisy/clean pair into:

```text
sure/models/<model>/fixture/se/
```

## Shared Fixture Set

Use `fleurs_noise_smoke/`. It contains one 8.16-second mono 16 kHz pair:

- `noisy.wav`: deterministic white-noise mixture at 10 dB target SNR.
- `clean.wav`: PCM16 conversion of the public FLEURS source utterance.
- `gt.jsonl`: explicit `audio` (noisy) and `reference_audio` (clean) roles.
- `provenance.json`: source revision, CC-BY-4.0 terms, transforms, and hashes.

The fixture is for smoke and integration testing, not benchmark claims.
Full-reference evaluation must never substitute `audio` for
`reference_audio`; doing so would score the noisy input against itself.

## Adapter Contract

Expose `enhance_speech`:

```json
{"audio_path":"noisy.wav","output_path":"enhanced.wav"}
```

Return `{"audio_path":"enhanced.wav"}`. The output must be the Harness-assigned,
non-symlink, non-empty PCM WAV in the run's writable output directory; it must
not alias either input. Formal evaluation maps the
roles to standalone `sure-evaluation` as `noisy_audio`, `reference_audio`, and
`enhanced_audio`.
