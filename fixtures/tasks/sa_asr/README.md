# SA-ASR Fixture Index

SA-ASR fixtures validate speaker-attributed ASR outputs during deployment smoke
tests. The model should return speaker time segments with transcript text.

## Shared Fixture Set

Use:

```text
fixtures/tasks/sa_asr/librispeech_2spk_smoke/
```

Source: LibriSpeech `test-clean`, using short utterances from two different
speakers per smoke sample.

Files:

- `gt.jsonl`
- `provenance.json`
- `librispeech_2spk_001.wav`
- `librispeech_2spk_002.wav`
- `librispeech_2spk_003.wav`

`gt.jsonl` is the primary deployment-validation input. Each row contains one
two-speaker recording, an audio path, and speaker-attributed transcript
segments:

```json
{"key":"librispeech_2spk_001","audio":"librispeech_2spk_001.wav","segments":[{"speaker":"spk1","start":0.0,"end":3.275,"text":"STUFF IT INTO YOU HIS BELLY COUNSELLED HIM"}]}
```

Speaker labels are local to each row and normalized to `spk1` and `spk2`.
The provenance manifest fixes the six source utterances and the deterministic
0.4-second silence concatenation used to build each WAV.
