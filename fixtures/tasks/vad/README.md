# VAD Fixture Index

VAD fixtures validate voice activity detection wrappers on bounded PCM WAV
inputs. Use:

```text
fixtures/tasks/vad/librispeech_vad_smoke/
```

The set contains two LibriSpeech-derived recordings with an exact 0.4-second
zero-valued gap and one deterministic pure-silence control. `gt.jsonl` stores
reference `speech_segments`; those annotations are evaluation roles and must
never be passed to model inference.

The canonical model output is:

```json
{"speech_segments":[{"start":0.0,"end":3.275}],"frame_scores":[{"start":0.0,"end":9.77,"score":0.8}]}
```

`speech_segments` is required and may be empty only for a byte-level
pure-silence smoke sample. `frame_scores` is optional; when present it must be
non-empty, ordered, non-overlapping, gap-free, and cover the complete WAV
timebase from zero through its measured duration. All times and scores must be
finite, and scores must be within `[0, 1]`.

See `librispeech_vad_smoke/provenance.json` for the locked upstream revision,
source utterances, deterministic transforms, and output hashes.
