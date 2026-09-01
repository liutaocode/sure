# KWS Fixture Index

Use this index for keyword spotting and wake-word detection models. Copy selected
positive and negative samples into:

```text
sure/models/<model>/fixture/kws/
```

## Shared Fixture Set

Use:

```text
fixtures/tasks/kws/wenwen_smoke/kws/
```

Source:

```text
src/sure_eval/models/daydream_factory__keyword-spot-fsmn-ctc-wenwen/fixture/kws/
```

## Included Source

| Source | Files | Notes |
|--------|-------|-------|
| `src/sure_eval/models/daydream_factory__keyword-spot-fsmn-ctc-wenwen/fixture/kws/` | `audio/*.wav`, `gt.jsonl` | Positive and negative wake-word samples. |

## Expected Model-Local Layout

```text
sure/models/<model>/fixture/kws/
├── gt.jsonl
└── audio/
    ├── positive_*.wav
    └── negative_*.wav
```

`gt.jsonl` must contain 2 to 5 rows with unique `key` values, safe relative
audio paths, a non-empty `keywords` string/list, and an explicit positive or
negative annotation. The staged form records boolean `expected_detected`,
`expected_keyword` (`null` for negatives), and positive finite `duration`.

## Validation Metrics

Task-formatted namespace:

```text
src/sure_eval/evaluation/tasks/kws/
```

Formal evaluation uses the standalone `sure-evaluation` KWS route with
`reference_jsonl + sample_output`. `sample_output` is a JSON array, or an
object with a `rows` array, whose entries have this shape:

```json
{"key":"sample-id","result":{"detected":false,"keyword":null,"score":null}}
```

The model adapter must emit the direct `detected`, `keyword`, and `score`
summary. An untyped event list is evidence only and is not converted into a
detection by the Harness.

The formal operating threshold is `0.5`. Scores, when present, must be finite
values in `[0,1]`, and `detected` must agree with `score >= 0.5`. Accuracy-only
evaluation accepts a correctly rejected sample with `score: null`; macro-recall
and DET evaluation require a score for every sample.

Validation should include at least one positive and one negative sample.

## Related Tool-Agent Memory

- `docs/agents/model_tool_agent/task_playbooks/KWS.md`
- `docs/agents/model_tool_agent/contracts/fixture_policy.md`
- `docs/agents/model_tool_agent/contracts/minimal_validation.md`
