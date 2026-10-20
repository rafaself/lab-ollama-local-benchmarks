# Local Ollama benchmark

Run: 2026-10-20T16:19:22.235179-03:00 to 2026-10-20T16:20:53.131841-03:00.

Hardware: Ryzen 7 6800H (8 cores / 16 threads), RTX 3060 Laptop GPU with 6 GiB VRAM, approximately 15 GiB usable system RAM. Ollama 0.40.6.

Status: **complete exploratory benchmark**. This expanded repository report derives from the original [source report](source-report.md). The original [results](results.json) and [script](ollama-benchmark.py) are unchanged.

Evidence: [configuration](configuration.json), [hardware/provenance](hardware.json), [excluded attempts](excluded-attempts/README.md), [checksums](SHA256SUMS).

## Findings

These measurements compare response speed, memory use, and six small synthetic document-question checks. Thinking is enabled for both models. Token throughput includes reasoning tokens and answer tokens; it is not a measurement of final-answer tokens alone. These tests do not measure training, embeddings, retrieval quality, or broad model accuracy.

| Model | Short output tokens/s | Long output tokens/s | Short first token | Long first token | Loaded model allocation in VRAM | Document checks |
|---|---:|---:|---:|---:|---:|---:|
| qwen3:4b | 93.4 | 83.9 | 0.258 s | 0.879 s | 2.96 GiB | 6/6 |
| qwen3:0.6b | 302.1 | 234.8 | 0.197 s | 0.266 s | 0.87 GiB | 4/6 |

The 0.6B model generated tokens 3.23 times as fast on short prompts and 2.80 times as fast on long prompts.

Recommendation: use qwen3:4b as the main answer-generation model for this RAG lab. It fit entirely in GPU memory at the tested context and passed all six checks. Use 0.6B when quick iteration is the priority; the observed failures show why its output needs closer checking.

## Additional measurements

| Model | Short request wall time | Long request wall time | First-load duration | Peak whole-GPU memory used | Peak sampled GPU utilization | Peak sampled temperature |
|---|---:|---:|---:|---:|---:|---:|
| qwen3:4b | 3.01 s | 3.93 s | 2.73 s | 3.12 GiB | 100% | 85 C |
| qwen3:0.6b | 1.04 s | 1.35 s | 2.38 s | 1.04 GiB | 96% | 86 C |

Loaded model allocation is reported by `/api/ps`; peak whole-GPU memory is sampled using `nvidia-smi` and includes any other GPU activity. Both models reported their full model allocation in VRAM. First-load durations are not controlled disk-cold measurements and may include CUDA initialization, disk/OS caching, and runner startup. They are excluded from the warm throughput measurements.

## Document-question checks

Each check uses a self-contained synthetic document set with a predetermined JSON answer. Exact parsed JSON equality is required; JSON fences are accepted. Extra or missing fields fail. No schema-constrained decoding was used.

| Check | qwen3:4b | qwen3:0.6b |
|---|---|---|
| lookup | PASS | PASS |
| exception | PASS | PASS |
| dated_override | PASS | PASS |
| multiple_conditions | PASS | FAIL |
| calculation | PASS | PASS |
| abstention | PASS | FAIL |

The six checks cover direct lookup, an exception to a general policy, a dated policy override, multiple eligibility conditions, a billing calculation, and abstaining when a fact is absent. This is a small one-shot diagnostic, not a statistically reliable general quality ranking. Each quality request allows up to 2,048 output tokens, including reasoning. Inspect the exact prompts, expected values, reasoning, responses, and completion reasons in the JSON results.

In this run, 0.6B incorrectly included Cedar (backups disabled) and Dune (fewer than 100 users) as eligible workspaces. For the missing-support-time question, it returned arrays of null values instead of the required scalar null and empty source array. The latter is an output-format failure, not evidence that it invented a support-response time. The 4B model returned the expected answer on all six tasks.

Quality-task median complete-response wall times: qwen3:4b: 5.62 s, 0 truncated; qwen3:0.6b: 1.12 s, 0 truncated. These prompts do not require equal output lengths; complete-response times reflect both model speed and the amount of reasoning generated.

## Method and limits

- Both models use Q4_K_M weights, 4,096 context tokens, one request at a time, temperature 0, seed 42, and thinking enabled using `think: true` and each installed model's native template.
- Validation found different thinking metadata: the installed 4B tag only advertises `true`, whereas 0.6B advertises `false` and `true`. Attempts at a common non-thinking baseline were discarded after 4B continued emitting reasoning. The final comparison therefore explicitly enables thinking for both and provides a larger output budget for quality checks. No installed template or server configuration was modified.
- Each model is loaded alone, receives a short warmup, and runs three short and three long prompts. Performance requests allow up to 256 generated tokens. Performance rows show medians across three requests; GPU figures show sampled peaks. Exact output token counts appear below.
- Short prompts ask for a long numbered list of document-organization tips. Long prompts supply 60 synthetic workspace records and ask for a source-attributed listing. These deliberately long outputs reach the token cap; they are throughput workloads, not scored answers.
- The normal Ollama prompt cache is enabled. A differing request number appears at the start of each performance prompt, but shared system/template prefixes can still be cached. TTFT is client wall time to the first nonempty thinking or answer chunk; this is not necessarily the first final-answer token. Each request also records `answer_ttft_s`. Tokens/s is Ollama eval_count divided by eval_duration.
- Model order is 4B then 0.6B; thermal state, power policy, and other desktop activity are not controlled. No exact speed or memory guarantees should be inferred for other contexts, thinking budgets, concurrent requests, or workloads.
- The models are unloaded after testing; both downloads remain installed. The study repository is unchanged.

Long-workload input token counts: qwen3:4b: [2664, 2664, 2664]; qwen3:0.6b: [2664, 2664, 2664].

Performance output token counts: qwen3:4b: short=[256, 256, 256], long=[256, 256, 256]; qwen3:0.6b: short=[256, 256, 256], long=[256, 256, 256].

## Reproduce

With Ollama serving locally and both models installed:

```bash
cd runs/2026-10-20-qwen3-4b-vs-0.6b
python3 ollama-benchmark.py --output another-run.json
```

The script requires only Python's standard library and `nvidia-smi`. It refuses to begin if another model is already loaded. All synthetic inputs are embedded in the script. The run changes model loading state but not server configuration.

API/model references: [generation metrics](https://docs.ollama.com/api/generate), [loaded model memory](https://docs.ollama.com/api/ps), [thinking controls](https://docs.ollama.com/capabilities/thinking), [Qwen no-thinking controls](https://qwenlm.github.io/blog/qwen3/), [Qwen native chat template](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json).

## Environment detail and provenance

The original result file records baseline memory and GPU snapshots, loaded-model metadata, templates, full digests, and per-model telemetry. CPU, GPU identity, driver, kernel, and Python were verified on this host while preparing the repository and are stored under `supplementary_snapshot` in `hardware.json`. Its collection timestamp is later than this benchmark; it must not be interpreted as another concurrent resource measurement. The archived raw results were not retroactively modified to add that snapshot.

The recorded comparison uses the following exact model identities:

| Model | Digest | Quantization | Ollama-advertised thinking values |
| --- | --- | --- | --- |
| `qwen3:4b` | `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` | Q4_K_M | `[True]` |
| `qwen3:0.6b` | `7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435` | Q4_K_M | `[False, True]` |

## Every performance request

Generation rates include reasoning and answer tokens. TTFT includes the first thinking or answer chunk. These requests deliberately reached the 256-token output cap and are not assessed for completed task quality.

| Model | Workload | Repetition | Input tokens | Output tokens | Tokens/s | TTFT (s) | Request wall time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3:4b | short | 1 | 91 | 256 | 93.490 | 0.0414 | 2.7821 |
| qwen3:4b | long | 1 | 2664 | 256 | 83.952 | 0.8779 | 3.9318 |
| qwen3:4b | short | 2 | 91 | 256 | 93.390 | 0.2625 | 3.0059 |
| qwen3:4b | long | 2 | 2664 | 256 | 83.900 | 0.8789 | 3.9348 |
| qwen3:4b | short | 3 | 91 | 256 | 92.847 | 0.2577 | 3.0175 |
| qwen3:4b | long | 3 | 2664 | 256 | 83.680 | 0.8840 | 3.9487 |
| qwen3:0.6b | short | 1 | 91 | 256 | 308.401 | 0.0249 | 0.8484 |
| qwen3:0.6b | long | 1 | 2664 | 256 | 236.677 | 0.2708 | 1.3454 |
| qwen3:0.6b | short | 2 | 91 | 256 | 302.056 | 0.1970 | 1.0383 |
| qwen3:0.6b | long | 2 | 2664 | 256 | 233.885 | 0.2660 | 1.3537 |
| qwen3:0.6b | short | 3 | 91 | 256 | 301.921 | 0.2005 | 1.0420 |
| qwen3:0.6b | long | 3 | 2664 | 256 | 234.796 | 0.2660 | 1.3515 |

## Every document check

### lookup

Documents:

```text
[P1] Unopened standard items can be returned within 21 days of delivery. The handling fee is 12 USD.
```

Question:

What is the return deadline in days? Return only JSON with keys "days" (integer) and "source" (passage ID).

Expected answer:

```json
{
  "days": 21,
  "source": "P1"
}
```

**qwen3:4b: PASS**

Actual final response:

```text
{"days": 21, "source": "P1"}
```

Completion: `stop`; generated tokens including reasoning: 243; wall time: 2.8816 s; first answer token: 2.7282 s. Full reasoning is retained in the raw result.

**qwen3:0.6b: PASS**

Actual final response:

```text
{
  "days": 21,
  "source": "P1"
}
```

Completion: `stop`; generated tokens including reasoning: 226; wall time: 0.9462 s; first answer token: 0.8838 s. Full reasoning is retained in the raw result.

### exception

Documents:

```text
[P1] Standard items are returnable within 30 days if unused.
[P2] Clearance items are final sale and cannot be returned, even if unused.
[P3] Defective products have a separate repair program.
```

Question:

An unused, non-defective clearance item was delivered 10 days ago. Is it returnable? Return only JSON with keys "eligible" (boolean) and "source" (the decisive passage ID).

Expected answer:

```json
{
  "eligible": false,
  "source": "P2"
}
```

**qwen3:4b: PASS**

Actual final response:

```text
{"eligible": false, "source": "P2"}
```

Completion: `stop`; generated tokens including reasoning: 441; wall time: 4.8397 s; first answer token: 4.7058 s. Full reasoning is retained in the raw result.

**qwen3:0.6b: PASS**

Actual final response:

```text
```json
{"eligible": false, "source": "P2"}
```
```

Completion: `stop`; generated tokens including reasoning: 425; wall time: 1.4943 s; first answer token: 1.4278 s. Full reasoning is retained in the raw result.

### dated_override

Documents:

```text
[P1] January policy: Standard workspaces allow 24 members; Pro workspaces allow 64 members.
[P2] February policy: Pro trials have a limit of 8 members.
[P3] March policy: The paid Pro limit is reduced to 32 members, replacing the January paid Pro limit. Trial limits are unchanged.
```

Question:

Under the latest policy, what is the member limit for a paid Pro workspace that is not on a trial? Return only JSON with keys "limit" (integer) and "source" (decisive passage ID).

Expected answer:

```json
{
  "limit": 32,
  "source": "P3"
}
```

**qwen3:4b: PASS**

Actual final response:

```text
{"limit": 32, "source": "P3"}
```

Completion: `stop`; generated tokens including reasoning: 545; wall time: 6.0228 s; first answer token: 5.8658 s. Full reasoning is retained in the raw result.

**qwen3:0.6b: PASS**

Actual final response:

```text
{
  "limit": 32,
  "source": "P3"
}
```

Completion: `stop`; generated tokens including reasoning: 308; wall time: 1.1028 s; first answer token: 1.0390 s. Full reasoning is retained in the raw result.

### multiple_conditions

Documents:

```text
[P1] A pilot requires an EU region and enabled backups.
[P2] Workspace roster: Aster: EU, backups enabled, 120 active users; Birch: US, backups enabled, 150 active users; Cedar: EU, backups disabled, 200 active users; Dune: EU, backups enabled, 60 active users.
[P3] The pilot also requires at least 100 active users.
```

Question:

Which workspaces meet every pilot requirement? Return only JSON with one key "workspaces", an alphabetically sorted array of names.

Expected answer:

```json
{
  "workspaces": [
    "Aster"
  ]
}
```

**qwen3:4b: PASS**

Actual final response:

```text
{"workspaces": ["Aster"]}
```

Completion: `stop`; generated tokens including reasoning: 852; wall time: 9.4643 s; first answer token: 9.3718 s. Full reasoning is retained in the raw result.

**qwen3:0.6b: FAIL**

Actual final response:

```text
```json
{
  "workspaces": ["Aster", "Cedar", "Dune"]
}
```
```

Completion: `stop`; generated tokens including reasoning: 327; wall time: 1.1450 s; first answer token: 1.0654 s. Full reasoning is retained in the raw result.

### calculation

Documents:

```text
[P1] The monthly base price is 120 USD and includes 120 requests.
[P2] Each additional request costs 0.08 USD.
[P3] Annual-contract customers receive a 25% discount on the additional-request charges only. The base price is not discounted. No taxes or other charges apply.
```

Question:

An annual-contract customer uses 170 requests in one month. What is the total monthly bill? Return only JSON with one numeric key "total_usd".

Expected answer:

```json
{
  "total_usd": 123
}
```

**qwen3:4b: PASS**

Actual final response:

```text
{"total_usd": 123}
```

Completion: `stop`; generated tokens including reasoning: 830; wall time: 9.2459 s; first answer token: 9.1316 s. Full reasoning is retained in the raw result.

**qwen3:0.6b: PASS**

Actual final response:

```text
```json
{"total_usd": 123}
```
```

Completion: `stop`; generated tokens including reasoning: 1298; wall time: 4.6141 s; first answer token: 4.5586 s. Full reasoning is retained in the raw result.

### abstention

Documents:

```text
[P1] The Basic plan costs 15 USD per month and includes email support.
[P2] The Pro plan costs 40 USD per month and includes phone support.
[P3] Both plans allow data export as CSV.
```

Question:

How many hours is the guaranteed Pro phone-support response time? If the documents do not specify it, use null and no sources. Return only JSON with keys "hours" and "sources" (array of passage IDs).

Expected answer:

```json
{
  "hours": null,
  "sources": []
}
```

**qwen3:4b: PASS**

Actual final response:

```text
{"hours": null, "sources": []}
```

Completion: `stop`; generated tokens including reasoning: 467; wall time: 5.2234 s; first answer token: 5.1105 s. Full reasoning is retained in the raw result.

**qwen3:0.6b: FAIL**

Actual final response:

```text
```json
{
  "hours": [
    null,
    null
  ],
  "sources": [
    null,
    null
  ]
}
```
```

Completion: `stop`; generated tokens including reasoning: 166; wall time: 0.6592 s; first answer token: 0.5499 s. Full reasoning is retained in the raw result.

## Evidence verification

The repository validator recalculates generation rates, medians, telemetry peaks, parsed answers, check scores, and truncation counts from `results.json`, then checks the complete run-file checksum manifest. It verifies artifact consistency without loading either model. The raw results, original script, and source report retain the exact bytes of the chat-produced artifacts.

```bash
python3 benchmarks/validate_run.py runs/2026-10-20-qwen3-4b-vs-0.6b
```

The current reusable runner and separate dataset live in `benchmarks/` and `datasets/`; see the repository README for new runs. The complete protocol and future confirmatory hypotheses are documented in `docs/methodology.md`.
