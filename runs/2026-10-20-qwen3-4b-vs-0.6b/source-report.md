# Local Ollama benchmark

Run: 2026-10-20T16:19:22.235179-03:00 to 2026-10-20T16:20:53.131841-03:00.

Hardware: Ryzen 7 6800H (8 cores / 16 threads), RTX 3060 Laptop GPU with 6 GiB VRAM, approximately 15 GiB usable system RAM. Ollama 0.40.6.

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
python3 ollama-benchmark.py --output another-run.json
```

The script requires only Python's standard library and `nvidia-smi`. It refuses to begin if another model is already loaded. All synthetic inputs are embedded in the script. The run changes model loading state but not server configuration.

API/model references: [generation metrics](https://docs.ollama.com/api/generate), [loaded model memory](https://docs.ollama.com/api/ps), [thinking controls](https://docs.ollama.com/capabilities/thinking), [Qwen no-thinking controls](https://qwenlm.github.io/blog/qwen3/), [Qwen native chat template](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json).
