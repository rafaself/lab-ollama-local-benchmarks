# Methodology

## Research questions

How much GPU memory do the installed Qwen3 tags use on this computer? How quickly do they generate tokens on short and document-heavy prompts? Do they follow the constraints in six self-contained document questions?

The first run is descriptive. The task set was prepared before scoring, but there was no preregistered statistical analysis. The six checks were each run once per model. The report does not infer population accuracy or statistical significance.

For a future confirmatory comparison, record the hypotheses and decision rule before execution:

- **H0:** There is no systematic difference in generation throughput between the two configurations on the specified workloads and hardware.
- **H1:** The smaller configuration has higher generation throughput on those workloads.

Preselect sample size and a paired analysis, alternate or randomize model order across independent sessions, and retain all sessions. These are proposed future controls; they were not applied retroactively to the first run. Quality comparisons require a substantially larger, separately specified task set.

## Baseline procedure

1. Verify that no models are loaded. Record the available RAM, swap, GPU memory, and Ollama version.
2. Load 4B alone, then issue a short warmup. Keep the context at 4,096 tokens and use one request at a time.
3. Alternate a short and long workload for three repetitions. Each performance request permits 256 generated tokens. The recorded baseline reached that cap on all requests.
4. Run the six document checks with a 2,048-token output budget, including reasoning.
5. Unload 4B, then repeat the procedure for 0.6B. Unload the final model.

Controls: Q4_K_M weights; temperature 0; seed 42; batch size 512; thinking enabled. Both models use their installed native templates. The original model order, complete options, templates, digests, prompts, and responses are preserved in each run's raw evidence. The output is reproducible as a procedure, not guaranteed to be bit-identical across hardware or runtime versions.

The runner samples GPU memory, utilization, and temperature at roughly 0.4-second intervals plus subprocess execution time. Sampling includes model loading and the complete workload period. It is not an exhaustive record of every instantaneous peak. Power telemetry was excluded because this host returned an implausible reading during inspection.

## Workloads and checks

The short performance workload requests a long numbered list of document-organization tips. The long workload supplies 60 synthetic workspace records and requests a source-attributed listing. A changing request number appears near the start. Performance output is intentionally capped and is not scored for answer completeness.

The six checks in `datasets/rag_checks.json` cover lookup, an exception, a dated override, multiple simultaneous requirements, arithmetic, and missing information. Each embeds its documents and asks for specific JSON. These are document-answering checks with supplied passages; the experiment does not execute an embedding or retrieval pipeline.

Scoring uses parsed JSON equality against the predetermined object. A surrounding Markdown code fence is accepted. Object key order is irrelevant; extra or missing keys fail. Numeric values follow Python JSON equality (for example `123` equals `123.0`). Any response stopped by the token limit fails even if it contains a parseable answer. Report factual errors, formatting errors, and truncation separately.

## Metrics

| Metric | Definition |
| --- | --- |
| Generation tokens/second | `eval_count / (eval_duration / 1e9)` from Ollama; includes thinking and answer tokens |
| Time to first token (TTFT) | Client wall time until the first nonempty thinking **or** answer chunk |
| Time to first answer token | Client wall time until the first nonempty final-answer chunk (`answer_ttft_s`); may be absent for capped performance requests |
| Complete-response time | Client wall time for the full request, including prompt evaluation and reasoning |
| Initial load duration | Ollama's first-request `load_duration`; OS disk caches and CUDA state are not reset |
| Model allocation in VRAM | `size_vram` from `/api/ps`; compare with total allocated model `size` to establish GPU placement |
| Whole-GPU memory peak | Maximum sampled `nvidia-smi` memory used, including activity outside Ollama |
| GPU utilization/temperature peak | Maximum sampled value across the model's entire test window |
| Quality diagnostic | Number of exact parsed-answer matches out of six, with truncation counted as failure |

Speed/latency summaries use three-request medians per workload. Quality-task median wall time pools six different tasks with unequal output lengths and is descriptive. Peak memory is not the model file size. System memory snapshots describe the whole machine and do not isolate model RAM consumption.

## Baseline validation issue

The installed 4B tag advertised thinking values `[true]`; 0.6B advertised `[false, true]`. Preliminary attempts at a common non-thinking baseline left 4B emitting reasoning, which exhausted a short answer budget. Those attempts were excluded from the recorded comparison. The final run explicitly enabled thinking for both models, separated reasoning from answers, and allowed up to 2,048 output tokens for quality questions. All six answers completed for each model.

This is an observation about these installed model tags and this Ollama build. It is not a claim that every Qwen3 4B release lacks a non-thinking mode. The code, model templates, and metadata in the archived run preserve the exact setup.

## Confounds and limits

- One laptop, one model order, three performance repetitions, and six one-shot quality checks.
- Desktop activity, thermal state, power policy, and CPU/GPU clocks were not controlled. The second model started after the first warmed the machine.
- The normal Ollama prompt cache was enabled; shared system/template prefixes may be reused.
- The 256-token performance budget measures token generation, not completion of a full long answer.
- No controlled disk-cold startup, energy measurement, multi-request concurrency, or long endurance test.
- The score combines answer correctness and strict output-shape compliance. A formatting failure is not automatically a hallucination.
- Full RAG retrieval and embeddings were not measured. Better retrieval cannot be inferred from a generation-model score.

## Evidence and provenance

Each run should retain raw API results, model digests, configuration, hardware metadata with collection times, exact prompts/expected answers, a detailed report, and SHA-256 checksums. Preserve the script used for historical runs. Derived tables must be recomputable from the raw evidence. Supplementary hardware snapshots taken later must be labelled as later observations.

The validator checks stored result summaries and quality scores against their underlying measurements, and verifies checksums when present. It does not rerun inference, prove that the operating environment was controlled, or authenticate the origin of a dataset. Git history provides the publication record.

## Primary references

- [Ollama generation API and durations](https://docs.ollama.com/api/generate)
- [Ollama loaded-model API](https://docs.ollama.com/api/ps)
- [Ollama thinking controls](https://docs.ollama.com/capabilities/thinking)
- [Qwen3 release and thinking modes](https://qwenlm.github.io/blog/qwen3/)
- [Ollama Qwen3 4B tag](https://ollama.com/library/qwen3:4b)
- [Ollama Qwen3 0.6B tag](https://ollama.com/library/qwen3:0.6b)
