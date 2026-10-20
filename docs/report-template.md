# Experiment: <model/configuration comparison>

Status: <complete or incomplete>. Run ID: <unique ID>. Start/end timestamps: <with offsets>.

## Question and scope

State the practical question, workloads, and any hypotheses fixed before the run. Identify exploratory decisions made after examining results. State which aspects of RAG or inference were measured.

## Environment and provenance

Record CPU, usable RAM, initial available RAM and swap, GPU and VRAM, driver, OS/kernel, Python, Ollama, model digests, quantization, model order, and power/thermal controls. Give collection times and distinguish concurrent measurements from supplementary snapshots. Link the raw results, hardware metadata, configuration, dataset, and runner revision.

## Procedure

State context length, thinking mode, seed, temperature, output limits, concurrency, warmup, repetition count, cache behavior, model unload procedure, and GPU sampling cadence. Describe excluded or incomplete attempts and why they were excluded.

## Measured performance

| Model | Workload | Repetitions | Generation tokens/s | First-token latency | Full-request time | GPU allocation | Peak whole-GPU memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <model> | <workload> | <count> | <summary> | <summary> | <summary> | <placement/size> | <sampled peak> |

Define units and aggregation. State whether generation and first-token latency include reasoning. Provide per-request measurements and output token counts. Treat initial load time separately from warm inference. Report sampled temperature and utilization.

## Document checks

For every task, provide its documents, question, expected answer, actual final answer, score, completion reason, latency, and explanation of any error. Distinguish factual errors, output-format failures, and truncation. Link retained reasoning traces in the raw results when available.

## Interpretation and limits

Explain what the measured evidence supports, the practical recommendation, the sample size, and uncontrolled factors. Distinguish observations from hypotheses. Do not claim general accuracy, significance, or retrieval improvements without the corresponding experiment.

## Reproduction and validation

Give the exact command and dependency/runtime requirements. Record the validator outcome and evidence checksums. List missing checks or evidence explicitly.
