# Lab: local Ollama benchmarks

Measured comparisons of local language models for a personal RAG study. Each experiment includes a detailed report, reproducible workloads, raw responses, and GPU telemetry.

## Latest comparison: Qwen3 4B vs 0.6B

Measured on **October 20, 2026**, using an **RTX 3060 Laptop GPU with 6 GiB VRAM**.

| Measurement | `qwen3:4b` | `qwen3:0.6b` |
| --- | ---: | ---: |
| Short-prompt generation | 93.4 tokens/s | 302.1 tokens/s |
| Document-prompt generation | 83.9 tokens/s | 234.8 tokens/s |
| Median complete document answer | 5.62 s | 1.12 s |
| Peak whole-GPU memory used | 3.12 GiB | 1.04 GiB |
| Model placement | 100% GPU | 100% GPU |
| Document checks passed | 6/6 | 4/6 |

Both models used a **4,096-token context with thinking enabled**. Generation rates include reasoning and answer tokens and show medians of three warm requests per workload. Complete-answer time is the median across six different document questions with unequal output lengths. The six synthetic checks are a small diagnostic, not a general accuracy ranking.

The initial results support **4B for the RAG lab** and **0.6B for faster iteration**. The smaller model generated tokens about three times faster, but missed eligibility conditions and returned an incorrect JSON structure on another check.

Read the [full comparison report](runs/2026-10-20-qwen3-4b-vs-0.6b/report.md) for individual answers, timings, error explanations, and limitations.

## Contents

| Resource | What it contains |
| --- | --- |
| [Benchmark runner](benchmarks/compare_models.py) | Runs the model comparison using Python's standard library |
| [Offline validator](benchmarks/validate_run.py) | Recalculates metrics and scores, then verifies evidence checksums |
| [Document checks](datasets/rag_checks.json) | Synthetic documents, questions, and expected answers |
| [Methodology](docs/methodology.md) | Metric definitions, experimental controls, limitations, and future protocol |
| [Report template](docs/report-template.md) | Structure for documenting subsequent experiments |
| [Recorded runs](runs/) | Reports and supporting evidence organized by experiment |

### Evidence from the first run

| Artifact | Purpose |
| --- | --- |
| [Detailed report](runs/2026-10-20-qwen3-4b-vs-0.6b/report.md) | Findings, every performance request, and every document check |
| [Raw results](runs/2026-10-20-qwen3-4b-vs-0.6b/results.json) | Original measurements, prompts, reasoning, and final answers |
| [Configuration](runs/2026-10-20-qwen3-4b-vs-0.6b/configuration.json) | Exact model digests, templates, and generation settings |
| [Hardware and provenance](runs/2026-10-20-qwen3-4b-vs-0.6b/hardware.json) | Recorded resource snapshots and labelled supplementary environment details |
| [Archived runner](runs/2026-10-20-qwen3-4b-vs-0.6b/ollama-benchmark.py) | Exact original script, including its embedded dataset |
| [Original report](runs/2026-10-20-qwen3-4b-vs-0.6b/source-report.md) | Unchanged report produced before repository preparation |
| [Excluded attempts](runs/2026-10-20-qwen3-4b-vs-0.6b/excluded-attempts/README.md) | Preliminary validation attempts and reasons for exclusion |
| [SHA-256 checksums](runs/2026-10-20-qwen3-4b-vs-0.6b/SHA256SUMS) | Integrity manifest covering the complete run directory |

## Run a comparison

### Requirements

| Component | Requirement |
| --- | --- |
| Operating system | Linux |
| Python | 3.10 or newer; no pip packages required |
| Ollama | Running locally at `http://127.0.0.1:11434` |
| GPU telemetry | NVIDIA `nvidia-smi` available |
| Initial model state | No models loaded; check with `ollama ps` |

The baseline environment used Python 3.14.8 and Ollama 0.40.6. Other versions may behave differently; see [hardware provenance](runs/2026-10-20-qwen3-4b-vs-0.6b/hardware.json) for collection times.

### 1. Download the models

```bash
ollama pull qwen3:4b
ollama pull qwen3:0.6b
```

### 2. Run the benchmark

From the repository root, choose an unused run directory:

```bash
python3 benchmarks/compare_models.py \
  --output runs/local-qwen3-comparison-001/results.json
```

Use a new run ID for each experiment. The runner refuses to overwrite existing results, loads one model at a time, and unloads it after testing. Installed downloads remain available. Prompts go to the local Ollama service.

To select other models or a dataset, inspect the available options:

```bash
python3 benchmarks/compare_models.py --help
```

Download tags can change. For baseline reproduction, compare the installed model digests with the recorded [configuration](runs/2026-10-20-qwen3-4b-vs-0.6b/configuration.json). The current runner adds environment capture and overwrite protection while preserving the original default prompts, settings, scoring, and model order.

## Verify results and add a report

### Validate the recorded experiment

This command checks stored evidence without loading a model:

```bash
python3 benchmarks/validate_run.py runs/2026-10-20-qwen3-4b-vs-0.6b
```

### Finalize a new experiment

Create `report.md`, `hardware.json`, and `configuration.json` in the new run directory. Use the [report template](docs/report-template.md) to describe the measured environment, results, and limitations. After reviewing those files, validate the measurements and write their checksums:

```bash
python3 benchmarks/validate_run.py \
  runs/local-qwen3-comparison-001 --write-checksums
```

Keep earlier run directories intact and label incomplete experiments clearly. See [methodology](docs/methodology.md) for evidence requirements and interpretation rules.

## Scope

These experiments measure local inference and small document-answering tasks. They do not measure model training, embedding quality, retrieval quality, production concurrency, battery endurance, or general model intelligence. Full model weights are not stored in Git.

No software licence has been selected for this repository. Model licences are separate and are recorded in the original model metadata when supplied by Ollama.
