# Lab: local Ollama benchmarks

Measured comparisons of local language models, with reproducible workloads, raw responses, GPU telemetry, and detailed reports. The initial experiment compares `qwen3:4b` and `qwen3:0.6b` for a personal RAG study on an RTX 3060 Laptop GPU.

## Recorded experiments

| Run | Hardware | Short generation | Document generation | Document checks |
| --- | --- | --- | --- | --- |
| [2026-10-20: Qwen3 4B vs 0.6B](runs/2026-10-20-qwen3-4b-vs-0.6b/report.md) | RTX 3060 Laptop, 6 GiB | 4B: 93.4; 0.6B: 302.1 tokens/s | 4B: 83.9; 0.6B: 234.8 tokens/s | 4B: 6/6; 0.6B: 4/6 |

Both models ran entirely on the GPU at a 4,096-token context. Peak whole-GPU memory use was 3.12 GiB for 4B and 1.04 GiB for 0.6B. Thinking was enabled; generation rates include reasoning and answer tokens. Speed figures are medians of three warm requests per workload. Six synthetic checks provide a small diagnostic, not a general accuracy ranking.

The initial measurements support using 4B for the RAG lab and 0.6B for faster iteration. See the report for every check, exact answers, timings, limitations, and the distinction between factual and output-format failures.

## Contents

```text
benchmarks/compare_models.py     Current runner; Python standard library only
benchmarks/validate_run.py       Offline evidence and integrity validation
datasets/rag_checks.json         Synthetic documents, questions, expected answers
docs/methodology.md              Metrics, controls, interpretation, future protocol
docs/report-template.md          Template for subsequent reports
runs/<run-id>/                  Recorded evidence and reports for one experiment
```

The first run contains the exact original `results.json` and `ollama-benchmark.py`, a detailed `report.md`, extracted `configuration.json`, a provenance-labelled `hardware.json`, and SHA-256 checksums. The archived script keeps its original embedded dataset. The current runner loads the same dataset from `datasets/` and records supplementary environment information for new runs.

## Reproduce

Requirements: Linux, Python 3.10 or newer, Ollama running at `http://127.0.0.1:11434`, and NVIDIA `nvidia-smi` available. The baseline used Python 3.14.8 and Ollama 0.40.6; other versions may behave differently. No pip packages are required.

Install the models if needed:

```bash
ollama pull qwen3:4b
ollama pull qwen3:0.6b
```

Confirm `ollama ps` has no loaded models, then use a new run directory:

```bash
python3 benchmarks/compare_models.py \
  --output runs/<new-run-id>/results.json
```

Replace `<new-run-id>` with a unique identifier, for example a timestamp and model names. The current runner refuses to overwrite an existing results file. It loads one model at a time and unloads it after testing; installed downloads remain available. Prompts go to the local Ollama service. No hosted inference service or third-party dataset is required.

To choose models or a dataset:

```bash
python3 benchmarks/compare_models.py --help
```

For exact baseline reproduction, check the full model digests in the run's [configuration](runs/2026-10-20-qwen3-4b-vs-0.6b/configuration.json). Download tags can change. The archived script preserves the procedure that produced the baseline; the current runner adds environment capture and file-preservation checks without changing prompts, options, scoring, or model order by default.

## Verify and add reports

Validate existing results without loading a model:

```bash
python3 benchmarks/validate_run.py runs/2026-10-20-qwen3-4b-vs-0.6b
```

For a new run, use [the report template](docs/report-template.md), describe the measured environment, and distinguish observations from interpretations. Add SHA-256 checksums after reviewing the final report and supporting files:

```bash
python3 benchmarks/validate_run.py runs/<new-run-id> --write-checksums
```

Use a new directory for every experiment and retain incomplete runs as incomplete. Do not substitute new measurements into an old run. See [methodology](docs/methodology.md) for exact metric definitions and limitations.

## Scope

These experiments measure local inference and small document-answering tasks. They do not measure model training, embedding quality, retrieval quality, production concurrency, battery endurance, or general model intelligence. Full model weights are not stored in Git.

No software licence has been selected for this repository. Model licences are separate and are recorded in the original model metadata when supplied by Ollama.
