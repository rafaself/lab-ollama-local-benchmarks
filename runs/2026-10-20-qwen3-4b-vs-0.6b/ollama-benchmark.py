#!/usr/bin/env python3
"""Local Ollama comparison. Python standard library only; no external datasets.

Runs models sequentially, unloads them between tests, and leaves downloads intact.
All prompts are synthetic. Thinking is enabled for both models.
"""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:11434"
OPTIONS = {"num_ctx": 4096, "num_predict": 256, "temperature": 0, "seed": 42, "num_batch": 512}
SYSTEM = "Follow the user's instructions. Use only the supplied documents for factual document questions."

TASKS = [
    {
        "id": "lookup",
        "documents": "[P1] Unopened standard items can be returned within 21 days of delivery. The handling fee is 12 USD.",
        "question": 'What is the return deadline in days? Return only JSON with keys "days" (integer) and "source" (passage ID).',
        "expected": {"days": 21, "source": "P1"},
    },
    {
        "id": "exception",
        "documents": "[P1] Standard items are returnable within 30 days if unused.\n[P2] Clearance items are final sale and cannot be returned, even if unused.\n[P3] Defective products have a separate repair program.",
        "question": 'An unused, non-defective clearance item was delivered 10 days ago. Is it returnable? Return only JSON with keys "eligible" (boolean) and "source" (the decisive passage ID).',
        "expected": {"eligible": False, "source": "P2"},
    },
    {
        "id": "dated_override",
        "documents": "[P1] January policy: Standard workspaces allow 24 members; Pro workspaces allow 64 members.\n[P2] February policy: Pro trials have a limit of 8 members.\n[P3] March policy: The paid Pro limit is reduced to 32 members, replacing the January paid Pro limit. Trial limits are unchanged.",
        "question": 'Under the latest policy, what is the member limit for a paid Pro workspace that is not on a trial? Return only JSON with keys "limit" (integer) and "source" (decisive passage ID).',
        "expected": {"limit": 32, "source": "P3"},
    },
    {
        "id": "multiple_conditions",
        "documents": "[P1] A pilot requires an EU region and enabled backups.\n[P2] Workspace roster: Aster: EU, backups enabled, 120 active users; Birch: US, backups enabled, 150 active users; Cedar: EU, backups disabled, 200 active users; Dune: EU, backups enabled, 60 active users.\n[P3] The pilot also requires at least 100 active users.",
        "question": 'Which workspaces meet every pilot requirement? Return only JSON with one key "workspaces", an alphabetically sorted array of names.',
        "expected": {"workspaces": ["Aster"]},
    },
    {
        "id": "calculation",
        "documents": "[P1] The monthly base price is 120 USD and includes 120 requests.\n[P2] Each additional request costs 0.08 USD.\n[P3] Annual-contract customers receive a 25% discount on the additional-request charges only. The base price is not discounted. No taxes or other charges apply.",
        "question": 'An annual-contract customer uses 170 requests in one month. What is the total monthly bill? Return only JSON with one numeric key "total_usd".',
        "expected": {"total_usd": 123},
    },
    {
        "id": "abstention",
        "documents": "[P1] The Basic plan costs 15 USD per month and includes email support.\n[P2] The Pro plan costs 40 USD per month and includes phone support.\n[P3] Both plans allow data export as CSV.",
        "question": 'How many hours is the guaranteed Pro phone-support response time? If the documents do not specify it, use null and no sources. Return only JSON with keys "hours" and "sources" (array of passage IDs).',
        "expected": {"hours": None, "sources": []},
    },
]


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def gpu_sample():
    command = ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        used, free, util, temp = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
        return {"monotonic_s": time.monotonic(), "used_mib": used, "free_mib": free, "utilization_pct": util, "temperature_c": temp}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def memory_sample():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            values[key + "_kib"] = int(value.strip().split()[0])
    return values


def generate(model, prompt, **overrides):
    body = {"model": model, "prompt": prompt, "system": SYSTEM, "think": True, "stream": True,
            "keep_alive": "10m", "options": {**OPTIONS, **overrides}}
    req = urllib.request.Request(BASE + "/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    first_answer = None
    text = []
    thinking = []
    final = None
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            item = json.loads(line)
            if "error" in item:
                raise RuntimeError(item["error"])
            chunk = item.get("response", "")
            if (chunk or item.get("thinking")) and first is None:
                first = time.monotonic() - start
            if chunk and first_answer is None:
                first_answer = time.monotonic() - start
            text.append(chunk)
            thinking.append(item.get("thinking", ""))
            if item.get("done"):
                final = item
    if final is None:
        raise RuntimeError("No final metrics received")
    elapsed = time.monotonic() - start
    metrics = {key: value for key, value in final.items() if key not in ("response", "thinking", "context")}
    count, duration = metrics.get("eval_count", 0), metrics.get("eval_duration", 0)
    metrics.update({"response": "".join(text), "thinking": "".join(thinking), "wall_s": elapsed,
                    "ttft_s": first, "answer_ttft_s": first_answer,
                    "generation_tokens_s": count / (duration / 1e9) if duration else None})
    return metrics


def parse_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    return json.loads(text)


def workload(kind, repetition):
    if kind == "short":
        return (f"Benchmark request {repetition}. Write 1000 numbered, distinct, one-sentence tips for organizing "
                "a personal document library. Include tips about filenames, backups, tags, folders, and search. "
                "Do not add an introduction or conclusion. Continue until all 1000 tips are written.")
    rows = []
    for i in range(1, 61):
        rows.append(f"[P{i:02d}] Record W{i:03d}: region={'EU' if i % 3 else 'US'}; "
                    f"plan={'Pro' if i % 2 else 'Basic'}; backups={'enabled' if i % 4 else 'disabled'}; "
                    f"active_users={40 + i}; owner=team-{i:02d}. This record is an independent workspace snapshot.")
    return (f"Benchmark request {repetition}. Use only the following workspace records.\n" + "\n".join(rows) +
            "\nList every EU workspace as a numbered list. For each, report its ID, plan, backup status, "
            "active user count, owner, and source passage. Include every matching workspace; do not summarize.")


def summarize(model_result):
    summary = {}
    for kind in ("short", "long"):
        runs = [r for r in model_result["runs"] if r["kind"] == kind]
        summary[kind] = {"median_generation_tokens_s": statistics.median(r["metrics"]["generation_tokens_s"] for r in runs),
                         "median_ttft_s": statistics.median(r["metrics"]["ttft_s"] for r in runs),
                         "median_wall_s": statistics.median(r["metrics"]["wall_s"] for r in runs),
                         "prompt_tokens": [r["metrics"].get("prompt_eval_count") for r in runs],
                         "generated_tokens": [r["metrics"].get("eval_count") for r in runs]}
    summary["quality_passes"] = sum(r["passed"] for r in model_result["quality"])
    summary["quality_total"] = len(model_result["quality"])
    summary["quality_median_wall_s"] = statistics.median(r["metrics"]["wall_s"] for r in model_result["quality"])
    summary["quality_truncated"] = sum(r["metrics"].get("done_reason") == "length" for r in model_result["quality"])
    samples = model_result["gpu_samples"]
    if samples:
        summary.update({"peak_gpu_used_mib": max(r["used_mib"] for r in samples),
                        "peak_gpu_utilization_pct": max(r["utilization_pct"] for r in samples),
                        "peak_temperature_c": max(r["temperature_c"] for r in samples)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen3:4b", "qwen3:0.6b"])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial = api("/api/ps")
    if initial.get("models"):
        raise SystemExit("A model is already loaded. Finish existing work before running this benchmark.")
    result = {"started_at": datetime.datetime.now().astimezone().isoformat(), "options": OPTIONS,
              "think": True, "quality_num_predict": 2048, "system": SYSTEM, "tasks": TASKS,
              "prompt_rendering": "Installed native templates via standard /api/generate. Both models advertise support for think=true; the installed 4B tag only advertises true. Reported eval_count includes reasoning and answer tokens.",
              "method": "Sequential models; fresh model load then warmup; three short and three long 256-token capped requests; six one-shot synthetic RAG checks; greedy decoding; no concurrent inference; GPU sampled every 0.4 seconds. First load has OS caches in their existing state, not a controlled disk-cold load. Repeated prompt prefixes can use Ollama's normal cache.",
              "version": api("/api/version"), "installed_models": api("/api/tags"),
              "baseline_gpu": gpu_sample(), "baseline_memory": memory_sample(), "models": []}

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for model in args.models:
        print(f"Starting {model}", flush=True)
        entry = {"model": model, "show": api("/api/show", {"model": model}), "runs": [], "quality": [], "gpu_samples": []}
        result["models"].append(entry)
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                sample = gpu_sample()
                if sample:
                    entry["gpu_samples"].append(sample)
                stop.wait(0.4)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            entry["first_request"] = generate(model, "Reply with only READY.", num_predict=16)
            entry["loaded_state"] = api("/api/ps")
            entry["loaded_memory"] = memory_sample()
            entry["warmup"] = generate(model, "Write one short sentence about organizing documents.", num_predict=32)
            for repetition in range(1, 4):
                for kind in ("short", "long"):
                    prompt = workload(kind, repetition)
                    metrics = generate(model, prompt)
                    entry["runs"].append({"kind": kind, "repetition": repetition, "prompt": prompt, "metrics": metrics})
                    save()
                    print(f"{model} {kind} #{repetition}: {metrics['generation_tokens_s']:.1f} tok/s, "
                          f"TTFT={metrics['ttft_s']:.3f}s, wall={metrics['wall_s']:.3f}s, "
                          f"input={metrics.get('prompt_eval_count')}, output={metrics.get('eval_count')}", flush=True)
            for task in TASKS:
                prompt = f"Documents:\n{task['documents']}\n\nQuestion:\n{task['question']}"
                metrics = generate(model, prompt, num_predict=2048)
                parsed = None
                try:
                    parsed = parse_json(metrics["response"])
                except (ValueError, TypeError):
                    pass
                passed = parsed == task["expected"] and metrics.get("done_reason") != "length"
                entry["quality"].append({"task": task["id"], "prompt": prompt, "expected": task["expected"],
                                         "parsed": parsed, "passed": passed, "metrics": metrics})
                save()
                print(f"{model} quality {task['id']}: {'PASS' if passed else 'FAIL'} {metrics['response']!r}", flush=True)
        finally:
            api("/api/generate", {"model": model, "keep_alive": 0, "stream": False})
            stop.set()
            thread.join(timeout=6)
            entry["unloaded_state"] = api("/api/ps")
            save()
        entry["summary"] = summarize(entry)
        save()
        print(json.dumps({"model": model, "summary": entry["summary"]}), flush=True)
    result["finished_at"] = datetime.datetime.now().astimezone().isoformat()
    result["final_gpu"] = gpu_sample()
    result["final_memory"] = memory_sample()
    save()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
