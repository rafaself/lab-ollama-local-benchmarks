#!/usr/bin/env python3
"""Recalculate stored benchmark metrics and verify run checksums, without inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    if isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), label)
    else:
        require(actual == expected, label)


def parsed_answer(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        return json.loads(text)
    except ValueError:
        return None


def validate_results(data):
    require(data.get("finished_at"), "Run is incomplete: no finished_at timestamp")
    require(data.get("models"), "No measured models")
    tasks = {task["id"]: task for task in data["tasks"]}
    require(len(tasks) == len(data["tasks"]), "Duplicate task IDs")
    for model in data["models"]:
        name = model["model"]
        require(len(model["runs"]) == 6, f"{name}: expected six performance requests")
        require({check["task"] for check in model["quality"]} == set(tasks), f"{name}: task set mismatch")
        require(len(model["quality"]) == len(tasks), f"{name}: duplicate or missing checks")
        for call in [model["first_request"], model["warmup"]] + [r["metrics"] for r in model["runs"] + model["quality"]]:
            require(call.get("done") is True, f"{name}: unfinished request")
            require(call["eval_duration"] > 0, f"{name}: invalid generation duration")
            equal(call["generation_tokens_s"], call["eval_count"] / (call["eval_duration"] / 1e9), f"{name}: token rate mismatch")
        for check in model["quality"]:
            task = tasks[check["task"]]
            equal(check["expected"], task["expected"], f"{name}: changed expected answer")
            equal(check["prompt"], f"Documents:\n{task['documents']}\n\nQuestion:\n{task['question']}", f"{name}: quality prompt mismatch")
            parsed = parsed_answer(check["metrics"]["response"])
            equal(check["parsed"], parsed, f"{name}: parsed answer mismatch")
            passed = parsed == task["expected"] and check["metrics"].get("done_reason") != "length"
            equal(check["passed"], passed, f"{name}: check score mismatch")
        summary = model["summary"]
        for kind in ("short", "long"):
            runs = [r for r in model["runs"] if r["kind"] == kind]
            require(len(runs) == 3 and {r["repetition"] for r in runs} == {1, 2, 3}, f"{name}/{kind}: missing repetitions")
            for label, metric in (("median_generation_tokens_s", "generation_tokens_s"), ("median_ttft_s", "ttft_s"), ("median_wall_s", "wall_s")):
                equal(summary[kind][label], statistics.median(r["metrics"][metric] for r in runs), f"{name}/{kind}: {label} mismatch")
            equal(summary[kind]["prompt_tokens"], [r["metrics"]["prompt_eval_count"] for r in runs], f"{name}/{kind}: prompt counts mismatch")
            equal(summary[kind]["generated_tokens"], [r["metrics"]["eval_count"] for r in runs], f"{name}/{kind}: output counts mismatch")
        equal(summary["quality_passes"], sum(r["passed"] for r in model["quality"]), f"{name}: total score mismatch")
        equal(summary["quality_total"], len(tasks), f"{name}: total checks mismatch")
        equal(summary["quality_truncated"], sum(r["metrics"].get("done_reason") == "length" for r in model["quality"]), f"{name}: truncation count mismatch")
        equal(summary["quality_median_wall_s"], statistics.median(r["metrics"]["wall_s"] for r in model["quality"]), f"{name}: quality latency mismatch")
        samples = model["gpu_samples"]
        for label, metric in (("peak_gpu_used_mib", "used_mib"), ("peak_gpu_utilization_pct", "utilization_pct"), ("peak_temperature_c", "temperature_c")):
            if samples:
                equal(summary[label], max(sample[metric] for sample in samples), f"{name}: {label} mismatch")
        require(not model["unloaded_state"]["models"], f"{name}: other loaded models at completion")
        print(f"{name}: metrics and {len(tasks)} quality checks verified ({summary['quality_passes']}/{len(tasks)})")


def run_files(directory):
    files = {}
    for path in directory.rglob("*"):
        require(not path.is_symlink(), f"Run contains a symlink: {path}")
        if path.is_file() and path.name != "SHA256SUMS":
            files[path.relative_to(directory).as_posix()] = path
    return files


def verify_checksums(directory):
    manifest = directory / "SHA256SUMS"
    require(manifest.is_file(), "No SHA256SUMS: finalize the run with --write-checksums")
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Malformed checksum")
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe checksum path")
        require(name not in expected, "Duplicate checksum entry")
        expected[name] = digest
    actual_files = run_files(directory)
    equal(set(actual_files), set(expected), "Checksum file list differs from the run directory")
    for name, path in actual_files.items():
        equal(hashlib.sha256(path.read_bytes()).hexdigest(), expected[name], f"Checksum mismatch: {name}")
    print(f"Verified SHA-256 checksums for {len(expected)} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--write-checksums", action="store_true", help="Finalize a new run after validating its results; existing manifests are never replaced")
    args = parser.parse_args()
    try:
        data = json.loads((args.run / "results.json").read_text())
        validate_results(data)
        if args.write_checksums:
            for name in ("report.md", "hardware.json", "configuration.json"):
                require((args.run / name).is_file(), f"Missing required report artifact: {name}")
            contents = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n" for name, path in sorted(run_files(args.run).items()))
            with (args.run / "SHA256SUMS").open("x") as manifest:
                manifest.write(contents)
        verify_checksums(args.run)
    except (OSError, ValueError, KeyError, TypeError, statistics.StatisticsError) as error:
        parser.exit(1, f"Validation failed: {error}\n")


if __name__ == "__main__":
    main()
