"""Warm-up and repeat inference, sampling system memory separately from CUDA/RSS."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from compare_predictions import prediction_files


def system_memory():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0]) * 1024
    return fields["MemTotal"], fields["MemTotal"] - fields["MemAvailable"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.interval <= 0:
        parser.error("Require warmups >= 0, repeats >= 1 and interval > 0")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for rep in range(args.warmups + args.repeats):
        dest = args.output_root / f"run{rep}"
        dest.mkdir(exist_ok=False)
        total, baseline = system_memory()
        samples = [(0.0, baseline)]
        errors = []
        done = threading.Event()
        start = time.perf_counter()

        def sample():
            try:
                while not done.wait(args.interval):
                    samples.append((time.perf_counter() - start, system_memory()[1]))
            except Exception as exc:
                errors.append(str(exc))

        monitor = threading.Thread(target=sample, daemon=True)
        command = [sys.executable, str(Path(__file__).with_name("measure_inference.py")),
                   f"--metrics={dest}/resource_metrics.json", "predict",
                   f"--query_json={args.query_json}", f"--output_dir={dest}",
                   "--use_templates=false", "--use_msa_server=false"]
        load = os.getloadavg()
        monitor.start()
        try:
            with (dest / "inference.log").open("w") as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        finally:
            wall = time.perf_counter() - start
            done.set()
            monitor.join()
            samples.append((wall, system_memory()[1]))
            with (dest / "system_memory.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["elapsed_seconds", "system_nonavailable_bytes"])
                writer.writerows(samples)
        if code or errors:
            raise RuntimeError(f"Inference/sampler failed: exit={code}, sampler={errors}")
        files = prediction_files(dest)
        for name, path in files.items():
            if name.endswith(".json"):
                json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        metrics = json.loads((dest / "resource_metrics.json").read_text())
        if not metrics["cli_returned_successfully"] or metrics["peak_cuda_allocated_bytes"] <= 0:
            raise RuntimeError("Incomplete GPU resource measurement")
        inference = sum(json.loads(p.read_text())["runtime_s"] for p in dest.rglob("timing.json"))
        peak_system = max(value for _, value in samples)
        record = {"replicate": rep, "warmup": rep < args.warmups,
                  "wall_seconds": wall, "inference_seconds": inference,
                  **metrics, "system_memory_total_bytes": total,
                  "system_nonavailable_baseline_bytes": baseline,
                  "sampled_system_nonavailable_peak_bytes": peak_system,
                  "sampled_system_increase_peak_bytes": peak_system - baseline,
                  "system_sampling_interval_seconds": args.interval,
                  "samples": len(samples), "prediction_files": len(files),
                  "start_loadavg": load, "end_loadavg": os.getloadavg()}
        records.append(record)
        (args.output_root / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
