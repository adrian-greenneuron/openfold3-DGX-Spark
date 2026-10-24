"""Run the normal OpenFold3 CLI and record process/CUDA memory high-water marks."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    args, cli = parser.parse_known_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The GPU benchmark requires CUDA")
    # Initialise once before the CLI, then include all of its allocations,
    # including model loading. No model settings or precision flags are changed.
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    sys.argv = ["run_openfold", *cli]
    completed = False
    try:
        runpy.run_module("openfold3.run_openfold", run_name="__main__")
        completed = True
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        completed = True
    finally:
        torch.cuda.synchronize()
        metrics = {
            "cli_returned_successfully": completed,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_main_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "openfold3": importlib.metadata.version("openfold3"),
            "deepspeed": importlib.metadata.version("deepspeed"),
            "device": torch.cuda.get_device_name(),
        }
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
