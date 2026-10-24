"""Opt-in deterministic inference for the pinned DGX Spark runtime.

Usage: python3 /opt/openfold3-deployment/tools/deterministic_inference.py predict
       --query_json=... --output_dir=... --use_templates=false --use_msa_server=false

Normal run_openfold invocations do not import this wrapper.
"""
import os
import sys


def main():
    # Hash randomisation must be fixed before Python starts. Re-exec only this
    # process, preserving all CLI arguments, before importing PyTorch/OpenFold3.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import importlib.metadata
    import json
    from pathlib import Path
    import runpy
    import torch
    from openfold3.projects.of3_all_atom.model import OpenFold3

    def configure():
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    configure()
    original_forward = OpenFold3.forward
    record = {
        "completed": False,
        "model_forward_calls": 0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deepspeed": importlib.metadata.version("deepspeed"),
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "use_deepspeed_evo_attention": False,
    }

    def save_record():
        target = os.environ.get("OF3_VERIFICATION_RECORD")
        if target:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n")

    def forward(self, *args, **kwargs):
        configure()
        self.settings.memory.eval.use_deepspeed_evo_attention = False
        record["model_forward_calls"] += 1
        save_record()
        return original_forward(self, *args, **kwargs)

    OpenFold3.forward = forward
    save_record()
    print("Verification mode: deterministic PyTorch operations; standard attention.", flush=True)
    sys.argv = ["run_openfold", *sys.argv[1:]]
    try:
        runpy.run_module("openfold3.run_openfold", run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    if record["model_forward_calls"] == 0:
        raise RuntimeError("Verification did not execute a model forward pass")
    record["completed"] = True
    save_record()


if __name__ == "__main__":
    main()
