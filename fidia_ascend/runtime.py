"""Load the native code after activating the requested accelerator backend."""
import os
from pathlib import Path
import runpy
import sys


def project_root(explicit=None):
    candidates = [Path(explicit).expanduser()] if explicit else [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[1]]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "src/main.py").is_file() and (candidate / "config/FIDIA.py").is_file():
            return candidate
    raise FileNotFoundError("FIDIA source checkout not found; supply --project-root /path/to/FIDIA-code")


def enable_backend(backend):
    import torch
    if backend == "npu":
        import torch_npu
        from torch_npu.contrib import transfer_to_npu  # Import activates the official runtime API mapping.
        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable; no CPU fallback is selected")
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    else:
        raise ValueError(backend)


def launch_native(root, arguments, backend="npu", motif_indexing="sequence", expected="train"):
    from .mapping import wrap_motif_sampler
    root = project_root(root)
    os.chdir(root)
    sys.path.insert(0, str(root))
    enable_backend(backend)
    sys.argv = [str(root / "src/main.py"), *arguments]
    namespace = runpy.run_path(sys.argv[0], run_name="fidia_ascend_native")
    globals_ = namespace["main"].__globals__
    globals_["Lmpnn_with_logprob"] = wrap_motif_sampler(
        globals_["Lmpnn_with_logprob"], lambda: globals_["FLAGS"].config.task, motif_indexing)
    # absl parses the untouched native flags/configuration before invoking main.
    def checked_main(argv):
        cfg = globals_["FLAGS"].config
        if expected == "infer" and not (cfg.train.mode == "EVAL" and cfg.eval_bmk):
            raise ValueError("The infer command requires EVAL mode and eval_bmk=True")
        if expected == "train" and cfg.train.mode not in {"IAPO", "GRPO"}:
            raise ValueError("The train command requires the native IAPO or GRPO configuration")
        return namespace["main"](argv)
    namespace["app"].run(checked_main)
