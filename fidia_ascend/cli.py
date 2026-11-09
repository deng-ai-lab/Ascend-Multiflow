"""Portable entry points; importing this module does not initialize an accelerator."""
import argparse
import json
from pathlib import Path

from . import __version__
from .runtime import launch_native, project_root


def main(argv=None):
    parser = argparse.ArgumentParser(description="FIDIA Ascend motif adapter (upstream use restrictions apply)")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, config in [("train", "IAPO"), ("infer", "EVAL")]:
        sub = commands.add_parser(name, help=f"Run the native {config} entry point")
        sub.add_argument("--project-root", type=Path)
        sub.add_argument("--backend", choices=["npu", "cuda"], default="npu")
        sub.add_argument("--motif-indexing", choices=["sequence", "pdb"], default="sequence")
        sub.add_argument("--dry-run", action="store_true")
        sub.set_defaults(default_config=config)
    options, native = parser.parse_known_args(argv)
    if native and native[0] == "--":
        native = native[1:]
    root = project_root(options.project_root)
    if not any(a == "--config" or a.startswith("--config=") for a in native):
        native.insert(0, f"--config=config/FIDIA.py:{options.default_config}")
    if not any(a.startswith("--config.task") for a in native):
        native.append("--config.task=motif")
    if options.command == "infer" and not any(a.startswith("--config.checkpoint_path") for a in native):
        native.append("--config.checkpoint_path=data/ckpt/motif-IAPO.pt")
    if options.dry_run:
        print(json.dumps({"project_root": str(root), "backend": options.backend,
                          "motif_indexing": options.motif_indexing,
                          "native_arguments": native}, indent=2))
        return
    launch_native(root, native, options.backend, options.motif_indexing, options.command)
