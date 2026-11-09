# Changes from upstream

Upstream: [deng-ai-lab/FIDIA-code](https://github.com/deng-ai-lab/FIDIA-code), commit `eef3bff91f7d7ea7e6e03933d5fdc8d2b4d0151d`.

## Existing source files

- `openfold/setup.py`: add `OPENFOLD_BUILD_CUDA=0` to install OpenFold without compiling its CUDA attention extension. The default CUDA build is retained.
- `openfold/openfold/utils/kernel/attention_core.py`: make the CUDA extension optional and use differentiable PyTorch attention operations on non-CUDA devices or when the extension is absent.
- `openfold/openfold/model/structure_module.py`: make the same extension optional and use the existing PyTorch softmax module (`self.softmax`) for the corresponding non-CUDA operation.
- `README.md`: replace the installation and usage instructions with Ascend motif instructions; retain upstream attribution and the paper citation.
- `.gitignore`: exclude local configuration, environments, weights, generated structures, data downloads, caches and distribution archives; retain the upstream CSV indices.

The original `src/`, `config/`, `LigandMPNN/`, `utils/` Python code, scripts, data indices, requirements and `LICENSE` are unchanged, except that the bundled x86 ligand docking distribution under `utils/vina/` is omitted. Original model weights are downloaded separately.

## Added files

- `fidia_ascend/`: CLI and runtime adapter. The NPU backend imports `torch_npu.contrib.transfer_to_npu` before loading the native entry point. Training and inference select the original configurations. An inference guard requires native EVAL mode.
- `fidia_ascend/mapping.py`: translate 1-based motif sequence positions to actual PDB residue numbers at the sampler boundary. The input mask and native scoring coordinates are unchanged. `--motif-indexing pdb` selects PDB-numbered input masks explicitly.
- `environment-ascend.yml`, `requirements-ascend*.txt`, `ascend/install.sh` and `ascend/env.sh`: independent Linux ARM64 Python environment, dependency versions, OpenFold installation and process-local CANN setup.
- `.env.ascend.example` and `ascend/assets.json`: external data/cache/output paths and downloadable data/weight sources with checksums.
- `pyproject.toml`: installable adapter package and the `fidia-ascend` command. The full FIDIA checkout and runtime dependencies are required.
- `tests/test_adapter.py`: CPU unit checks for motif numbering, argument forwarding and inference-mode selection.
- `vendor/dllogger/` and `licenses/Apache-2.0.txt`: pinned DLLogger source dependency and third-party license text.
- `.gitattributes`: LF line endings for portable Python and shell files.
- `NOTICE_ASCEND.md`: upstream and dependency attribution.
