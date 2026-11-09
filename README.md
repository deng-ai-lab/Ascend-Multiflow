# Ascend-Multiflow

Ascend-Multiflow is a protein sequence design toolkit for Huawei Ascend NPUs, focused on motif-conditioned generation. It provides training and inference entry points, pretrained checkpoint loading, and ESMFold-based structure evaluation.

See [CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md) for implementation changes and [NOTICE_ASCEND.md](NOTICE_ASCEND.md) for source attribution and research references.

## Environment

| Component | Version |
| --- | --- |
| OS / architecture | Ubuntu 22.04 / Linux ARM64 |
| NPU | Ascend 910B3 |
| Python | 3.10.12 |
| PyTorch / torch_npu | 2.1.0 / 2.1.0.post12 |
| CANN Toolkit and 910B operator package | 8.1.RC1 |
| Driver | 24.1.rc3 |

Create an independent Python environment in the repository directory:

```bash
conda env create -f environment-ascend.yml
conda activate fidia-ascend

export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh
source ascend/env.sh
bash ascend/install.sh
```

Alternatively, create and activate a Python 3.10 venv before running the same environment and installation scripts:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

The installer uses [requirements-ascend.lock.txt](requirements-ascend.lock.txt), installs the bundled OpenFold with `OPENFOLD_BUILD_CUDA=0`, and installs the `fidia-ascend` command. CANN and the driver are installed separately. For each new shell, activate the Python environment, set `CANN_SET_ENV`, and source `ascend/env.sh`.

The installer creates `.env` from [.env.ascend.example](.env.ascend.example) if it does not exist. Adjust the data and output paths there. The default data, model caches and generated structures are stored in sibling directories outside the repository.

## Dataset and Model Weights

Download the motif data and benchmark archives from the [dataset repository](https://zenodo.org/records/20353469). Prepare both the CATH motif data and benchmark structures for the native entry points. The original CSV indices remain in `data/dataset/`.

Download URLs, file sizes and SHA256 checksums for the data and model weights are listed in [ascend/assets.json](ascend/assets.json). Arrange the external files as follows, relative to the repository root:

```text
../fidia-data/
  cath/dompdbRn/...
  benchmark/bmk_5/targets/...

../fidia-assets/
  weights/motif-IAPO.pt
  weights/proteinmpnn_v_48_020.pt
  torch/hub/checkpoints/esmfold_3B_v1.pt
  torch/hub/checkpoints/esm2_t36_3B_UR50D.pt
  torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt
```

Keep the archive directory layout when extracting the data. `DATA_STORE_FOLDER` points to `../fidia-data`; `TORCH_HOME` points to `../fidia-assets/torch`. The entry points also accept an explicit checkpoint path.

## Evaluation

Run motif evaluation with the pretrained motif checkpoint:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 fidia-ascend infer --project-root . \
  --config=config/FIDIA.py:EVAL --config.task=motif \
  --config.checkpoint_path=../fidia-assets/weights/motif-IAPO.pt
```

Use `../fidia-assets/weights/proteinmpnn_v_48_020.pt` as the checkpoint path to evaluate ProteinMPNN. Sampling temperature, candidate count, seed and evaluation settings come from `config/FIDIA.py` and can be passed through the original `--config.*` flags. Generated structures are written to `DATA_SAVE_FOLDER`; native logs are written under `logs/`.

Motif masks in the CSV use 1-based sequence positions. The adapter maps these positions to the reference PDB residue numbers before sampling. For inputs whose masks already contain PDB residue numbers, select `--motif-indexing pdb` explicitly.

## Training

Run motif IAPO training from the ProteinMPNN checkpoint:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 fidia-ascend train --project-root . \
  --config=config/FIDIA.py:IAPO --config.task=motif \
  --config.checkpoint_path=../fidia-assets/weights/proteinmpnn_v_48_020.pt \
  --config.save_dir=../fidia-outputs/checkpoints \
  --config.wandb=False
```

Select `--config=config/FIDIA.py:GRPO` to use the upstream GRPO configuration. Other training options retain the original configuration values. To inspect arguments without loading a model, append `--dry-run`.

The original CUDA installation script and experiment scripts are retained. This README describes the Ascend motif entry points.

## License and Acknowledgments

The original [LICENSE](LICENSE) and its non-commercial and use restrictions remain in effect. See [NOTICE_ASCEND.md](NOTICE_ASCEND.md) for attribution and dependency information.

Ascend-Multiflow integrates OpenFold, ProteinMPNN/LigandMPNN and ESM/ESMFold. The Ascend entry points use Huawei TorchNPU. Third-party source files retain their copyright and license notices.

## Citation

When using Ascend-Multiflow in research, cite the underlying methods and components listed in [NOTICE_ASCEND.md](NOTICE_ASCEND.md#research-citation).

## Contact

Use this repository's Issues for installation, Ascend training, and inference questions.
