#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python - <<'PY'
import os, platform, sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Use Python 3.10; the recorded build used 3.10.12.')
if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
    raise SystemExit('This environment recipe is validated for Linux ARM64 only. Do not install unverified wheel variants automatically.')
if sys.prefix == sys.base_prefix and os.environ.get('CONDA_PREFIX') != sys.prefix:
    raise SystemExit('Activate an independent venv/Conda environment before installation.')
PY
export DS_BUILD_OPS=0
python -m pip install setuptools==68.1.2 wheel==0.48.0 numpy==1.23.5
# Install Torch first: DeepSpeed metadata generation imports Torch.
python -m pip install torch==2.1.0 torch-npu==2.1.0.post12 torchvision==0.16.0 torchaudio==2.1.0
python -m pip install -r requirements-ascend.lock.txt
OPENFOLD_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e openfold
python -m pip install --no-build-isolation --no-deps -e .
if [[ ! -e .env ]]; then cp .env.ascend.example .env; fi
python -m pip check
