# Attribution and license notice

This is an unofficial Ascend adaptation of [deng-ai-lab/FIDIA-code](https://github.com/deng-ai-lab/FIDIA-code), based on upstream commit `eef3bff91f7d7ea7e6e03933d5fdc8d2b4d0151d`. It is not an official Huawei or upstream release.

The original `LICENSE` is retained verbatim. It starts with CC BY-NC 4.0 and contains additional restrictions concerning commercial use, drug development/discovery and use of outputs for other models. Read the complete file. This release does not grant broader rights, change those terms to MIT/Apache, or assert OSI-approved open-source licensing. Dependencies, model weights and datasets retain their own terms.

Upstream includes code derived from OpenFold (AlQuraishi Laboratory; Apache License 2.0) and ProteinMPNN/LigandMPNN (Dauparas et al.; see the original projects). Existing copyright headers are retained. A copy of Apache License 2.0 is provided in `licenses/Apache-2.0.txt`. `vendor/dllogger/` contains the installation sources from NVIDIA's DLLogger repository at commit `0478734ff7be75adde8d160e04872664d1c62e5f`, together with its original README and license.

Only three existing Python source files are changed for hardware compatibility, all in OpenFold. The new launcher uses Huawei's `torch_npu.contrib.transfer_to_npu`. The sequence-position-to-PDB-number mapping preserves CSV motif semantics at the sampler boundary without editing reference PDBs, CSVs, the policy loss or the native scoring code. See `CHANGES_FROM_UPSTREAM.md`.

The project is presented as Ascend-Multiflow. Its underlying source attribution and research citation are retained here. Also credit the original ProteinMPNN, OpenFold, ESM/ESMFold, TorchNPU and TM-align/tmtools projects when using their components.

## Research citation

Please cite the original FIDIA paper:

```bibtex
@inproceedings{
anonymous2026fidia,
title={{FIDIA}: Function-Informed Sequence Design via Inference-Aligned Policy Optimization},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=pvbJsa0ia0}
}
```
