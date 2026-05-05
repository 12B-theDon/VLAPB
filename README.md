# VLAPB

Code release for the VLAPB paper.

This repository contains benchmark suites, LIBERO-based tooling, and supporting scripts for running and analyzing VLAPB experiments.

## Repository

- `VLAPB_suites/`: Generated VLAPB task suites and summaries.
- `libero/`: LIBERO integration, examples, documentation, and teleoperation tools.

## Documentation

- [Teleoperation with MoveIt](libero/teleop_moveit/readme.md)
- [VLAPB results documentation](libero/docs/README_vlapb_results_overleaf.md)
- [Finetuning procedures](libero/evaluations/finetune_models/README.md)

## Setup

Use a separate conda environment for each upstream model repository. The model codebases have different Python, CUDA, JAX, PyTorch, and robotics dependencies, so a shared environment is not recommended.

Example environment layout:

```bash
conda create -n vlapb_libero python=3.8
conda create -n vlapb_octo python=3.10
conda create -n vlapb_openvla python=3.10
conda create -n vlapb_openvla_oft python=3.10
conda create -n vlapb_openpi python=3.11
```

Docker images for reproducing the experiments will be provided soon. Until then, install each model from its official repository and follow the environment notes in the [finetuning procedures](libero/evaluations/finetune_models/README.md).

## Finetuning Models

The finetuning experiments use pretrained weights from the official model releases linked below.

- `libero_data/`: Local LIBERO datasets used for finetuning and evaluation (`/home/artemis/libero_data`).
- [Octo](https://github.com/octo-models/octo): Generalist robot policy pretrained on Open X-Embodiment robot trajectories; local code path: `/home/artemis/Documents/octo`.
- [OpenVLA](https://github.com/openvla/openvla): Open-source vision-language-action model for robotic manipulation; local code path: `/home/artemis/Documents/openvla`.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft): Optimized finetuning recipe for OpenVLA with action chunking and continuous action prediction; local code path: `/home/artemis/Documents/openvla-oft`.
- [OpenPI](https://github.com/Physical-Intelligence/openpi): Physical Intelligence's open-source pi0/pi0-FAST/pi0.5 VLA models and finetuning code; local code path: `/home/artemis/Documents/openpi`.

The reported finetuning setting uses full finetuning rather than LoRA. In our setup, full finetuning requires 4 NVIDIA RTX A6000 GPUs and approximately two weeks of training.

Related benchmark papers:

- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
- [VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks](https://arxiv.org/abs/2412.18194)

## Citation

Citation information will be added with the paper release.

## License

License information will be added before public release.
