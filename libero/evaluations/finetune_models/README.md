# VLAPB Finetuning

This folder contains launchers and notes for finetuning pretrained VLA policies on VLAPB data. The procedure follows the official upstream repositories and uses their released pretrained weights.

## Compute

The paper setting uses full finetuning, not LoRA finetuning. Full finetuning is expected to require 4 NVIDIA RTX A6000 GPUs for approximately two weeks.

LoRA or other parameter-efficient settings can be useful for debugging, but they are not the main finetuning setting reported for VLAPB.

## Environments

Use a separate conda environment for each model repository. The upstream projects depend on different Python, CUDA, PyTorch, JAX, and robotics package versions.

Create the environments first:

```bash
conda create -n vlapb_libero python=3.8
conda create -n vlapb_octo python=3.10
conda create -n vlapb_openvla python=3.10
conda create -n vlapb_openvla_oft python=3.10
conda create -n vlapb_openpi python=3.11
```

Then activate the matching environment before installing or running each upstream repository:

```bash
conda activate vlapb_openvla
cd /home/artemis/Documents/openvla
```

Recommended local layout:

- LIBERO data: `/home/artemis/libero_data`
- Octo: `/home/artemis/Documents/octo`
- OpenVLA: `/home/artemis/Documents/openvla`
- OpenVLA-OFT: `/home/artemis/Documents/openvla-oft`
- OpenPI: `/home/artemis/Documents/openpi`
- VLAPB: `/home/artemis/Documents/VLAPB`

Docker images will be provided soon. Until then, install each model using the official setup instructions from its repository.

## Data

Generate VLAPB episodes first, then convert or register them in the format expected by each upstream model.

- Octo expects TFDS/RLDS-style data.
- OpenVLA expects RLDS-style data.
- OpenVLA-OFT expects RLDS-style data for LIBERO-style training.
- OpenPI expects a LeRobot-style dataset or a dataset registered through its OpenPI configuration.

The local LIBERO dataset root is:

```bash
/home/artemis/libero_data
```

## Launchers

This directory provides thin launchers for common experiments:

- `train_octo.py`: Octo finetuning.
- `train_openvla.py`: OpenVLA finetuning launcher.
- `train_openvla_oft.py`: OpenVLA-OFT finetuning launcher.
- `train_pi05.py`: OpenPI pi0.5 finetuning launcher.
- `eval_finetuned_checkpoints.py`: Evaluation launcher for finetuned checkpoints.

Run each launcher from the VLAPB repository root. Start with `--dry-run` to inspect the upstream command before launching a long job.

```bash
cd /home/artemis/Documents/VLAPB

python libero/evaluations/finetune_models/train_octo.py \
  --episodes-dir /path/to/vlapb_rlds \
  --dataset-name vlapb_episodes \
  --num-gpus 4 \
  --dry-run
```

```bash
python libero/evaluations/finetune_models/train_openvla.py \
  --episodes-dir /path/to/vlapb_rlds \
  --dataset-name vlapb_episodes \
  --num-gpus 4 \
  --dry-run
```

```bash
python libero/evaluations/finetune_models/train_openvla_oft.py \
  --episodes-dir /path/to/vlapb_rlds \
  --dataset-name vlapb_episodes \
  --num-gpus 4 \
  --dry-run
```

```bash
python libero/evaluations/finetune_models/train_pi05.py \
  --episodes-dir /path/to/vlapb_lerobot \
  --dataset-name physical-intelligence/libero \
  --num-gpus 4 \
  --dry-run
```

For full finetuning, use the full-finetuning configuration in the corresponding upstream repository and keep the pretrained checkpoint source fixed to the official model release. Do not report LoRA runs as the full-finetuning setting.

## Evaluation

After training, evaluate the saved checkpoints with:

```bash
python libero/evaluations/finetune_models/eval_finetuned_checkpoints.py \
  --models openvla openvla_oft octo pi05 \
  --checkpoint-root /path/to/checkpoints \
  --dry-run
```

Default checkpoint paths:

- Octo: `/home/artemis/Documents/octo`
- OpenVLA: `/home/artemis/Documents/openvla`
- OpenVLA-OFT: `/home/artemis/Documents/openvla-oft`
- OpenPI/pi0.5: `/home/artemis/Documents/openpi`

## References

- [LIBERO](https://arxiv.org/abs/2306.03310)
- [VLABench](https://arxiv.org/abs/2412.18194)
- [Octo](https://github.com/octo-models/octo)
- [OpenVLA](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
