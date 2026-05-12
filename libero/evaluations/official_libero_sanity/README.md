# Official LIBERO Sanity Evals

These scripts evaluate the models on official LIBERO benchmark tasks, not VLAPB tasks.
Each script writes `run_config.json`, `results.jsonl`, `summary.json`, optional videos, and first-action debug values.

## OpenVLA

```bash
/home/artemis/miniconda3/envs/openvla/bin/python \
  libero/evaluations/official_libero_sanity/eval_openvla_official_libero.py \
  --task-suite-name libero_10 \
  --task-id 0 \
  --num-trials-per-task 1 \
  --checkpoint /home/artemis/libero_data/openvla_checkpoints/openvla-7b-finetuned-libero-10
```

## OpenVLA-OFT

```bash
/home/artemis/miniconda3/envs/openvla-oft/bin/python \
  libero/evaluations/official_libero_sanity/eval_openvla_oft_official_libero.py \
  --task-suite-name libero_object \
  --task-id 0 \
  --num-trials-per-task 1 \
  --checkpoint /home/artemis/libero_data/openvlaOFT_checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

Use `--unnorm-key` if the checkpoint was trained with a different normalization key.

## pi0.5

```bash
/home/artemis/miniconda3/envs/pi_vla/bin/python \
  libero/evaluations/official_libero_sanity/eval_pi05_official_libero.py \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --num-trials-per-task 1 \
  --checkpoint /home/artemis/libero_data/pi_checkpoints/openpi-assets/checkpoints/pi05_libero
```

For a quick no-video run, add `--no-save-videos`.
