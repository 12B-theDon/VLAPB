# Data Generation Scripts

## `extract_LIBERO.py`

Extracts LIBERO task/object metadata from the four local datasets:
`libero_10`, `libero_90`, `libero_object`, and `libero_spatial`.

Outputs:
- `/home/artemis/Documents/VLAPB/libero/docs/libero_objects.json`
- `/home/artemis/Documents/VLAPB/libero/docs/libero_objects_summary.md`
- `/home/artemis/Documents/VLAPB/libero/docs/libero_fixed_location_consistency.txt`

Run:

```bash
/home/artemis/miniconda3/envs/openvla/bin/python \
  /home/artemis/Documents/VLAPB/libero/scripts/data_generation/extract_LIBERO.py \
  --pretty
```

## `spawnObjects.py`

Reads the extracted LIBERO metadata and batch-checks whether movable LIBERO
objects can physically settle at candidate fixed-object locations in MuJoCo.
It treats `in` and `contain` as the same placement family, and `on` and `cook`
as the same placement family. `akita_black_bowl` is excluded as a fixed target.

Stable placements are written to:
- `/home/artemis/Documents/VLAPB/libero/scripts/data_generation/possible_spawn_positions`

Unstable or failed placements are written to:
- `/home/artemis/Documents/VLAPB/libero/scripts/data_generation/holded_spawn_positions`

Smoke test a small number of MuJoCo cases:

```bash
/home/artemis/miniconda3/envs/openvla/bin/python \
  /home/artemis/Documents/VLAPB/libero/scripts/data_generation/spawnObjects.py \
  --max-cases 10
```

Run the full batch, skipping cases that already have output JSON:

```bash
/home/artemis/miniconda3/envs/openvla/bin/python \
  /home/artemis/Documents/VLAPB/libero/scripts/data_generation/spawnObjects.py \
  --skip-existing
```

Run existing LIBERO pairs again instead of skipping them:

```bash
/home/artemis/miniconda3/envs/openvla/bin/python \
  /home/artemis/Documents/VLAPB/libero/scripts/data_generation/spawnObjects.py \
  --include-existing-libero-pairs \
  --skip-existing
```
