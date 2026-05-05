This folder contains BDDL/metadata generators for the three VLAPB suites: `belongings`, `placements`, and `sequences`.

# Current Dataset Summary

The numbers below are from the latest generated `generation_summary.json` files.

| Suite | Possible episodes | Written episodes | Split distribution |
| --- | ---: | ---: | --- |
| `belongings` | 5,783,376 | 1,100 | `type1`: 300, `type2`: 300, `type3`: 300, `adaptability`: 100, `multiuser`: 100 |
| `placements` | 666 | 666 | `type1`: 21, `type2`: 115, `type3`: 345, `type4`: 115, `adaptability`: 16, `multiuser`: 49, `consistency`: 5 |
| `sequences` | 120 | 120 | `type1`: 40, `type2`: 40, `adaptability`: 10, `consistency`: 30 |

Current config:

```yaml
belongings:
  type_episodes: 900
  adaptability_episodes: 100
  multiuser_episodes: 100

sequences:
  total_episodes: 1000

placements:
  total_episodes: 1000
```

Run all generators:

```bash
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_belongings.py
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_placements.py
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_sequences.py
```

Dry-run:

```bash
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_belongings.py --dry-run
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_placements.py --dry-run
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_sequences.py --dry-run
```

Debug cap:

```bash
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_belongings.py --max-episodes-per-split 10
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_placements.py --max-episodes-per-split 10
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_sequences.py --max-episodes-per-split 10
```

# Common

- Each generator uses profile data from `/home/artemis/Documents/VLAPB/libero/profiles`.
- Each generator creates BDDL files that define the task description, participating user information, graspable objects, interrupting or distractor objects, fixture objects, and scene description.
- Generated suites are written under `/home/artemis/Documents/VLAPB/VLAPB_suites` by default.
- Episode budgets are controlled by `VLAPB_config.yaml` in this folder.
- These generators are metadata/BDDL writers and do not require GPU. Run them on CPU unless a future validation step explicitly needs rendering or simulation.

# Task Suites

## 1. `generate_VLAPB_belongings.py`

This suite evaluates whether the model can identify the correct user belongings in a scene.

- Fixture objects: `basket`, `wooden_tray`, `flat_stove`, and `plate`.
- Goal relation by fixture:
  - `basket`: `In`
  - `wooden_tray`: `In`
  - `flat_stove`: `On`
  - `plate`: `On`
- The fixture object defines the scene type. Table variations can also be used to add background diversity.
- Each episode always contains exactly four graspable objects and one fixed object.
- Object placement should be roughly balanced around the fixture object. Place the objects in a circular region around the fixture and randomly sample the final object positions within that region.
- For high-neck fixtures such as `basket`, avoid spawning target objects behind the fixture where they may be occluded.
- The generator reads fixed-object/graspable-object pair frequencies from `/home/artemis/Documents/VLAPB/libero/docs/libero_objects_summary.md`.
- Pair frequencies are used to balance common, rare, and unseen pairs. They are not used as a strict priority ranking.
- Fixed objects and target object frequency buckets are mixed with round-robin ordering so the generated suite is not dominated by the most common LIBERO pairs.

Inputs and outputs:

- Input profile file: `/home/artemis/Documents/VLAPB/libero/profiles/profiles.json`
- Pair statistics file: `/home/artemis/Documents/VLAPB/libero/docs/libero_objects_summary.md`
- Output root: `/home/artemis/Documents/VLAPB/VLAPB_suites/belongings`
- Output files: episode-level `.bddl` files and readable metadata `.json` files.

Episode budget config:

```yaml
belongings:
  type_episodes: 900
  adaptability_episodes: 100
  multiuser_episodes: 100
```

- `type_episodes` is divided evenly across `type1`, `type2`, and `type3`.
- `adaptability_episodes` controls only adaptability episodes.
- `multiuser_episodes` controls only multi-user episodes.
- The generator still logs the maximum possible episode count before applying the configured budget.

Run examples:

```bash
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_belongings.py --dry-run
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_belongings.py
python3 /home/artemis/Documents/VLAPB/libero/scripts/suites/generate_VLAPB_belongings.py --max-episodes-per-split 10
```

- `--dry-run` prints counts and planned budgets without writing files.
- `--max-episodes-per-split` is only a debugging cap.
- `--use-gpu` is currently a reserved flag and is not needed for this generator.

Task types:

- Type 1: Spawn three unique non-user items and one user belonging. Requires one user's personal information.
- Type 2: Spawn different belongings from different users, then choose one interacting user's item as the target. Requires four users' personal information.
- Type 3: Spawn three different user belongings and one interacting user's item. Requires two users' personal information.

Multi-user setting:

- Start from two user objects, one for user A and one for user B, and check whether the model understands ownership correctly.
- A harder setting can use four objects, including objects from two other people. This can share data with the Type 2 setting above.

Adaptability setting:

- A user's belonging can change across episodes.
- The changed belonging should not overwrite the original personal profile. Save the updated information separately, indexed by the adaptability episode number.
- The episode description should mention what the previous belonging was and what it changed to.
- The changed belonging is selected from the object pool in `profiles.json`.
- Prefer changes that do not overlap with other users' belongings. If needed, allow overlap up to the configured overlap limit.

## 2. `generate_VLAPB_placements.py`

This suite evaluates whether the model can place a user's item in the correct location according to that user's placement preference.

- Fixture objects: `basket`, `cabinet`, `flat_stove`, `microwave`, `white_cabinet`, `wooden_cabinet`, `wooden_tray`, and `wooden_two_layer_shelf`.
- For each episode, spawn one selected user's item at the center of the table.
- The model must move the item to the correct fixture or fixture region based on the user's placement preference.

Task types:

- Type 1: Spawn one fixture object on either the left or right side.
- Type 2: Spawn two fixture objects, one on the left and one on the right. One is the correct fixture for the user's item, and the other is randomly selected from the fixture list above.
- Type 3: Add a distractor object that does not occupy the user's target placement region. The distractor should be irrelevant to the user.
- Type 4: Randomize door or drawer states, so the model may need to fix the fixture state before grasping or placing the object. This is the harder setting.

Fixture-state rules:

- For `microwave`, the door should always be open.
- For `cabinet`, `white_cabinet`, and `wooden_cabinet`, the door or drawer corresponding to the correct placement region should be open.
- If the target region is the top drawer, the top drawer should be spawned open.
- If the target region is in front of the fixture, the door should be closed.

Multi-user setting:

- Task 1: Spawn two user belonging items with their corresponding fixture objects.
- Task 2: Add task-irrelevant objects while keeping the related fixture doors or drawers accessible.

Adaptability setting:

- A user's preferred placement can change, for example from the front of a drawer to the top of a drawer.
- The episode description should mention the previous placement preference and the updated one.

Consistency setting:

- Provide an unseen belonging object for a known user.
- The model should place the unseen object according to the user's existing placement preference.

## 3. `generate_VLAPB_sequences.py`

This suite evaluates whether the model can bring objects back to a `wooden_tray` in the correct user-specific sequence.

- Only `wooden_tray` is used as the fixture object.
- Placement position is not important. The evaluation only checks whether the object is brought back to the `wooden_tray` in the correct order.
- For each selected user, spawn three objects randomly around the `wooden_tray`.
- The model should pick and place the objects according to the given sequence.

Task types:

- Type 1: Spawn three random user-independent graspable objects.
- Type 2: Spawn three target-user belongings and require the sequence to follow the user's preference.

Consistency setting:

- Spawn two seen user belongings and one unseen user belonging.
- The model should infer the correct sequence preference and bring all objects back in the correct order.

Adaptability setting:

- The user's sequence preference can change.
- The model should follow the new sequence, not the old one.
- Adaptability uses only user belongings and does not include unseen objects.

Multi-user setting:

- Multi-user sequence tasks are not included for this suite.

# Future Work

- Control the granularity of textual personal information, such as keyword-level, sentence-level, or paragraph-level descriptions.
- Control how much visual information is provided for each episode.
