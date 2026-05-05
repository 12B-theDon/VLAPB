# LIBERO Fixed Item Placement Summary

- Tasks: `120`
- Couplings: `116`

## Grouped Fixed Items

| Fixed item | Areas | Target count | Placement labels |
| --- | --- | ---: | --- |
| `akita_black_bowl` | kitchen, living_room | 4 | on(4), stacking(4) |
| `basket` | libero_object, living_room | 25 | in(25), contain(25) |
| `caddy` | study | 11 | in(11), front(3), back(2), left(3), right(3), contain(11) |
| `drawer` | kitchen | 15 | on(5), in(10), top(12), bottom(3) |
| `flat_oven` | kitchen | 8 | on(8), cook(8) |
| `kitchen_table` | kitchen | 2 | on(2), front(1), right(1) |
| `living_room_table` | living_room | 3 | on(3), left(1), right(2) |
| `microwave` | kitchen | 1 | in(1) |
| `plate` | kitchen, libero_spatial, living_room | 25 | on(25), stacking(25) |
| `shelf` | kitchen, study | 8 | on(3), in(5), top(6), bottom(2) |
| `study_table` | study | 3 | on(3), right(3) |
| `tray` | living_room | 10 | in(10), contain(10) |
| `wine_rack` | kitchen | 1 | on(1), top(1) |

## Raw Fixed Object Types

| Fixed item | Areas | Target count | Placement labels |
| --- | --- | ---: | --- |
| `akita_black_bowl` | kitchen, living_room | 4 | on(4), stacking(4) |
| `basket` | libero_object, living_room | 25 | in(25), contain(25) |
| `desk_caddy` | study | 11 | in(11), front(3), back(2), left(3), right(3), contain(11) |
| `flat_stove` | kitchen | 8 | on(8), cook(8) |
| `kitchen_table` | kitchen | 2 | on(2), front(1), right(1) |
| `living_room_table` | living_room | 3 | on(3), left(1), right(2) |
| `microwave` | kitchen | 1 | in(1) |
| `plate` | kitchen, libero_spatial, living_room | 25 | on(25), stacking(25) |
| `study_table` | study | 3 | on(3), right(3) |
| `white_cabinet` | kitchen | 7 | on(2), in(5), top(4), bottom(3) |
| `wine_rack` | kitchen | 1 | on(1), top(1) |
| `wooden_cabinet` | kitchen | 8 | on(3), in(5), top(8) |
| `wooden_tray` | living_room | 10 | in(10), contain(10) |
| `wooden_two_layer_shelf` | kitchen, study | 8 | on(3), in(5), top(6), bottom(2) |

## Table-Bounded Fixed Location Consistency

A row is **consistent** when the same raw fixed object type has exactly one initial location for each `(scene, table)` context. Different scenes may still have different table-bounded coordinates.

| Raw fixed type | Contexts | Unique locations | Max per context | Verdict |
| --- | ---: | ---: | ---: | --- |
| `akita_black_bowl` | 2 | 4 | 2 | varies |
| `basket` | 3 | 3 | 1 | consistent |
| `desk_caddy` | 3 | 3 | 1 | consistent |
| `flat_stove` | 3 | 3 | 1 | consistent |
| `kitchen_table` | 2 | 2 | 1 | consistent |
| `living_room_table` | 1 | 1 | 1 | consistent |
| `microwave` | 1 | 1 | 1 | consistent |
| `plate` | 7 | 8 | 2 | varies |
| `study_table` | 2 | 2 | 1 | consistent |
| `white_cabinet` | 2 | 2 | 1 | consistent |
| `wine_rack` | 1 | 1 | 1 | consistent |
| `wooden_cabinet` | 3 | 3 | 1 | consistent |
| `wooden_tray` | 2 | 2 | 1 | consistent |
| `wooden_two_layer_shelf` | 2 | 2 | 1 | consistent |

## Instruction Pair Frequencies

Counts below summarize how often each fixed object is paired with a graspable object in the LIBERO instruction files. These counts are used by VLAPB suite generators to balance common, rare, and unseen episode pairs instead of always prioritizing the most frequent pairs.

- `basket`: `alphabet_soup`(5), `cream_cheese`(4), `tomato_sauce`(4), `butter`(3), `ketchup`(2), `milk`(2), `orange_juice`(2), `salad_dressing`(1), `bbq_sauce`(1), `chocolate_pudding`(1)
- `wooden_tray`: `akita_black_bowl`(5), `alphabet_soup`(1), `butter`(1), `cream_cheese`(1), `ketchup`(1), `tomato_sauce`(1), `chocolate_pudding`(1), `new_salad_dressing`(1)
- `flat_stove`: `moka_pot`(5), `chefmate_8_frypan`(3)
- `plate`: `akita_black_bowl`(15), `porcelain_mug`(5), `white_yellow_mug`(3), `chocolate_pudding`(3), `red_coffee_mug`(3), `white_bowl`(2)
