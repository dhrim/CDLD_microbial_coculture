# Deterministic input reconstruction

## Source version

Download only the 17 files listed in `preprocessing/source_manifest.json` from upstream commit `7a2e0a97ea26fd0eb26d0611ccbce57b2962b213`. Verify every SHA256 before processing. Files are downloaded to `.cache/upstream/` by default. No upstream code is downloaded or executed.

## Filtering and identifier mapping

Read source-name to identifier mappings from columns R and Q of the upstream `Data/Strains.xlsx`. Preserve the original mapping when interpreting the four-strain data. Harmonize Ecoli/EColi, EColi_gfp/Ecoli_gfp, RP1_A/RP1_B and PAg1_gfp as explicitly implemented in `preprocessing/pilot.py` and `preprocessing/full.py`. Sort mapped training IDs lexicographically to assign zero-based model IDs. Target and affecting roles have distinct IDs.

For three-strain 24 h observations, use the supplied growth effects after requiring distinct affecting strains, the target-specific autofluorescence inclusion list, no monoculture placeholders and at least three replicates. Retain original CSV row indices in `pair_` observation identifiers. This yields 5,357 observations.

For the corresponding 48 h responses, read paired chip files in target order EC, EA, RP1, BI, CF, PAg. Require `0.7 * modal_initial_area < initial_area < 1.3 * modal_initial_area` and two droplets. Subtract initial fluorescence and clip the corrected value at 1. Compute the target monoculture median before harmonizing Blank/Mono sample labels. Group the remaining well observations by their original ordered sample key, use the median and require at least three valid 48 h measurements. Normalize by the target monoculture median and take the natural logarithm. Keep the same observation identities and split labels. There are 5,289 valid 48 h observations.

For four-strain 24 h observations, require three distinct affecting strains, the EC inclusion list and at least three replicates. Then retain only combinations of IDs represented in discovery training. Of 3,009 eligible observations, 2,691 remain in the full-data task.

`preprocessing/measured_effects.py` reconstructs 48 h individual effects from duplicate-strain or strain-plus-monoculture wells and constructs comparison rules. Source and four-strain individual effects use the supplied values. Growth-weighted rules use `Data/Isolate_profiling/gc_data.csv`.

## Combination sorting and splitting

A split key joins lexicographically sorted mapped affecting-strain IDs with `|`. Target IDs are excluded from the key, so a combination never spans splits even across different targets. Start from lexicographically sorted unique keys and permute them with NumPy `default_rng`: seed 1232026 for pairs and 1232027 for triples. Assign the first `int(0.7*n)` keys to training, the next keys up to `int(0.85*n)` to validation and the rest to test.

The reported full-data split inherits the pilot assignments. Therefore the pilot selection must be reconstructed, not replaced by a new 70:15:15 split. In each pilot split, sort keys and apply a separate permutation with seed+100, +101 or +102. Select `max(1, round(0.1*n_split_keys))` keys, retaining all target observations of each selected key.

## Training-ID coverage moves

For the pilot, repeatedly find the lexicographically first ID absent from selected training observations. Move the lexicographically first selected combination containing that ID into training. This historical pilot rule can move a selected validation or test combination. It uses identities, not response values. Reconstruct the pilot pair split first, then filter triple observations to pilot-known IDs before constructing the pilot triple split.

For the full-data split, restore the original full assignments and overlay the selected pilot assignments after those moves. For a key absent from the pilot's full assignment table, convert the first eight hex digits of SHA256(`str(seed) + key`) to an integer and divide by `2**32`; assign training below 0.7, validation below 0.85 and test otherwise.

If an ID remains absent from full training, move its lexicographically first non-test combination into training. Do not move a test combination at this stage; raise an error if no eligible combination exists. The alternative split retains this same test assignment, permutes the sorted non-test keys with seed+9000, assigns `round(n * 0.7 / 0.85)` to training, and applies the same non-test coverage rule.

All three-strain and four-strain arrays retain source-row order. Each model input orders affecting IDs lexicographically and places the target ID last. Standardization uses each task's training mean and population standard deviation (`ddof=0`).

## Nested 48 h training subsets

Use the primary 48 h training array. For repeat r=0..9, permute lexicographically sorted unique combination keys using seed `2026100201 + r`. For each fraction, take the shortest prefix reaching at least that fraction of the training-row count. Preserve original row order within the subset. Use 10%, 25% and 50% in every repeat, with 100% only in repeat 0. This produces 31 subsets. Do not resample or move combinations to restore ID coverage in these subsets.

Copy the same validation and test arrays to every subset. For 10%, 25% and 50%, recompute scaling from the subset’s float32 training responses. At 100%, retain the full-data scale calculated before conversion to float32, so the full-data validation reproduction check compares identical scaling. Discovery and downstream prediction use that same subset-specific scale. Exact subset indices, counts and scales are generated in `fractions/subsets.json`; they are not distributed as source inputs.

## Outputs

The workflow writes CSV/NPZ splits, identifier maps, task metadata, diagnostic summaries and subset configurations. These are ignored by Git. `validate_inputs.py` checks counts, disjoint combination keys, coverage and subset nesting without loading trained models.
