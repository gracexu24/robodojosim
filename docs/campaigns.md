# Policy and world-model campaigns

## Targets

The `policy` profile collects **50 successful completed episodes**. Failed episodes remain available for diagnosis
but do not count toward the quota. The runner chooses at most 25 layouts per simulator launch and reduces the final
batch so a fully successful run ends at exactly 50.

The `world_model` profile collects **at least 6.0 hours of complete recorded interactions**. Duration is defined as
the sum of `episode_frames / episode_frequency`; at 25 Hz the target is 540,000 frames. Isaac Sim startup and
rendering between recorded observations do not count. A full task reward is intentionally not required: randomized
push/exploration episodes are valid world-model dynamics data as long as they finish and commit cleanly.

Override targets when experimenting:

```bash
robodojosim campaign ... --profile policy --target-episodes 10
robodojosim campaign ... --profile world_model --target-hours 0.25
```

Use `--max-passes 1` to run one batch and pause cleanly. Invoke the same command later to resume.

## Variation and diversity

RoboDojo's current public bottle task supplies 25 deterministic layout IDs. Campaign pass `N` uses:

- a unique episode offset (`N * 100000`);
- trajectory variant `N`;
- policy seed `base_seed + N`;
- deterministic changes to bottle order, approach/lift height, and bin drop point;
- a randomized push on the first selected bottle in most episodes;
- a variable airborne hold and two to five randomized carry waypoints for a grasped bottle.

This creates useful action and interaction variation, but initial bottle/object scenes still come from those 25
layouts. Six hours therefore contains repeated initial configurations. Broader world-model pretraining should mix in
other RoboDojo tasks or genuinely randomized/generated layouts once those are available.

## Storage

Campaign RGB frames are stored as XPolicyLab-compatible JPEG at quality 90. Actual size depends heavily on camera
resolution and scene texture. Run one 25-layout pass, measure its directory, and extrapolate before launching six
hours. Keep extra space for failed episodes, RoboDojo videos, partial files, and training conversions.

## Resume and failure behavior

`campaign_state.json` stores the campaign UUID, next pass, and current statistics. HDF5 files are committed atomically.
Policy campaigns preserve successful files and retry failed variants. World-model campaigns preserve every complete
interaction, including deliberately non-task-completing push trajectories. Three consecutive passes with no usable
episodes stop the campaign rather than consuming unlimited GPU time.

Do not mix policy and world-model profiles in one directory. A profile-specific report is available with:

```bash
robodojosim report --dataset /path/to/dataset --profile policy
```

## Splits

On quota completion the runner writes `splits.json`. Policy splits include successful episodes; world-model splits
include all complete episodes. Both are grouped by original RoboDojo layout seed, so repeated trajectory variants of
one layout cannot leak across training, validation, and test. The default ratios are 80/10/10 by layout group;
integer group counts mean episode ratios may differ slightly.
