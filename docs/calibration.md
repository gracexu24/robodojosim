# Controller calibration

Calibration is the only part that genuinely requires the Ubuntu/Isaac Sim machine. The JSON config is intentionally
the complete tuning surface; no controller code should need editing.

## Coordinate conventions

- Poses are relative to environment 0 and use meters.
- Quaternions are `[qw, qx, qy, qz]`.
- `left_grasp_quaternion: null` means “preserve the episode-start left EE orientation.”
- Gripper values are normalized: `1.0` open and `0.0` closed.

For the shipped dual-X5 embodiment, RoboDojo declares the left arm's preferred direction as
`top_down_little_right` and the right arm's as `top_down_little_left`. The corresponding calibrated wxyz quaternions
are `[-0.61239, 0.353523, -0.61239, -0.353524]` and
`[-0.353523, 0.61239, -0.353524, -0.61239]`. Preserving the home orientation can make low bottle poses
unreachable even though the hand looks roughly vertical.

## Recommended sequence

1. Temporarily set `direct_left_max_x` high enough to route all reachable test objects to the left arm.
2. Determine a top-down quaternion from a manually verified EE pose and put it in `left_grasp_quaternion`.
3. Tune `grasp_clearance` in 5 mm increments. The fingers should close around the bottle body, not its top cap.
4. Increase `approach_height`/`lift_height` only as much as collision clearance requires.
5. Restore `direct_left_max_x` and calibrate the right grasp quaternion.
6. Tune `handover_right_position`, then bring `handover_left_position` toward it gradually. Avoid simultaneous arm
   collision; start farther apart and close the gap in 5–10 mm increments.
7. Verify both arms drop inside the shared tabletop training dustbin. Keep the four `drop_slot_offsets` separated so
   later bottles do not strike or stack on earlier placements. The experimental handoff settings are retained for
   future non-top-down strategies, but are not used by the production profiles.
8. Tune `drop_clearance` to clear the rim, then `place_clearance` to lower the held bottle close to the bin floor
   before release without contacting it.
9. Validate the small profile jitters separately: `policy_data.json` is deterministic; `world_model_data.json`
   varies height by 15 mm and each drop slot by 10 mm. Reduce either value if success degrades.

## Safety invariants

`workspace_min` and `workspace_max` are hard software limits applied before the episode starts. Keep them tight around
the actual table/bin workspace. `max_translation_step` controls waypoint spacing, not simulator substeps. Smaller
values make paths smoother but consume more of the 700-action budget.

The CLI prints the planned action count:

```bash
robodojosim plan --config /workspace/configs/bottle-calibrated.json
```

Keep substantial headroom below 700 so failed IK calls or future settling holds can be added safely.

## Evidence to save

For every accepted calibration, save:

- The exact JSON config.
- RoboDojo, XPolicyLab, and this repository's commit hashes.
- Seeds used for the smoke test.
- Success count and representative failure descriptions.
- A short video of left and right direct pickups, tabletop-bin releases, and final home state.
