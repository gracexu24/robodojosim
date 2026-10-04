# bottle_scripted

XPolicyLab environment-side scripted teacher for RoboDojo's `put_bottles_into_dustbin` task. The policy server is a
lightweight heartbeat; `deploy.py` reads privileged simulator scene state, produces EE actions with `robodojosim`,
and records XPolicyLab-compatible trajectories.

Required environment variables:

- `ROBODOJOSIM_DATASET_DIR`: output directory.
- `ROBODOJOSIM_CONFIG`: optional controller JSON path.
- `ROBODOJOSIM_EPISODE_OFFSET`: optional integer added to each RoboDojo layout ID.
- `ROBODOJOSIM_TRAJECTORY_VARIANT`: deterministic order/height/drop variation ID.
- `ROBODOJOSIM_JPEG_QUALITY`: enable XPolicyLab-compatible JPEG storage at this quality.

Only `env_cfg_type=arx_x5`, `action_type=ee`, and non-batched evaluation are supported.
