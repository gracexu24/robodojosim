# bottle_lerobot

XPolicyLab policy-server adapter for LeRobot ACT and Diffusion Policy checkpoints trained on the RoboDojo bottle
dataset. The policy environment loads the checkpoint and performs inference; the separate RoboDojo environment owns
Isaac Sim and sends policy-safe camera/state observations over XPolicyLab's WebSocket protocol.

`ckpt_name` must be a local `pretrained_model` directory. Only `env_cfg_type=arx_x5`, `action_type=ee`, and
`eval_batch=false` are supported. Set `ROBODOJOSIM_EVAL_OUTPUT` to collect atomic per-episode JSON metrics and
`ROBODOJO_HEADLESS=0` to display Isaac Sim on the lab computer.
