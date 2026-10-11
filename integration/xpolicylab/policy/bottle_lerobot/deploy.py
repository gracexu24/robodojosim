from __future__ import annotations

import json
import os
from pathlib import Path


def _seed(task_env) -> int:
    seed_map = getattr(task_env, "current_env_seed_map", {})
    if 0 in seed_map:
        return int(seed_map[0])
    return int(os.environ.get("ROBODOJOSIM_SEED", "0"))


def _write_result(output: Path, episode_id: int, result: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"episode_{episode_id:06d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def eval_one_episode(TASK_ENV, model_client):
    """Run one closed-loop learned-policy episode and record its outcome."""

    model_client.call(func_name="reset")
    observation = TASK_ENV.get_obs()
    layout_seed = _seed(TASK_ENV)
    episode_id = layout_seed + int(os.environ.get("ROBODOJOSIM_EPISODE_OFFSET", "0"))
    maximum_steps = int(os.environ.get("ROBODOJOSIM_POLICY_MAX_STEPS", "900"))
    steps = 0
    termination_reason = "environment_terminal"
    while not TASK_ENV.is_episode_end() and steps < maximum_steps:
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        if not isinstance(actions, list) or not actions:
            raise RuntimeError("bottle_lerobot returned an empty or invalid action chunk")
        for action in actions:
            TASK_ENV.take_action(action)
            steps += 1
            if TASK_ENV.is_episode_end() or steps >= maximum_steps:
                break
        if not TASK_ENV.is_episode_end() and steps < maximum_steps:
            observation = TASK_ENV.get_obs()

    if not TASK_ENV.is_episode_end():
        termination_reason = "policy_timeout"
        TASK_ENV.success[0] = False
        TASK_ENV.end_flag[0] = True
    success = bool(getattr(TASK_ENV, "success", [False])[0]) and bool(getattr(TASK_ENV, "end_flag", [False])[0])
    if success:
        termination_reason = "task_success"
    output = Path(os.environ.get("ROBODOJOSIM_EVAL_OUTPUT", "evaluations/bottle_lerobot"))
    _write_result(
        output,
        episode_id,
        {
            "episode_id": episode_id,
            "layout_seed": layout_seed,
            "steps": steps,
            "success": success,
            "termination_reason": termination_reason,
            "checkpoint": os.environ.get("ROBODOJOSIM_CHECKPOINT", ""),
        },
    )
    print(f"[bottle_lerobot] episode={episode_id} steps={steps} success={success} reason={termination_reason}")


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("bottle_lerobot supports only eval_batch=false")
