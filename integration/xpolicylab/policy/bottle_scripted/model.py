from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    """Heartbeat model; actions are generated inside the simulator client."""

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg

    def update_obs(self, obs):
        return None

    def update_obs_batch(self, obs_list):
        return None

    def get_action(self):
        return []

    def get_action_batch(self, env_idx_list=None):
        return [[] for _ in (env_idx_list or [])]

    def reset(self):
        return None
