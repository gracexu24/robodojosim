from env.robot_manager.robot_class.x5 import X5


class X5Aloha(X5):
    """ARX X5 arm with ALOHA fingers and four VT-Refine tactile pads."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.tactile_sensor_links = {
            "left_pad": "aloha_left_pad",
            "right_pad": "aloha_right_pad",
        }
