import numpy as np

from robodojosim.mock_env import MockBottleEnv
from robodojosim.robodojo_adapter import RoboDojoSceneAdapter, teacher_frame


class FakeRigid:
    def __init__(self, state):
        self.state = state

    def get_bbox(self, is_relative=True):
        return self.state.pose.position, self.state.pose.quaternion, self.state.bbox


class FakeGeometry:
    def __init__(self, state):
        self.state = state

    def get_state(self, is_relative=True):
        return {"root_pose": self.state.pose.as_array()}


class FakeSceneManager:
    def __init__(self, mock):
        self.mock = mock
        self.requests = []

    def get_objects(self, env_ids, object_name, object_type):
        self.requests.append((env_ids, object_name, object_type))
        label = object_name.removesuffix("_instance")
        if object_type == "rigid":
            return {f"env0_rigid_{object_name}": FakeRigid(self.mock.bottles[label].state)}
        return {f"env0_geometry_{object_name}": FakeGeometry(self.mock.dustbin)}


class FakeLayoutManager:
    def get_instance_name(self, env_idx, label):
        assert env_idx == 0
        return f"{label}_instance"


class FakeTaskEnv:
    def __init__(self, mock):
        self.scene_manager = FakeSceneManager(mock)
        self.layout_manager = FakeLayoutManager()


def test_scene_adapter_extracts_privileged_state_without_isaac_imports():
    mock = MockBottleEnv(5)
    task_env = FakeTaskEnv(mock)
    snapshot = RoboDojoSceneAdapter(task_env).snapshot(mock.observation())
    assert set(snapshot.bottles) == {"bottle0", "bottle1", "bottle2", "bottle3"}
    assert np.isclose(snapshot.dustbin.pose.position[0], -0.63)
    requested_names = [request[1] for request in task_env.scene_manager.requests]
    assert requested_names == [
        "bottle0_instance",
        "bottle1_instance",
        "bottle2_instance",
        "bottle3_instance",
        "dustbin_instance",
    ]
    teacher = teacher_frame(snapshot, "grasp", "bottle0", "left")
    assert set(teacher) >= {"phase", "bottle", "active_arm", "bottle0_pose", "dustbin_pose"}
