"""Portable pieces of the RoboDojo scripted-data pipeline."""

from .controller import BottleController, ControllerConfig, Phase
from .recording import EpisodeRecorder, validate_episode
from .types import Pose, SceneSnapshot

__all__ = [
    "BottleController",
    "ControllerConfig",
    "EpisodeRecorder",
    "Phase",
    "Pose",
    "SceneSnapshot",
    "validate_episode",
]

__version__ = "0.2.0"
