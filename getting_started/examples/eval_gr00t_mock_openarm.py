# Minimal example of using the gr00t eval service with a mock OpenArm robot

from gr00t.eval.service import ExternalRobotInferenceClient
from typing import Dict, Any
import numpy as np

raw_obs_dict: Dict[str, Any] = {
    "video.ego_view": np.zeros((480, 640, 3), dtype=np.uint8),
    "state.single_arm": np.zeros((1, 7)),
    "state.gripper": np.zeros((1, 1)),
    "annotation.human.action.task_description": ["pick up the bottle from the counter and place it inside the bin."],
} 

policy = ExternalRobotInferenceClient(host="localhost", port=8888)
raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)