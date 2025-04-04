# Minimal example of using the gr00t eval service with a mock OpenArm robot

from gr00t.eval.service import ExternalRobotInferenceClient
from typing import Dict, Any
import numpy as np

raw_obs_dict: Dict[str, Any] = {
    "video.ego_view": np.zeros((1, 480, 640, 3), dtype=np.uint8),
    "state.single_arm": np.zeros((1, 7)),
    "state.gripper": np.zeros((1, 1)),
    "annotation.human.action.task_description": "pick up the bottle from the counter and place it inside the bin.",
} 

policy = ExternalRobotInferenceClient(host="localhost", port=8888)
raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)
print("Raw action chunk:", raw_action_chunk)
for key, value in raw_action_chunk.items():
    print(f"Action: {key}: {value.shape}")
MODALITY_KEYS = ["single_arm", "gripper"]

concat_action = np.concatenate([np.atleast_1d(raw_action_chunk[f"action.{key}"][0]) for key in MODALITY_KEYS],axis=0,)
assert concat_action.shape == (8,), concat_action.shape