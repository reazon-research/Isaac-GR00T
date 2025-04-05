# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# OpenArm Real Robot
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from gr00t.experiment.data_config import OpenArmDataConfig
from drivers.openarm import OpenArm, POS0
from drivers.camera import CameraArray
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.embodiment_tags import EmbodimentTag

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from gr00t.eval.service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################

class RobotError(Exception):
    pass

class OpenArmRobot:
    def __init__(self, enable_camera=True, camera_index=0, device="can0"):
        self.enable_camera = enable_camera
        self.camera_index = camera_index
        if not enable_camera:
            self.cameras = {}
        else:
            self.cameras = {"ego_view": CameraArray()}

        # Create the robot
        print(f"starting OpenArm on {device}")
        self.robot = OpenArm(device) # todo

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.enable_torque():
            print("robot torque enabled")
        else:
            raise RobotError("Failed to enable torque") 

        print("robot present position:", self.motor_bus.read("Present_Position"))

        self.camera = self.cameras["ego_view"] if self.enable_camera else None
        if self.camera is not None:
            self.camera.connect()
        print("================> OpenArm is fully connected =================")
        

    def move_to_initial_pose(self):
        self.robot.safe_set_position(POS0)
        time.sleep(2)
        print("----------------> OpenArm moved to initial pose")

    def go_home(self):
        self.move_to_initial_pose()

    def get_observation(self):
        return self.robot.capture_observation() and self.camera.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        img = self.get_observation()["observation.images.ego_view"].data.numpy()
        # convert bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.set_position(target_state)

    def disable(self):
        self.robot.disable_torque()

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        print("================> OpenArm disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=8888,
        language_instruction="pick up the bottle from the counter and place it inside the bin.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img, state):
        obs_dict = {
            "video.webcam": img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 7)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/openarm-pnp-checkpoints")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="10.110.17.183")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=1500)
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["single_arm", "gripper"]

    if USE_POLICY:
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction="pick up the bottle from the counter and place it inside the bin.",
        )

        robot = OpenArmRobot(enable_camera=True, camera_index=args.camera_index)
        with robot.activate():
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                img = robot.get_current_img()
                view_img(img)
                state = robot.get_current_state()
                action = client.get_action(img, state)
                start_time = time.time()
                for i in range(ACTION_HORIZON):
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    assert concat_action.shape == (8,), concat_action.shape
                    robot.set_target_state(torch.from_numpy(concat_action))
                    time.sleep(0.01)

                    # get the realtime image
                    img = robot.get_current_img()
                    view_img(img)

                    # 0.05*16 = 0.8 seconds
                    print("executing action", i, "time taken", time.time() - start_time)
                print("Action chunk execution time taken", time.time() - start_time)
    else:

        dataset_path = "/home/thomason/src/Isaac-GR00T/demo_data/openarm.PickNPlace"

        data_config = DATA_CONFIG_MAP["openarm"]

        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=data_config.modality_config(),
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            video_backend="torchvision_av",
        )

        print("Running playback of actions, this is NOT inference")
        import matplotlib.pyplot as plt
        view_img(dataset[0]["observation.images.ego_view"].data.numpy().transpose(1, 2, 0))
        robot = OpenArmRobot(enable_camera=True, camera_index=args.camera_index)
        with robot.activate():
            actions = []
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
                action = dataset[i]["action"]
                img = dataset[i]["observation.images.ego_view"].data.numpy()
                # original shape (3, 480, 640) for image data
                realtime_img = robot.get_current_img()

                img = img.transpose(1, 2, 0)
                view_img(img, realtime_img)
                actions.append(action)

            # plot the actions
            plt.plot(actions)
            plt.show()

            print("Done initial pose")

            # Use tqdm to create a progress bar
            for action in tqdm(actions, desc="Executing actions"):
                img = robot.get_current_img()
                view_img(img)

                robot.set_target_state(action)
                time.sleep(0.05)

            print("Done all actions")
            robot.go_home()
            print("Done home")
