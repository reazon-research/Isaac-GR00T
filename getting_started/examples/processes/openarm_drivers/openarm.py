import dataclasses
import sys
print(sys.path)
from .motor import DM_Motor_Type, DamiaoPort, Control_Type
import numpy as np


POS0 = np.zeros((8,), dtype=np.float32)

# -----
# Utils
# -----

@dataclasses.dataclass
class ArmState:
    position: np.float32
    velocity: np.float32
    torque: np.float32

class OpenArm:
    def __init__(self, device_path):
        self.port = DamiaoPort(device_path,
                              [DM_Motor_Type.DM4340, DM_Motor_Type.DM4340,
                               DM_Motor_Type.DM4340, DM_Motor_Type.DM4340,
                               DM_Motor_Type.DM4310, DM_Motor_Type.DM4310, 
                               DM_Motor_Type.DM4310, DM_Motor_Type.DM4310],
                              [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
                              [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
                              [True, True, True, True, True, True, True, True])
        SHOULDER = 1
        ROTATOR_CUFF = 2
        ELBOW = 3
        WRIST_RAISE = 5
        WRIST = 6

        self.KP = [15.0] * 6 + [.5] * 2
        self.KP[SHOULDER] = 130.0
        self.KP[ROTATOR_CUFF] = 90.0
        self.KP[ELBOW] = 130.0
        self.KP[WRIST_RAISE] = 17.0
        self.KP[WRIST] = 0.8

        self.KD = [1.5] * 6 + [0.1] * 2
        self.KD[SHOULDER] = 2.0
        self.KD[ELBOW] = 2.0

        self.read_only = True

    def enable_torque(self) -> bool:
        for motor in self.port.motors:
            self.port.control.enable(motor)
        self.read_only = False
        return True

    def disable_torque(self) -> bool:
        self.port.disable()
        self.read_only = True
        return True

    def disconnect(self):
        self.port.disconnect()
 
    def safe_set_position(self, radians):
        """Use hardcoded K values for """
        for i, rad in enumerate(radians):
            self.port.controlMIT(i, 1.0, 0, rad, 0.0, 0.0)

    def set_position(self, radians):
        for i, rad in enumerate(radians):
            self.port.controlMIT(i, self.KP[i], 0, rad, 0.0, 0.0)

        return True

    def get_state(self):
        if self.read_only:
            for i in range(len(self.port.motors)):
                self.port.controlMIT(i, 0.0, 0.0, 0.0, 0.0, 0.0)

        position = []
        velocity = []
        torque = []
        for motor in self.port.motors:
            position.append(motor.getPosition())
            velocity.append(motor.getVelocity())
            torque.append(motor.getTorque())

        return ArmState(
            np.float32(position),
            np.float32(velocity),
            np.float32(torque)
        )
