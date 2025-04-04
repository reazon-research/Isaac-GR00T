import time
import numpy as np
import dataclasses
import pyrealsense2 as rs
import cv2

FRAME_W = 640
FRAME_H = 480
FPS = 30

@dataclasses.dataclass
class Frame:
    timestamp: float
    buffer: object

class CameraArray:
    def __init__(self, serials):
        self.serials = serials
        self.pipelines = {}

        for serial in self.serials:
            ctx = rs.context()
            config = rs.config()
            config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, FPS)
            config.enable_device(serial)

            pipe = rs.pipeline(ctx)
            pipe.start(config)
            self.pipelines[serial] = pipe

    def wait_for_frames(self):
        ret = {}
        for serial in self.serials:
            pipeline = self.pipelines[serial]
            frames = pipeline.wait_for_frames()

            color = frames.get_color_frame()
            buffer = np.asanyarray(color.get_data()).copy()

            # NOTE We use the system timestamp instead of frame.timestamp,
            # to accomodate delayed signals.
            ret[serial] = Frame(time.time(), buffer)
        return ret
