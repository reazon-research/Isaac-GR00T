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
    def __init__(self, serials, names=None):
        self.serials = serials
        self.names = names if names else serials
        assert len(self.serials) == len(self.names), "serials and names must have the same length"
        assert len(set(self.names)) == len(self.names), "names must be unique"
        assert len(set(self.serials)) == len(self.serials), "serials must be unique"
        self.pipelines = {}

        for name, serial in zip(self.names, self.serials):
            ctx = rs.context()
            config = rs.config()
            config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, FPS)
            config.enable_device(serial)

            pipe = rs.pipeline(ctx)
            pipe.start(config)
            self.pipelines[name] = pipe


    def wait_for_frames(self):
        ret = {}
        for name in self.names:
            pipeline = self.pipelines[name]
            frames = pipeline.wait_for_frames()

            color = frames.get_color_frame()
            buffer = np.asanyarray(color.get_data()).copy()

            # NOTE We use the system timestamp instead of frame.timestamp,
            # to accomodate delayed signals.
            ret[name] = Frame(time.time(), buffer)
        return ret

if __name__ == "__main__":
    names = ["ego_view"]
    serials = ["213622072129"]
    camera_array = CameraArray(serials, names)

    while True:
        frames = camera_array.wait_for_frames()
        for name, frame in frames.items():
            frame.buffer = cv2.cvtColor(frame.buffer, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Camera {name}", frame.buffer)
            cv2.waitKey(1)