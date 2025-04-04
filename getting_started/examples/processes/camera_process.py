import logging
import os
import time
import cv2
from drivers import camera
from logging_common import configure_worker_logger

_logger = logging.getLogger(__name__)

# CAM_HIGH = "213222079892"        # D455
CAM_HEAD = "922612071413"       # D451
# CAM_RIGHT_WRIST = "215222079832" # D451

def save_to_disk(dataset, outdir):
    os.makedirs(os.path.join(outdir, "observations/cam_high"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "observations/cam_right_wrist"), exist_ok=True)

    for frames in dataset:
        frame = frames[CAM_HEAD]
        path = os.path.join(outdir, 'observations/cam_high', "%.4f.jpg" % frame.timestamp)
        cv2.imwrite(path, cv2.cvtColor(frame.buffer, cv2.COLOR_RGB2BGR))

        # frame = frames[CAM_RIGHT_WRIST]
        # path = os.path.join(outdir, 'observations/cam_right_wrist', "%.4f.jpg" % frame.timestamp)
        # cv2.imwrite(path, cv2.cvtColor(frame.buffer, cv2.COLOR_RGB2BGR))

def main(running, outdir, log_queue):
    configure_worker_logger(log_queue, _logger)
    _logger.info('Initializing camera...')
    try:
        ca = camera.CameraArray([CAM_HEAD])
    except Exception as e:
        _logger.exception("Failed to initialize camera:")
        return

    _logger.info('Initialized camera successfully!')

    while not running.is_set():
        ca.wait_for_frames()

    dataset = []
    while running.is_set():
        frames = ca.wait_for_frames()
        dataset.append(frames)
        frame = frames[CAM_HEAD]

        frame_bgr = cv2.cvtColor(frame.buffer, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera Head", frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running.clear()
            break
        
        if len(dataset) % 50 == 0:
            _logger.info('%i frames' % len(dataset))

    cv2.destroyAllWindows()
    _logger.info("Camera stream ended")
    save_to_disk(dataset, outdir)
    _logger.info("Camera stream saved")
