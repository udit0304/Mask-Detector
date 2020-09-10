import os

os.environ["DARKNET_PATH"] = "/home/dokania/darknet"
from models.Yolo import Yolo


class MaskDetector:
    def __init__(self, detector='yolo', config='cfg/yolov4_mask.ini', resize=False, confidence=0.65):
        self.mask_detector = None
        if detector == 'yolo':
            self.mask_detector = Yolo(config, resize)
        self.min_score = confidence

    def get_licence_plate(self, frame):
        plate_results = self.mask_detector.detect(frame, self.min_score)
        return self.mask_detector.draw_box(plate_results, frame)
