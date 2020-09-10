import cv2
import os
from models import darknet
import configparser


# def convertBack(x, y, w, h):
#     xmin = int(round(x - (w / 2)))
#     xmax = int(round(x + (w / 2)))
#     ymin = int(round(y - (h / 2)))
#     ymax = int(round(y + (h / 2)))
#     return xmin, ymin, xmax, ymax
#
#
# def convertdetections(detections):
#     detections2 = []
#     for detection in detections:
#         if detection[0] != b'person':  # maybe dont want only the people so could remove it
#             continue
#         x, y, w, h = detection[2][0], \
#                      detection[2][1], \
#                      detection[2][2], \
#                      detection[2][3]
#         detections2.append(convertBack(
#             float(x), float(y), float(w), float(h)))
#     return detections2


class Yolo:
    def __init__(self, ini_file, resize=False):
        self.classNames = None
        self.netMain = None
        self.colors = None
        self.resize = resize

        config = configparser.RawConfigParser()
        config.read(ini_file)
        configPath = config.get('yolo', 'cfg_file')
        weightPath = config.get('yolo', 'weight_file')
        metaPath = config.get('yolo', 'data_file')

        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        if self.netMain is None:
            self.netMain, self.classNames, self.colors = darknet.load_network(configPath, metaPath, weightPath,
                                                                              batch_size=1)

        if self.resize:
            self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                    darknet.network_height(self.netMain), 3)
        else:
            self.darknet_image = None

    def detect(self, img, in_thresh=0.25):
        if self.resize:
            frame_resized = cv2.resize(img, (darknet.network_width(self.netMain), darknet.network_height(self.netMain)),
                                       interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        else:
            if self.darknet_image is None:
                self.darknet_image = darknet.make_image(img.shape[1], img.shape[0], 3)
            if self.darknet_image.w != img.shape[1] or self.darknet_image.h != img.shape[0]:
                self.darknet_image = darknet.make_image(img.shape[1], img.shape[0], 3)

            darknet.copy_image_from_bytes(self.darknet_image, img.tobytes())

        detections = darknet.detect_image(self.netMain, self.classNames, self.darknet_image, thresh=in_thresh)
        return detections

    def get_bbpoints(self, bb):
        return darknet.bbox2points(bb)

    def draw_box(self, detection, frame):
        return darknet.draw_boxes(detection, frame, self.colors)
