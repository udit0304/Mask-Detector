import cv2
import os
import numpy as np
from models import darknet
import configparser

class Yolo:
    def __init__(self, ini_file, resize=False, batch_size=1):
        self.classNames = None
        self.netMain = None
        self.colors = None
        self.resize = resize
        self.batch_size = batch_size

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
                                                                              batch_size=self.batch_size)

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

    def prepare_batch(self, images, channels=3):
        width = darknet.network_width(self.netMain)
        height = darknet.network_height(self.netMain)

        darknet_images = []
        for image in images:
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            custom_image = image_resized.transpose(2, 0, 1)
            darknet_images.append(custom_image)

        batch_array = np.concatenate(darknet_images, axis=0)
        batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32) / 255.0
        darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
        return darknet.IMAGE(width, height, channels, darknet_images)

    def batch_detect(self, img_arr, in_thresh=0.25, hier_thresh=.5, nms=.45):
        image_height, image_width, _ = self.check_batch_shape(img_arr, self.batch_size)
        width = darknet.network_width(self.netMain)
        height = darknet.network_height(self.netMain)
        darknet_images = self.prepare_batch(img_arr)
        batch_detections = darknet.network_predict_batch(self.netMain, darknet_images, self.batch_size, image_width, image_height, in_thresh, hier_thresh, None, 0, 0)
        batch_predictions = []
        for idx in range(self.batch_size):
            num = batch_detections[idx].num
            detections = batch_detections[idx].dets
            if nms:
                darknet.do_nms_obj(detections, num, len(self.classNames), nms)
            predictions = darknet.remove_negatives(detections, self.classNames, num)
            img_arr[idx] = self.draw_box(predictions, img_arr[idx]) #darknet.draw_boxes(predictions, img_arr[idx], self.colors)
            batch_predictions.append(predictions)
        darknet.free_batch_detections(batch_detections, self.batch_size)
        return img_arr, batch_predictions


    def get_bbpoints(self, bb):
        return darknet.bbox2points(bb)

    def draw_box(self, detection, frame):
        return darknet.draw_boxes(detection, frame, self.colors)

    def check_batch_shape(self, images, batch_size):
        """
            Image sizes should be the same width and height
        """
        shapes = [image.shape for image in images]
        if len(set(shapes)) > 1:
            raise ValueError("Images don't have same shape")
        if len(shapes) > batch_size:
            raise ValueError("Batch size higher than number of images")
        return shapes[0]
