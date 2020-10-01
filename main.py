import os
import cv2
import glob
import time
import threading
from yolo_anno import generate_config_files
from libs.mask_detector import MaskDetector


class myThread(threading.Thread):
    def __init__(self, video, detector, batch):
        threading.Thread.__init__(self)
        self.video = video
        self.detector = detector
        self.batch = batch

    def run(self):
        threadLimiter.acquire()
        try:
            print("Detecting for Video: " + self.video)
            main(self.video, self.detector, self.batch)
            # del self.detector
        finally:
            print("Finish detection for video: " + self.video)
            threadLimiter.release()


threadLimiter = threading.BoundedSemaphore(2)


def main(video, pmd, batch):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    img_arry = []
    while cap.isOpened():
        ret, frame = cap.read()
        if video_writer is None:
            video_writer = cv2.VideoWriter("detection/" + os.path.basename(video), fourcc, fps,
                                           (frame.shape[1], frame.shape[0]))
        if not ret:
            print(len(img_arry))
            cap.release()
            video_writer.release()
            continue
        if frame is not None and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_arry.append(frame)
            if len(img_arry) != batch:
                continue
            # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = pmd.get_licence_plate(img_arry)
            for detection in detections:
                detection_bgr = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
                video_writer.write(detection_bgr)
            img_arry = []


if __name__ == "__main__":
    generate_config_files()
    videos = glob.glob("./test/*.mp4")
    threads = []
    running_t = []
    batch = 4
    pmd = MaskDetector(detector='yolo', config='cfg/yolov4_mask.ini', resize=False, batch=batch)
    start = time.time()
    for video in videos:
        t = myThread(video, pmd, batch)
        t.start()
        running_t.append(t)
    for t in running_t:
        t.join()
    end = time.time()
    duration = (end - start)
    del pmd
    print("duration {}s".format(round(duration, 2)))
