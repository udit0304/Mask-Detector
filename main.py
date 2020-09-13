import os
import cv2
import glob
import time
import threading
from yolo_anno import generate_config_files
from libs.mask_detector import MaskDetector


class myThread(threading.Thread):
    def __init__(self, video, detector):
        threading.Thread.__init__(self)
        self.video = video
        self.detector = detector

    def run(self):
        threadLimiter.acquire()
        try:
            print("Detecting for Video: " + self.video)
            main(self.video, self.detector)
            # del self.detector
        finally:
            print("Finish detection for video: " + self.video)
            threadLimiter.release()


threadLimiter = threading.BoundedSemaphore(1)


def main(video, pmd):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    while cap.isOpened():
        ret, frame = cap.read()
        if video_writer is None:
            video_writer = cv2.VideoWriter("detection/" + os.path.basename(video), fourcc, fps,
                                           (frame.shape[1], frame.shape[0]))
        if not ret:
            cap.release()
            video_writer.release()
            continue
        if frame is not None and frame.shape[2] == 3:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = pmd.get_licence_plate(image_rgb)
            detection_bgr = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
            video_writer.write(detection_bgr)


if __name__ == "__main__":
    generate_config_files()
    videos = glob.glob("./test/*.mp4")
    threads = []
    running_t = []
    pmd = MaskDetector(detector='yolo', config='cfg/yolov4_mask.ini', resize=False)
    start = time.time()
    for video in videos:
        t = myThread(video, pmd)
        t.start()
        running_t.append(t)
    for t in running_t:
        t.join()
    end = time.time()
    duration = (end - start)
    del pmd
    print("duration {}s".format(round(duration, 2)))
