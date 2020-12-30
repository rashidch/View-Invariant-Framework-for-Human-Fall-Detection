import os
import cv2
import time
import torch
import numpy as np

from queue import Queue
from threading import Thread, Lock

class CamLoader:
    """Use threading and queue to capture a frame and store to queue for pickup in sequence.
    Recommend for video file.

    Args:
        camera: (int, str) Source of camera or video.,
        batch_size: (int) Number of batch frame to store in queue. Default: 1,
        queue_size: (int) Maximum queue size. Default: 256,
        preprocess: (Callable function) to process the frame before return.
    """
    def __init__(self, camera, batch_size=1, queue_size=256, preprocess=None, output='1.avi'):
        self.stream = cv2.VideoCapture(camera)
        assert self.stream.isOpened(), 'Cannot read camera source!'
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.stream.get(3)),int(self.stream.get(4)))
        self.frame_width  = int(self.stream.get(3))
        self.frame_height = int(self.stream.get(4))
        self.out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (self.frame_width,self.frame_height))
        # Queue for storing each frames.
        self.stopped = False
        self.batch_size = batch_size
        self.Q = Queue(maxsize=queue_size)

        self.preprocess_fn = preprocess
    
    def save_video(self, frame, file_name='1.avi'):

        self.out.write(frame.astype('uint8'))

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        time.sleep(0.5)
        return self

    def update(self):
        while not self.stopped:
            if not self.Q.full():
                frames = []
                for k in range(self.batch_size):
                    ret, frame = self.stream.read()
                    if not ret:
                        self.stop()
                        return

                    if self.preprocess_fn is not None:
                        frame = self.preprocess_fn(frame)

                    frames.append(frame)
                    frames = np.stack(frames)
                    self.Q.put(frames)
            else:
                with self.Q.mutex:
                    self.Q.queue.clear()
            #time.sleep(0.05)

    def grabbed(self):
        """Return `True` if can read a frame."""
        return self.Q.qsize() > 0

    def getitem(self):
        return self.Q.get().squeeze()

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        #self.stream.release()

    def __len__(self):
        return self.Q.qsize()

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream.isOpened():
            self.stream.release()