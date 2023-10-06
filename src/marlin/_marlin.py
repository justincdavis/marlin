from __future__ import annotations

from queue import Queue, Empty
from threading import Thread

import cv2
import numpy as np

from _tracker import MultiBoxTracker
 

class Marlin:
    def __init__(
            self, 
            dnn: callable[[np.ndarray], list[tuple[int, tuple[int, int, int, int], float]]],
            ncc_threshold: float = 0.3,
            nfeatures: int = 500,
        ) -> None:
        """Use to create a Marlin object."""
        self._tracker = MultiBoxTracker(nfeatures=nfeatures)
        self._dnn = dnn
        self._ncc_threshold = ncc_threshold
        self._last_frame = None
        self._use_dnn = True
        self._last_bboxs: list[tuple[int, tuple[int, int, int, int], float]] | None = None
        self._last_ncc: float | None = None
        self._stopped = False
        self._dnn_queue: Queue[np.ndarray] = Queue()
        self._dnn_thread = Thread(target=self._dnn_worker, daemon=True)
        self._dnn_thread.start()
        self._change_dect_queue: Queue[np.ndarray] = Queue()
        self._change_dect_thread = Thread(target=self._change_dect_worker, daemon=True)
        self._change_dect_thread.start()

    def __del__(self) -> None:
        self._stopped = True

    def _run_dnn(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Run the DNN on the frame"""
        self._last_bboxs = self._dnn(frame)
        self._last_frame = frame
        self._tracker.init(self._last_frame, self._last_bboxs)
        return self._last_bboxs
    
    def _dnn_worker(self) -> None:
        """Worker for the DNN thread"""
        while not self._stopped:
            try:
                frame = self._dnn_queue.get(block=True, timeout=0.25)
                self._run_dnn(frame)
                self._dnn_queue.task_done()
            except Empty:
                continue

    def _run_change_dect(self, frame: np.ndarray) -> bool:
        """Run the change detection on the frame"""
        # Step (i): White out objects
        new_frame = frame.copy()
        for label, bbox, conf in self._last_bboxs:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(new_frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # Step (ii): Resize and compute histograms
        resized_colored_image = cv2.resize(new_frame, (128, 128))
        hist_red = cv2.calcHist([resized_colored_image], [0], None, [256], [0, 256])
        hist_green = cv2.calcHist([resized_colored_image], [1], None, [256], [0, 256])
        hist_blue = cv2.calcHist([resized_colored_image], [2], None, [256], [0, 256])

        # Step (iii): Flatten the resized_colored_image and append histograms
        feature_vector = resized_colored_image.reshape(1, -1)  # Flatten as a 1D array
        feature_vector = feature_vector.astype(float)  # Convert to float

        # Append histograms to the feature vector
        feature_vector = np.concatenate([feature_vector, hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()], axis=None)

        

    def _change_dect_worker(self) -> None:
        """Worker for the change detection thread"""
        while not self._stopped:
            try:
                frame = self._change_dect_queue.get(block=True, timeout=0.25)
                self._run_change_dect(frame)
                self._change_dect_queue.task_done()
            except Empty:
                continue
    
    def _run_tracker(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Run the tracker on the frame"""
        self._last_bboxs = self._tracker.run(frame)
        self._last_frame = frame
        self._last_ncc = min(self._last_bboxs, key=lambda x: x[2])
        return self._last_bboxs

    def __call__(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Run the Marlin algorithm on the (next) frame"""
        self._change_dect_queue.put(frame)
        if self._last_bboxs is None or self._use_dnn:
            if self._use_dnn:
                self._use_dnn = False
            self._dnn_queue.put(frame)
            if self._last_bboxs is None:  # first frame, otherwise let it update in the background
                self._dnn_queue.join()
            return self._last_bboxs
        else:
            new_bboxs = self._tracker.run(frame)
            self._last_ncc = min(new_bboxs, key=lambda x: x[2])[2]
            self._use_dnn = self._last_ncc <= self._ncc_threshold
            return new_bboxs

if __name__ == "__main__":
    def fake_dnn(img: np.ndarray):
        return [(0, (1400, 400, 1500, 500), 0.9)]

    m = Marlin(fake_dnn, nfeatures=5000)

    vid = cv2.VideoCapture("P2D2.mp4")

    while True:
        ret, frame = vid.read()
        display = frame.copy()
        if not ret:
            break
        detections = m(frame)
        for label, bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Frame", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    