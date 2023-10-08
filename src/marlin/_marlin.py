from __future__ import annotations

import pathlib
from queue import Queue, Empty
from threading import Thread

import joblib
import cv2
import numpy as np

from ._tracker import MultiBoxTracker
 

class Marlin:
    def __init__(
            self, 
            dnn: callable[[np.ndarray], list[tuple[int, tuple[int, int, int, int], float]]],
            forest: str | pathlib.Path,
            ncc_threshold: float = 0.3,
            nfeatures: int = 500,
        ) -> None:
        """Use to create a Marlin object."""
        self._tracker = MultiBoxTracker(nfeatures=nfeatures)
        self._dnn = dnn
        self._forest = joblib.load(forest)
        self._ncc_threshold = ncc_threshold
        self._last_frame = None
        self._use_dnn = True
        self._last_bboxs: list[tuple[int, tuple[int, int, int, int], float]] | None = None
        self._last_ncc: float | None = None
        self._stopped = False
        self._dnn_running = False
        self._dnn_queue: Queue[np.ndarray] = Queue()
        self._dnn_thread = Thread(target=self._dnn_worker, daemon=True)
        self._dnn_thread.start()
        self._change_dect_queue: Queue[np.ndarray] = Queue()
        self._change_dect_thread = Thread(target=self._change_dect_worker, daemon=True)
        self._change_dect_thread.start()

    def __del__(self) -> None:
        self._stopped = True

    def _run_dnn(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]] | None:
        """Run the DNN on the frame"""
        self._last_bboxs = self._dnn(frame)
        # this is a fix for allowing PyTorch style shapes (B, C, H, W)
        # convert the frame to (H, W, C) and assume batch size is one
        if len(frame.shape) == 4:
            frame = np.transpose(frame.squeeze(), (1, 2, 0))
        self._last_frame = frame
        if self._last_bboxs is None:
            return None
        self._tracker.init(self._last_frame, self._last_bboxs)
        return self._last_bboxs
    
    def _dnn_worker(self) -> None:
        """Worker for the DNN thread"""
        while not self._stopped:
            try:
                frame = self._dnn_queue.get(block=True, timeout=0.25)
                self._dnn_running = True
                self._run_dnn(frame)
                self._dnn_running = False
                self._dnn_queue.task_done()
            except Empty:
                continue

    def _run_change_dect(self, frame: np.ndarray) -> None:
        """Run the change detection on the frame"""
        new_frame = frame.copy()
        if self._dnn_running:
            return
        if self._last_bboxs is None:
            if self._dnn_running:
                return
            else:
                self._use_dnn = True
                return
        # Step (i): White out objects
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

        result = self._forest.predict([feature_vector])

        if result[0]:
            self._use_dnn = True

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
        self._last_ncc = min(self._last_bboxs, key=lambda x: x[2])[2]
        self._use_dnn = self._last_ncc <= self._ncc_threshold
        return self._last_bboxs

    def __call__(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Run the Marlin algorithm on the (next) frame"""
        dnn_frame = frame.copy()
        # this is a fix for allowing PyTorch style shapes (B, C, H, W)
        # convert the frame to (H, W, C) and assume batch size is one
        if len(frame.shape) == 4:
            frame = np.transpose(frame.squeeze(), (1, 2, 0))
        change_dect_frame = frame.copy()
        tracker_frame = frame.copy()
        self._change_dect_queue.put(change_dect_frame)
        if self._last_bboxs is None or self._use_dnn:
            if self._use_dnn:
                self._use_dnn = False
            self._dnn_queue.put(dnn_frame)
            if self._last_bboxs is None:  # first frame, otherwise let it update in the background
                self._dnn_queue.join()
            return self._last_bboxs
        else:
            new_bboxs = self._run_tracker(tracker_frame)
            return new_bboxs
