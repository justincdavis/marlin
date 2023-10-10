from __future__ import annotations

import pathlib
from queue import Queue, Empty
from threading import Thread

import joblib
import cv2
import numpy as np

from ._change import ChangeDetector
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
        self._change_dect = ChangeDetector(forest)
        self._ncc_threshold = ncc_threshold
        self._last_frame = None
        self._use_dnn = True
        self._last_bboxs: list[tuple[int, tuple[int, int, int, int], float]] | None = None
        self._last_ncc: float | None = None

    def __call__(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Run the Marlin algorithm on the (next) frame"""
        orig_frame = frame.copy()
        if len(frame.shape) == 4:
            frame = np.transpose(frame.squeeze(), (1, 2, 0))
        cv_frame = frame.copy() 
        cd_frame = frame.copy()
        if self._use_dnn or self._last_bboxs is None:
            self._use_dnn = False
            self._last_bboxs = self._dnn(orig_frame)
            self._last_frame = cv_frame
            self._tracker.init(self._last_frame, self._last_bboxs)
        else:
            self._last_bboxs = self._tracker.run(cv_frame)
            self._last_frame = cv_frame
            self._last_ncc = min(self._last_bboxs, key=lambda x: x[2])[2]
            self._use_dnn = self._last_ncc <= self._ncc_threshold

        # RUN CHANGE DECT ALWAYS
        result = self._change_dect(cd_frame, self._last_bboxs)
        if result[0]:
            self._use_dnn = True
        
        return self._last_bboxs
    