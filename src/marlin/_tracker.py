from __future__ import annotations

import math

import cv2
import numpy as np


def sanitize_bbox(bbox: tuple[int, int, int, int], width: int, height: int, min_size: int = 10) -> tuple[int, int, int, int]:        
    def change_pair(cords: tuple[int, int], maxval: int, minval: int, min_size: int = 10) -> tuple[int, int]:
        c1, c2 = cords
        counter = 0
        while counter < 3:
            diff = c2 - c1
            if diff < min_size:
                offset = int((min_size - diff) / 2)
                if c1 > (minval + offset):
                    c1 -= offset
                if c2 < (maxval - offset):
                    c2 += offset
            else:
                break
            counter += 1
        return c1, c2
    x1, y1, x2, y2 = bbox
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, width)
    x1, x2 = change_pair((x1, x2), width, 0)
    y1, y2 = change_pair((y1, y2), height, 0)
    return x1, y1, x2, y2


class LucasKanadeTracker:
    def __init__(
            self, 
            nfeatures: int = 500, 
            lk_params: dict = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
        ) -> None:
        self._prev_frame = None
        self._prev_roi = None
        self._prev_keypoints = None
        self._orb = cv2.ORB_create(nfeatures=nfeatures)
        self._lk_params = lk_params

    @staticmethod
    def _ncc(img1: np.ndarray, img2: np.ndarray) -> float:
        """Normalized cross correlation between two images"""
        if img1.shape != img2.shape:
            h1, w1 = img1.shape
            h2, w2 = img2.shape
            h, w = (h1, w1) if (h1 * w1) < (h2 * h2) else (h2, w2)
            # print(f"  Image shapes do not match: {img1.shape} != {img2.shape}")
            # print(f"Resizing images to: {h, w}")
            img1 = cv2.resize(img1, (h, w))
            img2 = cv2.resize(img2, (h, w))
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        std1 = np.std(img1, ddof=2)
        std2 = np.std(img2, ddof=2)
        norm1 = (img1 - mean1) / std1
        norm2 = (img2 - mean2) / std2
        area = norm1.shape[0] * norm1.shape[1]
        return (1.0 / area) * np.sum(norm1 * norm2)

    def init(self, frame, bounding_box):
        # print("Starting LK Tracker init")
        if len(frame.shape) == 3:
            # print("  Converted to gray")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = bounding_box
        # print(frame.shape)
        bounding_box = sanitize_bbox(bounding_box, frame.shape[1], frame.shape[0])
        # print(f"   BBOX: {bounding_box}")
        self._prev_roi = frame[x1:x2, y1:y2]
        # print(f"   ROI: {self._prev_roi.shape}")
        self._prev_frame = frame
        self._prev_keypoints = self._detect_keypoints(self._prev_roi)

    def update(self, frame):
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_frame is None or self._prev_roi is None or self._prev_keypoints is None:
            raise Exception("Tracker has not been initialized!")

        bbox = self._track(frame)
        if bbox is None:
            self._prev_frame = None
            self._prev_roi = None
            self._prev_keypoints = None
            return None
        
        x1, y1, x2, y2 = bbox
        # print(f"bbox: {bbox}")
        bbox = sanitize_bbox(bbox, frame.shape[1], frame.shape[0])
        # print(f"bbox: {bbox}")
        ncc = self._ncc(self._prev_roi, frame[x1:x2, y1:y2])
        self.init(frame, bbox)

        return ncc, bbox

    def _detect_keypoints(self, image):
        keypoints = self._orb.detect(image, None)
        return np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    
    def _track(self, frame):
        try:
            current_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_frame, frame, self._prev_keypoints, None, **self._lk_params)
        except Exception:
            return None

        mask = status.ravel() == 1
        current_keypoints = current_keypoints[mask]
        x, y, w, h = cv2.boundingRect(current_keypoints)
        x1, y1, x2, y2 = x, y, x + w, y + h

        return x1, y1, x2, y2
    

class MultiBoxTracker:
    def __init__(
            self, 
            nfeatures: int = 500, 
            lk_params: dict = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        ) -> None:
        self._nfeatures = nfeatures
        self._lk_params = lk_params
        self._prev_frame: np.ndarray | None = None
        self._prev_dects: list[tuple[int, tuple[int, int, int, int], float]] | None = None
        self._trackers: list[LucasKanadeTracker] = []
    
    def init(self, frame: np.ndarray, detections: list[tuple[int, tuple[int, int, int, int], float]]) -> None:
        """Initialize the tracker"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_frame = frame
        self._prev_dects = detections
        self._trackers = []
        for detection in self._prev_dects:
            # print(f"Detection: {detection}")
            tracker = LucasKanadeTracker()
            bbox = detection[1]
            bbox = sanitize_bbox(bbox, frame.shape[1], frame.shape[0])
            # if bbox != detection[1]:
            #     print(f"Changed bbox: {detection[1]} -> {bbox}")
            tracker.init(self._prev_frame, bbox)
            self._trackers.append(tracker)

    def run(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Runs tracking on the next frame"""
        if self._prev_frame is None or self._prev_dects is None:
            raise RuntimeError("Tracker must be initialized before running")
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_dects = []
        for (object_id, _, _), tracker in zip(self._prev_dects, self._trackers):
            data = tracker.update(frame)
            if data is None:
                continue
            score, bbox = data
            new_dects.append((object_id, bbox, score))
        self._prev_frame = frame
        self._prev_dects = new_dects
        return new_dects
    