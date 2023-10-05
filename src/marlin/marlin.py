from __future__ import annotations

import cv2
import numpy as np
 

class Marlin:
    def __init__(self, dnn: callable[[np.ndarray], tuple[int, tuple[int, int, int, int], float]]):
        self._dnn = dnn
        self._orb = cv2.ORB_create()
        self._lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self._last_frame = None
        self._last_bboxs: list[tuple[int, tuple[int, int, int, int], float]] | None = None
        self._last_kp = None
        self._last_des = None

    @staticmethod
    def _ncc(img1: np.ndarray, img2: np.ndarray) -> float:
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        std1 = np.std(img1)
        std2 = np.std(img2)
        norm1 = (img1 - mean1) / std1
        norm2 = (img2 - mean2) / std2
        area = norm1.shape[0] * norm1.shape[1]
        return (1.0 / area) * np.sum(norm1 * norm2)

    def _track_objects(
        self,
        img: np.ndarray, 
        prev_img: np.ndarray,
        prev_detections: list[tuple[int, tuple[int, int, int, int], float]], 
    ) -> list[tuple[int, tuple[int, int, int, int], float]]:
        new_dect = []
        for object_id, bbox, conf in prev_detections:
            # extract orb in prev_frame
            x1, y1, x2, y2 = bbox
            prev_bbox_frame = prev_img[x1:x2, y1:y2]
            prev_kp, prev_desc = self._orb.detectAndCompute(prev_bbox_frame, None)
            
            # extract orb in current_frame
            h, w, _ = img.shape
            x1 = max(0, x1 - 15)
            y1 = max(0, y1 - 15)
            x2 = min(w, x2 + 15)
            y2 = min(h, y2 + 15)
            curr_bbox_frame = img[x1:x2, y1:y2]
            curr_kp, prev_desc = self._orb.detectAndCompute(curr_bbox_frame, None)
            # Calculate optical flow using Lucas-Kanade
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, img, prev_kp, curr_kp, **self._lk_params)
            # Filter out points with status 1 (successfully tracked)
            good_new = p1[st == 1]
            good_old = self._last_kp[st == 1]

        return new_dect

    def __call__(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        if self._last_bboxs is None:
            label, bbox, conf = self._dnn(frame)
            self._last_bboxs = [(label, bbox, conf)]
            self._last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return self._last_bboxs
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, desc = self._orb.detectAndCompute(gray, None)
            if self._last_kp is None:
                self._last_kp, self._last_des = self._orb.detectAndCompute(self._last_frame, None)
            # Calculate optical flow using Lucas-Kanade
            p1, st, err = cv2.calcOpticalFlowPyrLK(self._last_frame, gray, self._last_kp, None, **self._lk_params)

            # Filter out points with status 1 (successfully tracked)
            good_new = p1[st == 1]
            good_old = self._last_kp[st == 1]

            for pt in good_new:


if __name__ == "__main__":

    m = Marlin()

    