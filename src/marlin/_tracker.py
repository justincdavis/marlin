import cv2
import numpy as np


class MultiBoxTracker:
    def __init__(self, nfeatures: int = 500):
        self._orb = cv2.ORB_create(nfeatures=nfeatures)
        self._lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self._prev_frame: np.ndarray | None = None
        self._prev_dects: list[tuple[int, tuple[int, int, int, int], float]] | None = None

    @staticmethod
    def _ncc(img1: np.ndarray, img2: np.ndarray) -> float:
        """Normalized cross correlation between two images"""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        std1 = np.std(img1)
        std2 = np.std(img2)
        norm1 = (img1 - mean1) / std1
        norm2 = (img2 - mean2) / std2
        area = norm1.shape[0] * norm1.shape[1]
        return (1.0 / area) * np.sum(norm1 * norm2)

    @staticmethod
    def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        """Calculate bounding box from list of points"""
        x_coords, y_coords = zip(*points)
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    
    @staticmethod
    def _resize_bbox(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Resizes a bounding box while keeping the same center"""
        x1, y1, x2, y2 = bbox
        w, h = size
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return max(cx - w // 2, 0), max(cy - h // 2, 0), cx + w // 2, cy + h // 2

    def _track_objects(
        self,
        img: np.ndarray, 
        prev_img: np.ndarray,
        prev_detections: list[tuple[int, tuple[int, int, int, int], float]], 
    ) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Track objects using optical flow and ncc"""
        new_dect = []
        for object_id, bbox, _ in prev_detections:
            try:
                # extract orb in prev_frame
                x1, y1, x2, y2 = bbox
                prev_bbox_frame = prev_img[x1:x2, y1:y2]
                prev_kp, prev_desc = self._orb.detectAndCompute(prev_bbox_frame, None)
                prev_kp = np.asarray([p.pt for p in prev_kp], dtype=np.float32)
                # lucas-kanade
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, img, prev_kp, None, **self._lk_params)
                # filter out bad points 
                good_new = np.where(st == 1, p1, None)
                good_new = [p.ravel() for p in good_new]
                good_new = [(int(x), int(y)) for x, y in good_new if x is not None and y is not None]
                # generate bbox
                bbox = self._bbox_from_points(good_new)
                # bbox = self._resize_bbox(bbox, (x2 - x1, y2 - y1))
                bx1, by1, bx2, by2 = bbox
                # # offset back to original frame
                # bx1, by1, bx2, by2 = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                # compute ncc as the new confidence
                ncc = self._ncc(prev_bbox_frame, img[bx1:bx2, by1:by2])
                new_dect.append((object_id, (bx1, by1, bx2, by2), ncc))
            except Exception:
                new_dect.append((object_id, bbox, 0.0))
        return new_dect
    
    def init(self, frame: np.ndarray, detections: list[tuple[int, tuple[int, int, int, int], float]]) -> None:
        """Initialize the tracker"""
        self._prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_dects = detections

    def run(self, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Runs tracking on the next frame"""
        if self._prev_frame is None or self._prev_dects is None:
            raise RuntimeError("Tracker must be initialized before running")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_dects = self._track_objects(gray, self._prev_frame, self._prev_dects)
        self._prev_frame = gray
        self._prev_dects = new_dects
        return new_dects
    