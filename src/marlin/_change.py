from __future__ import annotations

import pathlib

import cv2
import joblib
import numpy as np


class ChangeDetector:
    def __init__(self, path: str | pathlib.Path) -> None:
        self._forest = joblib.load(path)

    def __call__(self, frame, detections):
        # Step (i): White out objects
        for label, bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # Step (ii): Resize and compute histograms
        resized_colored_image = cv2.resize(frame, (128, 128))
        hist_red = cv2.calcHist([resized_colored_image], [0], None, [256], [0, 256])
        hist_green = cv2.calcHist([resized_colored_image], [1], None, [256], [0, 256])
        hist_blue = cv2.calcHist([resized_colored_image], [2], None, [256], [0, 256])

        # Step (iii): Flatten the resized_colored_image and append histograms
        feature_vector = resized_colored_image.reshape(1, -1)  # Flatten as a 1D array
        feature_vector = feature_vector.astype(float)  # Convert to float

        # Append histograms to the feature vector
        feature_vector = np.concatenate([feature_vector, hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()], axis=None)

        result = self._forest.predict([feature_vector])

        return result
