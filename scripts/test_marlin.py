import cv2
import numpy as np
from marlin import Marlin

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", type=str, help="Path to the forest file", required=True)
    parser.add_argument("--video", type=str, help="Path to the video file", required=True)
    args = parser.parse_args()

    def fake_dnn(img: np.ndarray):
        return [(0, (1400, 400, 1500, 500), 0.9)]

    m = Marlin(fake_dnn, args.forest)

    vid = cv2.VideoCapture(args.video)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        display = frame.copy()
        detections = m(frame)
        for label, bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Frame", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
