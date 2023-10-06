from __future__ import annotations

import os

import joblib
import cv2
from tqdm import tqdm
import numpy as np
import pybboxes as pbx
from sklearn.ensemble import RandomForestClassifier


def get_feature_vector(image: np.ndarray) -> np.ndarray:
    hist_red = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_blue = cv2.calcHist([image], [2], None, [256], [0, 256])

    # Step (iii): Flatten the resized_colored_image and append histograms
    feature_vector = image.reshape(1, -1)  # Flatten as a 1D array
    feature_vector = feature_vector.astype(float)  # Convert to float

    # Append histograms to the feature vector
    feature_vector = np.concatenate([feature_vector, hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()], axis=None)
    return feature_vector

def preprocess_image(image: np.ndarray, labels: list[tuple[int, int, int, int]]) -> list[(np.ndarray, bool)]:
    """
    Preprocesses an image for training.

    :param image: Image to preprocess.
    :return: Preprocessed image.
    """
    frames = [image.copy() for _ in range(4)]
    feature_vectors = []
    for frame in frames:
        if len(feature_vectors) == 0:
            for bbox in labels:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
            resized_colored_image = cv2.resize(frame, (128, 128))
            feature_vector = get_feature_vector(resized_colored_image)
            feature_vectors.append((feature_vector, False))
        elif len(feature_vectors) == 1:
            resized_colored_image = cv2.resize(frame, (128, 128))
            feature_vector = get_feature_vector(resized_colored_image)
            feature_vectors.append((feature_vector, True))
    return feature_vectors

def main():
    forest = RandomForestClassifier(n_estimators=50, max_depth=20)

    train_dir = args.train
    image_dir = os.path.join(train_dir, "images")
    label_dir = os.path.join(train_dir, "labels")
    # get the image names from the label dir
    image_names = [n.split(".")[0] for n in os.listdir(image_dir)]
    label_names = [n.split(".")[0] for n in os.listdir(label_dir)]
    # get the common names
    common_names = set(image_names).intersection(set(label_names))
    def load_feature_vectors(common_name: str) -> list[(np.ndarray, bool)]:
        try:
            image_path = os.path.join(image_dir, common_name + ".jpg")
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            with open(os.path.join(label_dir, common_name + ".txt")) as f:
                labels = f.readlines()
            labels = [map(float, label.split(" ")) for label in labels]
            labels = [(x, y, w, h) for _, x, y, w, h in labels]
            labels = [pbx.convert_bbox(bbox, from_type="yolo", to_type="voc", image_size=(width, height)) for bbox in labels]
            vectors = preprocess_image(image, labels)
            return vectors
        except AttributeError:
            return None
    for common_name in tqdm(common_names):
        feature_vectors = load_feature_vectors(common_name)
        if feature_vectors is None:
            continue
        vectors, labels = zip(*feature_vectors)
        forest.fit(vectors, labels)
    joblib.dump(forest, "forest.joblib")

    # load the test data and test the forest
    test_dir = args.test
    image_dir = os.path.join(test_dir, "images")
    label_dir = os.path.join(test_dir, "labels")
    # get the image names from the label dir
    image_names = [n.split(".")[0] for n in os.listdir(image_dir)]
    label_names = [n.split(".")[0] for n in os.listdir(label_dir)]
    # get the common names
    common_names = set(image_names).intersection(set(label_names))
    correct = 0
    num_features = 0
    for common_name in tqdm(common_names):
        feature_vectors = load_feature_vectors(common_name)
        if feature_vectors is None:
            continue
        num_features += len(feature_vectors)
        vectors, labels = zip(*feature_vectors)
        predictions = forest.predict(vectors)
        for prediction, label in zip(predictions, labels):
            if prediction == label:
                correct += 1
    print(f"Accuracy: {correct / num_features}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Directory containing training images and labels in the yolo format")
    parser.add_argument("--test", type=str, help="Directory containing testing images and labels in the yolo format")
    args = parser.parse_args()
    main()
