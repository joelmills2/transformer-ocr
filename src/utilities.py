import os
import json
import cv2
from pathlib import Path


def load_dataset(directory):
    """
    Load images and corresponding annotations from the dataset directory (FUNSD dataset).

    Args:
        dataset_dir (str): The directory where the FUNSD dataset is stored.

    Returns:
        A list of tuples, where each tuple contains an image and its corresponding annotations.
    """
    data = []
    images_directory = Path(directory) / "images"
    annotations_directory = Path(directory) / "annotations"

    for annotation_path in annotations_directory.glob("*.json"):
        image_path = images_directory / (annotation_path.stem + ".png")

        if image_path.exists():
            image = cv2.imread(str(image_path))
            with open(annotation_path, "r") as f:
                annotations = json.load(f)
            data.append((image, annotations))

    return data
