import os
import json
import cv2
from pathlib import Path
import random


def load_dataset(directory, sample_size=15, seed=9):
    """
    Load images and corresponding annotations from the dataset directory (FUNSD dataset).

    Args:
        directory (str): The directory where the FUNSD dataset is stored.
        sample_size (int): The number of images to randomly select for testing.
        seed (int): The seed for the random number generator.

    Returns:
        A list of tuples, where each tuple contains an image and its corresponding annotations.
    """
    random.seed(seed)  # Set the seed for the random number generator

    data = []
    images_directory = Path(directory) / "images"
    annotations_directory = Path(directory) / "annotations"

    # Get a random sample of annotation files
    annotation_paths = list(annotations_directory.glob("*.json"))
    annotation_sample = random.sample(annotation_paths, sample_size)

    for annotation_path in annotation_sample:
        image_path = images_directory / (annotation_path.stem + ".png")

        if image_path.exists():
            image = cv2.imread(str(image_path))
            with open(annotation_path, "r") as f:
                annotations = json.load(f)
            data.append((image, annotations))

    return data
