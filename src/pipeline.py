import cv2
import numpy as np
from detection import get_detection, preprocess_image
from recognition import get_recognition, load_trocr_model


def ocr_pipeline(image_path):
    """
    Run the OCR pipeline: detection -> recognition.

    CRAFT (Clova) is used for text detection and TrOCR (Microsoft) is used for text recognition.
    """
    # Load the models
    processor, model = load_trocr_model()

    # Preprocess image
    preprocessed_image = preprocess_image(image_path)

    # Detect text in the image
    boxes = get_detection(image_path)

    # Load the original image for recognition
    image = cv2.imread(image_path)

    # Recognize text in the detected regions
    texts = get_recognition(image, boxes)

    return texts


if __name__ == "__main__":
    image_path = "../data/images/data_test_image.png"
    recognized_texts = ocr_pipeline(image_path)
    for text in recognized_texts:
        print(f"Recognized text: {text}")
