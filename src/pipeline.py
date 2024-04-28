import cv2
import numpy as np
from detection import get_detection, preprocess_image
from recognition import get_recognition, load_trocr_model
from utilities import load_dataset
from evaluation import calculate_metrics


def ocr_pipeline(dataset_dir):
    """
    Run the OCR pipeline: detection -> recognition.

    CRAFT (Clova) is used for text detection and TrOCR (Microsoft) is used for text recognition.
    """
    # Load the dataset
    data = load_dataset(dataset_dir)

    # Load the models
    processor, model = load_trocr_model()

    for i, (image, annotations) in enumerate(data):
        print("Detecting text...")
        boxes = get_detection(image)
        true_boxes = [ann["box"] for ann in annotations["form"]]
        output_path = os.path.join("images", f"plot_{i}.png")
        plot_boxes(image, true_boxes, boxes, output_path)
        print("Detection complete...")

        print("Recognizing text...")
        texts, boxes = get_recognition(image, boxes)
        true_texts = [ann["text"] for ann in annotations["form"]]
        print("____________________________________________________")
        print(f"True texts: {true_texts}")
        print("____________________________________________________")
        print(f"Predicted texts: {texts}")
        print("______________________________________________________")
        metrics = calculate_metrics(boxes, texts, true_boxes, true_texts)


if __name__ == "__main__":
    directory = "../data/funsd/testing_data"
    recognized_texts = ocr_pipeline(directory)
    for result in recognized_texts:
        print(result)
