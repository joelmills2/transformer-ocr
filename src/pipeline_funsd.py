import cv2
import numpy as np
from detection import get_detection, preprocess_image
from recognition import get_recognition, load_trocr_model
from utilities import load_dataset
from evaluation import calculate_metrics
import os


def plot_boxes(image, true_boxes, detected_boxes, output_path):
    """
    Plot the detected and true boxes on the image.
    """
    # Draw true boxes in red
    for box in true_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    # Draw detected boxes in green
    for box in detected_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Save the image
    cv2.imwrite(output_path, image)


def ocr_pipeline(dataset_dir):
    """
    Run the OCR pipeline: detection -> recognition.

    CRAFT (Clova) is used for text detection and TrOCR (Microsoft) is used for text recognition.
    """
    # Load the dataset
    data = load_dataset(dataset_dir)

    metrics_sum = {"precision": 0, "recall": 0, "f1": 0, "cer": 0, "wer": 0}

    num_tests = len(data)

    # Load the models
    processor, model = load_trocr_model()

    all_true_texts = ""
    all_predicted_texts = ""

    for i, (image, annotations) in enumerate(data):
        boxes = get_detection(image)
        annotations["form"] = sorted(
            annotations["form"], key=lambda ann: (ann["box"][1], ann["box"][0])
        )
        true_boxes = [ann["box"] for ann in annotations["form"]]
        output_path = os.path.join("images", f"plot_{i}.png")
        plot_boxes(image, true_boxes, boxes, output_path)
        texts, boxes = get_recognition(image, boxes)
        true_texts = [ann["text"] for ann in annotations["form"]]
        all_true_texts += " ".join(true_texts)
        all_predicted_texts += " ".join(texts)
        # print("____________________________________________________")
        # print(f"True texts: {all_true_texts}")
        # print("____________________________________________________")
        # print(f"Predicted texts: {all_predicted_texts}")
        # print("______________________________________________________")

        metrics = calculate_metrics(
            boxes, all_predicted_texts, true_boxes, all_true_texts
        )
        for key in metrics_sum:
            metrics_sum[key] += metrics[key]

    avg_metrics = {key: value / num_tests for key, value in metrics_sum.items()}

    return avg_metrics


def main():
    directory = "../data/funsd/testing_data"
    avg_metrics = ocr_pipeline(directory)
    print("Average metrics after all testing:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
