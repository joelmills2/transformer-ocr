"""
File for evaluation of the model. Metrics such as precision, recall, and F1 score are calculated for the OCR pipeline.
"""

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from jiwer import wer, cer
import numpy as np


def calculate_iou(detected_box, true_box, epsilon=1e-3):
    """
    Implementation of Intersection over Union (IoU) for text detection evaluation.
    Modified to consider a detected box as correct if it is fully within a true box or vice versa,
    with a small margin of error.
    """
    # Check if detected box is fully within true box
    if (
        detected_box[0] >= true_box[0] - epsilon
        and detected_box[1] >= true_box[1] - epsilon
        and detected_box[2] <= true_box[2] + epsilon
        and detected_box[3] <= true_box[3] + epsilon
    ):
        return 1.0

    # Check if true box is fully within detected box
    if (
        true_box[0] >= detected_box[0] - epsilon
        and true_box[1] >= detected_box[1] - epsilon
        and true_box[2] <= detected_box[2] + epsilon
        and true_box[3] <= detected_box[3] + epsilon
    ):
        return 1.0

    # Calculate intersection coordinates
    x_left = np.maximum(detected_box[0], true_box[0])
    y_top = np.maximum(detected_box[1], true_box[1])
    x_right = np.minimum(detected_box[2], true_box[2])
    y_bottom = np.minimum(detected_box[3], true_box[3])

    # Check if there is no intersection
    if np.any(x_right < x_left) or np.any(y_bottom < y_top):
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    det_area = (detected_box[2] - detected_box[0]) * (detected_box[3] - detected_box[1])
    true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    union_area = det_area + true_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def calculate_detection_metrics(detected_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculate detection metrics (precision, recall, F1) based on IoU scores.
    """

    # Initialize counters
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = len(true_boxes)  # All true boxes are initially considered as False Negatives

    for det_box in detected_boxes:
        # Find the best matching true box
        best_iou = 0
        for true_box in true_boxes:
            iou = calculate_iou(det_box, true_box)
            if iou > best_iou:
                best_iou = iou
        if best_iou >= iou_threshold:
            tp += 1
            fn -= 1  # One less false negative since we found a match
        else:
            fp += 1  # No match found, it's a false positive

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    # print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    # print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    return precision, recall, f1


def calculate_recognition_metric(predicted_texts, true_texts):
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER) for text recognition evaluation.
    """
    wer_score = wer(true_texts, predicted_texts)
    cer_score = cer(true_texts, predicted_texts)
    return wer_score, cer_score


def calculate_metrics(detected_boxes, detected_texts, true_boxes, true_texts):
    """
    Calculate precision, recall, F1 score, and other metrics for the OCR pipeline.
    """
    precision, recall, f1 = calculate_detection_metrics(detected_boxes, true_boxes)

    cer_score, wer_score = calculate_recognition_metric(detected_texts, true_texts)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": cer_score,
        "wer": wer_score,
    }

    # print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    # print(f"CER: {cer_score}, WER: {wer_score}")
    return metrics
