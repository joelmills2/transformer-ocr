from craft_text_detector import Craft
import cv2
import numpy as np


def load_craft_model(text_threshold=0.5, link_threshold=0.4, low_text=0.4):
    """
    Load the CRAFT model.
    """
    craft = Craft(
        output_dir="craft_output",
        crop_type="poly",  # Polygon or rectangle
        cuda=False,  # Use GPU acceleration, False for CPU
        text_threshold=text_threshold,  # 0.7 is default, higher value means less text detected but with higher confidence
        link_threshold=link_threshold,  # 0.4 is default, higher value means less links between text detected but with higher confidence
        low_text=low_text,  # 0.4 is default, higher value means less low-confidence text detected
    )
    return craft
    # should run a grid search to find the best parameters for the model


def detect_text(image, craft):
    """
    Detect text in the image using the CRAFT model.
    """
    if image is None or image.size == 0:
        print("The image is empty.")
        return None
    prediction_result = craft.detect_text(image)  # Perform text detection
    boxes = prediction_result["boxes"]

    # Convert the detected coordinates to integer values and adjust the format
    adjusted_boxes = []
    for box in boxes:
        # Ensure box coordinates are integers and arrange them as [min_x, min_y, max_x, max_y]
        int_box = np.array(box, dtype=np.float32).astype(np.int32)
        min_x, min_y = np.min(int_box[:, 0]), np.min(int_box[:, 1])
        max_x, max_y = np.max(int_box[:, 0]), np.max(int_box[:, 1])
        adjusted_boxes.append([min_x, min_y, max_x, max_y])

    return adjusted_boxes


def preprocess_image(image):
    """
    Preprocess the image before detecting text.
    """
    if image is None or image.size == 0:
        print("The image is empty.")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
        1
    ]  # Apply thresholding (white or black)
    return image


def get_detection(image_path, text_threshold=0.6, link_threshold=0.4, low_text=0.4):
    """
    Perform detection and return bounding boxes.
    """
    craft = load_craft_model(text_threshold, link_threshold, low_text)
    image = preprocess_image(image_path)
    boxes = detect_text(image, craft)
    # craft.unload_craftnet_model()  # Unload the CRAFT model
    return boxes
