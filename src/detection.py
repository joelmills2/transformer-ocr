from craft_text_detector import Craft
import cv2


def load_craft_model():
    """
    Load the CRAFT model.
    """
    craft = Craft(
        output_dir="craft_output",
        crop_type="poly",  # Polygon or rectangle
        cuda=False,  # Use GPU acceleration, False for CPU
        text_threshold=0.6,  # 0.7 is default, higher value means less text detected but with higher confidence
        link_threshold=0.4,  # 0.4 is default, higher value means less links between text detected but with higher confidence
        low_text=0.4,  # 0.4 is default, higher value means less low-confidence text detected
    )
    return craft


def detect_text(image, craft):
    """
    Detect text in the image using the CRAFT model.
    """
    prediction_result = craft.detect_text(image) # Perform text detection
    return prediction_result["boxes"]


def preprocess_image(image_path):
    """
    Preprocess the image before detecting text.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # Apply thresholding (white or black)
    return image


def get_detection(image_path):
    """
    Perform detection and return bounding boxes.
    """
    craft = load_craft_model()
    image = preprocess_image(image_path)
    boxes = detect_text(image, craft)
    craft.unload_craftnet_model() # Unload the CRAFT model
    return boxes
