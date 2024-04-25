from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import cv2


def load_trocr_model():
    """
    Load the TrOCR model and processor.
    """
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    return processor, model


def recognize_text(image_np, model, processor):
    """
    Recognize text from an image numpy array using TrOCR.
    """
    image_pil = Image.fromarray(image_np).convert("RGB") # Convert to PIL image
    inputs = processor(images=image_pil, return_tensors="pt") # Preprocess the image
    generated_ids = model.generate(inputs["pixel_values"]) # Generate text
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # Decode the generated text
    return predicted_text


def get_recognition(image, boxes):
    """
    Recognize text for all detected boxes.
    """
    processor, model = load_trocr_model()
    texts = []
    for box in boxes:
        # Ensure box coordinates are integer values
        box = np.array(box).astype(np.int32)
        min_x, min_y, max_x, max_y = box[0][0], box[0][1], box[2][0], box[2][1]

        # Crop the image to the detected text region
        cropped_region = image[min_y:max_y, min_x:max_x]

        # Convert color space from BGR to RGB which is expected by TrOCR
        cropped_region_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)

        # Recognize the text in the cropped image
        text = recognize_text(cropped_region_rgb, model, processor)
        texts.append(text)
    return texts
