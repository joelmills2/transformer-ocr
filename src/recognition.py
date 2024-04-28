from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2


def load_trocr_model():
    """
    Load the TrOCR model and processor.

    Models: https://huggingface.co/models?search=microsoft/trocr
    """
    # 11 models available - test to see which one works best for your use case
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    return processor, model


def recognize_text(image_np, model, processor):
    """
    Recognize text from an image numpy array using TrOCR.
    """
    image_pil = Image.fromarray(image_np).convert("RGB")  # Convert to PIL image
    inputs = processor(images=image_pil, return_tensors="pt")  # Preprocess the image
    generated_ids = model.generate(inputs["pixel_values"])  # Generate text
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]  # Decode the generated text
    return predicted_text


def get_recognition(image, boxes):
    """
    Recognize text for all detected boxes.
    """
    processor, model = load_trocr_model()
    texts = []
    for box in boxes:
        min_x, min_y, max_x, max_y = box
        cropped_region = image[min_y:max_y, min_x:max_x]
        cropped_region_rgb = cv2.cvtColor(
            cropped_region, cv2.COLOR_BGR2RGB
        )  # Convert BGR to RGB

        # Recognize text using TrOCR
        text = recognize_text(cropped_region_rgb, model, processor)
        texts.append(text)
    return texts, boxes
