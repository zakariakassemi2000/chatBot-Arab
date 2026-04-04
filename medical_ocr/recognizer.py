from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import logging
import torch

class RecognizerSingleton:
    """Singleton implementation to ensure TrOCR loads only once."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logging.info("Initializing TrOCR Engine (Fallback)...")
            cls._instance = super(RecognizerSingleton, cls).__new__(cls)
            # Using microsoft/trocr-small-handwritten as fallback
            cls._instance.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten", use_fast=False)
            cls._instance.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.model.to(cls._instance.device)
        return cls._instance

class Recognizer:
    def __init__(self):
        instance = RecognizerSingleton()
        self.processor = instance.processor
        self.model = instance.model
        self.device = instance.device

    def recognize_crop(self, crop_img) -> dict:
        """
        Runs TrOCR on a clean cropped image matrix.
        Returns recognized text.
        """
        if crop_img.size == 0:
            return {"text": "", "confidence": 0.0, "source": "trocr"}
            
        # Convert OpenCV format (BGR) to RGB Pillow Image
        rgb_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        # Preprocess and Generate
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        # We output scores to calculate a proxy confidence for TrOCR
        generated_ids = self.model.generate(
            pixel_values, 
            output_scores=True, 
            return_dict_in_generate=True, 
            max_length=64
        )
        
        generated_text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
        
        # Simple proxy for confidence using transition scores
        transition_scores = self.model.compute_transition_scores(
            generated_ids.sequences, generated_ids.scores, normalize_logits=True
        )
        # Average probability of tokens
        prob_scores = torch.exp(transition_scores[0])
        confidence = torch.mean(prob_scores).item()

        return {
            "text": generated_text.strip(),
            "confidence": confidence,
            "source": "trocr"
        }
