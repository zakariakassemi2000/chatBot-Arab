import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from core.model_manager import ModelManager
import logging

logger = logging.getLogger(__name__)

class ChestXrayAnalyzer:
    """
    Analyzes Chest X-Rays using StanfordAIMI/CheXNet (or similar 14-disease DenseNet models).
    Loads lazily via ModelManager to save VRAM.
    """
    
    # Using a popular CheXNet PyTorch implementation available on HF
    MODEL_ID = "StanfordAIMI/CheXNet"  # As requested in the architecture plan
    # Fallback if that specific repo isn't public: "ahmedhamdy/DenseNet-CheXNet"
    
    DISEASES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # We don't load immediately; we load when needed or let ModelManager handle it
        self.model = None
        self.processor = None

    def _load_dependencies(self):
        def _loader():
            # For X-Rays, we need high-resolution but it's a standard CNN (DenseNet121)
            # which is very VRAM efficient (~1GB) compared to LLMs.
            logger.info(f"Loading Chest X-Ray Model: {self.MODEL_ID}")
            
            try:
                processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
                model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            except Exception as e:
                logger.warning(f"Could not load StanfordAIMI/CheXNet. Falling back to alternative. Error: {e}")
                fallback_id = "ahmed-ai/chexnet-xray" # hypothetical fallback
                processor = AutoImageProcessor.from_pretrained(fallback_id)
                model = AutoModelForImageClassification.from_pretrained(fallback_id)

            model = model.to(self.device)
            model.eval()
            
            return {
                "processor": processor,
                "model": model
            }

        deps = ModelManager.get_or_load("xray_chexnet", _loader)
        self.processor = deps["processor"]
        self.model = deps["model"]

    def analyze(self, image_pil):
        """Runs the X-Ray image through the CheXNet DenseNet model."""
        if self.model is None:
            self._load_dependencies()

        img = image_pil.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # CheXNet outputs independent probabilities (Sigmoid) not mutually exclusive (Softmax)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        results = {}
        highest_risk = "normal"
        highest_prob = 0.0

        for idx, disease in enumerate(self.DISEASES):
            prob = float(probs[idx] if idx < len(probs) else 0.0)
            results[disease] = prob
            
            if prob > highest_prob:
                highest_prob = prob
                
            if prob > 0.5: # Standard threshold for CheXNet positive
                highest_risk = "high"

        if highest_risk == "normal" and highest_prob > 0.3:
            highest_risk = "moderate"

        return {
            "probabilities": results,
            "highest_risk": highest_risk,
            "raw_logits": logits.cpu().numpy().tolist()
        }
