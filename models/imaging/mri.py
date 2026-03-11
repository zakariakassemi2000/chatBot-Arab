import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from core.model_manager import ModelManager
import logging

logger = logging.getLogger(__name__)

class BrainTumorAnalyzer:
    """
    Segements and classifies Brain Tumors from MRI images.
    Modes: Glioma, Meningioma, Pituitary Tumor.
    """
    
    # We will use the HuggingFace Vision Transformer (ViT) finetuned for Brain MRI
    MODEL_ID = "ahmedhamdy/brain-tumor-vit"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # We don't load immediately; we load when needed or let ModelManager handle it
        self.model = None
        self.processor = None

    def _load_dependencies(self):
        def _loader():
            logger.info(f"Loading Brain MRI Model: {self.MODEL_ID}")
            processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID).to(self.device)
            model.eval()
            
            return {
                "processor": processor,
                "model": model,
                "labels": model.config.id2label
            }

        deps = ModelManager.get_or_load("mri_brain_tumor", _loader)
        self.processor = deps["processor"]
        self.model = deps["model"]
        self.labels = deps["labels"]

    def analyze(self, image_pil):
        """Runs the MRI image through the ViT model."""
        if self.model is None:
            self._load_dependencies()

        img = image_pil.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        predictions = []
        for i, val in enumerate(probs):
            label_name = self.labels.get(i, self.labels.get(str(i), f"Class {i}"))
            predictions.append({
                "tumor_type": label_name,
                "probability": float(val)
            })

        # Sort by highest probability
        predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
        
        highest = predictions[0]
        risk = "high" if "tumor" in highest["tumor_type"].lower() or "glioma" in highest["tumor_type"].lower() or "meningioma" in highest["tumor_type"].lower() else "normal"
        if highest["probability"] < 0.6:
            risk = "moderate"

        return {
            "predictions": predictions,
            "top_prediction": highest["tumor_type"],
            "risk_level": risk
        }
