"""
Multimodal medical image + text reasoning model.
Uses med-flamingo/med-flamingo (9B parameters).
Requires careful VRAM management (quantization recommended).
"""
import torch
from core.model_manager import ModelManager
import logging

logger = logging.getLogger(__name__)

class MedFlamingoAnalyzer:
    MODEL_ID = "med-flamingo/med-flamingo"

    def __init__(self):
        # We load this lazily to avoid out-of-memory errors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_dependencies()

    def _load_dependencies(self):
        def _loader():
            try:
                from open_flamingo import create_model_and_transforms
                import huggingface_hub
            except ImportError as e:
                logger.error("Missing open_flamingo library. Please install via: pip install open-flamingo")
                raise e

            logger.info("Initializing Med-Flamingo 9B (This will take significant VRAM!)")
            
            # Since this is a 9B model, we STRONGLY recommend loading in 8-bit or 4-bit,
            # but for demonstration we'll set up the standard loading process.
            # Real production implementation requires DeepSpeed or BitsAndBytes for 9B models on 16GB VRAM.
            
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="decapoda-research/llama-7b-hf",
                tokenizer_path="decapoda-research/llama-7b-hf",
                cross_attn_every_n_layers=4
            )
            
            # Load Med-Flamingo weights
            hf_token = os.environ.get("HF_TOKEN")
            checkpoint_path = huggingface_hub.hf_hub_download(
                "med-flamingo/med-flamingo", 
                "model.pt"
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
            
            # Move to target device safely
            if self.device == "cuda":
                # For a 9B model this will occupy ~18GB in FP16!
                model = model.half().to("cuda")
            else:
                model.eval()

            return {
                "model": model,
                "image_processor": image_processor,
                "tokenizer": tokenizer
            }

        # The ModelManager ensures we don't hold X-Ray AND Med-Flamingo BOTH in VRAM
        deps = ModelManager.get_or_load("med_flamingo", _loader)
        self.model = deps["model"]
        self.image_processor = deps["image_processor"]
        self.tokenizer = deps["tokenizer"]

    def analyze(self, image_pils, prompts, max_new_tokens=100):
        """
        Runs generative medical VQA utilizing few-shot inputs or zero-shot.
        image_pils: List of PIL images.
        prompts: String containing the <image> tags matching the images provided.
        """
        # (Simplified inference pipeline for Flamingo architecture)
        vision_x = [self.image_processor(img).unsqueeze(0) for img in image_pils]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)
        
        if self.device == "cuda":
             vision_x = vision_x.half()
             
        # Tokenize with Flamingo-specific handling
        lang_x = self.tokenizer(
            prompts,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_text = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
            )

        output_text = self.tokenizer.decode(generated_text[0])
        return output_text
