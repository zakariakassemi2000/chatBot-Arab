import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.model_manager import ModelManager
import logging
import os

logger = logging.getLogger(__name__)

class BioMistralChatbot:
    """
    BioMistral-7B Chatbot for medical questions.
    Because this is a 7B model, it must be loaded via 4-bit or 8-bit quantization
    if running on consumer hardware (e.g., bitsandbytes).
    """
    
    # Very powerful open-source medical LLM: BioMistral/BioMistral-7B
    MODEL_ID = "BioMistral/BioMistral-7B"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def _load_dependencies(self):
        def _loader():
            try:
                from transformers import BitsAndBytesConfig
                import huggingface_hub
            except ImportError as e:
                logger.error("Missing libraries for LLM quantization. Run: pip install bitsandbytes accelerate")
                raise e

            logger.info("Initializing BioMistral 7B (Requiring 8-Bit Quantization)")
            
            # Use strict 8-bit quantization to fit on 16GB VRAM GPUs (or even 10GB for inference)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            
            hf_token = os.environ.get("HF_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID, token=hf_token, use_fast=True
            )
            
            # Load the heavy LLM into VRAM carefully
            model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                token=hf_token,
                quantization_config=bnb_config,
                device_map="auto" # Let accelerate figure out the best GPU distribution
            )
            model.eval()

            return {
                "model": model,
                "tokenizer": tokenizer
            }

        deps = ModelManager.get_or_load("biomistral_llm", _loader)
        self.model = deps["model"]
        self.tokenizer = deps["tokenizer"]

    def generate_answer(self, user_prompt: str, context: str = "", max_new_tokens=256):
        """Generates a medical reasoning text utilizing BioMistral-7B."""
        if self.model is None:
            self._load_dependencies()

        # Simple prompt template for BioMistral
        prompt = f"""<s>[INST] You are an expert medical AI assistant. Answer the medical question using the context.
        
Context:
{context}

Question:
{user_prompt}
[/INST]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if "cuda" in self.device else "cpu")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3, # Low temperature for accurate medical facts
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Slice the response to remove the prompt part
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
