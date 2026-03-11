import gc
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized Model Manager to prevent Out-Of-Memory (OOM) errors.
    Uses an LRU (Least Recently Used) cache strategy.
    Loads models dynamically when requested, and evicts older ones when VRAM is full.
    """
    
    _active_models: Dict[str, Any] = {}
    MAX_MODELS_IN_MEMORY = 2  # Keep this low so we don't crash the GPU/RAM

    @classmethod
    def get_or_load(cls, model_key: str, load_function: callable) -> Any:
        """
        Retrieves a model if it's in memory, or loads it using the provided function.
        Handles eviction of old models if MAX is reached.
        """
        if model_key in cls._active_models:
            logger.info(f"[ModelManager] Model '{model_key}' retrieved from cache.")
            return cls._active_models[model_key]

        # Evict oldest model if we are at capacity
        if len(cls._active_models) >= cls.MAX_MODELS_IN_MEMORY:
            # In Python 3.7+ dictionaries maintain insertion order, 
            # so the first item is the oldest.
            oldest_key = list(cls._active_models.keys())[0]
            logger.warning(f"[ModelManager] Memory capacity reached ({cls.MAX_MODELS_IN_MEMORY}). Evicting oldest model: '{oldest_key}'")
            cls._unload_model(oldest_key)

        logger.info(f"[ModelManager] Loading model '{model_key}' into memory...")
        try:
            model_data = load_function()
            cls._active_models[model_key] = model_data
            logger.info(f"[ModelManager] Successfully loaded '{model_key}'. Active models: {list(cls._active_models.keys())}")
            return model_data
        except Exception as e:
            logger.error(f"[ModelManager] Failed to load model '{model_key}': {e}", exc_info=True)
            raise e

    @classmethod
    def _unload_model(cls, model_key: str):
        """Removes a model from memory and clears the PyTorch CUDA cache."""
        if model_key in cls._active_models:
            del cls._active_models[model_key]
            cls.clear_vram()
            logger.info(f"[ModelManager] Successfully unloaded '{model_key}' and cleared VRAM.")

    @classmethod
    def clear_vram(cls):
        """Forces Python garbage collection and clears PyTorch CUDA memory."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # If using torch 2.x, the below helps free memory reserved by the allocator
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"[ModelManager] Error clearing VRAM: {e}")
