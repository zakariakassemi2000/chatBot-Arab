import os
import logging

# Disable oneDNN (Intel MKL-DNN) — causes ConvertPirAttribute crash on some Windows CPUs
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

class DetectorSingleton:
    """Singleton implementation to ensure PaddleOCR loads only once."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logging.info("Initializing PaddleOCR Engine (Detection + Recognition) - French & Arabic...")
            cls._instance = super(DetectorSingleton, cls).__new__(cls)
            # Support French ('fr') and Arabic ('ar') text recognition
            # paddleocr detects language from model
            cls._instance.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            # Note: For strict mixed French-Arabic, PaddleOCR has a multi-language model
            # For simplicity, 'ar' language sometimes supports mixed-en/fr numerals but using the custom
            # server models is better. We use 'ar' as the base for Moroccan context.
            # You can also initialize a secondary 'fr' model if needed, but a single multilingual model fits best.
        return cls._instance

class Detector:
    def __init__(self):
        self.engine = DetectorSingleton().ocr

    def detect_and_recognize(self, processed_image):
        """
        Runs PaddleOCR detection and recognition on an image matrix.
        Returns a structured list of boxes and recognized texts.
        Compatible with PaddleOCR v3+ which uses .predict() API.
        """
        results_formatted = []
        
        try:
            # PaddleOCR v3+ uses predict() method
            result = self.engine.predict(processed_image)
        except TypeError:
            # Fallback for older PaddleOCR versions
            result = self.engine.ocr(processed_image)

        if result is None or len(result) == 0:
            return results_formatted

        # Handle v3+ output format: list of dicts with 'rec_text', 'rec_score', 'dt_polys'
        for item in result:
            if isinstance(item, dict):
                # New API format (v3+)
                texts = item.get("rec_text", [])
                scores = item.get("rec_score", [])
                boxes = item.get("dt_polys", [])
                for i, text in enumerate(texts):
                    bbox = boxes[i].tolist() if i < len(boxes) else []
                    confidence = float(scores[i]) if i < len(scores) else 0.0
                    results_formatted.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": confidence,
                        "source": "paddleocr"
                    })
            elif isinstance(item, list):
                # Old API format
                for line in item:
                    if line is None:
                        continue
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    results_formatted.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": confidence,
                        "source": "paddleocr"
                    })

        return results_formatted
