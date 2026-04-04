import logging
from .preprocessor import Preprocessor
from .detector import Detector
from .recognizer import Recognizer
from .extractor import Extractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class OCRPipeline:
    def __init__(self, fallback_threshold=0.85):
        """
        Initializes the Hybrid OCR Pipeline.
        :param fallback_threshold: If PaddleOCR confidence is below this, TrOCR is invoked.
        """
        self.fallback_threshold = fallback_threshold
        self.detector = Detector()
        self.recognizer = Recognizer()
        
        # Adjust path if script is run from different working dir
        self.extractor = Extractor(db_path="data/medicaments_maroc.json")

    def process_image(self, image_path: str) -> dict:
        """
        End-to-End processing of a prescription image.
        """
        logging.info(f"Processing image: {image_path}")
        
        try:
            # 1. Preprocessing
            processed_image = Preprocessor.process(image_path)
            
            # 2. Primary Engine (PaddleOCR)
            detections = self.detector.detect_and_recognize(processed_image)
            
            if not detections:
                logging.warning("No text detected in the image.")
                return {"medications": [], "overall_confidence": 0.0, "status": "empty_output"}

            medications = []
            overall_confidence = 0.0
            valid_meds_count = 0

            # 3. Iterate through text lines and fallback if needed
            for det in detections:
                bbox = det["bbox"]
                text = det["text"]
                conf = det["confidence"]
                source = det["source"]

                # TrOCR Fallback
                if conf < self.fallback_threshold:
                    logging.info(f"Low confidence ({conf:.2f}) from PaddleOCR. Enacting TrOCR fallback...")
                    crop = Preprocessor.crop_bbox(processed_image, bbox)
                    fallback_res = self.recognizer.recognize_crop(crop)
                    
                    # Assume TrOCR is better if invoked, or pick best confidence
                    if fallback_res["confidence"] > conf:
                        text = fallback_res["text"]
                        conf = fallback_res["confidence"]
                        source = fallback_res["source"]
                
                # 4. Filter empty noise
                if len(text.strip()) < 3:
                    continue

                # 5. Extract Entities and Fuse Confidence
                extracted = self.extractor.extract_entities(text, conf)
                
                # We only append if it resembles a drug (either matched or has posology/dosage)
                if extracted["drug_name"] or extracted["dosage"] or extracted["posology"]:
                    # Refine JSON format
                    medications.append({
                        "drug_name": extracted["drug_name"] or text, # raw fallbacks to text if not matched
                        "dosage": extracted["dosage"],
                        "posology": extracted["posology"],
                        "duration": extracted["duration"],
                        "confidence": extracted["confidence"],
                        "source": source,
                        "bbox": bbox
                    })
                    overall_confidence += extracted["confidence"]
                    valid_meds_count += 1

            if valid_meds_count > 0:
                overall_confidence = round(overall_confidence / valid_meds_count, 4)

            # Warning for ambiguous results
            if overall_confidence < 0.6 and valid_meds_count > 0:
                logging.warning("Low overall scan confidence. Please verify results manually.")

            return {
                "medications": medications,
                "overall_confidence": overall_confidence,
                "status": "success"
            }

        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
            return {
                "medications": [],
                "overall_confidence": 0.0,
                "status": f"error: {str(e)}"
            }
