from medical_ocr.pipeline import OCRPipeline
import json

def test_ocr():
    # Example Usage Script for the OCR engine
    print("Initialize the OCR Pipeline (Models will load as Singletons)...")
    pipeline = OCRPipeline(fallback_threshold=0.85)

    # Note: Create a 'sample_ordonnance.jpeg' in your directory to test
    image_path = "sample_ordonnance.jpeg"
    
    import os
    if not os.path.exists(image_path):
        print(f"File {image_path} not found. Please provide an image to test.")
        return

    print("Processing Image...")
    result_json = pipeline.process_image(image_path)
    
    print("\n--- EXTRACTED JSON STRUCTURE ---")
    print(json.dumps(result_json, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    test_ocr()
