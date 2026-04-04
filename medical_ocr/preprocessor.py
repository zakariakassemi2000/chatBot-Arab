import cv2
import numpy as np

class Preprocessor:
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Corrects the rotation of the document image using minimum area rectangle."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Inverse binarize for contour finding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Get coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        # Compute minimum bounding box
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle to be horizontal
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if abs(angle) < 0.5:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def apply_clahe(image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced_img
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def denoise_and_binarize(image: np.ndarray) -> np.ndarray:
        """Removes noise and applies adaptive thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Adaptive Thresholding for Binarization
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary

    @classmethod
    def process(cls, image_path: str) -> np.ndarray:
        """Main preprocessing pipeline."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        deskewed = cls.deskew_image(image)
        enhanced = cls.apply_clahe(deskewed)
        # Binarization is often best done only if PaddleOCR struggles with colors, 
        # but PaddleOCR works well with colored/grayscale enhanced images.
        # We will return the CLAHE enhanced image, but keep binarization available
        # if you want to feed it to TrOCR later on specifically.
        return enhanced

    @staticmethod
    def crop_bbox(image: np.ndarray, bbox: list) -> np.ndarray:
        """Properly crop bounding boxes for TrOCR fallback to avoid noisy crops."""
        points = np.array(bbox, dtype=np.int32)
        x_min = max(0, np.min(points[:, 0]) - 2)
        x_max = min(image.shape[1], np.max(points[:, 0]) + 2)
        y_min = max(0, np.min(points[:, 1]) - 2)
        y_max = min(image.shape[0], np.max(points[:, 1]) + 2)
        
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped
