from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from vision_engine import DocumentVisionEngine
import torch
import cv2
import numpy as np

class DocumentOCREngine:
    """
    Vanguard v5.0: Enhanced OCR Intelligence Engine.
    Supports both DocTR and TrOCR for best handwriting recognition.
    """

    def __init__(self):
        print("[OCR] Loading DocTR model...")
        # Load DocTR model
        try:
            # Use lightweight models for faster initial load
            self.model = ocr_predictor(
                det_arch='db_mobilenet_v3_large',
                reco_arch='crnn_vgg16_bn',
                pretrained=True,
                assume_straight_pages=True
            )
            if torch.cuda.is_available():
                self.model.cuda()
            print("[OCR] DocTR model loaded successfully!")
        except Exception as e:
            print(f"[OCR] Error loading DocTR model: {e}")
            self.model = None
        
        # Initialize Vision Engine
        try:
            self.vision = DocumentVisionEngine()
        except Exception as e:
            print(f"[OCR] Error initializing Vision Engine: {e}")
            self.vision = None
        
        # TrOCR will be loaded lazily (only when High-Fidelity mode is used)
        # This speeds up initial app loading
        self.troc_model = None
        self.troc_processor = None
        self.troc_loaded = False

    def _load_troc_model(self):
        """Load TrOCR model for better handwriting recognition (lazy loading)."""
        if self.troc_loaded:
            return  # Already loaded
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            print("[OCR] Loading TrOCR model for handwriting (first time, may take 20-30 seconds)...")
            # Use larger TrOCR model for better accuracy on clean handwriting
            # Falls back safely if system does not have enough memory
            try:
                model_id = 'microsoft/trocr-large-handwritten'
                self.troc_processor = TrOCRProcessor.from_pretrained(model_id)
                self.troc_model = VisionEncoderDecoderModel.from_pretrained(model_id)
                print("[OCR] TrOCR-LARGE model loaded successfully!")
            except Exception:
                # Fallback to base model if large model cannot be loaded
                print("[OCR] Large TrOCR model failed, falling back to BASE")
                model_id = 'microsoft/trocr-base-handwritten'
                self.troc_processor = TrOCRProcessor.from_pretrained(model_id)
                self.troc_model = VisionEncoderDecoderModel.from_pretrained(model_id)
                print("[OCR] TrOCR-BASE model loaded successfully!")
            if torch.cuda.is_available():
                self.troc_model.cuda()
            self.troc_loaded = True
            print("[OCR] TrOCR model ready!")
        except Exception as e:
            print(f"[OCR] TrOCR not available: {e}")
            print("[OCR] Using DocTR as fallback")
            self.troc_loaded = False

    def extract_text_from_image(self, image_path, pre_cleaned=None, high_fidelity=True, **kwargs):
        """
        DocTR primary pipeline with optional TrOCR *refinement* for low-confidence words.
        Whole-page TrOCR is intentionally avoided (it hallucinates on complex forms).
        """
        import os
        
        # Validate input
        if not image_path or not os.path.exists(image_path):
            print(f"[OCR] Error: Invalid image path: {image_path}")
            return []
        
        # Check if base model is loaded
        if self.model is None:
            print("[OCR] Error: DocTR model not loaded. Cannot process image.")
            return []
        
        # Lazy load TrOCR if High-Fidelity mode is enabled and not already loaded
        if high_fidelity and not self.troc_loaded:
            self._load_troc_model()
        
        try:
            # Enhance image
            if pre_cleaned is None:
                if self.vision is not None:
                    pre_cleaned = self.vision.enhance_image(image_path)
                else:
                    print("[OCR] Warning: Vision engine not available, skipping enhancement")
            
            # Save enhanced image to a temporary path
            temp_path = "temp_enhanced.jpg"
            if pre_cleaned is not None:
                # Ensure pre_cleaned is in correct format for cv2.imwrite
                if len(pre_cleaned.shape) == 2:  # Grayscale image
                    cv2.imwrite(temp_path, pre_cleaned)
                else:  # BGR image
                    cv2.imwrite(temp_path, pre_cleaned)
            else:
                temp_path = image_path

            # Always run DocTR first (stable layout-aware OCR)
            print("[OCR] Using DocTR pipeline (primary)")
            result = self._extract_with_doctr(temp_path)

            # Optional: refine only low-confidence words with TrOCR when enabled
            if (
                high_fidelity
                and result
                and self.troc_model is not None
                and self.troc_processor is not None
            ):
                print("[OCR] High-Fidelity mode ON -> refining low-confidence words with TrOCR")
                result = self._refine_with_troc_on_low_conf(temp_path, result)
            
            # Return empty list if no results
            if not result:
                print("[OCR] Warning: No text extracted from image")
            
            return result
        except Exception as e:
            print(f"[OCR] Error in extract_text_from_image: {e}")
            return []

    def _extract_with_troc(self, image_path):
        """Extract text using TrOCR (better for handwriting)."""
        from PIL import Image
        import math
        
        # Check if TrOCR is loaded
        if self.troc_processor is None or self.troc_model is None:
            print("[OCR] TrOCR not available, falling back to DocTR")
            return []
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process with TrOCR
            pixel_values = self.troc_processor(images=image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            generated_ids = self.troc_model.generate(pixel_values)
            decoded = self.troc_processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"[OCR] TrOCR processing failed: {e}")
            return []
        
        if not decoded:
            return []

        full_text = decoded[0].strip()
        if not full_text:
            return []

        # Split into lines and words and synthesize DocTR-style geometry
        lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
        if not lines:
            lines = [full_text]
        
        num_lines = len(lines)
        results = []

        for line_idx, line in enumerate(lines):
            words = [w for w in line.split(" ") if w]
            if not words:
                continue
            num_words = len(words)

            for word_idx, word in enumerate(words):
                # Normalized synthetic boxes in [0,1]
                x0 = word_idx / max(num_words, 1)
                x1 = (word_idx + 1) / max(num_words, 1)
                y0 = line_idx / max(num_lines, 1)
                y1 = (line_idx + 1) / max(num_lines, 1)

                results.append({
                    "text": word,
                    "confidence": 0.95,
                    "box": ((float(x0), float(y0)), (float(x1), float(y1)))
                })

        return results

    def _refine_with_troc_on_low_conf(self, image_path, words, min_conf: float = 0.7):
        """
        Use TrOCR only on low-confidence words from DocTR.
        This avoids hallucinations while improving tough handwriting.
        """
        from PIL import Image
        import math

        if self.troc_processor is None or self.troc_model is None:
            # TrOCR not available – nothing to do
            return words

        try:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
        except Exception as e:
            print(f"[OCR] TrOCR refine: failed to open image: {e}")
            return words

        refined = []

        for w in words:
            try:
                conf = float(w.get("confidence", 0.0) or 0.0)
                box = w.get("box")

                # Skip high-confidence or missing-box words
                if conf >= min_conf or not box:
                    refined.append(w)
                    continue

                (x0n, y0n), (x1n, y1n) = box

                # Convert normalized coordinates to pixel space, with padding
                def clamp01(v):
                    try:
                        return max(0.0, min(1.0, float(v)))
                    except Exception:
                        return 0.0

                x0 = clamp01(x0n) * width
                x1 = clamp01(x1n) * width
                y0 = clamp01(y0n) * height
                y1 = clamp01(y1n) * height

                pad_x = 0.02 * width
                pad_y = 0.01 * height

                x0 = int(max(0, x0 - pad_x))
                x1 = int(min(width, x1 + pad_x))
                y0 = int(max(0, y0 - pad_y))
                y1 = int(min(height, y1 + pad_y))

                if x1 <= x0 or y1 <= y0:
                    refined.append(w)
                    continue

                crop = img.crop((x0, y0, x1, y1))

                # Run TrOCR on the cropped word/segment
                pixel_values = self.troc_processor(images=crop, return_tensors="pt").pixel_values
                if torch.cuda.is_available():
                    pixel_values = pixel_values.cuda()

                generated_ids = self.troc_model.generate(pixel_values)
                decoded = self.troc_processor.batch_decode(generated_ids, skip_special_tokens=True)
                new_text = decoded[0].strip() if decoded else ""

                if new_text and new_text.lower() != str(w.get("text", "")).lower():
                    w = w.copy()
                    w["text"] = new_text
                    # Boost confidence since we re-verified it
                    w["confidence"] = max(conf, 0.9)

            except Exception as e:
                print(f"[OCR] TrOCR refine failed for word '{w.get('text', '')}': {e}")

            refined.append(w)

        return refined

    def _extract_with_doctr(self, image_path):
        """Extract text using DocTR."""
        import os
        
        # Check if model is loaded
        if self.model is None:
            print("[OCR] Error: DocTR model not loaded")
            return []
        
        # Validate image path
        if not os.path.exists(image_path):
            print(f"[OCR] Error: Image file not found: {image_path}")
            return []
        
        try:
            doc = DocumentFile.from_images(image_path)
            result = self.model(doc)
            
            extracted_data = []

            # DocTR Hierarchy: Pages -> Blocks -> Lines -> Words
            if not hasattr(result, 'pages') or not result.pages:
                print("[OCR] Warning: No pages detected in document")
                return []
            
            for page in result.pages:
                if not hasattr(page, 'blocks') or not page.blocks:
                    continue
                for block in page.blocks:
                    if not hasattr(block, 'lines') or not block.lines:
                        continue
                    for line in block.lines:
                        if not hasattr(line, 'words') or not line.words:
                            continue
                        for word in line.words:
                            extracted_data.append({
                                "text": word.value,
                                "confidence": word.confidence,
                                "box": word.geometry
                            })

            return extracted_data
        except Exception as e:
            print(f"[OCR] DocTR processing failed: {e}")
            return []

if __name__ == "__main__":
    print("Enhanced OCR Engine Ready.")

