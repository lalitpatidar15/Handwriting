from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from vision_engine import DocumentVisionEngine
from universal_parser import UniversalDataIntelligence
import torch
import cv2
import numpy as np
import re

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
        
        # Initialize Parser
        try:
            self.parser = UniversalDataIntelligence()
        except Exception as e:
            print(f"[OCR] Error initializing Parser: {e}")
            self.parser = None
        
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
                model_id = 'microsoft/trocr-base-handwritten'
                self.troc_processor = TrOCRProcessor.from_pretrained(model_id)
                self.troc_model = VisionEncoderDecoderModel.from_pretrained(model_id)
                print("[OCR] TrOCR-BASE model loaded successfully!")
            except Exception:
                # Fallback to small model if base model cannot be loaded
                print("[OCR] BASE TrOCR model failed, falling back to SMALL")
                model_id = 'microsoft/trocr-small-handwritten'
                self.troc_processor = TrOCRProcessor.from_pretrained(model_id)
                self.troc_model = VisionEncoderDecoderModel.from_pretrained(model_id)
                print("[OCR] TrOCR-SMALL model loaded successfully!")
            if torch.cuda.is_available():
                self.troc_model.cuda()
            self.troc_loaded = True
            print("[OCR] TrOCR model ready!")
        except Exception as e:
            print(f"[OCR] TrOCR not available: {e}")
            print("[OCR] Using DocTR as fallback")
            self.troc_loaded = False

    def extract_text_from_image(self, image_path, pre_cleaned=None, high_fidelity=True, preserve_form=False, **kwargs):
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

            # Check if preserve form and it's likely a printable template form.
            if preserve_form:
                looks_like_form = self._looks_like_form_template(result)
                parser_form = False
                if self.parser:
                    doc_analysis, _, _ = self.parser.parse_universal(result)
                    parser_form = doc_analysis.get('type') == 'STRUCTURED_FORM'

                if looks_like_form or parser_form:
                    print("[OCR] Detected template form, generating filled form image")
                    filled_path = self._generate_filled_form_image(image_path, result)
                    if filled_path:
                        return {'type': 'filled_form', 'path': filled_path}

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

    def generate_digital_form_image(self, image_path, pre_cleaned=None):
        """Generate digital form output (white/black template + typed entries).

        This method is used in Gemini-first mode: Gemini provides structure/text,
        while local DocTR provides word geometry for placing typed values.
        """
        try:
            temp_path = image_path
            if pre_cleaned is not None:
                temp_path = "temp_enhanced_form.jpg"
                cv2.imwrite(temp_path, pre_cleaned)

            ocr_words = self._extract_with_doctr(temp_path)
            # Always attempt digital form rendering even if OCR words are empty.
            # In that case, we still return a clean digital template (no handwritten values typed).
            return self._generate_filled_form_image(image_path, ocr_words or [])
        except Exception as e:
            print(f"[OCR] Error generating digital form image: {e}")
            return None

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

    def _generate_filled_form_image(self, original_image_path, ocr_results):
        """Generate a digitalized form image.

        Output style:
        - White background
        - Dark template lines/text (print-like)
        - Handwritten values replaced by typed text in entry zones
        """
        try:
            img = cv2.imread(original_image_path)
            if img is None:
                print(f"[OCR] Failed to load image: {original_image_path}")
                return None

            h, w = img.shape[:2]
            digital = self._build_digital_form_template(img)
            field_boxes = self._detect_form_field_rects(img)

            # Filter to likely handwritten entries instead of overlaying every OCR word.
            entry_words = self._select_form_entry_words(ocr_results, field_boxes=field_boxes)
            if not entry_words:
                print("[OCR] No entry-like words detected; returning clean digital template")

            for word in entry_words:
                box = word.get('box')
                text = str(word.get('text', '') or '').strip()
                if not box or not text:
                    continue

                # Convert normalized coordinates to pixel coordinates
                x0 = max(0, int(box[0][0] * w))
                y0 = max(0, int(box[0][1] * h))
                x1 = min(w - 1, int(box[1][0] * w))
                y1 = min(h - 1, int(box[1][1] * h))
                if x1 <= x0 or y1 <= y0:
                    continue

                # Remove pen strokes near the detected word before typing replacement.
                pad_x = max(2, int((x1 - x0) * 0.08))
                pad_y = max(2, int((y1 - y0) * 0.25))
                rx0 = max(0, x0 - pad_x)
                ry0 = max(0, y0 - pad_y)
                rx1 = min(w - 1, x1 + pad_x)
                ry1 = min(h - 1, y1 + pad_y)
                digital[ry0:ry1, rx0:rx1] = 255

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # adjust font_scale if it doesn't fit
                if text_width > (x1 - x0):
                    font_scale = font_scale * (x1 - x0) / text_width
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                text_x = x0
                text_y = max(y0 + text_height, y1 - baseline)

                # Draw dark typed text for clean digital look.
                cv2.putText(digital, text, (text_x, text_y), font, font_scale, 0, thickness, cv2.LINE_AA)

            # Convert grayscale digital form to RGB before saving.
            digital_bgr = cv2.cvtColor(digital, cv2.COLOR_GRAY2BGR)

            filled_path = original_image_path.replace('.jpg', '_filled.jpg').replace('.png', '_filled.png')
            if cv2.imwrite(filled_path, digital_bgr):
                print(f"[OCR] Filled form saved to: {filled_path}")
                return filled_path
            else:
                print("[OCR] Failed to save filled form image")
                return None
        except Exception as e:
            print(f"[OCR] Error generating filled form: {e}")
            return None

    def _looks_like_form_template(self, ocr_results):
        """Detect if OCR output appears to be a fixed form template with fillable fields."""
        if not ocr_results:
            return False

        text_blob = " ".join(str(w.get("text", "") or "") for w in ocr_results).lower()
        template_tokens = [
            "application", "form", "name", "student", "id", "date", "birth", "phone",
            "email", "address", "city", "state", "postal", "zip", "yes", "no",
            "signature", "dob", "mm/dd/yyyy"
        ]
        token_hits = sum(1 for tok in template_tokens if tok in text_blob)

        # field-like markers commonly present in blank forms
        field_markers = 0
        for item in ocr_results:
            txt = str(item.get("text", "") or "").strip().lower()
            if not txt:
                continue
            if txt in {"yes", "no"}:
                field_markers += 1
            if txt.endswith(":"):
                field_markers += 1
            if "mm/dd/yyyy" in txt or "zip" in txt:
                field_markers += 1

        return token_hits >= 4 or field_markers >= 4

    def _build_digital_form_template(self, img):
        """Create clean monochrome template from colored/scanned form image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Flatten illumination/background texture.
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Keep text/lines dark and remove soft background graphics.
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )

        # Light morphological cleanup for crisp digital appearance.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        return clean

    def _detect_form_field_rects(self, img):
        """Detect likely form input rectangles in normalized coordinates."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        rects = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            x, y, rw, rh = cv2.boundingRect(approx)
            if rw < 40 or rh < 16:
                continue
            aspect = rw / float(max(1, rh))
            if 1.2 <= aspect <= 40:
                rects.append((x / w, y / h, (x + rw) / w, (y + rh) / h))
        return rects

    def _select_form_entry_words(self, ocr_results, field_boxes=None):
        """Pick words that likely belong to filled blanks, not printed labels.

        Heuristics:
        - Exclude known printed form/header vocabulary.
        - Prefer words with lower OCR confidence (common for handwriting).
        - Prefer words appearing after a label separator (like ':' in same line).
        """
        if not ocr_results:
            return []
        if field_boxes is None:
            field_boxes = []

        form_label_tokens = {
            "name", "date", "address", "phone", "email", "gender", "age", "dob",
            "city", "state", "country", "zip", "pincode", "department", "designation",
            "amount", "total", "description", "qty", "quantity", "price", "signature",
            "patient", "doctor", "hospital", "bill", "invoice", "receipt", "form"
        }

        # Group words into approximate lines using ymin proximity.
        lines = {}
        for item in ocr_results:
            box = item.get("box")
            if not box:
                continue
            y = float(box[0][1])
            key = None
            for existing in lines:
                if abs(existing - y) < 0.015:
                    key = existing
                    break
            if key is None:
                lines[y] = [item]
            else:
                lines[key].append(item)

        selected = []

        for _, row in sorted(lines.items(), key=lambda x: x[0]):
            row_sorted = sorted(row, key=lambda x: float(x.get("box", ((0, 0), (0, 0)))[0][0]))

            label_split_index = -1
            for i, token in enumerate(row_sorted):
                txt = str(token.get("text", "") or "").strip()
                if txt.endswith(":") or txt == ":":
                    label_split_index = i
                    break

            for i, token in enumerate(row_sorted):
                txt = str(token.get("text", "") or "").strip()
                if not txt:
                    continue

                normalized = re.sub(r"[^a-zA-Z0-9]", "", txt).lower()
                conf = float(token.get("confidence", 1.0) or 1.0)

                # Ignore common printed labels.
                if normalized in form_label_tokens:
                    continue

                # Ignore purely decorative separators.
                if set(txt) <= {"_", "-", "/", ".", ":", "|"}:
                    continue

                # Accept probable handwritten entries.
                after_label = label_split_index >= 0 and i > label_split_index
                low_confidence = conf < 0.9
                mixed_token = any(c.isdigit() for c in txt) or any(c.islower() for c in txt)
                cx = (float(token["box"][0][0]) + float(token["box"][1][0])) / 2.0
                cy = (float(token["box"][0][1]) + float(token["box"][1][1])) / 2.0
                in_field_box = any((bx0 <= cx <= bx1 and by0 <= cy <= by1) for (bx0, by0, bx1, by1) in field_boxes)

                if after_label or (low_confidence and mixed_token) or (in_field_box and mixed_token):
                    selected.append(token)

        return selected

if __name__ == "__main__":
    print("Enhanced OCR Engine Ready.")

