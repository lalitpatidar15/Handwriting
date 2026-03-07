from typing import List, Dict, Any, Tuple, cast
from medical_ai import MedicalIntelligence
from spell_corrector import GeneralSpellCorrector

class UniversalDataIntelligence:
    """
    Vanguard Brain v5.0: DocTR Optimized Intelligence.
    Uses normalized coordinate clustering for perfect line reconstruction.
    """
    
    def __init__(self):
        self.medical_ai = MedicalIntelligence()
        self.spell_corrector = GeneralSpellCorrector()
        self.med_dict = [
            "PATIENT", "HOSPITAL", "DIAGNOSIS", "KAILASH", "BLOOD", "REQUEST", "SAMPLE",
            "CLINICAL", "TRANSFUSION", "PLASMA", "THERAPY", "COVID", "POSITIVE", "NEGATIVE",
            "DOCTOR", "DATE", "AGE", "GENDER", "COMPONENT", "QUANTITY", "RESULT",
            "MEDICOS", "CHEMIST", "COSMETICS", "TABLET", "SYRUP", "INJECTION", "AMOUNT",
            "DESCRIPTION", "BATCH", "EXPIRAL", "PRICE", "TOTAL", "CASH", "MEMO"
        ]

    def fuzzy_correct(self, text: str) -> str:
        """
        Uses MedicalIntelligence for proper fuzzy matching.
        """
        if not text:
            return text
        # Use the medical AI for proper fuzzy matching
        corrected = self.medical_ai.correct_medication(text)
        return corrected
    
    def _fix_merged_words(self, text: str) -> str:
        """
        Try to fix common OCR issues where words are merged together.
        Example: "tellyou" -> "tell you"
        """
        if not text or len(text) < 4:
            return text
        
        # Common word patterns that might be merged
        common_patterns = [
            ("tellyou", "tell you"), ("youcan", "you can"), ("cando", "can do"),
            ("dont", "don't"), ("wont", "won't"), ("cant", "can't"),
            ("isnt", "isn't"), ("arent", "aren't"), ("wasnt", "wasn't"),
            ("werent", "weren't"), ("havent", "haven't"), ("hasnt", "hasn't"),
            ("hadnt", "hadn't"), ("wouldnt", "wouldn't"), ("shouldnt", "shouldn't"),
            ("couldnt", "couldn't"), ("mustnt", "mustn't"), ("didnt", "didn't"),
            ("doesnt", "doesn't"), ("thats", "that's"), ("its", "it's"),
            ("hes", "he's"), ("shes", "she's"), ("youre", "you're"),
            ("were", "we're"), ("theyre", "they're"), ("im", "i'm"),
            ("heres", "here's"), ("theres", "there's"), ("whats", "what's"),
            ("whos", "who's"), ("wheres", "where's"), ("hows", "how's"),
            ("whys", "why's"), ("whens", "when's"), ("lets", "let's"),
            ("got", "got to"), ("gotto", "got to"), ("gonna", "going to"),
            ("wanna", "want to"), ("gotta", "got to"), ("hafta", "have to"),
            ("oughta", "ought to"), ("useta", "used to"), ("supposta", "supposed to"),
            ("sposta", "supposed to"), ("outta", "out of"), ("kinda", "kind of"),
            ("sorta", "sort of"), ("lotta", "lot of"), ("lotsa", "lots of"),
            ("lemme", "let me"), ("gimme", "give me"), ("tellme", "tell me"),
            ("showme", "show me"), ("giveme", "give me"), ("letme", "let me"),
        ]
        
        text_lower = text.lower()
        for merged, fixed in common_patterns:
            if merged in text_lower:
                # Replace with proper spacing, preserving case
                if text_lower == merged:
                    return fixed
                # Replace in context
                text = text.replace(merged, fixed)
                text_lower = text.lower()
        
        return text

    def parse_universal(self, ocr_results: List[Any], force_mode: str = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[List[Any]]]:
        """
        DocTR Hierarchy Reconstruction (v5.0)
        """
        if not ocr_results: return {"type": "empty"}, {}, []

        # 1. Bucket Grouping (Fixed-Baseline Clustering)
        # DocTR geometry: ((xmin, ymin), (xmax, ymax))
        lines_dict = {}
        
        # Calculate average word height in normalized coords: ymax - ymin
        heights = [(b['box'][1][1] - b['box'][0][1]) for b in ocr_results if 'box' in b]
        avg_h = sum(heights) / len(heights) if heights else 0.01
        
        # Also calculate average word width for spacing detection
        widths = [(b['box'][1][0] - b['box'][0][0]) for b in ocr_results if 'box' in b]
        avg_w = sum(widths) / len(widths) if widths else 0.01

        for item in ocr_results:
            if 'box' not in item:
                continue
            # Group by Y-coordinate (ymin)
            y_val = item['box'][0][1] # ymin
            
            found_line_key = None
            for existing_y in lines_dict.keys():
                # Improved threshold: use 60% of average height for better line grouping
                if abs(y_val - existing_y) < (avg_h * 0.6):
                    found_line_key = existing_y
                    break
            
            if found_line_key is not None:
                lines_dict[found_line_key].append(item)
            else:
                lines_dict[y_val] = [item]

        # 2. Sort lines top-to-bottom and words left-to-right
        sorted_y_keys = sorted(lines_dict.keys())
        rows = [sorted(lines_dict[y], key=lambda x: x['box'][0][0]) for y in sorted_y_keys] # Sort by xmin

        # 3. Intelligent Decision: Table vs Paragraph
        all_text_upper = " ".join([str(b.get('text', '')).upper() for b in ocr_results])
        table_anchors = ["QTY", "DESCRIPTION", "HSN", "BATCH", "EXP", "AMOUNT", "PARTICULARS", "PRICE", "TOTAL"]
        force_table = any(anchor in all_text_upper for anchor in table_anchors)

        is_table = force_table
        if force_mode == "HANDWRITTEN_NOTE": is_table = False
        elif force_mode == "STRUCTURED_FORM": is_table = True

        doc_analysis = {
            "type": "STRUCTURED_FORM" if is_table else "HANDWRITTEN_NOTE",
            "confidence_score": 0.98 if is_table else 0.95,
            "stats": {"rows": len(rows), "avg_cols": round(len(ocr_results)/len(rows), 1) if rows else 0}
        }

        if not is_table:
            # --- PARAGRAPH MODE (CLEAN RECONSTRUCTION) ---
            full_content = []
            for row in rows:
                if not row:
                    continue
                line_parts: List[str] = []
                for i, item in enumerate(row):
                    if 'box' not in item:
                        continue
                    item_text = str(item.get('text', '')).strip()
                    
                    # Apply spelling correction for all handwritten text
                    if item_text:
                        # First fix merged words
                        item_text = self._fix_merged_words(item_text)
                        # Then use general spell corrector for handwritten notes
                        text = self.spell_corrector.correct(item_text)
                    else:
                        text = item_text
                    
                    if i > 0 and 'box' in row[i-1]:
                        prev_item = row[i-1]
                        # Improved spacing detection: use average word width as reference
                        # Calculate gap between previous word end and current word start
                        prev_xmax = prev_item['box'][1][0]
                        curr_xmin = item['box'][0][0]
                        visual_gap = curr_xmin - prev_xmax
                        
                        # Better threshold: use 30% of average word width for spacing
                        # This handles handwriting better where spacing can vary
                        spacing_threshold = avg_w * 0.3 if avg_w > 0 else 0.02
                        
                        if visual_gap > spacing_threshold:
                            line_parts.append(" " + text)
                        else:
                            # Very small gap - words are close together, add space anyway for readability
                            # unless they're clearly touching (negative gap)
                            if visual_gap > -0.005:  # Almost touching but not overlapping
                                line_parts.append(" " + text)
                            else:
                                line_parts.append(text)
                    else:
                        line_parts.append(text)
                
                if line_parts:
                    full_content.append("".join(line_parts))
            
            return doc_analysis, {"full_text": "\n".join(full_content)}, rows
        
        else:
            # --- STRUCTURED MODE (TABLE MAPPING) ---
            # FIXED: Proper table extraction logic
            kv_pairs = {}
            table_data = []
            
            # Detect table columns from first row
            if rows and len(rows) > 1:
                first_row = rows[0]
                first_row_text = " ".join([b['text'] for b in first_row]).upper()
                
                # Check if first row looks like headers
                header_anchors = ["QTY", "DESCRIPTION", "PRICE", "AMOUNT", "TOTAL", "HSN", "BATCH", "EXP", "PARTICULARS"]
                is_header_row = any(anchor in first_row_text for anchor in header_anchors)
                
                if is_header_row:
                    # Process data rows (skip header)
                    data_rows = rows[1:]
                    for row in data_rows:
                        row_text = " ".join([b['text'] for b in row]).strip()
                        if row_text:  # Only add non-empty rows
                            # Try to parse as structured data
                            parts = row_text.split()
                            if len(parts) >= 2:
                                # Extract qty (first number) and rest as item
                                row_dict = {"description": row_text}
                                # Try to find quantity
                                for word in parts:
                                    if word.isdigit():
                                        row_dict["qty"] = word
                                        break
                                table_data.append(row_dict)
                            else:
                                table_data.append({"description": row_text})
                else:
                    # No clear header, treat all rows as data
                    for row in rows:
                        row_text = " ".join([b['text'] for b in row]).strip()
                        if row_text:
                            table_data.append({"description": row_text})
            
            # Also extract key-value pairs (e.g., "Total: 500")
            for row in rows:
                row_text = " ".join([b['text'] for b in row]).strip()
                if ":" in row_text:
                    parts = row_text.split(":", 1)
                    kv_pairs[parts[0].strip()] = self.fuzzy_correct(parts[1].strip())
            
            return doc_analysis, {"kv": kv_pairs, "table": table_data}, rows

    def _fix_merged_words(self, text: str) -> str:
        """
        Heuristic to split words that might be merged by OCR.
        Example: 'Dontlet' -> 'Dont let'
        """
        if not text: return text
        
        # Simple heuristic: if a capital letter is in the middle of a word
        # (excluding common medical abbreviations)
        # Note: This is a basic starter implementation.
        import re
        # Splits 'WordWord' into 'Word Word'
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

if __name__ == "__main__":
    print("Vanguard v5.3 Intelligence Engine Ready.")
